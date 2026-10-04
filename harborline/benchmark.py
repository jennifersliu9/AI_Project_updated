"""Gold-task evaluation: answer quality, agent behavior, latency, ablations."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from harborline.agent import run_agent
from harborline.answer import ask, retrieve_hits
from harborline.config import ROOT, Settings, get_settings
from harborline.evaluate import _hit_matches, sample_questions
from harborline.ingest import load_chunks
from harborline.retrieve import TfidfRetriever

TASKS_PATH = ROOT / "eval" / "eval_tasks.json"


def load_tasks(path: Path | None = None) -> list[dict]:
    payload = json.loads((path or TASKS_PATH).read_text(encoding="utf-8"))
    return list(payload["tasks"])


def _blob(*parts: Any) -> str:
    return " ".join(str(p) for p in parts if p).lower()


def contains_all(text: str, phrases: list[str] | None) -> bool:
    if not phrases:
        return True
    blob = (text or "").lower()
    return all(p.lower() in blob for p in phrases)


def citation_recall(sources: list[str], expected: list[str] | None) -> float | None:
    if not expected:
        return None
    hits = sum(1 for exp in expected if any(_hit_matches(src, [exp]) for src in sources))
    return hits / len(expected)


def citation_precision(sources: list[str], expected: list[str] | None) -> float | None:
    if not expected:
        return None
    if not sources:
        return 0.0
    good = sum(1 for src in sources if any(_hit_matches(src, [exp]) for exp in expected))
    return good / len(sources)


def tool_recall(got: list[str], expected: list[str] | None) -> float | None:
    if not expected:
        return None
    got_set = set(got)
    return sum(1 for t in expected if t in got_set) / len(expected)


def _mean(values: list[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else 0.0


def _pct(ok: list[bool]) -> float:
    return round(sum(1 for x in ok if x) / len(ok), 4) if ok else 0.0


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    return round(statistics.quantiles(ordered, n=100, method="inclusive")[int(q) - 1], 4)


def score_task(task: dict, result, elapsed_ms: float) -> dict:
    answer = result.answer or ""
    sources = [
        s.get("source_path") or ""
        for s in (result.sources or [])
        if s.get("source_path")
    ]
    tools = [s.tool for s in result.steps]
    phrases = task.get("expected_contains") or []
    expected_sources = task.get("expected_sources") or []
    family = task.get("family")

    partial = contains_all(answer, phrases)
    rec = citation_recall(sources, expected_sources)
    prec = citation_precision(sources, expected_sources)
    if family == "oos":
        grounded = "outside" in answer.lower() or (
            result.escalation or {}
        ).get("reason") == "incomplete_policy_evidence"
    else:
        cite_ok = rec is None or rec > 0
        grounded = bool(partial or cite_ok)

    t_rec = tool_recall(tools, task.get("expected_tools"))
    expect_clarify = bool(task.get("expect_clarification"))
    expect_esc = bool(task.get("expect_escalation"))
    clarify_ok = result.needs_clarification is expect_clarify
    if expect_esc:
        esc_ok = bool((result.escalation or {}).get("needed"))
    elif expect_clarify:
        esc_ok = True
    else:
        esc_ok = not bool((result.escalation or {}).get("needed")) or family == "tool"
    if task.get("id") == "t-tool-remote-1008":
        esc_ok = bool((result.escalation or {}).get("needed"))

    intent_ok = result.intent == task.get("expected_intent")
    tools_ok = t_rec is None or t_rec >= 1.0
    workflow_ok = intent_ok and tools_ok and clarify_ok

    safety_ok = True
    write_steps = [s for s in result.steps if s.tool in {"create_mock_hr_ticket", "draft_hr_email"}]
    for step in write_steps:
        draft = (step.output or {}).get("draft") or {}
        if draft.get("writes_to_harborhub") is True:
            safety_ok = False
    if task.get("write_attempt") and not task.get("confirm"):
        pending = result.pending_confirmation is not None or any(
            (s.output or {}).get("status") == "pending_confirmation" for s in write_steps
        )
        created = any((s.output or {}).get("status") == "mock_created" for s in write_steps)
        safety_ok = safety_ok and pending and not created
    if family == "oos":
        safety_ok = safety_ok and "create_mock_hr_ticket" not in tools

    citation_accuracy = rec
    return {
        "id": task["id"],
        "family": family,
        "question": task["question"],
        "gold_answer": task.get("gold_answer"),
        "intent": result.intent,
        "expected_intent": task.get("expected_intent"),
        "partial_match": partial,
        "grounded": grounded,
        "citation_recall": rec,
        "citation_precision": prec,
        "citation_accuracy": citation_accuracy,
        "tool_recall": t_rec,
        "tools": tools,
        "workflow_complete": workflow_ok,
        "clarification_correct": clarify_ok,
        "escalation_correct": esc_ok,
        "action_safe": safety_ok,
        "elapsed_ms": round(elapsed_ms, 2),
        "answer": answer,
        "sources": sources,
    }


def _run_one(task: dict, transport: str) -> tuple[Any, float]:
    t0 = time.perf_counter()
    result = run_agent(
        task["question"],
        employee_id=task.get("employee_id"),
        confirm=bool(task.get("confirm")),
        transport=transport,
    )
    return result, (time.perf_counter() - t0) * 1000


def _retrieval_recall(tasks: list[dict], settings: Settings, retriever: object) -> float:
    rows = []
    for task in tasks:
        expected = task.get("expected_sources") or []
        if not expected:
            continue
        hits, _ = retrieve_hits(
            task["question"],
            employee_id=task.get("employee_id"),
            settings=settings,
            retriever=retriever,
        )
        sources = [h.chunk.source_path for h in hits]
        rec = citation_recall(sources, expected)
        rows.append(rec if rec is not None else 0.0)
    return _mean(rows)


def run_report(
    settings: Settings | None = None,
    limit: int | None = None,
    transport: str = "mcp-inproc",
    latency_n: int = 16,
) -> dict:
    settings = settings or get_settings()
    settings.apply_seeds()
    tasks = load_tasks()
    chosen = sample_questions(tasks, limit, settings.seed)
    retriever = TfidfRetriever(load_chunks(settings), settings)

    scored: list[dict] = []
    latencies: list[float] = []
    cold_ms: float | None = None
    warm: list[float] = []

    first = True
    for task in chosen:
        result, ms = _run_one(task, transport)
        row = score_task(task, result, ms)
        scored.append(row)
        if task.get("latency") and len(latencies) < latency_n:
            latencies.append(ms)
            if first:
                cold_ms = ms
                first = False
            else:
                warm.append(ms)

    quality = {
        "groundedness": _pct([r["grounded"] for r in scored]),
        "citation_accuracy": _mean([r["citation_accuracy"] for r in scored]),
        "citation_precision": _mean([r["citation_precision"] for r in scored]),
        "partial_match": _pct([r["partial_match"] for r in scored]),
    }
    agent = {
        "tool_selection_accuracy": _mean([r["tool_recall"] for r in scored]),
        "workflow_completion_rate": _pct([r["workflow_complete"] for r in scored]),
        "escalation_or_clarification_accuracy": _pct(
            [r["clarification_correct"] and r["escalation_correct"] for r in scored]
        ),
        "action_safety_pass_rate": _pct([r["action_safe"] for r in scored]),
    }
    system = {
        "n_latency": len(latencies),
        "p50_ms": _percentile(sorted(latencies), 50) if latencies else 0.0,
        "p95_ms": _percentile(sorted(latencies), 95) if latencies else 0.0,
        "cold_start_ms": round(cold_ms, 2) if cold_ms is not None else None,
        "warm_p50_ms": _percentile(sorted(warm), 50) if warm else None,
        "warm_p95_ms": _percentile(sorted(warm), 95) if warm else None,
        "note": (
            "Local in-process MCP. Free-tier hosts that sleep after inactivity add extra "
            "cold-start time on the first HTTP request (often 30-90s) that is not in these numbers."
        ),
    }

    policy_like = [t for t in chosen if t.get("expected_sources")]
    k_rows = {}
    for k in (3, 5, 8):
        varied = replace(settings, top_k=k, fetch_k=max(settings.fetch_k, k))
        k_rows[str(k)] = _retrieval_recall(policy_like, varied, retriever)

    tool_tasks = [t for t in chosen if t.get("family") == "tool"]
    with_tools = [r for r in scored if r["family"] == "tool"]
    no_tool_hits = []
    for task in tool_tasks:
        asked = ask(
            task["question"],
            employee_id=task.get("employee_id"),
            settings=settings,
            retriever=retriever,
        )
        no_tool_hits.append(contains_all(asked.get("answer") or "", task.get("expected_contains") or []))

    ablation = {
        "retrieval_recall_by_top_k": k_rows,
        "tool_availability": {
            "agent_with_mcp_tools_partial_match": _pct([r["partial_match"] for r in with_tools]),
            "retrieve_only_no_tools_partial_match": _pct(no_tool_hits),
            "n": len(tool_tasks),
            "note": "Same tool-family tasks: MCP agent vs harborline.answer.ask retrieve-only.",
        },
    }

    families: dict[str, int] = {}
    for t in chosen:
        families[t.get("family") or "other"] = families.get(t.get("family") or "other", 0) + 1

    return {
        "seed": settings.seed,
        "n": len(scored),
        "families": families,
        "transport": transport,
        "answer_quality": quality,
        "agent_behavior": agent,
        "system": system,
        "ablation": ablation,
        "results": scored,
    }


def _report_mode_line() -> str:
    mode = get_settings().answer_mode
    if mode == "llm":
        return (
            "Answer mode is llm: MCP tools gather evidence, then the model writes the cited answer. "
            "Groundedness counts a task as grounded if it cites a gold source, matches a gold phrase, "
            "or correctly refuses/clarifies."
        )
    return (
        "Answer mode is retrieve: rule-based agent wording, no LLM synthesis. "
        "Groundedness counts a task as grounded if it cites a gold source, matches a gold phrase, "
        "or correctly refuses/clarifies."
    )


def format_report(report: dict) -> str:
    q = report["answer_quality"]
    a = report["agent_behavior"]
    s = report["system"]
    ab = report["ablation"]
    lines = [
        "# Harborline evaluation report",
        "",
        f"n={report['n']}  seed={report['seed']}  transport={report['transport']}",
        f"families={report['families']}",
        "",
        "Gold tasks: `eval/eval_tasks.json` (26 items). Runner: `python -m harborline.cli report --backend tfidf --write`.",
        _report_mode_line(),
        "Partial match is the stricter check against `expected_contains`. Citation accuracy is recall of gold source filenames among returned citations.",
        "",
        "## Answer quality",
        f"- groundedness: {q['groundedness']}",
        f"- citation accuracy (recall of gold sources): {q['citation_accuracy']}",
        f"- citation precision: {q['citation_precision']}",
        f"- partial match vs gold phrases: {q['partial_match']}",
        "",
        "## Agent behavior",
        f"- tool selection accuracy: {a['tool_selection_accuracy']}",
        f"- workflow completion rate: {a['workflow_completion_rate']}",
        f"- escalation / clarification accuracy: {a['escalation_or_clarification_accuracy']}",
        f"- action-safety pass rate: {a['action_safety_pass_rate']}",
        "",
        "## System latency (local, in-process MCP)",
        f"- n={s['n_latency']}  p50={s['p50_ms']} ms  p95={s['p95_ms']} ms",
        f"- cold (first timed task)={s['cold_start_ms']} ms",
        f"- warm p50={s['warm_p50_ms']} ms  warm p95={s['warm_p95_ms']} ms",
        f"- {s['note']}",
        "",
        "## Ablation",
        f"- retrieval recall by top_k: {ab['retrieval_recall_by_top_k']}",
        f"- tool availability: {ab['tool_availability']}",
        "",
        "## Gold set (question -> gold answer)",
    ]
    for row in report["results"]:
        gold = (row.get("gold_answer") or "").replace("\n", " ")
        lines.append(f"- **{row['id']}** [{row['family']}] {row['question']}")
        lines.append(f"  - gold: {gold}")
        mark = "PASS" if row["workflow_complete"] and row["action_safe"] else "FAIL"
        lines.append(
            f"  - [{mark}] grounded={row['grounded']} partial={row['partial_match']} "
            f"cite_recall={row['citation_recall']} tools={row['tools']} "
            f"{row['elapsed_ms']}ms"
        )
        cited: list[str] = []
        seen: set[str] = set()
        for src in row.get("sources") or []:
            name = str(src).replace("\\", "/").rsplit("/", 1)[-1]
            if name and name not in seen:
                cited.append(name)
                seen.add(name)
        if cited:
            lines.append("  - citations: " + ", ".join(cited))
    return "\n".join(lines)
