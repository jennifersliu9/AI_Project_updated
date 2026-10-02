"""Command-line entry: ingest, ask, eval, tool, agent, mcp-probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace

from harborline.agent import run_agent
from harborline.answer import ask
from harborline.config import RETRIEVE_BACKENDS, get_settings
from harborline.benchmark import format_report, run_report
from harborline.evaluate import run_eval
from harborline.ingest import load_chunks, write_index
from harborline.mcp_client import open_mcp_bus
from harborline.mcp_server import MCP_TOOL_NAMES
from harborline.store import persist_chunks
from harborline.tools import (
    check_policy_compliance,
    check_pto_balance,
    create_mock_hr_ticket,
    draft_hr_email,
    get_policy_section,
    lookup_benefits_status,
    lookup_employee_profile,
    search_policy_documents,
)


def _optional_retriever(backend: str | None, settings):
    from harborline.retrieve import build_retriever

    chosen = (backend or settings.retrieve_backend).lower()
    return build_retriever(replace(settings, retrieve_backend=chosen))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harborline policy Q&A")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "ingest",
        help="Parse corpus, chunk, embed with the OpenAI API, and cache the cosine index",
    )

    ask_p = sub.add_parser("ask", help="Retrieve (and optionally generate) an answer")
    ask_p.add_argument("query", help="Employee question")
    ask_p.add_argument("--employee-id", dest="employee_id", default=None)
    ask_p.add_argument("--kind", default=None, help="Filter: policy or structured")
    ask_p.add_argument("--source-format", dest="source_format", default=None)
    ask_p.add_argument("--json", action="store_true")

    eval_p = sub.add_parser("eval", help="Run seeded retrieval evaluation")
    eval_p.add_argument("--limit", type=int, default=None, help="Sample N gold items")
    eval_p.add_argument("--json", action="store_true")

    report_p = sub.add_parser(
        "report",
        help="Run the 20-30 task eval: quality, agent behavior, latency, ablation",
    )
    report_p.add_argument("--limit", type=int, default=None, help="Sample N gold tasks")
    report_p.add_argument("--json", action="store_true")
    report_p.add_argument(
        "--write",
        action="store_true",
        help="Write eval/latest_report.json and eval/REPORT.md",
    )
    report_p.add_argument("--backend", choices=list(RETRIEVE_BACKENDS), default=None)

    tool_p = sub.add_parser(
        "tool",
        help="Call a tool implementation in-process (debugging only; the agent uses MCP)",
    )
    tool_p.add_argument("name", choices=list(MCP_TOOL_NAMES))
    tool_p.add_argument("query_or_id", nargs="?", default="", help="Query, policy id, or employee id")
    tool_p.add_argument("--employee-id", dest="employee_id", default=None)
    tool_p.add_argument("--kind", default=None)
    tool_p.add_argument("--section", default="")
    tool_p.add_argument("--topic", default="general")
    tool_p.add_argument("--summary", default="")
    tool_p.add_argument("--subject", default="Harborline HR note")
    tool_p.add_argument("--body", default="")
    tool_p.add_argument("--policy-id", dest="policy_id", default="")
    tool_p.add_argument("--confirm", action="store_true", help="Accept a MOCK write in this process")
    tool_p.add_argument("--backend", choices=list(RETRIEVE_BACKENDS), default=None)
    tool_p.add_argument("--json", action="store_true")

    agent_p = sub.add_parser("agent", help="Run the HR orchestrator through the MCP tool layer")
    agent_p.add_argument("query", help="Employee or HR request")
    agent_p.add_argument("--employee-id", dest="employee_id", default=None)
    agent_p.add_argument("--confirm", action="store_true", help="Accept MOCK irreversible actions")
    agent_p.add_argument("--backend", choices=list(RETRIEVE_BACKENDS), default=None)
    agent_p.add_argument(
        "--transport",
        choices=["mcp-stdio", "mcp-inproc"],
        default="mcp-stdio",
        help="mcp-stdio spawns python -m harborline.mcp_server (default)",
    )
    agent_p.add_argument("--json", action="store_true")

    probe_p = sub.add_parser("mcp-probe", help="Discover MCP tools over stdio or in-process FastMCP")
    probe_p.add_argument("--transport", choices=["mcp-stdio", "mcp-inproc"], default="mcp-stdio")
    probe_p.add_argument("--backend", choices=list(RETRIEVE_BACKENDS), default=None)

    args = parser.parse_args(argv)
    settings = get_settings()
    settings.apply_seeds()

    if args.command == "ingest":
        chunks = load_chunks(settings)
        json_path = write_index(settings)
        formats = Counter(c.source_format for c in chunks)
        print(f"Parsed and chunked {len(chunks)} records")
        print("  formats:", dict(formats))
        print(f"  json index: {json_path}")
        if settings.retrieve_backend == "tfidf":
            print("  vector index: skipped (tfidf backend does not call the embeddings API)")
            return 0
        stored = persist_chunks(chunks, settings)
        print(
            f"  vector index: OpenAI {settings.embedding_model} "
            f"({stored} embedded chunks, cached in {settings.cache_dir})"
        )
        print(f"  embedding: OpenAIEmbeddings {settings.embedding_model} (cloud API)")
        return 0

    if args.command == "ask":
        result = ask(
            args.query,
            employee_id=args.employee_id,
            kind=args.kind,
            source_format=args.source_format,
            settings=settings,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["rewritten_query"] != args.query:
                print(f"Rewritten query: {result['rewritten_query']}\n")
            print(result["answer"])
            if result["sources"]:
                print("\nCitations:")
                for src in result["sources"]:
                    print(
                        f"  [{src['n']}] {src['title']} | {src['section']}\n"
                        f"    {src['source_path']} ({src['kind']}, score={src['score']})\n"
                        f"    {src['snippet']}"
                    )
        return 0

    if args.command == "tool":
        retriever = _optional_retriever(args.backend, settings)
        name = args.name
        if name == "search_policy_documents":
            payload = search_policy_documents(
                args.query_or_id,
                employee_id=args.employee_id,
                kind=args.kind,
                retriever=retriever,
            )
        elif name == "get_policy_section":
            payload = get_policy_section(
                args.query_or_id or args.policy_id,
                section=args.section,
                retriever=retriever,
            )
        elif name == "lookup_employee_profile":
            payload = lookup_employee_profile(args.query_or_id or args.employee_id or "")
        elif name == "check_pto_balance":
            payload = check_pto_balance(args.query_or_id or args.employee_id or "")
        elif name == "lookup_benefits_status":
            payload = lookup_benefits_status(args.query_or_id or args.employee_id or "")
        elif name == "create_mock_hr_ticket":
            payload = create_mock_hr_ticket(
                topic=args.topic,
                summary=args.summary or args.query_or_id,
                employee_id=args.employee_id,
                confirm=args.confirm,
            )
        elif name == "draft_hr_email":
            payload = draft_hr_email(
                args.employee_id or args.query_or_id,
                args.subject,
                args.body or args.summary or args.query_or_id,
                confirm=args.confirm,
            )
        else:
            payload = check_policy_compliance(
                args.query_or_id,
                employee_id=args.employee_id,
                policy_id=args.policy_id,
                retriever=retriever,
            )
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload.get("ok", True) and "error" not in payload else 1

    if args.command == "mcp-probe":
        extra = {}
        if args.backend:
            extra["HARBORLINE_RETRIEVE_BACKEND"] = args.backend
        bus = open_mcp_bus(args.transport, extra_env=extra or None)
        try:
            report = {
                "transport": bus.transport,
                "available": bus.available,
                "discovered_tools": bus.discovered_tools,
                "schemas": getattr(bus, "tool_schemas", []),
            }
            print(json.dumps(report, indent=2, default=str))
            return 0 if bus.available and len(bus.discovered_tools) >= 5 else 1
        finally:
            bus.close()

    if args.command == "agent":
        extra = {}
        if args.backend:
            extra["HARBORLINE_RETRIEVE_BACKEND"] = args.backend
            os.environ["HARBORLINE_RETRIEVE_BACKEND"] = args.backend
            get_settings.cache_clear()
        result = run_agent(
            args.query,
            employee_id=args.employee_id,
            confirm=args.confirm,
            transport=args.transport,
            extra_env=extra or None,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print(result.format_trace())
            print()
            print(result.answer)
        return 0 if not result.needs_clarification else 2

    if args.command == "report":
        if getattr(args, "backend", None):
            os.environ["HARBORLINE_RETRIEVE_BACKEND"] = args.backend
            get_settings.cache_clear()
        settings = get_settings()
        report = run_report(settings=settings, limit=args.limit)
        if args.write:
            out_json = settings.root / "eval" / "latest_report.json"
            out_md = settings.root / "eval" / "REPORT.md"
            out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            out_md.write_text(format_report(report) + "\n", encoding="utf-8")
        text = json.dumps(report, indent=2, default=str) if args.json else format_report(report)
        try:
            print(text)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")
        return 0

    report = run_eval(settings=settings, limit=args.limit)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"backend={settings.retrieve_backend}  seed={report['seed']}  n={report['n']}  "
            f"passed={report['passed']}  recall@{report['top_k']}={report['recall_at_k']}"
        )
        for row in report["results"]:
            mark = "PASS" if row["pass"] else "FAIL"
            print(f"  [{mark}] {row['id']}: {row['question']}")
    return 0 if report["passed"] == report["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
