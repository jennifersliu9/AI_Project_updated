"""Tool implementations wrapped by the MCP server.

These functions are the implementation layer. The agent must not call them
directly during execution; it goes through harborline.mcp_client, which
discovers tools via MCP `tools/list` and invokes them via `tools/call`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harborline.answer import ask, retrieve_hits
from harborline.config import Settings, get_settings
from harborline.ingest import load_chunks

POLICY_ID_RE = re.compile(r"POL-[A-Z]+-\d{3}", re.I)
PRIMARY_POLICY_RE = re.compile(r"Policy ID:\*+\s*(POL-[A-Z]+-\d{3})\b", re.I)
MONEY_RE = re.compile(r"\$\s?(\d+(?:,\d{3})*(?:\.\d+)?)")

# Session-only mocks. Never written to data/*.json or a live HRIS.
_MOCK_TICKETS: list[dict] = []
_MOCK_EMAILS: list[dict] = []


def reset_mocks() -> None:
    _MOCK_TICKETS.clear()
    _MOCK_EMAILS.clear()


def _load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(name: str, settings: Settings) -> list[dict]:
    return _load_dataset(settings.data_dir / name)["records"]


def _hit_dicts(hits) -> list[dict]:
    return [
        {
            "chunk_id": h.chunk.chunk_id,
            "title": h.chunk.title,
            "section": h.chunk.section,
            "source_path": h.chunk.source_path,
            "snippet": h.chunk.snippet,
            "kind": h.chunk.kind,
            "score": round(h.score, 4),
        }
        for h in hits
    ]


def search_policy_documents(
    query: str,
    employee_id: str | None = None,
    kind: str | None = None,
    top_k: int | None = None,
    settings: Settings | None = None,
    retriever: object | None = None,
) -> dict:
    """Retrieve Harborline policy (and optional HarborHub) chunks from the RAG index."""
    settings = settings or get_settings()
    hits, rewritten = retrieve_hits(
        query,
        employee_id=employee_id,
        kind=kind,
        settings=settings,
        retriever=retriever,
    )
    if top_k is not None:
        hits = hits[:top_k]
    return {
        "tool": "search_policy_documents",
        "query": query,
        "rewritten_query": rewritten,
        "hits": _hit_dicts(hits),
    }


def _declared_policy_id(chunk) -> str | None:
    """Policy ID line in the document header, not a cross-reference in Related:."""
    match = PRIMARY_POLICY_RE.search(chunk.text[:1200])
    return match.group(1).upper() if match else None


def _section_payload(matched: list, pid: str, heading: str) -> dict:
    return {
        "tool": "get_policy_section",
        "found": True,
        "policy_id": pid,
        "section": heading or None,
        "sections": [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "section": c.section,
                "source_path": c.source_path,
                "snippet": c.snippet,
                "text": c.text[:1200],
            }
            for c in matched
        ],
    }


def get_policy_section(
    policy_id: str,
    section: str = "",
    settings: Settings | None = None,
    retriever: object | None = None,
) -> dict:
    """Return chunks from the document that declares this policy ID.

    Heading match uses the section title only. A mention of the same words in
    another policy (or in a Related: line) does not count, so repeated calls
    return the same sections.
    """
    settings = settings or get_settings()
    pid = (policy_id or "").strip().upper()
    heading = (section or "").strip()
    chunks = [c for c in load_chunks(settings) if c.kind == "policy"]
    owners: set[str] = set()
    if pid:
        seen_header: set[str] = set()
        for chunk in chunks:
            if chunk.source_path in seen_header:
                continue
            declared = _declared_policy_id(chunk)
            if declared is None:
                continue
            seen_header.add(chunk.source_path)
            if declared == pid:
                owners.add(chunk.source_path)
    owned = [c for c in chunks if c.source_path in owners] if owners else []
    if heading:
        headed = [c for c in owned if heading.lower() in (c.section or "").lower()]
    else:
        headed = []
    matched = (headed or owned)[:8]
    if matched:
        return _section_payload(matched, pid, heading)
    query = " ".join(part for part in (pid, heading) if part) or "Harborline policy"
    hits, rewritten = retrieve_hits(query, kind="policy", settings=settings, retriever=retriever)
    return {
        "tool": "get_policy_section",
        "found": bool(hits),
        "policy_id": pid,
        "section": heading or None,
        "fallback": "rag",
        "rewritten_query": rewritten,
        "sections": _hit_dicts(hits),
    }


def lookup_employee_profile(employee_id: str, settings: Settings | None = None) -> dict:
    """Look up a fictional HarborHub employee row (mock JSON)."""
    settings = settings or get_settings()
    eid = employee_id.strip().upper()
    employees = _records("employees.json", settings)
    person = next((r for r in employees if r.get("employee_id") == eid), None)
    if person is None:
        return {
            "tool": "lookup_employee_profile",
            "found": False,
            "employee_id": eid,
            "error": "No HarborHub profile for that id.",
        }
    manager = next((r for r in employees if r.get("employee_id") == person.get("manager_id")), None)
    return {
        "tool": "lookup_employee_profile",
        "found": True,
        "employee_id": eid,
        "employee": person,
        "manager": {
            "employee_id": manager.get("employee_id"),
            "legal_name": manager.get("legal_name"),
            "job_title": manager.get("job_title"),
            "email": manager.get("email"),
        }
        if manager
        else None,
    }


def check_pto_balance(employee_id: str, settings: Settings | None = None) -> dict:
    """Return the mock HarborHub PTO/sick/floating-holiday banks."""
    settings = settings or get_settings()
    eid = employee_id.strip().upper()
    row = next((r for r in _records("pto_balances.json", settings) if r.get("employee_id") == eid), None)
    if row is None:
        return {
            "tool": "check_pto_balance",
            "found": False,
            "employee_id": eid,
            "error": "No HarborHub PTO row for that id.",
        }
    return {
        "tool": "check_pto_balance",
        "found": True,
        "employee_id": eid,
        "pto": row,
    }


def lookup_benefits_status(employee_id: str, settings: Settings | None = None) -> dict:
    """Return the mock HarborHub benefits election row."""
    settings = settings or get_settings()
    eid = employee_id.strip().upper()
    row = next(
        (r for r in _records("benefits_elections.json", settings) if r.get("employee_id") == eid),
        None,
    )
    if row is None:
        return {
            "tool": "lookup_benefits_status",
            "found": False,
            "employee_id": eid,
            "error": "No HarborHub benefits election for that id.",
        }
    return {
        "tool": "lookup_benefits_status",
        "found": True,
        "employee_id": eid,
        "benefits": row,
    }


def create_mock_hr_ticket(
    topic: str,
    summary: str,
    employee_id: str | None = None,
    confirm: bool = False,
) -> dict:
    """Draft a HarborHub ticket. MOCK only; never writes tickets.json."""
    draft = {
        "id": f"MOCK-TCK-{len(_MOCK_TICKETS) + 1:04d}",
        "topic": topic,
        "employee_id": employee_id,
        "summary": summary,
        "persists": False,
        "writes_to_harborhub": False,
    }
    if not confirm:
        return {
            "tool": "create_mock_hr_ticket",
            "status": "pending_confirmation",
            "ok": True,
            "id": None,
            "draft": draft,
            "message": (
                "Irreversible ticket create is MOCK-only. "
                "Re-run with confirm=true to keep a session-only mock id. "
                "Nothing is written to tickets.json."
            ),
        }
    _MOCK_TICKETS.append(draft)
    return {
        "tool": "create_mock_hr_ticket",
        "status": "mock_created",
        "ok": True,
        "id": draft["id"],
        "draft": draft,
        "message": "MOCK created in this process only. Not a live HarborHub case.",
    }


def draft_hr_email(
    employee_id: str,
    subject: str,
    body: str,
    confirm: bool = False,
) -> dict:
    """Draft an HR email. Never sent; confirm only stores a session MOCK."""
    draft = {
        "id": f"MOCK-EML-{len(_MOCK_EMAILS) + 1:04d}",
        "employee_id": employee_id,
        "subject": subject,
        "body": body,
        "channel": "not_sent",
        "writes_to_harborhub": False,
    }
    if not confirm:
        return {
            "tool": "draft_hr_email",
            "status": "pending_confirmation",
            "ok": True,
            "id": None,
            "draft": draft,
            "message": (
                "Emails are never sent from this tool. "
                "Re-run with confirm=true to keep a session-only MOCK draft."
            ),
        }
    _MOCK_EMAILS.append(draft)
    return {
        "tool": "draft_hr_email",
        "status": "mock_created",
        "ok": True,
        "id": draft["id"],
        "draft": draft,
        "message": "MOCK draft stored in this process. No email was sent.",
    }


def check_policy_compliance(
    scenario: str,
    employee_id: str | None = None,
    policy_id: str = "",
    settings: Settings | None = None,
    retriever: object | None = None,
) -> dict:
    """Compare a scenario (and optional employee record) to retrieved policy evidence."""
    settings = settings or get_settings()
    query = " ".join(part for part in (scenario, policy_id) if part)
    hits, rewritten = retrieve_hits(
        query,
        employee_id=employee_id,
        kind="policy",
        settings=settings,
        retriever=retriever,
    )
    profile = lookup_employee_profile(employee_id, settings) if employee_id else None
    pto = check_pto_balance(employee_id, settings) if employee_id else None
    text = scenario.lower()
    verdict = "needs_review"
    reasons: list[str] = []
    amounts = [float(m.replace(",", "")) for m in MONEY_RE.findall(scenario)]

    if profile and profile.get("found"):
        emp = profile["employee"]
        cat = str(emp.get("location_category") or "")
        miles = emp.get("miles_from_assigned_hub")
        if any(tok in text for tok in ("remote", "hybrid", "hub", "wfh", "work from home")):
            if cat.startswith("hub"):
                verdict = "noncompliant_if_fully_remote"
                reasons.append(
                    f"{emp.get('legal_name')} is coded {cat} at {miles} miles; "
                    "hub staff owe at least 3 office days (POL-RMT-003)."
                )
            elif cat.startswith("remote"):
                verdict = "compliant"
                reasons.append(f"{emp.get('legal_name')} is already coded remote.")
        if any(tok in text for tok in ("pto", "vacation", "time off")) and pto and pto.get("found"):
            row = pto["pto"]
            if row.get("pto_eligible") is False:
                verdict = "noncompliant"
                reasons.append("This HarborHub row is not PTO-eligible.")
            elif row.get("eligible_to_use") is False:
                verdict = "noncompliant"
                reasons.append(
                    f"PTO use is locked until {row.get('eligible_to_use_on')} (30-day rule, POL-PTO-001)."
                )
            else:
                verdict = "compliant" if verdict == "needs_review" else verdict
                reasons.append(f"PTO balance {row.get('balance_hours')} hours; eligible_to_use=true.")
    if any(tok in text for tok in ("hotel", "lodging", "nightly cap")):
        over = [a for a in amounts if a > 225]
        if over:
            verdict = "noncompliant"
            reasons.append(f"Claimed lodging ${over[0]:.0f} exceeds the US hotel cap $225 (POL-EXP-004).")
        elif amounts:
            reasons.append("Claimed lodging is at or under the US hotel cap $225 (POL-EXP-004).")
            if verdict == "needs_review":
                verdict = "compliant"
        else:
            reasons.append("The US hotel nightly cap is $225 (POL-EXP-004).")
    if "receipt" in text:
        reasons.append("Receipts are required at $25 and above (POL-EXP-004).")
    if amounts and any(tok in text for tok in ("meal", "dinner", "lunch")):
        cap = 100 if "dinner" in text or "client" in text else 75
        over = [a for a in amounts if a > cap]
        if over:
            verdict = "noncompliant"
            reasons.append(f"Claimed meal ${over[0]:.0f} exceeds the meal cap ${cap} (POL-EXP-004).")

    if not reasons:
        reasons.append("See retrieved policy snippets; People Operations must confirm exceptions.")

    return {
        "tool": "check_policy_compliance",
        "scenario": scenario,
        "employee_id": employee_id,
        "verdict": verdict,
        "reasons": reasons,
        "rewritten_query": rewritten,
        "evidence": _hit_dicts(hits),
        "employee_found": bool(profile and profile.get("found")),
        "ok": True,
    }


def ask_policy(
    query: str,
    employee_id: str | None = None,
    kind: str | None = None,
    settings: Settings | None = None,
    retriever: object | None = None,
) -> dict:
    """Cited policy answer used by the ask CLI; not required on the MCP surface."""
    result = ask(
        query,
        employee_id=employee_id,
        kind=kind,
        settings=settings,
        retriever=retriever,
    )
    result["tool"] = "ask_policy"
    return result
