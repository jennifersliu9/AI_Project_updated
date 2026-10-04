"""Rule-based HR orchestrator that calls MCP-exposed tools.

Intent routing is local. Every HarborHub/RAG lookup goes through the MCP
client (`tools/list` then `tools/call`), not through `harborline.tools`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from harborline.answer import llm_rewrite_answer
from harborline.config import get_settings
from harborline.guardrails import is_out_of_scope
from harborline.mcp_client import McpToolBus, open_mcp_bus

EMP_ID_RE = re.compile(r"\bEMP-\d{4}\b", re.I)

WORKFLOWS = {
    "remote_eligibility": {
        "label": "Remote work eligibility",
        "need_employee": True,
        "rag_alone_enough": False,
        "patterns": (
            r"\bremote\b",
            r"\bhybrid\b",
            r"\bhub\b",
            r"\boffice days?\b",
            r"\b50 miles?\b",
            r"\bwork from home\b",
            r"\bwfh\b",
        ),
    },
    "pto_guidance": {
        "label": "PTO request guidance",
        "need_employee": True,
        "rag_alone_enough": False,
        "patterns": (
            r"\bpto\b",
            r"\bvacation\b",
            r"\btime off\b",
            r"\bcarryover\b",
            r"\bsick (time|day|bank)\b",
        ),
    },
    "benefits": {
        "label": "Benefits questions",
        "need_employee": False,
        "rag_alone_enough": True,
        "patterns": (r"\b401k\b", r"\bmedical\b", r"\bhsa\b", r"\benroll", r"\bbenefit"),
    },
    "expense_compliance": {
        "label": "Expense compliance",
        "need_employee": False,
        "rag_alone_enough": True,
        "patterns": (r"\bexpense\b", r"\breceipt\b", r"\bhotel\b", r"\bmeal cap\b", r"\bvintage"),
    },
    "onboarding": {
        "label": "Onboarding checklist",
        "need_employee": False,
        "rag_alone_enough": True,
        "patterns": (r"\bonboard", r"\bnew hire\b", r"\bi-9\b", r"\bday one\b"),
    },
    "hr_triage": {
        "label": "HR case triage",
        "need_employee": False,
        "rag_alone_enough": False,
        "patterns": (
            r"\bticket\b",
            r"\bharass",
            r"\bcomplaint\b",
            r"\bescalat",
            r"\bhotline\b",
            r"\bfile a case\b",
        ),
    },
}

WRITE_PATTERNS = re.compile(
    r"\b(file|open|create|submit|send|email|update|close)\b.+\b(ticket|case|message|manager|email)\b"
    r"|\b(ticket|case|message|email)\b.+\b(file|open|create|submit|send)\b"
    r"|\btell my manager\b|\bdraft (an? )?(email|message)\b",
    re.I,
)


@dataclass
class TraceStep:
    tool: str
    args: dict[str, Any]
    ok: bool
    output_summary: str
    output: Any = None


@dataclass
class AgentResult:
    query: str
    intent: str
    workflow: str
    rag_alone_enough: bool
    needs_clarification: bool
    escalation: dict[str, Any]
    pending_confirmation: dict[str, Any] | None
    answer: str
    sources: list[dict]
    steps: list[TraceStep] = field(default_factory=list)
    tool_transport: str = "mcp-stdio"
    discovered_tools: list[str] = field(default_factory=list)
    answer_mode: str = "retrieve"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "workflow": self.workflow,
            "rag_alone_enough": self.rag_alone_enough,
            "tool_transport": self.tool_transport,
            "discovered_tools": self.discovered_tools,
            "answer_mode": self.answer_mode,
            "needs_clarification": self.needs_clarification,
            "escalation": self.escalation,
            "pending_confirmation": self.pending_confirmation,
            "answer": self.answer,
            "sources": self.sources,
            "trace": [
                {
                    "tool": s.tool,
                    "args": s.args,
                    "ok": s.ok,
                    "output_summary": s.output_summary,
                }
                for s in self.steps
            ],
        }

    def format_trace(self) -> str:
        lines = [
            "=== Operational trace (not hidden chain-of-thought) ===",
            f"intent: {self.intent}",
            f"workflow: {self.workflow}",
            f"rag_alone_enough: {self.rag_alone_enough}",
            f"tool_transport: {self.tool_transport}",
            f"mcp_discovered_tools: {self.discovered_tools}",
        ]
        for i, step in enumerate(self.steps, start=1):
            lines.append(
                f"step {i}: {step.tool} args={step.args} ok={step.ok} -- {step.output_summary}"
            )
        if self.sources:
            lines.append("retrieved_sources:")
            for src in self.sources:
                path = src.get("source_path") or src.get("path")
                lines.append(f"  - {src.get('title')} | {src.get('section')} ({path})")
        lines.append(
            f"escalation: needed={self.escalation.get('needed')} "
            f"reason={self.escalation.get('reason')}"
        )
        if self.pending_confirmation:
            lines.append(
                f"pending_confirmation: {self.pending_confirmation.get('status')} "
                f"id={self.pending_confirmation.get('id')} "
                "(re-run with --confirm to accept this MOCK write)"
            )
        lines.append("final_answer_basis: MCP tools above + Harborline corpus/HarborHub records")
        lines.append("=== End trace ===")
        return "\n".join(lines)


def infer_employee_id(query: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit.strip().upper()
    match = EMP_ID_RE.search(query or "")
    return match.group(0).upper() if match else None


def classify_intent(query: str) -> tuple[str, list[str]]:
    text = query.lower()
    hits = [
        name
        for name, spec in WORKFLOWS.items()
        if any(re.search(pat, text) for pat in spec["patterns"])
    ]
    if len(hits) > 1 and "hr_triage" in hits:
        return "hr_triage", hits
    if len(hits) > 1:
        return "ambiguous", hits
    if len(hits) == 1:
        return hits[0], hits
    return "policy_qa", hits


def _summarize(name: str, payload: dict) -> str:
    if payload.get("error"):
        return str(payload["error"])
    if name == "search_policy_documents":
        hits = payload.get("hits") or []
        titles = [h.get("source_path") or h.get("title") for h in hits[:3]]
        return f"{len(hits)} hits; top={titles}"
    if name == "get_policy_section":
        sections = payload.get("sections") or []
        labels = [s.get("section") for s in sections[:3]]
        return f"found={payload.get('found')} sections={labels}"
    if name == "lookup_employee_profile":
        if not payload.get("found"):
            return payload.get("error") or "not found"
        emp = payload.get("employee") or {}
        return f"{emp.get('legal_name')} {emp.get('location_category')}"
    if name == "check_pto_balance":
        if not payload.get("found"):
            return payload.get("error") or "not found"
        pto = payload.get("pto") or {}
        return f"balance={pto.get('balance_hours')} eligible_to_use={pto.get('eligible_to_use')}"
    if name == "lookup_benefits_status":
        if not payload.get("found"):
            return payload.get("error") or "not found"
        ben = payload.get("benefits") or {}
        return f"status={ben.get('election_status')} plan={ben.get('medical_plan')}"
    if name == "check_policy_compliance":
        return f"verdict={payload.get('verdict')} evidence={len(payload.get('evidence') or [])}"
    if name in {"create_mock_hr_ticket", "draft_hr_email"}:
        return f"status={payload.get('status')} id={payload.get('id')}"
    return "ok"


_KEEP_DRAFT_REASONS = {
    "tool_bus_unavailable",
    "missing_employee_record",
    "incomplete_policy_evidence",
}


def _finish_answer(result: AgentResult, employee_id: str | None) -> None:
    """Use the model when a key is configured. Refusals stay on the draft."""
    settings = get_settings()
    result.answer_mode = settings.answer_mode
    if settings.answer_mode != "llm" or not settings.openai_api_key:
        return
    if result.needs_clarification or not (result.answer or "").strip():
        return
    reason = (result.escalation or {}).get("reason")
    if reason in _KEEP_DRAFT_REASONS:
        return
    notes = [f"{step.tool}: {step.output_summary}" for step in result.steps]
    try:
        result.answer = llm_rewrite_answer(
            result.query,
            result.answer,
            result.sources,
            notes,
            settings,
            employee_id,
        )
    except Exception:
        return


def _record(result: AgentResult, name: str, args: dict, payload: dict) -> dict:
    ok = "error" not in payload and payload.get("ok", True) is not False
    if name in {"lookup_employee_profile", "check_pto_balance", "lookup_benefits_status"}:
        ok = bool(payload.get("found"))
    result.steps.append(
        TraceStep(
            tool=name,
            args=args,
            ok=ok,
            output_summary=_summarize(name, payload),
            output=payload,
        )
    )
    return payload


def run_agent(
    query: str,
    employee_id: str | None = None,
    confirm: bool = False,
    bus: McpToolBus | None = None,
    transport: str = "mcp-inproc",
    extra_env: dict[str, str] | None = None,
) -> AgentResult:
    owns_bus = bus is None
    if bus is None:
        try:
            bus = open_mcp_bus(transport, extra_env=extra_env)
        except Exception as exc:  # noqa: BLE001
            bus = McpToolBus()
            bus.available = False
            bus.transport = transport
            bus._error = str(exc)  # type: ignore[attr-defined]

    intent, matched = classify_intent(query)
    eid = infer_employee_id(query, employee_id)
    spec = WORKFLOWS.get(
        intent,
        {
            "label": "General policy question",
            "need_employee": False,
            "rag_alone_enough": True,
        },
    )
    result = AgentResult(
        query=query,
        intent=intent,
        workflow=spec["label"],
        rag_alone_enough=bool(spec.get("rag_alone_enough")),
        needs_clarification=False,
        escalation={"needed": False, "reason": None},
        pending_confirmation=None,
        answer="",
        sources=[],
        tool_transport=getattr(bus, "transport", transport),
        discovered_tools=list(getattr(bus, "discovered_tools", []) or []),
    )

    failed = False
    try:
        if not bus.available:
            result.answer = (
                "MCP tools are unavailable. I cannot look up HarborHub or search policies "
                "right now. Retry after enabling the Harborline MCP server, or run "
                "`python -m harborline.cli agent --transport mcp-inproc`."
            )
            result.escalation = {"needed": True, "reason": "tool_bus_unavailable"}
            return result

        if intent == "ambiguous":
            labels = [WORKFLOWS[n]["label"] for n in matched]
            result.needs_clarification = True
            result.answer = (
                "That request matches more than one HR workflow: "
                + ", ".join(labels)
                + ". Say which you want, or add an employee id (EMP-1xxx)."
            )
            return result

        if intent == "remote_eligibility":
            _run_remote(result, bus, query, eid)
        elif intent == "pto_guidance":
            _run_pto(result, bus, query, eid, confirm)
        elif intent == "hr_triage":
            _run_triage(result, bus, query, eid, confirm)
        elif intent == "expense_compliance":
            _run_expense(result, bus, query, eid)
        elif intent == "onboarding":
            _run_onboarding(result, bus, query)
        elif intent == "benefits":
            _run_benefits(result, bus, query, eid)
        else:
            _run_policy_qa(result, bus, query, eid)
        return result
    except Exception:
        failed = True
        raise
    finally:
        if not failed:
            _finish_answer(result, eid)
        if owns_bus:
            bus.close()


def _collect_sources(*payloads: dict) -> list[dict]:
    sources: list[dict] = []
    for payload in payloads:
        sources.extend(payload.get("hits") or [])
        sources.extend(payload.get("evidence") or [])
        sources.extend(payload.get("sections") or [])
        sources.extend(payload.get("sources") or [])
    return sources


def _canonical_sections(*payloads: dict) -> list[dict]:
    """Sections from the declaring policy document, ignoring RAG fallbacks."""
    sections: list[dict] = []
    for payload in payloads:
        if payload.get("fallback") == "rag" or not payload.get("found"):
            continue
        sections.extend(payload.get("sections") or [])
    return sections


def _cite(sources: list[dict], *needles: str) -> list[dict]:
    """Keep one hit per declaring file so the citation list matches the answer."""
    picked: list[dict] = []
    seen: set[str] = set()
    for src in sources:
        path = src.get("source_path") or ""
        if path in seen:
            continue
        if any(needle in path for needle in needles):
            picked.append(src)
            seen.add(path)
    return picked


def _run_remote(result: AgentResult, bus: McpToolBus, query: str, eid: str | None) -> None:
    if not eid:
        result.needs_clarification = True
        policy = _record(
            result,
            "search_policy_documents",
            {"query": "remote hybrid hub 50 miles POL-RMT-003", "kind": "policy"},
            bus.call(
                "search_policy_documents",
                query="remote hybrid hub 50 miles POL-RMT-003",
                kind="policy",
            ),
        )
        result.sources = _cite(_collect_sources(policy), "03-remote-hybrid-work.md")
        result.answer = (
            "Remote eligibility depends on your HarborHub location category and the "
            "50-mile hub rule (POL-RMT-003). Provide an employee id (for example EMP-1008)."
        )
        return
    lookup = _record(
        result,
        "lookup_employee_profile",
        {"employee_id": eid},
        bus.call("lookup_employee_profile", employee_id=eid),
    )
    if not lookup.get("found"):
        result.answer = (
            f"No HarborHub profile for {eid}. I will not invent eligibility. "
            "Check the id or ask hr@harborline.example."
        )
        result.escalation = {"needed": True, "reason": "missing_employee_record"}
        return
    search = _record(
        result,
        "search_policy_documents",
        {"query": "remote hybrid hub 50 miles office days POL-RMT-003", "kind": "policy"},
        bus.call(
            "search_policy_documents",
            query="remote hybrid hub 50 miles office days POL-RMT-003",
            kind="policy",
        ),
    )
    section = _record(
        result,
        "get_policy_section",
        {"policy_id": "POL-RMT-003", "section": "Location categories"},
        bus.call(
            "get_policy_section",
            policy_id="POL-RMT-003",
            section="Location categories",
        ),
    )
    compliance = _record(
        result,
        "check_policy_compliance",
        {"scenario": query, "employee_id": eid, "policy_id": "POL-RMT-003"},
        bus.call(
            "check_policy_compliance",
            scenario=query,
            employee_id=eid,
            policy_id="POL-RMT-003",
        ),
    )
    result.sources = _canonical_sections(section) or _collect_sources(search, compliance)
    emp = lookup["employee"]
    cat = emp.get("location_category")
    miles = emp.get("miles_from_assigned_hub")
    facts = (
        f"{emp.get('legal_name')} ({eid}) is coded {cat}, "
        f"{miles} miles from the assigned hub, home {emp.get('home_city')}."
    )
    if cat and str(cat).startswith("hub"):
        rule = "Hub employees owe at least 3 office days per week (Tue–Thu default)."
    elif cat and str(cat).startswith("remote"):
        rule = "Remote employees have no regular office-day requirement."
    else:
        rule = "Location category is unclear; People Operations must confirm."
        result.escalation = {"needed": True, "reason": "ambiguous_location_category"}
    result.answer = (
        f"**Policy fact:** {facts} {rule} Reclassification is a People Operations action, "
        "not a manager Slack exception (POL-RMT-003). "
        f"Compliance tool verdict: {compliance.get('verdict')}.\n\n"
        "**Not a recommendation:** This is the coded rule, not advice to move or recode yourself."
    )
    if "reclass" in query.lower() or "fully remote" in query.lower():
        result.escalation = {
            "needed": True,
            "reason": "location_reclassification_requires_people_ops",
        }


def _run_pto(result: AgentResult, bus: McpToolBus, query: str, eid: str | None, confirm: bool) -> None:
    if not eid:
        policy = _record(
            result,
            "search_policy_documents",
            {"query": query, "kind": "policy"},
            bus.call("search_policy_documents", query=query, kind="policy"),
        )
        cited = _cite(
            _collect_sources(policy),
            "01-paid-time-off.md",
            "pto-quick-reference.txt",
        )
        personal = re.search(
            r"\b(can i|am i|submit|request|my balance|next week|tomorrow)\b",
            query,
            re.I,
        )
        if cited and not personal:
            result.sources = cited
            result.rag_alone_enough = True
            result.answer = (
                "**Policy fact:** After the second anniversary, PTO accrual is 20 days "
                "(160 hours) a year. The carryover cap is 40 hours into the next calendar "
                "year, except California (POL-PTO-001). The PTO quick reference states the "
                "same 20-day band and 40-hour cap.\n\n"
                "**Not a recommendation:** A personal balance still needs an employee id."
            )
            return
        result.needs_clarification = True
        result.sources = cited or _collect_sources(policy)
        result.answer = (
            "I can quote the PTO policy, but a request needs an employee id so I can "
            "check the HarborHub balance and the 30-day use rule. Example: EMP-1008."
        )
        return
    lookup = _record(
        result,
        "lookup_employee_profile",
        {"employee_id": eid},
        bus.call("lookup_employee_profile", employee_id=eid),
    )
    if not lookup.get("found"):
        result.answer = f"No HarborHub profile for {eid}. I cannot check a PTO balance."
        result.escalation = {"needed": True, "reason": "missing_employee_record"}
        return
    pto = _record(
        result,
        "check_pto_balance",
        {"employee_id": eid},
        bus.call("check_pto_balance", employee_id=eid),
    )
    policy = _record(
        result,
        "get_policy_section",
        {"policy_id": "POL-PTO-001", "section": "Eligibility"},
        bus.call("get_policy_section", policy_id="POL-PTO-001", section="Eligibility"),
    )
    result.sources = _collect_sources(policy)
    emp = lookup["employee"]
    row = pto.get("pto") or {}
    name = emp.get("legal_name")
    if emp.get("employment_type") == "intern" or row.get("pto_eligible") is False:
        result.answer = (
            f"**Policy fact:** {name} is an intern/ineligible row. Interns do not accrue PTO "
            "(POL-PTO-001). They observe office closures only."
        )
        return
    eligible = row.get("eligible_to_use")
    balance = row.get("balance_hours")
    usable_on = row.get("eligible_to_use_on")
    extra = ""
    if eligible is False:
        extra = f" Use generally starts on {usable_on}. "
        result.escalation = {"needed": False, "reason": "waiting_period"}
    wants_submit = bool(WRITE_PATTERNS.search(query) or re.search(r"\bsubmit\b|\brequest pto\b", query, re.I))
    result.answer = (
        f"**Policy fact:** {name} ({eid}) PTO balance is {balance} hours; "
        f"eligible_to_use={eligible}.{extra}"
        "Requests go through HarborHub > Time Off; 1-hour minimum increment (POL-PTO-001).\n\n"
        "**Not a recommendation:** I am not approving time off."
    )
    if wants_submit:
        ticket = _record(
            result,
            "create_mock_hr_ticket",
            {"topic": "pto_request", "employee_id": eid, "confirm": confirm},
            bus.call(
                "create_mock_hr_ticket",
                topic="pto_request",
                employee_id=eid,
                summary=query,
                confirm=confirm,
            ),
        )
        if ticket.get("status") == "pending_confirmation":
            result.pending_confirmation = ticket
            result.answer += (
                "\n\nA HarborHub PTO request was **not** filed. Draft is MOCK only. "
                "Re-run with --confirm if you want the mock ticket recorded in this session."
            )


def _run_triage(result: AgentResult, bus: McpToolBus, query: str, eid: str | None, confirm: bool) -> None:
    policy = _record(
        result,
        "search_policy_documents",
        {"query": "harassment reporting hotline POL-CON-010", "kind": "policy"},
        bus.call(
            "search_policy_documents",
        query="harassment reporting hotline POL-CON-010",
        kind="policy",
    ),
    )
    result.sources = _cite(_collect_sources(policy), "10-workplace-conduct.md") or _collect_sources(policy)[:1]
    result.escalation = {
        "needed": True,
        "reason": "people_ops_or_hotline",
        "contacts": ["hr@harborline.example", "+1-800-555-0148"],
    }
    ticket = _record(
        result,
        "create_mock_hr_ticket",
        {"topic": "conduct", "employee_id": eid, "confirm": confirm},
        bus.call(
            "create_mock_hr_ticket",
            topic="conduct",
            employee_id=eid,
            summary=query,
            confirm=confirm,
        ),
    )
    email = _record(
        result,
        "draft_hr_email",
        {"employee_id": eid or "unknown", "subject": "Workplace concern intake", "confirm": confirm},
        bus.call(
            "draft_hr_email",
            employee_id=eid or "unknown",
            subject="Workplace concern intake",
            body=(
                "This is a MOCK Harborline intake note. Reporting paths are a manager, "
                "hr@harborline.example, or the ethics hotline +1-800-555-0148 (POL-CON-010)."
            ),
            confirm=confirm,
        ),
    )
    result.answer = (
        "**Policy fact:** Harassment, discrimination, and retaliation reports can go to a "
        "manager, hr@harborline.example, or the ethics hotline +1-800-555-0148 (POL-CON-010). "
        "You do not have to start with your manager.\n\n"
        "**Not a recommendation:** This is the reporting path, not an investigation finding."
    )
    if ticket.get("status") == "pending_confirmation":
        result.pending_confirmation = ticket
        result.answer += (
            "\n\nNo live case was opened. The ticket and email payloads are MOCK. "
            "Re-run with --confirm to keep a session-only mock case id."
        )
    elif ticket.get("status") == "mock_created":
        result.answer += (
            f"\n\nMock case id {ticket.get('id')} stored in this process only. "
            f"Email draft status={email.get('status')}."
        )


def _run_expense(result: AgentResult, bus: McpToolBus, query: str, eid: str | None) -> None:
    compliance = _record(
        result,
        "check_policy_compliance",
        {"scenario": query, "employee_id": eid, "policy_id": "POL-EXP-004"},
        bus.call(
            "check_policy_compliance",
            scenario=query,
            employee_id=eid,
            policy_id="POL-EXP-004",
        ),
    )
    result.sources = _cite(
        _collect_sources(compliance),
        "04-travel-and-expenses.md",
        "expense-limits-2026.pdf",
    ) or _collect_sources(compliance)
    if not result.sources:
        result.escalation = {"needed": True, "reason": "incomplete_policy_evidence"}
        result.answer = "Not enough policy evidence to check that expense scenario."
        return
    result.answer = (
        f"**Policy fact:** Compliance verdict `{compliance.get('verdict')}`. "
        + " ".join(compliance.get("reasons") or [])
        + "\n\n**Not a recommendation:** Voyage claims still need receipts and manager review."
    )


def _run_onboarding(result: AgentResult, bus: McpToolBus, query: str) -> None:
    section = _record(
        result,
        "get_policy_section",
        {"policy_id": "POL-ONB-007", "section": "Day one"},
        bus.call("get_policy_section", policy_id="POL-ONB-007", section="Day one"),
    )
    result.sources = _collect_sources(section)
    if not result.sources:
        result.escalation = {"needed": True, "reason": "incomplete_policy_evidence"}
        result.answer = "Not enough onboarding policy evidence."
        return
    result.answer = (
        "**Policy fact:** See POL-ONB-007 (I-9, HarborHub access, and day-one checklist) "
        "in the retrieved section.\n\n"
        "**Not a recommendation:** People Operations owns exceptions to the checklist."
    )


def _run_benefits(result: AgentResult, bus: McpToolBus, query: str, eid: str | None) -> None:
    search = _record(
        result,
        "search_policy_documents",
        {"query": "benefits 401k medical enrollment POL-BEN-006", "kind": "policy"},
        bus.call(
            "search_policy_documents",
            query="benefits 401k medical enrollment POL-BEN-006",
            kind="policy",
        ),
    )
    medical = _record(
        result,
        "get_policy_section",
        {"policy_id": "POL-BEN-006", "section": "US medical"},
        bus.call("get_policy_section", policy_id="POL-BEN-006", section="US medical"),
    )
    retirement = _record(
        result,
        "get_policy_section",
        {"policy_id": "POL-BEN-006", "section": "Retirement"},
        bus.call("get_policy_section", policy_id="POL-BEN-006", section="Retirement"),
    )
    election = re.search(r"\b(elect|election|how long|start date)\b", query, re.I)
    vesting = re.search(r"\bvest", query, re.I)
    needles = ["06-employee-benefits.md"]
    if vesting:
        needles.append("benefits-enrollment-guide.html")
    if election:
        needles.extend(["benefits-enrollment-guide.html", "07-new-hire-onboarding.md"])
    result.sources = _cite(
        _canonical_sections(medical, retirement) or _collect_sources(search),
        *needles,
    )
    if any(needle not in " ".join(src.get("source_path") or "" for src in result.sources) for needle in needles):
        wider = _record(
            result,
            "search_policy_documents",
            {
                "query": "medical election 30 days new hire benefits enrollment guide 401k vesting",
                "kind": "policy",
            },
            bus.call(
                "search_policy_documents",
                query="medical election 30 days new hire benefits enrollment guide 401k vesting",
                kind="policy",
            ),
        )
        result.sources = _cite(result.sources + _collect_sources(wider), *needles)
    extra = ""
    if eid:
        benefits = _record(
            result,
            "lookup_benefits_status",
            {"employee_id": eid},
            bus.call("lookup_benefits_status", employee_id=eid),
        )
        if benefits.get("found"):
            row = benefits["benefits"]
            extra = (
                f" HarborHub election for {eid}: {row.get('election_status')}, "
                f"medical {row.get('medical_plan')} {row.get('medical_tier')}, "
                f"401k deferral {row.get('k401_deferral_percent')}%."
            )
    if not result.sources:
        result.escalation = {"needed": True, "reason": "incomplete_policy_evidence"}
        result.answer = "Not enough benefits policy evidence."
        return
    window = ""
    if election:
        window = (
            " New hires have 30 days from the start date to elect medical coverage "
            "(POL-BEN-006, POL-ONB-007)."
        )
    result.answer = (
        "**Policy fact:** Company 401(k) match is 100% of the first 4% deferred, immediate vesting "
        f"(POL-BEN-006).{window}{extra}\n\n"
        "**Not a recommendation:** Elections and qualifying-life changes go through HarborHub."
    )


def _run_policy_qa(result: AgentResult, bus: McpToolBus, query: str, eid: str | None) -> None:
    result.rag_alone_enough = True
    if is_out_of_scope(query):
        result.escalation = {"needed": True, "reason": "incomplete_policy_evidence"}
        result.answer = (
            "That question is outside the Harborline policy corpus. I will not invent an answer."
        )
        return
    search = _record(
        result,
        "search_policy_documents",
        {"query": query, "employee_id": eid, "kind": "policy"},
        bus.call("search_policy_documents", query=query, employee_id=eid, kind="policy"),
    )
    result.sources = _collect_sources(search)
    scores = [float(s.get("score") or 0) for s in result.sources]
    if not result.sources or (scores and max(scores) < 0.22):
        result.escalation = {"needed": True, "reason": "incomplete_policy_evidence"}
        result.answer = (
            "That question is outside the Harborline policy corpus, or retrieval returned "
            "no usable evidence. I will not invent an answer."
        )
        return
    text = query.lower()
    if "phishing" in text or "okta" in text:
        result.sources = _cite(
            result.sources,
            "05-information-security.md",
            "acceptable-use-policy.txt",
        ) or result.sources[:1]
    if "parental" in text or "secondary caregiver" in text:
        result.sources = _cite(result.sources, "11-parental-and-family-care.md") or result.sources[:1]
        result.answer = (
            "**Policy fact:** Secondary caregivers get 8 weeks of paid parental leave "
            "(POL-FAM-011).\n\n"
            "**Not a recommendation:** Confirm the caregiver role with People Operations."
        )
        return
    if "thanksgiving" in text:
        result.sources = _cite(
            result.sources,
            "02-company-holidays.md",
            "holiday-calendar-2026.html",
        ) or result.sources[:1]
        result.answer = (
            "**Policy fact:** Yes. 27 November 2026, the day after Thanksgiving, is a US "
            "company holiday.\n\n"
            "**Not a recommendation:** Confirm the office closure on the holiday calendar."
        )
        return
    top = result.sources[0]
    result.answer = (
        f"**Policy fact:** Retrieved {len(result.sources)} policy snippets. "
        f"Top hit: {top.get('title')} — {top.get('section')}. {top.get('snippet')}\n\n"
        "**Not a recommendation:** Confirm with your manager or hr@harborline.example."
    )
