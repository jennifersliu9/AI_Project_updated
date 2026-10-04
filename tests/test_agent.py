from harborline.agent import classify_intent, infer_employee_id, run_agent
from harborline.mcp_client import InProcessMcpBus, McpToolBus
from harborline.tools import reset_mocks


def test_intent_and_employee_parse():
    intent, hits = classify_intent("Am I eligible for remote work from Tacoma?")
    assert intent == "remote_eligibility"
    assert infer_employee_id("check EMP-1008 please", None) == "EMP-1008"
    intent2, _ = classify_intent("Can I request PTO next Friday?")
    assert intent2 == "pto_guidance"


def test_remote_eligibility_workflow_via_mcp():
    bus = InProcessMcpBus()
    try:
        result = run_agent(
            "Am I eligible for fully remote work living in Tacoma?",
            employee_id="EMP-1008",
            bus=bus,
        )
    finally:
        bus.close()
    assert result.intent == "remote_eligibility"
    assert result.rag_alone_enough is False
    assert result.tool_transport == "mcp-inproc"
    assert "search_policy_documents" in result.discovered_tools
    assert "lookup_employee_profile" in result.discovered_tools
    tools = [s.tool for s in result.steps]
    assert tools[:2] == ["lookup_employee_profile", "search_policy_documents"]
    assert "get_policy_section" in tools
    assert "check_policy_compliance" in tools
    assert {src.get("source_path") for src in result.sources} == {"corpus/03-remote-hybrid-work.md"}
    assert all(s.ok for s in result.steps[:2])
    assert result.sources
    assert "Alex Kim" in result.answer
    assert "hub" in result.answer.lower()
    assert result.escalation["needed"] is True
    assert "reclass" in result.escalation["reason"]
    trace = result.format_trace()
    assert "Operational trace" in trace
    assert "lookup_employee_profile" in trace
    assert "mcp_discovered_tools" in trace


def test_pto_policy_facts_do_not_require_an_employee():
    tenure = run_agent(
        "How many PTO days do I get after my second anniversary?",
        transport="mcp-inproc",
    )
    assert tenure.needs_clarification is False
    assert "20" in tenure.answer
    assert any("01-paid-time-off.md" in (src.get("source_path") or "") for src in tenure.sources)
    carry = run_agent(
        "What is the PTO carryover cap in hours?",
        transport="mcp-inproc",
    )
    assert carry.needs_clarification is False
    assert "40" in carry.answer


def test_pto_guidance_waiting_period():
    result = run_agent(
        "Can I take PTO next week?",
        employee_id="EMP-1014",
        transport="mcp-inproc",
    )
    assert result.intent == "pto_guidance"
    tools = [s.tool for s in result.steps]
    assert "lookup_employee_profile" in tools
    assert "check_pto_balance" in tools
    assert "get_policy_section" in tools
    assert "Devon Walsh" in result.answer
    assert "eligible_to_use=False" in result.answer or "2026-10-08" in result.answer
    assert result.pending_confirmation is None


def test_pto_submit_requires_confirm():
    reset_mocks()
    result = run_agent(
        "Please submit a PTO request for next Friday",
        employee_id="EMP-1008",
        confirm=False,
        transport="mcp-inproc",
    )
    assert result.pending_confirmation is not None
    assert result.pending_confirmation["status"] == "pending_confirmation"
    assert "MOCK" in result.answer


def test_missing_employee_and_clarification():
    missing = run_agent(
        "Am I remote eligible?",
        employee_id="EMP-9999",
        transport="mcp-inproc",
    )
    assert missing.escalation["reason"] == "missing_employee_record"
    clarify = run_agent("Am I eligible for remote work?", transport="mcp-inproc")
    assert clarify.needs_clarification is True
    assert "EMP-" in clarify.answer


def test_ambiguous_and_unavailable_bus():
    ambiguous = run_agent("remote PTO hybrid vacation", transport="mcp-inproc")
    assert ambiguous.intent == "ambiguous"
    assert ambiguous.needs_clarification is True
    bus = McpToolBus()
    bus.available = False
    bus.transport = "unavailable"
    down = run_agent("What is the hotel cap?", bus=bus)
    assert down.escalation["reason"] == "tool_bus_unavailable"
    assert "unavailable" in down.answer.lower()


def test_out_of_scope_incomplete_evidence():
    result = run_agent("Should I buy bitcoin with my bonus?", transport="mcp-inproc")
    assert result.intent == "policy_qa"
    assert result.rag_alone_enough is True
    assert result.escalation["reason"] == "incomplete_policy_evidence"
