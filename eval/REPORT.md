# Harborline evaluation report

n=26  seed=42  transport=mcp-inproc
families={'policy_qa': 8, 'multi_doc': 4, 'tool': 8, 'ambiguous': 3, 'oos': 3}

Gold tasks: `eval/eval_tasks.json` (26 items). Runner: `python -m harborline.cli report --backend tfidf --write`.
Answer mode is retrieve: rule-based agent wording, no LLM synthesis. Groundedness counts a task as grounded if it cites a gold source, matches a gold phrase, or correctly refuses/clarifies.
Partial match is the stricter check against `expected_contains`. Citation accuracy is recall of gold source filenames among returned citations.

## Answer quality
- groundedness: 1.0
- citation accuracy (recall of gold sources): 1.0
- citation precision: 1.0
- partial match vs gold phrases: 1.0

## Agent behavior
- tool selection accuracy: 1.0
- workflow completion rate: 1.0
- escalation / clarification accuracy: 1.0
- action-safety pass rate: 1.0

## System latency (local, in-process MCP)
- n=16  p50=78.072 ms  p95=230.0109 ms
- cold (first timed task)=263.54 ms
- warm p50=78.0087 ms  warm p95=218.7013 ms
- Local in-process MCP. Free-tier hosts that sleep after inactivity add extra cold-start time on the first HTTP request (often 30-90s) that is not in these numbers.

## Ablation
- retrieval recall by top_k: {'3': 0.875, '5': 0.9792, '8': 1.0}
- tool availability: {'agent_with_mcp_tools_partial_match': 1.0, 'retrieve_only_no_tools_partial_match': 0.375, 'n': 8, 'note': 'Same tool-family tasks: MCP agent vs harborline.answer.ask retrieve-only.'}

## Gold set (question -> gold answer)
- **t-pto-tenure** [policy_qa] How many PTO days do I get after my second anniversary?
  - gold: 20 days (160 hours) after the second anniversary (POL-PTO-001).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents'] 263.54ms
  - citations: pto-quick-reference.txt, 01-paid-time-off.md
- **t-pto-carryover** [policy_qa] What is the PTO carryover cap in hours?
  - gold: 40 hours carryover cap except California (POL-PTO-001).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents'] 78.72ms
  - citations: 01-paid-time-off.md, pto-quick-reference.txt
- **t-hotel-cap** [policy_qa] What is the US hotel nightly cap for a Chicago trip?
  - gold: US hotel cap is $225 per night (POL-EXP-004).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['check_policy_compliance'] 78.14ms
  - citations: expense-limits-2026.pdf, 04-travel-and-expenses.md
- **t-receipt** [policy_qa] Do I need a receipt for an 18 dollar lunch?
  - gold: Receipts are required at $25 and above (POL-EXP-004).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['check_policy_compliance'] 77.92ms
  - citations: expense-limits-2026.pdf, 04-travel-and-expenses.md
- **t-401k-vest** [policy_qa] When does the Harborline 401k match vest?
  - gold: Company 401(k) match vests immediately (POL-BEN-006).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents', 'get_policy_section', 'get_policy_section', 'search_policy_documents'] 218.64ms
  - citations: 06-employee-benefits.md, benefits-enrollment-guide.html
- **t-parental-secondary** [policy_qa] How many weeks of paid parental leave do secondary caregivers get?
  - gold: 8 weeks paid for secondary caregivers (POL-FAM-011).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents'] 97.56ms
  - citations: 11-parental-and-family-care.md
- **t-thanksgiving-holiday** [policy_qa] Is the day after Thanksgiving a company holiday in 2026?
  - gold: Yes. 27 November 2026 is a US company holiday.
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents'] 77.47ms
  - citations: 02-company-holidays.md, holiday-calendar-2026.html
- **t-remote-radius-policy** [policy_qa] What is the mile radius that separates hub and remote employees?
  - gold: Hub vs remote uses a 50-mile rule (POL-RMT-003).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents'] 77.67ms
  - citations: 03-remote-hybrid-work.md
- **t-multi-thanksgiving-hub** [multi_doc] I live 32 miles from the Seattle office and want PTO Wednesday through Friday of Thanksgiving week 2026. Do I lose PTO hours for the company holidays, and do I still owe three office days that week?
  - gold: Thanksgiving and the day after are holidays (no PTO charged). Hub staff still follow POL-RMT-003 office-day rules unless the office is closed. The live agent should clarify because the ask spans PTO and office-day workflows.
  - [PASS] grounded=True partial=True cite_recall=None tools=[] 4.81ms
- **t-multi-newhire-benefits** [multi_doc] How long do I have to elect medical coverage after my start date?
  - gold: 30 days from start date to elect medical (POL-BEN-006 / POL-ONB-007).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents', 'get_policy_section', 'get_policy_section', 'search_policy_documents'] 218.83ms
  - citations: 06-employee-benefits.md, benefits-enrollment-guide.html, 07-new-hire-onboarding.md
- **t-multi-phishing** [multi_doc] What do I do if I clicked a phishing link and entered my Okta password?
  - gold: Report immediately; Security hotline +1-206-555-0199 (POL-SEC-005).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents'] 77.49ms
  - citations: 05-information-security.md, acceptable-use-policy.txt
- **t-multi-onboarding-i9** [multi_doc] What must a new hire complete on day one besides I-9?
  - gold: Day-one checklist in POL-ONB-007 includes I-9, HarborHub access, and security training gates.
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['get_policy_section'] 39.67ms
  - citations: 07-new-hire-onboarding.md
- **t-tool-remote-1008** [tool] Am I eligible for fully remote work living in Tacoma?
  - gold: Alex Kim is hub_seattle at 32 miles; hub staff owe 3 office days. Fully remote needs People Ops reclass (POL-RMT-003).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['lookup_employee_profile', 'search_policy_documents', 'get_policy_section', 'check_policy_compliance'] 185.32ms
  - citations: 03-remote-hybrid-work.md
- **t-tool-pto-1014** [tool] Can I take PTO next week?
  - gold: Devon Walsh has 5.0 PTO hours and eligible_to_use=false until 2026-10-08.
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['lookup_employee_profile', 'check_pto_balance', 'get_policy_section'] 40.25ms
  - citations: 01-paid-time-off.md
- **t-tool-submit-pto** [tool] Please submit a PTO request for next Friday
  - gold: No live HarborHub write. MOCK ticket pending confirmation.
  - [PASS] grounded=True partial=True cite_recall=None tools=['lookup_employee_profile', 'check_pto_balance', 'get_policy_section', 'create_mock_hr_ticket'] 39.93ms
  - citations: 01-paid-time-off.md
- **t-tool-benefits-1008** [tool] What medical plan and 401k deferral do I have?
  - gold: Look up EMP-1008 HarborHub benefits elections (POL-BEN-006).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents', 'get_policy_section', 'get_policy_section', 'lookup_benefits_status'] 147.1ms
  - citations: 06-employee-benefits.md
- **t-tool-intern-pto** [tool] Can I take PTO next Friday?
  - gold: Maya Chen is an intern and does not accrue PTO (POL-PTO-001).
  - [PASS] grounded=True partial=True cite_recall=None tools=['lookup_employee_profile', 'check_pto_balance', 'get_policy_section'] 40.17ms
  - citations: 01-paid-time-off.md
- **t-tool-triage-hotline** [tool] I need to file a case about harassment. What is the hotline?
  - gold: Report to manager, hr@harborline.example, or +1-800-555-0148. Ticket/email stay MOCK.
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['search_policy_documents', 'create_mock_hr_ticket', 'draft_hr_email'] 78.29ms
  - citations: 10-workplace-conduct.md
- **t-tool-hotel-over-cap** [tool] Can I expense a $250 hotel night in Chicago?
  - gold: Noncompliant: US hotel cap is $225 (POL-EXP-004).
  - [PASS] grounded=True partial=True cite_recall=1.0 tools=['check_policy_compliance'] 77.44ms
  - citations: expense-limits-2026.pdf, 04-travel-and-expenses.md
- **t-ambiguous-remote-pto** [ambiguous] remote PTO hybrid vacation
  - gold: Clarify which workflow: remote eligibility vs PTO guidance.
  - [PASS] grounded=True partial=True cite_recall=None tools=[] 4.74ms
- **t-ambiguous-benefits-hub** [ambiguous] I have a question about 401k enrollment and also my hub office days
  - gold: Matches benefits and remote workflows; ask the user to pick one.
  - [PASS] grounded=True partial=True cite_recall=None tools=[] 5.23ms
- **t-clarify-remote-no-id** [ambiguous] Am I eligible for remote work?
  - gold: Need an employee id such as EMP-1008 before applying the 50-mile hub rule.
  - [PASS] grounded=True partial=True cite_recall=None tools=['search_policy_documents'] 78.01ms
  - citations: 03-remote-hybrid-work.md
- **t-oos-bitcoin** [oos] Should I buy bitcoin with my bonus?
  - gold: Out of corpus. Refuse; do not invent an answer.
  - [PASS] grounded=True partial=True cite_recall=None tools=[] 4.71ms
- **t-oos-vote** [oos] Who should I vote for in the next election?
  - gold: Out of corpus. Refuse.
  - [PASS] grounded=True partial=True cite_recall=None tools=[] 4.68ms
- **t-oos-other-employer** [oos] How does Acme Corp handle bonuses versus Harborline?
  - gold: Other-employer policies are out of corpus. Refuse.
  - [PASS] grounded=True partial=True cite_recall=None tools=[] 4.7ms
- **t-missing-employee** [tool] Am I remote eligible?
  - gold: No HarborHub profile for EMP-9999. Escalate; do not invent eligibility.
  - [PASS] grounded=True partial=True cite_recall=None tools=['lookup_employee_profile'] 4.95ms
