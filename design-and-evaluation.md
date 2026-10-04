# Design and evaluation

Harborline answers employee questions about a fictional 2026 handbook and mock HarborHub records. A hand-written orchestrator calls one MCP server. Retrieval uses OpenAI `text-embedding-3-small` and an in-process cosine index. When `HARBORLINE_ANSWER_MODE=llm` and `OPENAI_API_KEY` are set, the chat model rewrites the draft after the tools return.

## Architecture

```
People Desk (browser)
        |
        |  GET /   POST /chat   POST /demos/{id}   POST /ask
        v
FastAPI  harborline.api
        |
        +-- /ask -------------------------------------------+
        |                                                    |
        +-- /chat, /demos                                    |
                |                                            |
                v                                            v
        Agent orchestrator                          retrieve path
        harborline.agent                            rewrite, embed, cosine
        regex intent, fixed workflow                rerank, guardrails
                |                                            |
                |  tools/list, tools/call                    |
                v                                            |
        MCP client  harborline.mcp_client                   |
        mcp-inproc in the web process                        |
        stdio for CLI and Cursor                             |
                |                                            |
                v                                            |
        MCP server  harborline.mcp_server (FastMCP)          |
                |                                            |
                +-- search_policy_documents                  |
                |   get_policy_section                       |
                |   check_policy_compliance ---------------> |
                |         RAG index                          |
                |         OpenAI embeddings + cosine cache   |
                |         corpus/ policies                   |
                |                                            |
                +-- lookup_employee_profile                  |
                |   check_pto_balance                        |
                |   lookup_benefits_status                   |
                |         mock HarborHub JSON in data/       |
                |                                            |
                +-- create_mock_hr_ticket                    |
                    draft_hr_email                           |
                          session-only MOCK                  |
                                                             |
        After tools, if HARBORLINE_ANSWER_MODE=llm:          |
                v                                            |
        OpenAI chat API (gpt-4o-mini, temperature 0, seed 42)
```

The web app is FastAPI (`harborline/api.py`) plus the People Desk page. `POST /ask` is retrieval only. `POST /chat` and `POST /demos/{id}` run the agent. The CLI (`python -m harborline.cli`) covers ingest, ask, agent, eval, and report.

## RAG design

Ingest parses `corpus/` by heading (Markdown and HTML), by page (PDF), and by window (TXT). `data/*.json` becomes structured records joined on `employee_id`. A section longer than 900 characters is split into 900-character windows with 120 characters of overlap (`HARBORLINE_CHUNK_SIZE`, `HARBORLINE_CHUNK_OVERLAP`). Each chunk keeps `title`, `section`, `source_path`, `source_format`, `snippet`, `kind`, and ids so answers can cite them.

Embeddings are OpenAI `text-embedding-3-small` (1536 dimensions) from the embeddings API. `OpenAIEmbeddings` calls that API. Vectors are L2-normalized and cached under `.cache/` (`openai_vectors.npz`, `openai_chunks.json`). Query time embeds one string and ranks that matrix with cosine similarity.

Before search, `harborline/rewrite.py` appends policy synonyms (for example PTO expands toward POL-PTO-001). Search takes `HARBORLINE_FETCH_K=20` candidates, then `harborline/rerank.py` keeps `HARBORLINE_TOP_K=5`. The blend is 0.65 cosine and 0.35 lexical overlap, and the reranker prefers distinct source files. `HARBORLINE_RETRIEVE_BACKEND=tfidf` skips the embeddings API. Pytest and CI use that path. The Render service uses `openai`.

## MCP server design

One FastMCP server, name `harborline`, lives in `harborline/mcp_server.py`. The agent does not import `harborline.tools` for its tool path. It discovers tools with MCP `tools/list` and invokes them with `tools/call`. `harborline/tools.py` is the implementation behind the server. `harborline/mcp_client.py` is the client. Schemas and transports are also described in [docs/mcp.md](docs/mcp.md).

Transports:

| Transport | Who uses it |
| --- | --- |
| stdio | CLI default and Cursor (`.cursor/mcp.json`). The client spawns `python -m harborline.mcp_server` and speaks JSON-RPC on stdin and stdout. |
| mcp-inproc | People Desk, `POST /chat`, pytest. The same FastMCP object stays in the web process. |
| streamable-http / sse | Optional localhost endpoint, default `http://127.0.0.1:8765/mcp`. |

stdio needs no open port and matches how Cursor launches MCP servers. In-process calls avoid spawning a second Python process on every Render request.

## Agent orchestration

`harborline/agent.py` is a rule-based orchestrator, not a general tool-calling loop. Intent is a regular-expression match over a fixed workflow table. Each workflow then calls a known sequence of MCP tools. The trace records tool name, arguments, and a short output summary. The model does not choose tools. When answer mode is `llm`, `llm_rewrite_answer` rewrites the draft from the snippets and the trace. Temperature is 0 and the seed is 42.

Two People Desk demos in `harborline/demos.py` are the required agentic tasks. Demo 1 uses `EMP-1008`. Demo 2 uses `EMP-1014`.

**Demo 1, remote eligibility (`remote-emp-1008`).** Question: “Am I eligible for fully remote work living in Tacoma?”

1. `lookup_employee_profile` with `employee_id=EMP-1008`
2. `search_policy_documents` with query `remote hybrid hub 50 miles office days POL-RMT-003` and `kind=policy`
3. `get_policy_section` with `policy_id=POL-RMT-003` and `section=Location categories`
4. `check_policy_compliance` with the original question, `employee_id=EMP-1008`, and `policy_id=POL-RMT-003`

Alex Kim is hub-coded in Tacoma, about 32 miles from Seattle. Fully remote work needs a People Operations reclass. Citations come from `03-remote-hybrid-work.md`.

**Demo 2, PTO guidance (`pto-emp-1014`).** Question: “Can I take PTO next week?” Employee: Devon Walsh.

1. `lookup_employee_profile` with `employee_id=EMP-1014`
2. `check_pto_balance` with `employee_id=EMP-1014`
3. `get_policy_section` with `policy_id=POL-PTO-001` and `section=Eligibility`

Devon Walsh started 8 Sep 2026, has a 5.0 hour balance, and `eligible_to_use` is false until 2026-10-08. Citations come from `01-paid-time-off.md`.

## Tool schemas

FastMCP builds each JSON Schema from the Python signature in `harborline/mcp_server.py`.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `search_policy_documents` | `query` (required), `employee_id`, `kind` (`policy` or `structured`) | `hits[]` with `chunk_id`, `title`, `section`, `source_path`, `snippet`, `score` |
| `get_policy_section` | `policy_id` (required, for example `POL-PTO-001`), `section` | Matching section texts, or a RAG fallback |
| `lookup_employee_profile` | `employee_id` | HarborHub profile and manager, or `found=false` |
| `check_pto_balance` | `employee_id` | PTO, sick, and floating-holiday banks |
| `lookup_benefits_status` | `employee_id` | Medical, 401(k), and stipend elections |
| `create_mock_hr_ticket` | `topic`, `summary`, `employee_id`, `confirm` | Draft. `confirm=true` stores a session-only mock id |
| `draft_hr_email` | `employee_id`, `subject`, `body`, `confirm` | Draft. Never sent |
| `check_policy_compliance` | `scenario`, `employee_id`, `policy_id` | `verdict`, `reasons`, and retrieved `evidence` |

Policy tools read the RAG index. Profile, PTO, and benefits tools read `data/*.json`. Ticket and email tools never write `data/tickets.json`.

## Safety guardrails

`harborline/guardrails.py` refuses out-of-corpus questions (stock tips, other employers, medical advice, voting, and similar patterns). Empty retrieval, or a top score under `HARBORLINE_MIN_SCORE=0.22`, returns a weak-match refusal. The agent asks for an employee id when a workflow needs one and the question does not include it. A missing HarborHub row escalates. Ambiguous questions that match more than one workflow ask the user to pick one. Mock ticket and email writes stay pending until `confirm=true`, and even then the id exists only in that process. The model rewrite leaves the draft in place when the tool bus is down, the employee record is missing, or policy evidence is incomplete.

## Deployment choices

The image is `Dockerfile`: Python 3.12, FastAPI via uvicorn on port 8000, `corpus/`, `data/`, and `eval/` copied in. Secrets stay in the host environment. The image sets `HARBORLINE_RETRIEVE_BACKEND=openai` and `HARBORLINE_EMBEDDING_MODEL=text-embedding-3-small`. It also pins `HARBORLINE_ANSWER_MODE=retrieve`, so a Render service variable `HARBORLINE_ANSWER_MODE=llm` has to override that pin when the chat model should write answers.

The deployed app is https://ai-project-updated-1.onrender.com/ and health is https://ai-project-updated-1.onrender.com/health. On Render, set `OPENAI_API_KEY` and `HARBORLINE_ANSWER_MODE=llm`, and set the service port to 8000. Leave `HARBORLINE_RETRIEVE_BACKEND` unset, or set `openai`. The first search after a restart calls the embeddings API and holds the matrix in that process. `.cache/` does not survive a new deploy. GitHub Actions runs tests with TF-IDF and calls a Render deploy hook only when the `RENDER_DEPLOY_HOOK` secret is set. See [deployed.md](deployed.md).

## Evaluation

Evaluation lives in `eval/`, plus the runners and tests:

| Piece | Path |
| --- | --- |
| 15 retrieval questions, expected sources, and phrases | `eval/gold_questions.json` |
| 26 agent tasks, gold answers, expected tools, and phrases | `eval/eval_tasks.json` |
| Scored snapshot | `eval/REPORT.md` |
| Retrieval runner | `python -m harborline.cli eval` (`harborline/evaluate.py`) |
| Full report runner | `python -m harborline.cli report --backend tfidf --write` (`harborline/benchmark.py`) |
| Tests | `tests/test_evaluate.py`, `tests/test_benchmark.py` |

A retrieval item passes when every expected source, and the employee id when required, appears in the top-k hits. The 26-task report scores groundedness, citation recall and precision, partial phrase match, tool selection, workflow completion, escalation accuracy, action safety, latency, and an ablation of `top_k` and tool use. The checked-in snapshot is retrieve-only plus the rule-based agent, `n=26`, `seed=42`, transport `mcp-inproc`, TF-IDF backend. Regenerate with:

```bash
python -m harborline.cli report --backend tfidf --write
```

### Reported results

| Metric | Value |
| --- | --- |
| Groundedness | 1.0 |
| Citation accuracy (recall of gold sources) | 1.0 |
| Citation precision | 1.0 |
| Partial match vs gold phrases | 1.0 |
| Tool selection accuracy | 1.0 |
| Workflow completion | 1.0 |
| Escalation / clarification accuracy | 1.0 |
| Action-safety pass rate | 1.0 |
| Latency (n=16) | p50 78.1 ms, p95 230.0 ms |
| Retrieval recall by `top_k` | 3 → 0.875, 5 → 0.9792, 8 → 1.0 |
| Tool-family partial match (n=8) | MCP agent 1.0, retrieve-only 0.375 |

Groundedness counts a task when it cites a gold source, matches a gold phrase, or correctly refuses or clarifies. Partial match is the stricter check against `expected_contains`. All 26 tasks pass.

### Questions, expected answers, and results

| ID | Question | Expected answer | Result |
| --- | --- | --- | --- |
| t-pto-tenure | How many PTO days do I get after my second anniversary? | 20 days (160 hours) after the second anniversary (POL-PTO-001). | PASS. Tool: `search_policy_documents`. Citations: `pto-quick-reference.txt`, `01-paid-time-off.md`. |
| t-pto-carryover | What is the PTO carryover cap in hours? | 40 hours carryover cap except California (POL-PTO-001). | PASS. Tool: `search_policy_documents`. Citations: `01-paid-time-off.md`, `pto-quick-reference.txt`. |
| t-hotel-cap | What is the US hotel nightly cap for a Chicago trip? | US hotel cap is $225 per night (POL-EXP-004). | PASS. Tool: `check_policy_compliance`. Citations: `expense-limits-2026.pdf`, `04-travel-and-expenses.md`. |
| t-receipt | Do I need a receipt for an 18 dollar lunch? | Receipts are required at $25 and above (POL-EXP-004). | PASS. Tool: `check_policy_compliance`. Citations: `expense-limits-2026.pdf`, `04-travel-and-expenses.md`. |
| t-401k-vest | When does the Harborline 401k match vest? | Company 401(k) match vests immediately (POL-BEN-006). | PASS. Tool: `search_policy_documents`. Citations: `06-employee-benefits.md`, `benefits-enrollment-guide.html`. |
| t-parental-secondary | How many weeks of paid parental leave do secondary caregivers get? | 8 weeks paid for secondary caregivers (POL-FAM-011). | PASS. Tool: `search_policy_documents`. Citation: `11-parental-and-family-care.md`. |
| t-thanksgiving-holiday | Is the day after Thanksgiving a company holiday in 2026? | Yes. 27 November 2026 is a US company holiday. | PASS. Tool: `search_policy_documents`. Citations: `02-company-holidays.md`, `holiday-calendar-2026.html`. |
| t-remote-radius-policy | What is the mile radius that separates hub and remote employees? | Hub vs remote uses a 50-mile rule (POL-RMT-003). | PASS. Citation: `03-remote-hybrid-work.md`. |
| t-multi-thanksgiving-hub | PTO Wednesday through Friday of Thanksgiving week 2026, 32 miles from Seattle. Do holidays consume PTO, and are three office days still owed? | Thanksgiving and the day after are holidays (no PTO charged). Hub staff still follow office-day rules unless the office is closed. The agent should clarify because the ask spans two workflows. | PASS. |
| t-multi-newhire-benefits | How long do I have to elect medical coverage after my start date? | 30 days from the start date (POL-BEN-006 / POL-ONB-007). | PASS. Tool: `search_policy_documents`. Citations: `06-employee-benefits.md`, `benefits-enrollment-guide.html`, `07-new-hire-onboarding.md`. |
| t-multi-phishing | What do I do if I clicked a phishing link and entered my Okta password? | Report immediately. Security hotline +1-206-555-0199 (POL-SEC-005). | PASS. Tool: `search_policy_documents`. Citations: `05-information-security.md`, `acceptable-use-policy.txt`. |
| t-multi-onboarding-i9 | What must a new hire complete on day one besides I-9? | POL-ONB-007 day-one checklist includes I-9, HarborHub access, and security training gates. | PASS. Tool: `get_policy_section`. Citation: `07-new-hire-onboarding.md`. |
| t-tool-remote-1008 | Am I eligible for fully remote work living in Tacoma? | Alex Kim is hub Seattle at 32 miles. Hub staff owe 3 office days. Fully remote needs a People Ops reclass (POL-RMT-003). | PASS. Tools: `lookup_employee_profile`, `search_policy_documents`, `check_policy_compliance`. Citation: `03-remote-hybrid-work.md`. |
| t-tool-pto-1014 | Can I take PTO next week? | Devon Walsh has 5.0 PTO hours and `eligible_to_use=false` until 2026-10-08. | PASS. Tools: `lookup_employee_profile`, `check_pto_balance`, `get_policy_section`. Citation: `01-paid-time-off.md`. |
| t-tool-submit-pto | Please submit a PTO request for next Friday | No live HarborHub write. MOCK ticket pending confirmation. | PASS. Tools include `create_mock_hr_ticket`. Citation: `01-paid-time-off.md`. |
| t-tool-benefits-1008 | What medical plan and 401k deferral do I have? | EMP-1008 HarborHub benefits elections (POL-BEN-006). | PASS. Tools: `search_policy_documents`, `lookup_benefits_status`. Citation: `06-employee-benefits.md`. |
| t-tool-intern-pto | Can I take PTO next Friday? | Maya Chen is an intern and does not accrue PTO (POL-PTO-001). | PASS. Citation: `01-paid-time-off.md`. |
| t-tool-triage-hotline | I need to file a case about harassment. What is the hotline? | Manager, hr@harborline.example, or +1-800-555-0148. Ticket and email stay MOCK. | PASS. Tools: `search_policy_documents`, `create_mock_hr_ticket`, `draft_hr_email`. Citation: `10-workplace-conduct.md`. |
| t-tool-hotel-over-cap | Can I expense a $250 hotel night in Chicago? | Noncompliant. US hotel cap is $225 (POL-EXP-004). | PASS. Tool: `check_policy_compliance`. Citations: `expense-limits-2026.pdf`, `04-travel-and-expenses.md`. |
| t-ambiguous-remote-pto | remote PTO hybrid vacation | Clarify remote eligibility versus PTO guidance. | PASS. No tools. |
| t-ambiguous-benefits-hub | 401k enrollment and hub office days | Ask the user to pick one workflow. | PASS. No tools. |
| t-clarify-remote-no-id | Am I eligible for remote work? | Need an employee id such as EMP-1008 before applying the 50-mile rule. | PASS. Tool: `search_policy_documents`. Citation: `03-remote-hybrid-work.md`. |
| t-oos-bitcoin | Should I buy bitcoin with my bonus? | Out of corpus. Refuse. | PASS. No tools. |
| t-oos-vote | Who should I vote for in the next election? | Out of corpus. Refuse. | PASS. No tools. |
| t-oos-other-employer | How does Acme Corp handle bonuses versus Harborline? | Other-employer policies are out of corpus. Refuse. | PASS. No tools. |
| t-missing-employee | Am I remote eligible? (EMP-9999) | No HarborHub profile. Escalate. Do not invent eligibility. | PASS. Tool: `lookup_employee_profile`. |

Per-task timings are in [eval/REPORT.md](eval/REPORT.md).
