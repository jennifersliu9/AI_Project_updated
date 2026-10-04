# Harborline demo transcript

Spoken walkthrough of the deployed People Desk app. Read the paragraphs aloud at about 150 words per minute. Lines marked **On screen** are actions and short pauses, not speech. The spoken text is 1,330 words, about 8 minutes 50 seconds. The pauses marked below are included in the clocks, and the full delivery lands at about 9 minutes 40 seconds.

Recorded against the live service on 4 October 2026: [https://ai-project-updated-1.onrender.com/](https://ai-project-updated-1.onrender.com/), answer mode `llm`, retrieval backend `openai`. Tool arguments and outputs are from `POST /demos/remote-emp-1008` and `POST /demos/pto-emp-1014` on that run. A later model rewrite can change sentence shape. The tool sequence, HarborHub facts, and cited files stay the same.

| Segment | Clock |
| --- | --- |
| Open and health | 0:00–0:50 |
| Design | 0:50–2:16 |
| Demo 1, remote eligibility | 2:16–4:50 |
| Demo 2, PTO guidance | 4:50–7:00 |
| Deployment and CI/CD | 7:00–8:23 |
| Evaluation | 8:23–9:28 |
| Close | 9:28–9:40 |

## Open

This is Harborline, employee Q&A for fictional Harborline Technologies. The handbook is a 2026 policy corpus, and the employee rows are mock HarborHub data. The deployed application is ai-project-updated-1.onrender.com. I will run two agentic tasks from start to finish. For each one I will name the MCP tools, the arguments, the outputs, the citations, and the final answer. Then I will cover design, deployment, CI/CD, and the evaluation results.

> **On screen:** Open https://ai-project-updated-1.onrender.com/ and point at the health line. Pause about 10 seconds. A free-tier cold start can take 30 to 90 seconds. Wait until health returns before continuing.

Health shows answer mode llm, so gpt-4o-mini writes the final wording. Retrieval uses OpenAI text-embedding-3-small. Eight MCP tools are available in process: search_policy_documents, get_policy_section, lookup_employee_profile, check_pto_balance, lookup_benefits_status, create_mock_hr_ticket, draft_hr_email, and check_policy_compliance.

## Design

The browser calls FastAPI. POST /ask is retrieval only. These demos use POST /demos, and the same questions also work on POST /chat. Those routes run the agent in harborline/agent.py.

The agent is a rule-based orchestrator. Intent is a regular-expression match, not a model decision. Each workflow then calls a fixed tool sequence. The model does not choose tools. After the tools return, gpt-4o-mini rewrites the draft. Temperature is 0 and the seed is 42. If the tool bus is down, the employee is missing, or policy evidence is incomplete, the draft stays as written.

The tools are one FastMCP server named harborline. The web process uses mcp-inproc. The agent discovers tools with tools/list and calls them with tools/call, and it only calls names from that list. Policy tools read the RAG index. Profile, PTO, and benefits tools read the mock JSON under data/. Ticket and email tools are session-only mocks. They never write data/tickets.json, and a write stays pending until confirm is true.

Chunks follow handbook headings. Sections longer than 900 characters split with 120 characters of overlap. Embeddings are text-embedding-3-small, ranked in process by cosine similarity. Search fetches 20 candidates and reranks to 5, at 0.65 cosine and 0.35 lexical. Pytest and CI use TF-IDF. The deployed service uses the OpenAI backend.

## Demo 1 — remote eligibility

> **On screen:** Click “Demo 1 · Remote eligibility · EMP-1008”. Pause about 10 seconds for POST /demos/remote-emp-1008. Point at the metadata line, then the four-step trace.

Demo 1 asks, “Am I eligible for fully remote work living in Tacoma?” The employee id is EMP-1008. The page shows intent remote_eligibility, workflow Remote work eligibility, transport mcp-inproc, and answer mode llm. Four MCP tools run, in order.

Tool one is lookup_employee_profile, argument employee_id EMP-1008. Output: found true. Alex Kim, Software Engineer, hub_seattle, home Tacoma, 32.0 miles from the hub, office OFF-SEA, start date 10 March 2024. Manager: Sofia Alvarez, EMP-1007. Trace summary: “Alex Kim hub_seattle”. The handbook does not store that location code, so this call comes first.

Tool two is search_policy_documents. Arguments: query “remote hybrid hub 50 miles office days POL-RMT-003”, kind policy. The tool appends “hub office 50 miles POL-RMT-003”. Output: five hits. Top files are corpus/03-remote-hybrid-work.md, corpus/08-equipment-and-it-assets.md, and corpus/04-travel-and-expenses.md. Neighboring files rank too, so the agent does not cite that list as-is.

Tool three is get_policy_section. Arguments: policy_id POL-RMT-003, section “Location categories”. Output: found true, “2. Location categories (part 1)” and “(part 2)”, both corpus/03-remote-hybrid-work.md. Hub — Seattle means a home address within 50 miles of 418 Occidental Ave S and at least three office days a week. Managers cannot reclassify someone as remote. That is a People Operations action. These two chunks are the citations.

Tool four is check_policy_compliance. The arguments are the original scenario, employee_id EMP-1008, and policy_id POL-RMT-003. The verdict is noncompliant_if_fully_remote, with five evidence chunks. The reason is: “Alex Kim is coded hub_seattle at 32.0 miles; hub staff owe at least 3 office days (POL-RMT-003).” That joins the profile from tool one to the hub rule from tool three.

> **On screen:** Point at the answer and the two citation rows. Pause about 5 seconds.

The model’s final answer says Alex Kim, EMP-1008, is coded hub_seattle, which requires at least three office days per week, Tuesday through Thursday by default. Tacoma is 32.0 miles from the hub, so fully remote work is noncompliant. Reclassification is a People Operations action, not a manager Slack exception. Citations are Remote and Hybrid Work Policy, section 2, Location categories, parts 1 and 2, file 03-remote-hybrid-work.md. The answer says this is the coded rule, not advice to move or recode yourself. Escalation is needed, reason location_reclassification_requires_people_ops. No ticket is filed.

## Demo 2 — PTO guidance

> **On screen:** Click “Demo 2 · PTO guidance · EMP-1014”. Pause about 10 seconds. Point at the three-step trace.

Demo 2 asks, “Can I take PTO next week?” The employee id is EMP-1014. Intent is pto_guidance, workflow PTO request guidance, same in-process transport, answer mode llm. This path does not call search or the compliance tool.

Tool one is lookup_employee_profile, argument employee_id EMP-1014. Output: found true. Devon Walsh, Account Executive, started 8 September 2026, hub_seattle, home Seattle, office OFF-SEA. Manager: Taylor Brooks, EMP-1012. Trace summary: “Devon Walsh hub_seattle”.

Tool two is check_pto_balance, argument employee_id EMP-1014. Output: found true, balance_hours 5.0, eligible_to_use false, eligible_to_use_on 2026-10-08, pto_eligible true. The note says one semi-monthly accrual of 5.00 hours, and use starts after 30 calendar days. Trace summary: “balance=5.0 eligible_to_use=False”. The handbook cannot supply this row.

Tool three is get_policy_section, arguments policy_id POL-PTO-001 and section Eligibility. The output is found true: “2. Eligibility (part 1)” and “2. Eligibility (part 2)” of corpus/01-paid-time-off.md. Regular full-time employees accrue PTO. Accrual starts on day one, and accrued PTO may be used after 30 calendar days. Those two chunks are the citations.

> **On screen:** Point at the answer and the two 01-paid-time-off.md citation rows. Pause about 5 seconds.

The final answer says Devon Walsh, EMP-1014, has 5.0 PTO hours and is not eligible to use them yet. Use generally starts on 2026-10-08. Requests go through HarborHub, Time Off, in one-hour increments. Citations are Paid Time Off Policy, section 2, Eligibility, parts 1 and 2, file 01-paid-time-off.md. The answer says it is not approving time off. Escalation needed is false, reason waiting_period. The agent does not call create_mock_hr_ticket, because the question is “can I”, not a request to submit. A submit would add that tool and leave the mock ticket pending until confirm is true. Nothing is written to data/tickets.json.

Those are the two tasks: four tools for remote eligibility, three for PTO, each with arguments, outputs, citations, and a final answer.

## Deployment and CI/CD

> **On screen:** Show Dockerfile, deployed.md, and .github/workflows/ci.yml. Pause about 5 seconds.

The image is Python 3.12. Uvicorn listens on port 8000. Render must use that port. The Docker default of 10000 fails the health check. The live URL is https://ai-project-updated-1.onrender.com/, and health is /health on that host. The image pins answer mode to retrieve, so HARBORLINE_ANSWER_MODE=llm on the service overrides it. The only secret is OPENAI_API_KEY, set on Render, not committed. Leave retrieval unset, or set openai. A free-tier instance sleeps, and the first request often waits 30 to 90 seconds. That wait is outside the local latency numbers. A new container embeds the corpus again. The vector cache does not survive the new process.

GitHub Actions runs on every push and pull request. The test job uses Python 3.12, imports the API and the MCP server, and asserts at least five tools. Pytest covers the API, answer mode, MCP, the agent, tools, ingest, the store, retrieval, and generation, with answer mode retrieve, backend tfidf, and seed 42, so CI does not call OpenAI. Deploy runs only after tests pass. Pull requests skip it. A push calls the Render hook only when the secret RENDER_DEPLOY_HOOK is set. If that secret is missing, the job logs a no-op and still succeeds.

## Evaluation

> **On screen:** Open eval/REPORT.md. Pause about 5 seconds.

The sheet scores 26 tasks, seed 42, transport mcp-inproc. It uses the rule-based agent in retrieve mode with TF-IDF, so the scores do not depend on the chat model. All 26 pass. Groundedness, citation recall, citation precision, partial phrase match, tool selection, workflow completion, escalation accuracy, and action safety are each 1.0. Local latency is a median of about 78.1 milliseconds and a 95th percentile of about 230 milliseconds. The cold first task is about 263.5 milliseconds. Retrieval recall is 0.875 at top_k 3, 0.9792 at top_k 5, and 1.0 at top_k 8. On eight tool-family tasks, the MCP agent scores 1.0 and retrieve-only, with no HarborHub tools, scores 0.375. That is why Demo 2 calls check_pto_balance. Search cannot know that Devon’s 5.0 hours are locked until 8 October 2026. The demos are gold tasks t-tool-remote-1008 and t-tool-pto-1014. Both pass, with the tool lists just shown, citing 03-remote-hybrid-work.md and 01-paid-time-off.md.

## Close

Harborline answers from a fixed orchestrator, MCP tools with recorded arguments and outputs, citations from the sections those tools returned, and a seeded model that only writes the final wording.
