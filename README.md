# Harborline

Employee Q&A for fictional **Harborline Technologies**. The repo holds a 2026 policy corpus, HarborHub-style employee records, and a seeded retrieval app that answers questions about PTO, holidays, remote work, expenses, security, benefits, onboarding, equipment, leave, and conduct.

Answer mode is `HARBORLINE_ANSWER_MODE=llm`. With `OPENAI_API_KEY` set, the chat model writes the answer from retrieved snippets. Retrieval calls OpenAI `text-embedding-3-small` and ranks those vectors in process, so that key is required unless you set `HARBORLINE_RETRIEVE_BACKEND=tfidf`. `HARBORLINE_ANSWER_MODE=retrieve` keeps extractive quotes. Tickets and emails are session-only mocks; nothing is written to HarborHub or to `data/tickets.json`.

Deeper references:

| Topic | Where |
| --- | --- |
| Policy set | [corpus/README.md](corpus/README.md) |
| Mock HarborHub rows | [data/README.md](data/README.md) |
| MCP transports, schemas, Cursor config | [docs/mcp.md](docs/mcp.md) |
| Retrieval gold set (15 questions) | [eval/gold_questions.json](eval/gold_questions.json) |
| Agent gold set (26 tasks) | [eval/eval_tasks.json](eval/eval_tasks.json) |
| Latest scored report | [eval/REPORT.md](eval/REPORT.md) |
| Architecture, RAG, MCP, guardrails, evaluation | [design-and-evaluation.md](design-and-evaluation.md) |
| AI coding tools used on this repo | [ai-tooling.md](ai-tooling.md) |
| Deployed URL and cold starts | [deployed.md](deployed.md) |

## What is in the repo

**Policies.** Twelve handbook documents (`POL-HB-000` through `POL-FAM-011`) plus HTML, TXT, and PDF companions. Shared facts are repeated on purpose: 15/20/25 PTO days by tenure, a 40-hour carryover cap, 11 holidays plus 2 floating days, a 50-mile hub rule, $225 US hotel / $75 meal caps, receipts at $25, immediate 401(k) vesting, and 16/8 weeks of parental leave. Corpus revision date is 1 September 2026. Rebuild the two PDFs with `python scripts/build_pdfs.py`.

**HarborHub records (as of 21 September 2026).** Three offices, 16 employees (`EMP-1001`–`EMP-1016`), matching PTO banks and benefits elections, and 14 existing tickets. Join on `employee_id`. Two rows the agent demos use:

| ID | Who | What the row is for |
| --- | --- | --- |
| EMP-1008 | Alex Kim, software engineer, Tacoma | Hub Seattle at 32 miles; fully remote needs a People Ops reclass |
| EMP-1014 | Devon Walsh, account executive, started 8 Sep 2026 | PTO not usable until 8 Oct 2026; benefits election still open |

**App.** Heading-aware chunking, OpenAI `text-embedding-3-small` embeddings, an in-process cosine index, query rewrite, lexical rerank, citations, and corpus guardrails. The Render process calls the OpenAI embeddings API and keeps the resulting matrix in memory. A rule-based HR agent calls eight tools through MCP. FastAPI serves a People Desk chat page plus `/chat`, `/ask`, `/health`, `/demos`, and `/eval`. GitHub Actions installs, starts the app, runs pytest, and deploys only after tests pass.

## Architecture

```
People Desk UI  /  CLI ask  /  CLI agent  /  POST /chat
        │
        ├─ ask  → rewrite → retrieve (OpenAI embeddings or TF-IDF) → rerank → guardrails → cited answer
        │
        └─ agent → intent → MCP tools/list + tools/call
                    │
                    ├─ search_policy_documents, get_policy_section, check_policy_compliance
                    │     RAG index (corpus/ + data/*.json)
                    ├─ lookup_employee_profile, check_pto_balance, lookup_benefits_status
                    │     mock HarborHub JSON
                    └─ create_mock_hr_ticket, draft_hr_email
                          session-only MOCK (never tickets.json)
```

The agent classifies intent locally, then calls only tools the MCP server listed. Hard-coded `harborline.tools` imports are the server implementation and the `cli tool` debugger. They are not the agent path.

| Intent | Needs an employee id | Tools |
| --- | --- | --- |
| Remote work eligibility | yes | `lookup_employee_profile`, `search_policy_documents`, `check_policy_compliance` |
| PTO guidance (submit stays mock) | yes | `lookup_employee_profile`, `check_pto_balance`, `get_policy_section`, optional `create_mock_hr_ticket` |
| Benefits | when the question is personal | `search_policy_documents`, `lookup_benefits_status` |
| Expense compliance | no | `check_policy_compliance` |
| Onboarding | no | `get_policy_section` |
| HR case triage | no | `search_policy_documents`, `create_mock_hr_ticket`, `draft_hr_email` |
| Single-policy Q&A | no | `search_policy_documents` |
| Ambiguous, out of corpus, or unknown employee | — | clarify, refuse, or escalate; no invented facts |

## Prerequisites

- Python 3.11+ (3.12 is what CI uses)
- Optional: Conda (`environment.yml`)
- Optional: Docker
- Optional: an OpenAI-compatible API key, read from the environment and never committed

## Setup

From the repository root.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt   # pytest, httpx
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt
```

Conda:

```bash
conda env create -f environment.yml
conda activate harborline
pip install -e .
```

Secrets stay in a local `.env`. Copy the example and edit it. `.gitignore` already excludes `.env`, `.venv/`, and `.cache/`.

```bash
cp .env.example .env
```

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | For OpenAI retrieval and LLM answers | empty | Embeddings and optional chat |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Chat model |
| `OPENAI_BASE_URL` | No | OpenAI | Azure or another compatible gateway |
| `HARBORLINE_ANSWER_MODE` | No | `llm` | `llm` writes the answer; `retrieve` quotes the corpus |
| `HARBORLINE_SEED` | No | `42` | Eval sampling and LLM seed |
| `HARBORLINE_CHUNK_SIZE` | No | `900` | Deterministic window |
| `HARBORLINE_CHUNK_OVERLAP` | No | `120` | Overlap between windows |
| `HARBORLINE_TOP_K` | No | `5` | Hits returned after rerank |
| `HARBORLINE_FETCH_K` | No | `20` | Candidate pool before rerank |
| `HARBORLINE_REWRITE` | No | `true` | Query expansion |
| `HARBORLINE_RERANK` | No | `true` | Lexical overlap plus diverse sources |
| `HARBORLINE_MIN_SCORE` | No | `0.22` | Guardrail floor |
| `HARBORLINE_RETRIEVE_BACKEND` | No | `openai` | `openai` or `tfidf` |
| `HARBORLINE_EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embeddings model |

## RAG pipeline

Activate the virtualenv and run commands from the repo root. The default `openai` backend needs `OPENAI_API_KEY`. `HARBORLINE_RETRIEVE_BACKEND=tfidf` needs no keys.

**Ingest** parses `corpus/` (Markdown and HTML by heading, PDF by page, TXT by window) and `data/*.json` as structured records. Sections longer than the window are split into 900-character chunks with 120-character overlap. Embeddings are OpenAI **text-embedding-3-small**, requested from the embeddings API. Vectors are L2-normalized and cached under `.cache/` (`openai_vectors.npz` and `openai_chunks.json`). Query time embeds one string and ranks that matrix with cosine similarity. The guardrail floor is `0.22`. Each chunk keeps `title`, `section`, `source_path`, `source_format`, `snippet`, `kind`, and ids so `ask` can cite them.

```bash
python -m harborline.cli ingest
```

**Ask** retrieves, prints citations, and refuses questions outside the corpus.

```bash
python -m harborline.cli ask "How many PTO days do I get after my second anniversary?"
python -m harborline.cli ask "Can I use PTO tomorrow?" --employee-id EMP-1014
python -m harborline.cli ask "What is the US hotel cap?" --json
python -m harborline.cli ask "Should I buy bitcoin with my bonus?"
python -m harborline.cli ask "I live 32 miles from the Seattle office and want PTO Wednesday through Friday of Thanksgiving week 2026. Do I lose PTO hours for the company holidays, and do I still owe three office days that week?"
```

Optional filters: `--kind policy`, `--kind structured`, `--source-format md`.

Skipping ingest makes the first `ask` embed the corpus on the fly and write the cache. Set `HARBORLINE_RETRIEVE_BACKEND=tfidf` to skip embeddings entirely (this is what pytest and CI use).

**LLM answers** keep the same retrieval, rewrite, rerank, citations, and guardrails. Put `OPENAI_API_KEY` in `.env` and set `HARBORLINE_ANSWER_MODE=llm`. Temperature is `0` and `seed` is `42`. The model writes the answer from the retrieved snippets. Set `OPENAI_BASE_URL` for Azure or another gateway. `/health` reports `answer_mode` of `llm`, `has_openai_key`, and `llm_answers` true.

## Agent

`python -m harborline.cli agent` picks a workflow, decides whether retrieval alone is enough, calls MCP tools, and prints an operational trace: discovered tools, selected tools, arguments, output summaries, sources, and any escalation. The trace is a log of those steps.

```bash
python -m harborline.cli agent "Am I eligible for fully remote work living in Tacoma?" --employee-id EMP-1008 --backend tfidf
python -m harborline.cli agent "Can I take PTO next week?" --employee-id EMP-1014 --backend tfidf
python -m harborline.cli agent "Can I take PTO next week?" --employee-id EMP-1014 --backend tfidf --transport mcp-inproc
```

Default transport is **stdio**: the CLI spawns `python -m harborline.mcp_server`. `--transport mcp-inproc` stays in one process (pytest and `/chat` use this). Missing employee ids, thin policy evidence, ambiguous requests, and an unavailable MCP bus return a clarification or escalation. `--confirm` accepts a **session-only MOCK** ticket or email id in that process. It does not update `data/tickets.json`.

Direct tool calls (debugging the implementation, not the agent path):

```bash
python -m harborline.cli mcp-probe --transport mcp-stdio --backend tfidf
python -m harborline.cli tool lookup_employee_profile EMP-1008
python -m harborline.cli tool search_policy_documents "PTO carryover 40 hours" --kind policy --backend tfidf
python -m harborline.cli tool get_policy_section POL-PTO-001 --section Eligibility
python -m harborline.cli tool create_mock_hr_ticket --topic pto_request --employee-id EMP-1008 --summary "Friday off"
```

## Local run

From the repository root, with the virtualenv active.

Offline lexical search (no API key):

```bash
python -m harborline.cli ask "How many PTO days do I get after my second anniversary?" --backend tfidf
python -m harborline.cli agent "Am I eligible for fully remote work living in Tacoma?" --employee-id EMP-1008 --backend tfidf
```

OpenAI embeddings and LLM answers (needs `OPENAI_API_KEY` in `.env` and `HARBORLINE_ANSWER_MODE=llm`):

```bash
python -m harborline.cli ingest
python -m harborline.cli ask "When does the 401k match vest?"
python -m harborline.cli agent "What medical plan and 401k deferral do I have?" --employee-id EMP-1008
```

People Desk HTTP. Run ingest first when the backend is `openai`, then:

```bash
uvicorn harborline.api:app --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The page has two grader demos: EMP-1008 remote eligibility, and EMP-1014 PTO (not usable until 8 Oct 2026). The same calls over HTTP:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/demos
curl -X POST http://127.0.0.1:8000/demos/remote-emp-1008
curl -X POST http://127.0.0.1:8000/demos/pto-emp-1014
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"Am I eligible for fully remote work living in Tacoma?\",\"employee_id\":\"EMP-1008\"}"
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"Can I take PTO next week?\",\"employee_id\":\"EMP-1014\"}"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"When does the 401k match vest?\"}"
```

| Route | Behavior |
| --- | --- |
| `GET /` | People Desk chat UI |
| `GET /health` | App status plus `mcp.available` and discovered tool names |
| `GET /demos` | The two grader prompts |
| `POST /chat` | MCP agent. Returns `answer`, `citations`, `snippets`, and `trace` |
| `POST /ask` | Retrieve-only. No agent |
| `GET /eval` | Seeded retrieval eval (`recall@k`) |

## MCP

Full write-up: [docs/mcp.md](docs/mcp.md). Server entrypoint: `python -m harborline.mcp_server` (FastMCP, name `harborline`).

| Transport | How it runs |
| --- | --- |
| **stdio** (default) | Cursor or the CLI spawns `.venv` Python with `-m harborline.mcp_server` |
| **mcp-inproc** | Same FastMCP object in-process (`list_tools` + `call_tool`) |
| **streamable-http** / **sse** | You start it: `python -m harborline.mcp_server --transport streamable-http --host 127.0.0.1 --port 8765` (endpoint `http://127.0.0.1:8765/mcp`) |

`.cursor/mcp.json` already defines two servers:

- `harborline` — stdio via `${workspaceFolder}/.venv/Scripts/python.exe` (Windows). On macOS/Linux change `command` to `${workspaceFolder}/.venv/bin/python`. Env sets the OpenAI retrieve backend. `envFile` points at `.env`.
- `harborline-http` — Streamable HTTP at `http://127.0.0.1:18765/mcp/`. Start the server on that port yourself if you use this entry (`--port 18765`).

Eight tools: `search_policy_documents`, `get_policy_section`, `check_policy_compliance`, `lookup_employee_profile`, `check_pto_balance`, `lookup_benefits_status`, `create_mock_hr_ticket`, `draft_hr_email`.

Steps that stay on your machine:

1. Install dependencies so the Cursor-hosted process can import `mcp` and `harborline`.
2. Enable MCP in Cursor Settings and allow the `harborline` server.
3. Restart Cursor after editing `.cursor/mcp.json`.
4. For OpenAI retrieval inside that server, put `OPENAI_API_KEY` in `.env` and run `python -m harborline.cli ingest` once, or set `HARBORLINE_RETRIEVE_BACKEND=tfidf` in `mcp.json`.
5. For LLM wording, put `OPENAI_API_KEY` in `.env` and set `HARBORLINE_ANSWER_MODE=llm`.
6. Start Streamable HTTP yourself if you want that transport. The CLI agent does not need Cursor Settings.

## Evaluation

`eval` scores **recall@k** on `eval/gold_questions.json` (15 items): a question passes when every expected source (and employee id, when required) appears in the top-k hits.

`report` scores `eval/eval_tasks.json` (26 tasks: 8 policy Q&A, 4 multi-document, 8 tool workflows, 3 ambiguous, 3 out of scope). It reports answer quality, agent behavior, latency, and an ablation. `--write` refreshes `eval/latest_report.json` and `eval/REPORT.md`. `--limit` samples with `HARBORLINE_SEED` (default 42).

```bash
python -m harborline.cli eval
python -m harborline.cli eval --limit 8 --json
python -m harborline.cli report --backend tfidf --write
pytest
```

`cli eval` uses the OpenAI embedding index after ingest. `pytest` and CI use TF-IDF so they stay offline.

### Latest report

Regenerate with the command above. The checked-in snapshot (`eval/REPORT.md`) is retrieve-only plus the rule-based agent, `n=26`, `seed=42`, transport `mcp-inproc`, TF-IDF backend.

| Metric | Value |
| --- | --- |
| Groundedness | 1.0 |
| Citation accuracy (recall of gold sources) | 0.9375 |
| Citation precision | 0.3679 |
| Partial match vs gold phrases | 0.7308 |
| Tool selection accuracy | 1.0 |
| Workflow completion | 0.9231 |
| Escalation / clarification accuracy | 0.9231 |
| Action-safety pass rate | 1.0 |
| Latency (n=16) | p50 145.1 ms, p95 394.1 ms |
| Cold first task / warm p50 / warm p95 | 637.3 ms / 131.6 ms / 259.2 ms |
| Retrieval recall by `top_k` | 3 → 0.875, 5 → 0.9792, 8 → 1.0 |
| Tool-family partial match (n=8) | MCP agent 1.0, retrieve-only 0.375 |

Groundedness counts a task when it cites a gold source, matches a gold phrase, or correctly refuses or clarifies. Partial match is the stricter check against `expected_contains`. Local latency is in-process MCP. A free-tier host that sleeps adds its own cold start (often 30–90 seconds) on the first HTTP request; that delay is not in these numbers.

Two policy-QA tasks in that snapshot (`t-pto-tenure`, `t-pto-carryover`) are marked FAIL on partial phrase match while still grounded with citation recall 1.0. Per-task lines are in [eval/REPORT.md](eval/REPORT.md).

## Deployment

The live service is [https://ai-project-updated-1.onrender.com/](https://ai-project-updated-1.onrender.com/). Health is [https://ai-project-updated-1.onrender.com/health](https://ai-project-updated-1.onrender.com/health). Cold-start notes are in [deployed.md](deployed.md).

The image serves FastAPI on Render. Pass secrets at runtime. Do not bake keys into the image. The image defaults to the OpenAI embedding backend. Its Dockerfile sets `HARBORLINE_ANSWER_MODE=retrieve`, so the Render service variable must override that. Set `OPENAI_API_KEY` and `HARBORLINE_ANSWER_MODE=llm` on the service. The model then writes the answer. The image copies `corpus/`, `data/`, and `eval/`. The first search after a restart calls the embeddings API and holds the matrix in that process. `.cache/` is local to the container and does not survive a new deploy. `/health` should show `answer_mode` of `llm` and `llm_answers` true.

```bash
docker build -t harborline-qa .
docker run --rm -p 8000:8000 --env-file .env -e HARBORLINE_ANSWER_MODE=llm harborline-qa
```

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request, with `HARBORLINE_ANSWER_MODE=retrieve`, `HARBORLINE_RETRIEVE_BACKEND=tfidf`, and seed 42.

1. Install `requirements.txt`, `requirements-dev.txt`, and `pip install -e .` on Python 3.12.
2. Import check: load `harborline.api:app` and the MCP server factory (at least five tools).
3. `pytest` for the API, MCP discovery and `lookup_employee_profile`, the agent, tools, ingest, retrieval eval, and generation.
4. The deploy job runs only after that test job succeeds. Pull requests never deploy. On push, CI calls a Render deploy hook only when the `RENDER_DEPLOY_HOOK` GitHub Actions secret is set. The live service is https://ai-project-updated-1.onrender.com/ and its health check is https://ai-project-updated-1.onrender.com/health.

## Reproducibility

| Knob | Default | Effect |
| --- | --- | --- |
| `HARBORLINE_SEED` | 42 | `random`, NumPy, eval sampling, LLM `seed` |
| `HARBORLINE_CHUNK_SIZE` / `OVERLAP` | 900 / 120 | The same text always yields the same chunks |
| `PYTHONHASHSEED` | set by `Settings.apply_seeds()` | Stable hashing in-process |
| OpenAI embeddings | `text-embedding-3-small`, cosine over a cached matrix | Vectors come from the embeddings API |
| TF-IDF | optional | Stable sort: score descending, `chunk_id` ascending |
| LLM | temperature 0, seed 42 | When `HARBORLINE_ANSWER_MODE=llm` and `OPENAI_API_KEY` is set |

## Layout

```
corpus/                  Policies (md, html, txt, pdf) and corpus/README.md
data/                    Mock offices, employees, PTO, benefits, tickets
eval/                    Gold questions, rubrics, REPORT.md
harborline/              Parse, chunk, OpenAI embed, cosine index, ask, tools, agent, API
harborline/mcp_server.py MCP server and tool definitions
harborline/mcp_client.py MCP client used by the agent
harborline/tools.py      Tool implementations behind the MCP server
harborline/evaluate.py   Retrieval eval runner
harborline/benchmark.py  26-task report runner
docs/mcp.md              MCP transport, schemas, discovery
design-and-evaluation.md Architecture, RAG, MCP, guardrails, eval results
ai-tooling.md            AI coding tools used on this repo
deployed.md              Public URL, health URL, cold-start notes
.cursor/mcp.json         Cursor MCP config (enable the server in Settings)
scripts/build_pdfs.py    Rebuild the companion policy PDFs
tests/                   Ingest, eval, tools, MCP, agent, API
.github/workflows/ci.yml Install, test, then optional Render deploy
requirements.txt         Runtime pins
requirements-dev.txt     pytest, httpx
environment.yml          Conda env
pyproject.toml           Package metadata (harborline 0.1.0)
.env.example             Variable names only
Dockerfile
```
