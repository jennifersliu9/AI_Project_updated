# Harborline MCP architecture

The agent does **not** call `harborline.tools` directly. It discovers tools with MCP `tools/list` and invokes them with MCP `tools/call`. The Python functions in `harborline.tools` are the implementation behind the MCP server only.

## Components

```
Cursor Agent / `python -m harborline.cli agent`
        │
        │  initialize → tools/list → tools/call
        ▼
harborline.mcp_client  (StdioMcpBus or InProcessMcpBus)
        │
        │  JSON-RPC MCP
        ▼
harborline.mcp_server  (FastMCP, name=`harborline`)
        │
        ├── search_policy_documents / get_policy_section / check_policy_compliance
        │       → RAG index (FAISS or TF-IDF) + citation metadata
        ├── lookup_employee_profile / check_pto_balance / lookup_benefits_status
        │       → mock HarborHub JSON under data/
        └── create_mock_hr_ticket / draft_hr_email
                → session-only MOCK operations (never tickets.json)
```

## Transport choice

| Transport | How it runs | Who uses it |
| --- | --- | --- |
| **stdio** (default) | Cursor or the CLI spawns `.venv` Python with `-m harborline.mcp_server` and speaks JSON-RPC on stdin/stdout | `.cursor/mcp.json`, `python -m harborline.cli agent` |
| **mcp-inproc** | Same FastMCP object in-process: `list_tools()` + `call_tool()` | pytest, `--transport mcp-inproc` |
| **streamable-http** / **sse** | Localhost HTTP MCP (`--host 127.0.0.1 --port 8765`) | Optional; you start the process yourself |

stdio is the supported Cursor transport. It needs no open port and matches how Cursor launches MCP servers.

Streamable HTTP (optional):

```powershell
python -m harborline.mcp_server --transport streamable-http --host 127.0.0.1 --port 8765
```

The MCP endpoint is then `http://127.0.0.1:8765/mcp`. Pointing Cursor at HTTP instead of stdio is an **EXTERNAL** settings change.

## How the agent discovers and calls tools

1. `open_mcp_bus("mcp-stdio")` starts `python -m harborline.mcp_server`.
2. The client sends MCP `initialize`.
3. The client sends `tools/list` and records `discovered_tools` plus each `inputSchema`.
4. Workflows call **only** names that appear in that list (`lookup_employee_profile`, `search_policy_documents`, …).
5. Each step is `tools/call` with JSON arguments. The operational trace records tool name, args, and a short output summary.

`python -m harborline.cli mcp-probe` prints the live discovery payload.

## Tool catalog (8 tools)

At least one tool uses the RAG index (`search_policy_documents`, `get_policy_section`, `check_policy_compliance`). At least one uses mock structured data (`lookup_employee_profile`, `check_pto_balance`, `lookup_benefits_status`) and at least one performs a mock write (`create_mock_hr_ticket`, `draft_hr_email`).

| Tool | Kind | Arguments | Returns |
| --- | --- | --- | --- |
| `search_policy_documents` | RAG retrieve | `query` (string), `employee_id` (string, optional), `kind` (string, optional: `policy` / `structured`) | `hits[]` with `chunk_id`, `title`, `section`, `source_path`, `snippet`, `score` |
| `get_policy_section` | RAG / index lookup | `policy_id` (string, e.g. `POL-PTO-001`), `section` (string, optional heading) | matching section texts, or RAG fallback |
| `lookup_employee_profile` | mock JSON | `employee_id` | HarborHub profile + manager, or `found=false` |
| `check_pto_balance` | mock JSON | `employee_id` | PTO / sick / floating-holiday banks |
| `lookup_benefits_status` | mock JSON | `employee_id` | medical / 401k / stipend elections |
| `create_mock_hr_ticket` | mock operation | `topic`, `summary`, `employee_id` optional, `confirm` bool | draft; `confirm=true` stores a session MOCK id only |
| `draft_hr_email` | mock operation | `employee_id`, `subject`, `body`, `confirm` bool | never sent; same confirmation rule |
| `check_policy_compliance` | RAG + mock | `scenario`, `employee_id` optional, `policy_id` optional | `verdict`, `reasons`, retrieved `evidence` |

FastMCP derives JSON Schema from those function signatures. Example for `search_policy_documents`:

```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "employee_id": { "type": "string", "default": "" },
    "kind": { "type": "string", "default": "" }
  },
  "required": ["query"]
}
```

## Cursor config

`.cursor/mcp.json` already points at this repo:

```json
{
  "mcpServers": {
    "harborline": {
      "command": "${workspaceFolder}/.venv/Scripts/python.exe",
      "args": ["-m", "harborline.mcp_server"],
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "env": {
        "HARBORLINE_RETRIEVE_BACKEND": "openai",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

On macOS/Linux the command is `${workspaceFolder}/.venv/bin/python`.

## FLAG — EXTERNAL steps (this chat cannot finish these)

1. **Enable MCP** in Cursor Settings and allow the `harborline` server when prompted.
2. **Restart Cursor** after any `.cursor/mcp.json` change so the stdio process is relaunched.
3. Confirm `.venv` exists and `pip install -r requirements.txt` plus `pip install -e .` have been run so `mcp` and `harborline` import in the server process.
4. For OpenAI retrieval in the Cursor-hosted server, put `OPENAI_API_KEY` in `.env`, then run `python -m harborline.cli ingest` once. Ingest calls the embeddings API and caches vectors under `.cache/`. Set `HARBORLINE_RETRIEVE_BACKEND=tfidf` in `mcp.json` to skip embeddings entirely.
5. LLM-worded answers: put `OPENAI_API_KEY` only in a local `.env`. Leave `HARBORLINE_ANSWER_MODE` unset so the server selects `llm`. Do not set `HARBORLINE_ANSWER_MODE` inside `env` in `mcp.json` — that block overrides `envFile` and would pin retrieve mode even when `.env` has a key. A key saved only in Cursor Settings → Models is for Cursor's chat, not this process. Never commit the key.
6. If you want Streamable HTTP instead of stdio, **you** start `python -m harborline.mcp_server --transport streamable-http` and add that URL in Cursor MCP settings.
7. A real HarborHub / email / ticket write is **not** in scope. `confirm=true` is still a session-only mock.
