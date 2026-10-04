# AI tooling

This repository was built and revised in Cursor, using Cursor cloud agents as the coding assistant. The assistant read the Harborline package, tests, and README, then edited Python, Docker, CI, and docs in the same tree. OpenAI’s API is the application’s embedding and chat provider. It was not used as the code editor.

## What was used

| Tool | How it was used |
| --- | --- |
| Cursor IDE and cloud agents | Explored `harborline/`, tests, and docs; applied patches; ran Python checks; opened pull requests. |
| Cursor agent shell | Ran `pytest` after installing it, compiled modules, and verified a live OpenAI embedding call without printing the key. |
| GitHub | Held the source, pull requests, and the Actions workflow that tests on every push. |
| OpenAI embeddings and chat APIs | Runtime only: `text-embedding-3-small` for vectors and `gpt-4o-mini` when `HARBORLINE_ANSWER_MODE=llm`. |

The agent sessions that touched this repo covered the move from a hosted Pinecone index to in-process cosine search over OpenAI embeddings, Render environment notes, the README answer-mode wording, and these submission documents.

## What worked well

Reading the existing tests before editing kept the embedding client, health payload, and MCP env forwarding aligned with `tests/test_store.py` and `tests/test_api.py`. After `pytest` was installed, the store, API, and answer-mode tests passed on the OpenAI backend change.

A live embed check confirmed the cloud embeddings path: one call to `text-embedding-3-small` returned a 1536-dimension vector with a norm of about 1.0. That check used a local `.env` and did not print the key.

Keeping orchestration in `harborline/agent.py` as explicit workflows made the two People Desk demos easy to trace. The model rewrites the draft. It does not pick tools, so the tool sequence in [design-and-evaluation.md](design-and-evaluation.md) matches the code.

## What did not

Inline Python on Windows PowerShell broke when the shell stripped `$` variables and quotes. Small script files were more reliable than one-line `python -c` snippets.

The first test run failed because that Python install had no `pytest`. Installing `pytest` and `httpx` unblocked the suite. The failure was the environment, not the retrieval change.

Pinecone was wired as the vector store and then removed. A Render deploy still needed `OPENAI_API_KEY`, `PINECONE_API_KEY`, and an index host, and an empty namespace would embed the whole corpus inside one HTTP request. Replacing that with the OpenAI embeddings API plus a local cosine cache dropped the second vendor. The tradeoff is that the matrix lives in the process and the cache does not survive a new container.

The Docker image still sets `HARBORLINE_ANSWER_MODE=retrieve`. A key in the environment does not by itself turn on LLM answers in that image. Render has to set `HARBORLINE_ANSWER_MODE=llm` to override the pin. Early docs described the image as if the key alone selected `llm`, which did not match the Dockerfile.

The deployed app is https://ai-project-updated-1.onrender.com/ and health is https://ai-project-updated-1.onrender.com/health. Actions still auto-deploys only when `RENDER_DEPLOY_HOOK` is set.
