# AI tooling

This repository was built and revised in Cursor, using Cursor cloud agents as the coding assistant. The assistant read the Harborline package, tests, and README, then edited Python, Docker, CI, and docs in the same tree. OpenAI’s API is the application’s embedding and chat provider. It was not used as the code editor.

## What was used

| Tool | How it was used |
| --- | --- |
| Cursor IDE and cloud agents | Explored `harborline/`, tests, and docs; applied patches; ran Python checks; opened pull requests. |
| Cursor agent shell | Ran `pytest` after installing it, compiled modules, and verified a live OpenAI embedding call without printing the key. |
| GitHub | Held the source, pull requests, and the Actions workflow that tests on every push. |
| OpenAI embeddings and chat APIs | Runtime only: `text-embedding-3-small` for vectors and `gpt-4o-mini` when `HARBORLINE_ANSWER_MODE=llm`. |

## What worked well

Reading `tests/test_store.py` and `tests/test_api.py` before the embedding change kept the client, `/health`, and the MCP child environment on one contract. The tests named the health fields and the env keys the process is allowed to forward. Matching those assertions meant the OpenAI path could land without a second rewrite of the API. Once `pytest` and `httpx` were installed, the store, API, and answer-mode tests passed on that change.

A live embed was the check the test suite cannot make. CI sets `HARBORLINE_RETRIEVE_BACKEND=tfidf`, so a green pytest run never calls OpenAI. One `text-embedding-3-small` request, using a local `.env` and never printing the key, returned a 1536-dimension vector with a norm of about 1.0. That confirmed the cloud model. The TF-IDF suite remains the offline check.

## What did not work well

FAISS was the original vector store, paired with a local MiniLM embedding model. The index and the model weights lived in the same process. That layout crashed with out-of-memory errors on a small host, which is the failure mode on Render. Retrieval moved to OpenAI `text-embedding-3-small`. The embedding model stays on OpenAI’s side. The process holds the returned 1536-dimension vectors, L2-normalizes them, and ranks them with cosine similarity. The matrix is cached under `.cache/`. That cache does not survive a new container, so the first search after a restart calls the embeddings API again. Pytest and CI use TF-IDF so they do not call the embeddings API.

Inline Python on Windows PowerShell failed while checking that embed. The shell stripped `$` variables and quotes, so a one-line `python -c` script was not the program that had been typed. Small script files were used instead, then deleted. The embedding check was fine. The shell quoting was what broke.

The first `pytest` run failed before any test executed, because that Python install had no `pytest`. That looked like a retrieval regression and was an environment gap. Installing `pytest` and `httpx` unblocked the suite.
