# AI tooling

This repository was built and revised in Cursor, using Cursor cloud agents as the coding assistant. The assistant read the Harborline package, tests, and README, then edited Python, Docker, CI, and docs in the same tree. OpenAI’s API is the application’s embedding and chat provider. It was not used as the code editor.

## What was used

| Tool | How it was used |
| --- | --- |
| Cursor IDE and cloud agents | Explored `harborline/`, tests, and docs; applied patches; ran Python checks; opened pull requests. |
| Cursor agent shell | Ran `pytest` after installing it, compiled modules, and verified a live OpenAI embedding call without printing the key. |
| GitHub | Held the source, pull requests, and the Actions workflow that tests on every push. |
| OpenAI embeddings and chat APIs | Runtime only: `text-embedding-3-small` for vectors and `gpt-4o-mini` when `HARBORLINE_ANSWER_MODE=llm`. |

The sessions followed the retrieval and demo changes below. Each pass kept the previous tests and rewrote only the piece that had failed.

## How the code got here

Retrieval started as a local `all-MiniLM` model plus a FAISS index in the same process (`harborline` commit that added the FAISS store). FAISS held the corpus index in memory. On a small host that index, together with the local model weights, ran the process out of memory. That is why the local index was dropped.

The replacement is OpenAI `text-embedding-3-small`. The model stays on OpenAI’s side, so the Render process does not load sentence-transformer weights. Ingest requests those 1536-dimension vectors, L2-normalizes them, and ranks them with cosine similarity in process. The matrix is cached under `.cache/`. Pytest and CI still use TF-IDF so they do not call the embeddings API.

A hosted Pinecone index sat between those two designs. It stored the OpenAI vectors so the web process would not keep a FAISS index. It was removed because a deploy then needed a Pinecone key and index host in addition to `OPENAI_API_KEY`, and an empty namespace embedded the whole corpus inside one HTTP request. The current index is the OpenAI matrix in the People Desk process.

Answer mode went through the same kind of correction. An MCP env block had pinned `HARBORLINE_ANSWER_MODE=retrieve`, so a key in `.env` never turned on chat wording. The mode is now explicit: `llm` when the chat model should write the answer. The Docker image still pins `retrieve`, so the Render service sets `HARBORLINE_ANSWER_MODE=llm` itself. The live service is https://ai-project-updated-1.onrender.com/ and health is https://ai-project-updated-1.onrender.com/health.

People Desk demo 2 was first the EMP-1008 benefits question. The README HarborHub table already used EMP-1014, Devon Walsh, whose PTO is not usable until 8 Oct 2026. The demo, the design write-up, and the API test were changed to that PTO question so the page matches the README.

The design write-up was generated as a PDF and then as a Word file. Those files were removed. The write-up that remains is `design-and-evaluation.md`.

## What worked well

Reading `tests/test_store.py` and `tests/test_api.py` before the embedding swap kept the client, `/health`, and the MCP child environment on one contract. The reason that mattered is that FAISS, Pinecone, and the OpenAI cosine cache each changed where vectors lived. The tests named the health fields and the env keys the process is allowed to forward. Matching those assertions meant the OpenAI path could land without a second rewrite of the API. Once `pytest` and `httpx` were installed, the store, API, and answer-mode tests passed on that change.

A live embed was the check the test suite cannot make. CI sets `HARBORLINE_RETRIEVE_BACKEND=tfidf`, so a green pytest run never calls OpenAI. One `text-embedding-3-small` request, using a local `.env` and never printing the key, returned a 1536-dimension vector with a norm of about 1.0. That confirmed the cloud model that replaced FAISS. The TF-IDF suite remains the offline check.

Fixed workflows in `harborline/agent.py` are why demo 2 could change without the model choosing tools. Intent is a regular expression, and each workflow calls a known MCP sequence. The PTO workflow already looked up the employee, checked the HarborHub balance, and read POL-PTO-001. Pointing demo 2 at “Can I take PTO next week?” for EMP-1014 reused that sequence. The design write-up and the API test can assert the same tool names and the 8 Oct 2026 date because the model only rewrites the draft.

## What did not

FAISS was the original vector store, paired with a local MiniLM embedding model. The index lived in the process. That layout crashed with out-of-memory errors once the corpus index and the local weights were both loaded, which is the failure mode on a small Render instance. Retrieval therefore moved to OpenAI `text-embedding-3-small`. The embedding model stays on OpenAI’s side. The process only holds the returned vectors and ranks them with cosine similarity. The Pinecone host that briefly stored those vectors was removed so the deploy depends on `OPENAI_API_KEY` alone. The remaining limit is real. `.cache/` does not survive a new container, so the first search after a restart calls the embeddings API again.

Inline Python on Windows PowerShell failed while checking that embed. The shell stripped `$` variables and quotes, so a one-line `python -c` script was not the program that had been typed. Small script files were used instead, then deleted. The embedding check was fine. The shell quoting was what broke.

The first `pytest` run failed before any test executed, because that Python install had no `pytest`. That looked like a retrieval regression and was an environment gap. Installing `pytest` and `httpx` unblocked the suite.

The Docker image and the README disagreed about answer mode. The image sets `HARBORLINE_ANSWER_MODE=retrieve`. A key in the environment does not by itself turn on LLM answers in that image. Render has to set `HARBORLINE_ANSWER_MODE=llm` to override the pin. Earlier wording described the image as if the key alone selected `llm`. The service env is what selects it on the deployed app. Actions auto-deploys only when `RENDER_DEPLOY_HOOK` is set.
