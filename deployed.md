# Deployment

The People Desk app is deployed on Render.

GitHub Actions runs the test job on every push and pull request. The deploy job runs only after tests pass, and only on pushes to the default branch. It calls a Render deploy hook when the Actions secret `RENDER_DEPLOY_HOOK` is set. Pull requests never deploy. The live service below was created on Render from this repo.

## URL

| Item | Value |
| --- | --- |
| Deployed app URL | https://ai-project-updated-1.onrender.com/ |
| Health endpoint | https://ai-project-updated-1.onrender.com/health |
| What `/health` should show | `retrieve_backend` of `openai`, `embedding_model` of `text-embedding-3-small`, `has_openai_key` true, `answer_mode` of `llm`, and `llm_answers` true. |

## Render settings

Create a Docker web service from this repo. Set the service port to **8000**. The image listens on 8000. Render’s Docker default is 10000, and a mismatch fails the health check.

Environment:

| Variable | Value |
| --- | --- |
| `OPENAI_API_KEY` | The same key used for embeddings and chat. Do not commit it. |
| `HARBORLINE_ANSWER_MODE` | `llm`. This overrides the image default of `retrieve`. |
| `HARBORLINE_RETRIEVE_BACKEND` | Leave unset, or set `openai`. Do not set `tfidf`, `pinecone`, or `faiss` on the service. |

`PINECONE_API_KEY`, `PINECONE_INDEX_HOST`, `PINECONE_INDEX_NAME`, and `PINECONE_NAMESPACE` are unused. Remove them if an older service still has them.

## Free-tier cold starts

A Render free-tier service sleeps after inactivity. The first HTTP request after sleep often waits 30–90 seconds before the process answers. That delay is the platform waking the container. It is not included in the local latency numbers in [eval/REPORT.md](eval/REPORT.md) (in-process MCP, p50 about 145 ms).

The first search after a fresh process also calls the OpenAI embeddings API for the corpus and keeps the matrix in memory. `.cache/` is local to the container and does not survive a new deploy, so that embed step runs again after each deploy and after a cold start that starts a new instance. Later questions in the same process only embed the query.
