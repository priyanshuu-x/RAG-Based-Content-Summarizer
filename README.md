# 🦜 RAG Content Summarizer

🔗 **Live demo:** [rag-content-summarizer.streamlit.app](https://rag-content-summarizer.streamlit.app)

A production-hardened Retrieval-Augmented Generation (RAG) app that summarizes and answers questions about **YouTube videos** and **websites** — built with LangChain, FAISS, Groq, and Streamlit .

Paste a URL, get a summary, then ask follow-up questions grounded in the actual source content — with citations showing exactly which chunks the answer came from.

---

## Features

- **YouTube summarization** — pulls the video transcript and summarizes it
- **Website summarization** — fetches and cleans page content (strips navigation/sidebar/footer boilerplate) before summarizing
- **Question answering (RAG)** — ask follow-up questions; answers are grounded only in the retrieved content
- **Hybrid retrieval** — combines BM25 (keyword search) with FAISS (semantic vector search) via an `EnsembleRetriever`, so both exact keyword matches and semantic similarity contribute to what gets retrieved
- **Source attribution** — every answer includes an expandable panel showing the exact chunks used, for transparency and trust
- **Persistent vector store** — FAISS indexes are cached to disk per URL (hashed), so reprocessing the same link is instant instead of re-embedding from scratch
- **Rate limiting** — per-session and global request limits protect the shared Groq API budget from abuse
- **Secrets-based API key handling** — reads `GROQ_API_KEY` from Streamlit secrets or environment variables; falls back to manual entry only if neither is set, so end users never need their own key
- **Specific error handling** — YouTube failures (disabled/missing transcripts, unavailable videos, malformed URLs) and website failures (timeouts, unreachable pages) surface clear messages instead of a generic crash

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain (`langchain`, `langchain-classic`, `langchain-community`) |
| LLM | Groq (`llama-3.1-8b-instant` via `langchain-groq`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (`faiss-cpu`) |
| Keyword retrieval | BM25 (`rank_bm25`) |
| Web scraping | `requests` + `BeautifulSoup` (custom content cleaner) |
| YouTube transcripts | `youtube-transcript-api` |
| Containerization | Docker (used for local testing/portability; not used by the current hosting platform) |
| Deployment | Streamlit Community Cloud (auto-deploys from GitHub on push) |
| CI | GitHub Actions — lint, test, and a Docker build check |
| Testing | `pytest` |
| Linting | `ruff` |

---

## Project Structure

```
rag-content-summarizer/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Adds pytest + ruff for CI/local dev
├── Dockerfile                      # Container build definition
├── .dockerignore
├── .gitignore
├── README.md
├── .streamlit/
│   └── secrets.toml.example        # Template — copy to secrets.toml locally, never commit the real one
├── tests/
│   └── test_app.py                 # pytest suite
└── .github/
    └── workflows/
        └── ci.yml                  # Lint → test → Docker build check (no deploy job — see below)
```

---

## How It Works (Architecture)

1. **Input** — user pastes a YouTube or website URL
2. **Load & clean**
   - YouTube: transcript pulled via `youtube-transcript-api`
   - Website: fetched via `requests`, parsed with `BeautifulSoup`; script/style/nav/footer/aside tags and known boilerplate containers (e.g. Wikipedia's sidebar) are stripped; semantic containers (`<article>`, `<main>`, or site-specific content divs) are preferred when present
3. **Chunking** — content is split via `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
4. **Embedding + indexing** — chunks are embedded with a HuggingFace sentence-transformer and stored in a FAISS index, cached to disk keyed by a hash of the URL
5. **Summary** — the first ~4000 characters are sent to the Groq LLM for an initial summary
6. **Q&A** — user questions are run through a **hybrid retriever** (BM25 + FAISS via `EnsembleRetriever`, weighted 0.4/0.6) to pull the most relevant chunks, which are passed to the LLM as context; the retrieved chunks are also shown to the user as sources

---

## Local Setup

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd rag-content-summarizer
```

### 2. Create a virtual environment (Python 3.11 — matches the Docker image)
```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements-dev.txt
```
(`requirements-dev.txt` includes `requirements.txt` plus `pytest` and `ruff`.)

### 4. Set your Groq API key
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Edit `.streamlit/secrets.toml` and paste your real key:
```toml
GROQ_API_KEY = "your-actual-groq-api-key"
```
Get a free key at [console.groq.com](https://console.groq.com).

### 5. Run the tests
```bash
pytest tests/ -v
```

### 6. Run the app
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

---

## Running with Docker

### 1. Create a local `.env` file (not committed)
```
GROQ_API_KEY=your-actual-groq-api-key
```

### 2. Build the image
```bash
docker build -t rag-app .
```

### 3. Run the container
```bash
docker run -p 8501:8501 --env-file .env rag-app
```
Open `http://localhost:8501`. This runs in the exact same environment as the production deployment.

### 4. Stop it
```bash
Ctrl+C
```
or, if running detached (`-d`):
```bash
docker ps
docker stop <container_id>
```

---

## Deployment (Streamlit Community Cloud)

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io), which deploys directly from `requirements.txt` — it does **not** use the `Dockerfile`.

1. Push the repo to GitHub (must be a public repo for the free tier)
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. **Create app** → **From existing repo** → select this repo, branch `main`, main file path `app.py`
4. Under **App settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-actual-groq-api-key"
   ```
5. Deploy — Streamlit installs `requirements.txt` and starts the app automatically

Streamlit Community Cloud auto-redeploys on every push to `main` — no manual deploy step, webhook, or secret needed on the GitHub side for this.

### Why the Dockerfile is still in this repo

The Dockerfile isn't used by this deployment, but it's kept for a few real reasons:
- **Local testing** — `docker build` + `docker run` lets you test the app in the exact environment it would run in on a container-based host, before pushing
- **Portability** — if this project ever moves to a Docker-based host (Render, Cloud Run, etc.), the image is already built and proven to work, no rework needed
- **CI validation** — the `docker-build` job in CI (below) keeps this file honest by actually building it on every push, so it doesn't silently rot

### CI pipeline
`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. **Lint** (`ruff`) + **test** (`pytest`)
2. **Docker build check** — confirms the Dockerfile still builds cleanly (a safety net for future portability, not a deploy gate for the current host)

There's deliberately no "deploy" job — Streamlit Community Cloud's own GitHub integration handles that step independently of this workflow.

---

## Rate Limiting

Two in-memory limits protect the shared Groq API key from abuse:
- **Per-session**: 5 requests / 60 seconds per browser session
- **Global**: 30 LLM calls / 60 seconds across all users combined

**Known limitation:** this is per-process, in-memory state — correct for a single container instance, but each replica would track its own counters if scaled horizontally. A distributed limit at that point would need a shared store (e.g. Redis) or limiting at the reverse-proxy/CDN layer.

---

## Known Limitations / Future Improvements

- **Website content cleaning is heuristic, not perfect.** Some sites (e.g. Wikipedia's newer page-tab UI — "Article/Talk," "Read/Edit/View history") place navigation text outside the tags currently filtered, so some boilerplate can still leak into scraped content. Planned fix: more targeted selectors per common site pattern, or switching to a readability-focused extraction library.
- **YouTube transcript fetching may fail on cloud-hosted deployments.** YouTube blocks most cloud-provider IP ranges (AWS, GCP, Azure, and platforms built on them, including Streamlit Community Cloud) to prevent scraping, which can cause `IpBlocked`/`RequestBlocked` errors — this is a known limitation of the `youtube-transcript-api` library on any shared cloud host, not a bug in this app. Website summarization is unaffected. The real fix would be routing YouTube requests through a residential proxy service (e.g. Webshare), which is a paid add-on not currently implemented.
- **Streamlit Community Cloud's free tier caps memory at 1GB.** The full stack (`torch` + `sentence-transformers` + FAISS + LangChain) fits within this, but there's less headroom than a dedicated container host would offer — worth monitoring under heavier use.
- No authentication — anyone with the URL can use the app (mitigated, not eliminated, by rate limiting)
- No SSRF protection on arbitrary user-submitted URLs (low risk at current scale)
- `vectorstore_cache/` is local disk per container instance — not shared across replicas, and wiped on redeploy unless a persistent disk is attached
- No monitoring/alerting (e.g. Sentry, cost alerts) yet
- No FastAPI backend — current architecture is a single Streamlit process; a separate API layer would be needed to serve other clients or add auth/queuing at scale

---

## Testing

```bash
pytest tests/ -v
```

Covers:
- YouTube video ID extraction (standard/short/shorts/embed URLs, invalid URLs)
- Vector store cache path hashing
- Rate limiter timestamp pruning
- Website content cleaner (mocked HTTP responses — no real network calls in CI)

---

