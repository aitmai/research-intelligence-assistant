# 🧭 Research Intelligence Assistant

![Author](https://img.shields.io/badge/author-aitmai-4f7cff)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

A shared internal RAG platform for internal research and analysis teams, built around one retrieval
pipeline and two product modes:

- **Opportunity Brief** — upload market research, competitor filings, or news
  about a new business idea and get back a structured brief: market size,
  competitors, risks, and open questions.
- **Signal Monitor** — pull recent SEC filings for a ticker and extract
  structured guidance/margin/sentiment signals, tracked over time.

Both modes share the same chunking, embedding, vector store, and two-tier
Claude extraction pipeline — only the prompt template and output schema
change per mode. See `core/retriever.py` for the shared orchestration layer.

## Features

- **LlamaIndex** chunking (`SentenceSplitter`, offline-safe custom tokenizer)
- **Chroma** vector store (local, persistent, swappable for pgvector or a
  hosted vector DB)
- **LangChain** (LCEL) orchestration for a two-tier extraction pipeline:
  Haiku for cheap relevance filtering, Sonnet for structured JSON synthesis
- Pluggable embedding provider — ships with a free, fully offline TF-IDF
  embedder for dev/demo, with a one-line swap point for a real embedding API
- Content-hash deduping so the same document is never re-chunked or
  re-embedded twice
- Confidence-scored structured outputs (`high`/`medium`/`low`) so low-signal
  extractions get flagged for human review instead of presented as fact
- SQLite/Postgres persistence via SQLAlchemy (documents, chunks, briefs,
  signals, run history/audit log)
- SEC EDGAR ingestion for live 10-K/10-Q pulls, with a bundled sample filing
  for offline/demo use
- Dark-themed Flask dashboard for both modes
- 27-test pytest suite, fully mocked (no live API/network calls required to
  run tests)

## Tech Stack

Flask · SQLAlchemy · LangChain · LlamaIndex · Chroma · Claude API (Haiku +
Sonnet) · scikit-learn · SQLite/Postgres · SEC EDGAR API

## Setup

1. **Clone and enter the project**
   ```bash
   git clone https://github.com/aitmai/research-intelligence-assistant.git
   cd research-intelligence-assistant
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and set at minimum `ANTHROPIC_API_KEY`. See the
   [Environment Variables](#environment-variables) table below for everything
   else.

4. **Create the database tables**
   ```bash
   python create_tables.py
   ```

5. **Seed demo data** (optional — populates one sample brief and one sample
   signal so the dashboard isn't empty on first launch)
   ```bash
   python load_data.py
   ```

6. **Run the app**
   ```bash
   python app.py
   ```
   Visit `http://localhost:5000`.

7. **Run the test suite**
   ```bash
   pytest tests/ -v
   ```
   All tests run fully offline — Claude API calls are mocked, so no API key
   or network access is required just to run `pytest`.

### Using the Signal Monitor without live EDGAR access

SEC EDGAR fetching (`ingestion/edgar_fetch.py`) makes real network calls to
`sec.gov` / `data.sec.gov`. If you're running this somewhere with restricted
egress, check the **"Use sample filing"** box on the Signal Monitor form — it
runs the exact same pipeline against a bundled sample 10-K excerpt in
`sample_data/demo_filing.txt` instead of a live fetch.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key from console.anthropic.com |
| `CLAUDE_TAG_MODEL` | No | `claude-haiku-4-5-20251001` | Cheap model for Tier-1 relevance filtering |
| `CLAUDE_SYNTHESIS_MODEL` | No | `claude-sonnet-5` | Model for Tier-2 structured synthesis |
| `DATABASE_URL` | No | `sqlite:///via.db` | SQLAlchemy connection string; swap for Postgres in production |
| `CHROMA_PATH` | No | `./chroma_store` | Local path for the persistent Chroma vector store |
| `EMBEDDING_PROVIDER` | No | `tfidf` | Embedding backend; `tfidf` is free/offline, extend `core/embeddings.py` to add a real one |
| `OPENAI_API_KEY` | No | — | Reserved for a future OpenAI-embedding provider option |
| `CHUNK_SIZE` | No | `800` | Target chunk size (words) for document splitting |
| `CHUNK_OVERLAP` | No | `100` | Overlap (words) between adjacent chunks |
| `TOP_K_RETRIEVAL` | No | `6` | Number of chunks retrieved per extraction run |
| `EDGAR_USER_AGENT` | No | placeholder | SEC requires a descriptive User-Agent with a real contact email |
| `N8N_WEBHOOK_URL` | No | — | Optional webhook for Slack alerting on flagged signals |
| `SECRET_KEY` | No | dev default | Flask session secret — set a real random value in production |
| `FLASK_DEBUG` | No | `1` | Set to `0` in production |
| `PORT` | No | `5000` | Port for the Flask dev server |

## Deployment (Render)

`gunicorn` is already in `requirements.txt` and a `render.yaml` Blueprint is
included for one-click infra-as-code deployment. Two options:

### Option A — Blueprint (recommended, requires a paid instance for the disk)

1. Push this repo to GitHub under `github.com/aitmai/research-intelligence-assistant`.
2. In the Render dashboard, go to **Blueprints → New Blueprint Instance** and
   point it at the repo. Render reads `render.yaml` automatically.
3. Render will prompt for the two `sync: false` secrets — enter your real
   `ANTHROPIC_API_KEY` and `EDGAR_USER_AGENT` there (never in the yaml file).
4. Deploy. The attached 1GB disk at `/data` persists the SQLite database and
   the Chroma vector store across restarts and deploys.
5. Once live, open a shell from the Render dashboard (or SSH) and run the
   one-time setup:
   ```bash
   python create_tables.py
   python load_data.py   # optional seed data
   ```

### Option B — Manual web service (works on the free tier, no persistent disk)

1. Push to GitHub as above.
2. In Render: **New → Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add environment variables from the table above in the Render dashboard
   (Environment tab) — at minimum `ANTHROPIC_API_KEY`, `EDGAR_USER_AGENT`,
   and a real `SECRET_KEY`.
6. Leave `DATABASE_URL` and `CHROMA_PATH` at their defaults
   (`sqlite:///via.db`, `./chroma_store`).
7. **Tradeoff:** without a persistent disk, Render's free-tier filesystem is
   ephemeral — the SQLite DB and Chroma index reset on every deploy and on
   any restart after idle sleep. Fine for demoing the pipeline live; not
   fine for real production data. If that matters, point `DATABASE_URL` at
   a Render-managed Postgres instance (free tier available) so briefs/signals
   persist even though the Chroma vector index still resets — you'd just
   re-run ingestion after a restart.
8. After first deploy, open the Shell tab and run:
   ```bash
   python create_tables.py
   python load_data.py
   ```

## Security Note

- Never commit `.env` — it's excluded via `.gitignore`. `ANTHROPIC_API_KEY`
  and any real `EDGAR_USER_AGENT` contact email should only live in your local
  `.env` or your deployment platform's secret manager.
- Uploaded documents are written to `uploads/` and are also excluded from
  git — don't commit real due-diligence materials to a public repo.
- This project ships with a permissive TF-IDF embedding default specifically
  so it runs with zero external calls beyond the Claude API. If you swap in a
  hosted embedding provider, review that provider's data-handling terms before
  sending sensitive filing or research content to it.

## License

MIT © 2026 aitmai
