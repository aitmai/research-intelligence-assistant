"""
Research Intelligence Assistant

One shared RAG pipeline (LlamaIndex chunking -> Chroma vector store ->
LangChain-orchestrated two-tier Claude extraction), two product modes:

  Mode A: /brief   - ad hoc "opportunity brief" for a new business idea,
                      built from uploaded research/competitor docs.
  Mode B: /monitor - recurring "signal monitor" tracking guidance/margin/
                      sentiment signals from a company's SEC filings over time.

Run with:  python app.py   (after create_tables.py and, optionally, load_data.py)
"""
from __future__ import annotations
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from config import Config
from models import Base, Document, Chunk, Brief, Signal, RunHistory
from core.indexer import VectorIndex, hash_text
from core.claude_client import ClaudeExtractor
from core.retriever import run_extraction
from ingestion.loaders import load_upload
from ingestion import edgar_fetch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Auto-create tables on startup. This makes the app self-sufficient on
# platforms/tiers with no shell access (e.g. Render's free instance type) —
# create_tables.py is still useful for local dev clarity, but the app no
# longer depends on someone running it manually before first launch.
Base.metadata.create_all(engine)

_index = None
_extractor = None


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index


def get_extractor() -> ClaudeExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ClaudeExtractor()
    return _extractor


def ingest_document(db, index, source_type, title, text, company_ticker=None, filing_date=None):
    """
    Shared ingestion step for both modes: dedupe by content hash, chunk with
    LlamaIndex, embed + store in Chroma, persist Document/Chunk rows.
    Returns the Document row (existing one if this exact text was seen before).
    """
    h = hash_text(text)
    existing = db.scalar(select(Document).where(Document.content_hash == h))
    if existing:
        return existing, True  # cache hit, no re-indexing needed

    doc = Document(
        source_type=source_type,
        title=title,
        company_ticker=company_ticker,
        filing_date=filing_date,
        content_hash=h,
    )
    db.add(doc)
    db.flush()  # assign doc.id

    chunks = index.chunk_text(text)
    index.refit_embedder(chunks)  # TF-IDF needs a fitted vocabulary per batch
    vector_ids = index.add_chunks(doc.id, chunks)
    for i, (chunk_text_, vid) in enumerate(zip(chunks, vector_ids)):
        db.add(Chunk(document_id=doc.id, chunk_index=i, text=chunk_text_, vector_id=vid))
    db.commit()
    return doc, False


def daily_extraction_count(db) -> int:
    """Count of RunHistory rows (i.e. Claude-calling extraction runs) in the
    trailing 24 hours, used to enforce MAX_DAILY_EXTRACTIONS below."""
    since = datetime.utcnow() - timedelta(hours=24)
    return db.scalar(
        select(func.count(RunHistory.id)).where(RunHistory.created_at >= since)
    ) or 0


def over_daily_extraction_cap(db) -> bool:
    return daily_extraction_count(db) >= Config.MAX_DAILY_EXTRACTIONS


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------- Mode A ---

@app.route("/brief", methods=["GET", "POST"])
def brief():
    if request.method == "GET":
        return render_template("brief.html")

    topic = request.form.get("topic", "").strip()
    files = request.files.getlist("documents")
    if not topic or not files or files == [None]:
        flash("Please provide a topic and at least one document.")
        return redirect(url_for("brief"))

    db = SessionLocal()
    if over_daily_extraction_cap(db):
        flash(f"Daily extraction limit reached ({Config.MAX_DAILY_EXTRACTIONS} Claude-calling runs in the "
              f"last 24h). This is a safety net to prevent runaway usage — raise MAX_DAILY_EXTRACTIONS "
              f"in .env if this was intentional, or try again later.")
        return redirect(url_for("brief"))

    index = get_index()
    extractor = get_extractor()
    document_ids = []
    cache_hits = 0

    try:
        for f in files:
            if not f or not f.filename:
                continue
            save_path = UPLOAD_DIR / f.filename
            f.save(save_path)
            text = load_upload(str(save_path))
            doc, was_cached = ingest_document(db, index, "upload", f.filename, text)
            document_ids.append(doc.id)
            cache_hits += int(was_cached)

        result = run_extraction(
            index, extractor, mode="brief",
            query=f"Market opportunity, competitors, and risks for: {topic}",
            document_ids=document_ids,
            topic=topic,
        )
    except Exception as e:
        # Covers PDF/HTML parsing failures, DB errors, and anything else
        # outside the already-guarded Claude API call. Logged so the full
        # traceback shows up in `render logs`; the short message is also
        # flashed directly so diagnosing doesn't require log access.
        logger.exception("Brief generation failed (topic=%r)", topic)
        db.rollback()
        flash(f"Something went wrong generating this brief: {type(e).__name__}: {e}")
        return redirect(url_for("brief"))

    if result.get("error"):
        flash(f"Extraction completed with an error: {result['error']}")

    brief_row = Brief(
        topic=topic,
        market_size=result.get("market_size"),
        competitors=result.get("competitors"),
        risks=result.get("risks"),
        open_questions=result.get("open_questions"),
        recommendation=result.get("recommendation"),
        confidence=result.get("confidence"),
        source_document_ids=document_ids,
    )
    db.add(brief_row)
    db.add(RunHistory(
        mode="brief", input_summary=topic,
        tag_model=Config.CLAUDE_TAG_MODEL, synthesis_model=Config.CLAUDE_SYNTHESIS_MODEL,
        chunks_considered=result.get("_meta", {}).get("chunks_considered"),
        chunks_used=result.get("_meta", {}).get("chunks_used"),
        cache_hit=int(cache_hits == len(document_ids) and len(document_ids) > 0),
    ))
    db.commit()

    return render_template("brief_result.html", brief=brief_row, result=result)


@app.route("/briefs")
def briefs_list():
    db = SessionLocal()
    rows = db.scalars(select(Brief).order_by(Brief.created_at.desc())).all()
    return render_template("briefs_list.html", briefs=rows)


# ---------------------------------------------------------------- Mode B ---

@app.route("/monitor", methods=["GET", "POST"])
def monitor():
    if request.method == "GET":
        return render_template("monitor.html")

    ticker = request.form.get("ticker", "").strip().upper()
    use_sample = request.form.get("use_sample") == "on"
    if not ticker:
        flash("Please provide a ticker.")
        return redirect(url_for("monitor"))

    db = SessionLocal()
    if over_daily_extraction_cap(db):
        flash(f"Daily extraction limit reached ({Config.MAX_DAILY_EXTRACTIONS} Claude-calling runs in the "
              f"last 24h). This is a safety net to prevent runaway usage — raise MAX_DAILY_EXTRACTIONS "
              f"in .env if this was intentional, or try again later.")
        return redirect(url_for("monitor"))

    index = get_index()
    extractor = get_extractor()

    filings_text = []  # list of (title, text, filing_date)

    if use_sample:
        sample_path = Path("sample_data/demo_filing.txt")
        filings_text.append((f"{ticker} sample 10-K excerpt", sample_path.read_text(), datetime.utcnow()))
    else:
        try:
            filings = edgar_fetch.list_recent_filings(ticker, limit=3)
            for filing in filings:
                text = edgar_fetch.fetch_filing_text(filing)
                filing_date = datetime.strptime(filing["filing_date"], "%Y-%m-%d")
                filings_text.append((f"{ticker} {filing['form']} {filing['filing_date']}", text, filing_date))
        except Exception as e:
            flash(f"EDGAR fetch failed ({e}). Falling back to the sample filing — "
                  f"check network access or try 'Use sample data' instead.")
            sample_path = Path("sample_data/demo_filing.txt")
            filings_text.append((f"{ticker} sample 10-K excerpt", sample_path.read_text(), datetime.utcnow()))

    signals_created = []
    try:
        for title, text, filing_date in filings_text:
            doc, _cached = ingest_document(
                db, index, "edgar_10k", title, text,
                company_ticker=ticker, filing_date=filing_date,
            )
            result = run_extraction(
                index, extractor, mode="monitor",
                query=f"Revenue guidance, margin commentary, and management sentiment for {ticker}",
                document_ids=[doc.id],
                ticker=ticker,
            )
            if result.get("error"):
                flash(f"Extraction completed with an error for {title}: {result['error']}")
            signal = Signal(
                document_id=doc.id,
                company_ticker=ticker,
                filing_date=filing_date,
                guidance_direction=result.get("guidance_direction"),
                margin_trend=result.get("margin_trend"),
                sentiment_score=result.get("sentiment_score"),
                key_quote=result.get("key_quote"),
                confidence=result.get("confidence"),
            )
            db.add(signal)
            db.add(RunHistory(
                mode="monitor", input_summary=f"{ticker}:{title}",
                tag_model=Config.CLAUDE_TAG_MODEL, synthesis_model=Config.CLAUDE_SYNTHESIS_MODEL,
                chunks_considered=result.get("_meta", {}).get("chunks_considered"),
                chunks_used=result.get("_meta", {}).get("chunks_used"),
            ))
            signals_created.append(signal)
    except Exception as e:
        logger.exception("Signal extraction failed (ticker=%r)", ticker)
        db.rollback()
        flash(f"Something went wrong extracting signals: {type(e).__name__}: {e}")
        return redirect(url_for("monitor"))

    db.commit()

    return render_template("monitor_result.html", ticker=ticker, signals=signals_created)


@app.route("/monitor/<ticker>")
def monitor_timeline(ticker):
    db = SessionLocal()
    rows = db.scalars(
        select(Signal).where(Signal.company_ticker == ticker.upper()).order_by(Signal.filing_date)
    ).all()
    return render_template("monitor_result.html", ticker=ticker.upper(), signals=rows)


if __name__ == "__main__":
    app.run(debug=Config.FLASK_DEBUG, port=int(os.getenv("PORT", 5000)))
