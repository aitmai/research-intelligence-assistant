"""
Seeds the database with a demo run so the dashboard isn't empty on first
launch: one sample opportunity brief and one sample signal-monitor entry,
both built from sample_data/demo_filing.txt using the real pipeline
(chunking, embedding, and Chroma storage) but with the Claude synthesis
step mocked to avoid requiring an API key just to see the UI populated.

For a real run against your own documents/tickers, just use the /brief
and /monitor forms in the running app instead of this script.

Usage:
    python load_data.py
"""
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Config
from models import Document, Chunk, Brief, Signal, RunHistory
from core.indexer import VectorIndex, hash_text

SAMPLE_TEXT = Path("sample_data/demo_filing.txt").read_text()


def main():
    engine = create_engine(Config.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()
    index = VectorIndex()

    # --- ingest the sample filing through the real pipeline ---
    content_hash = hash_text(SAMPLE_TEXT)
    doc = Document(
        source_type="edgar_10k",
        title="DEMO 10-K excerpt",
        company_ticker="DEMO",
        filing_date=datetime(2026, 3, 31),
        content_hash=content_hash,
    )
    db.add(doc)
    db.flush()

    chunks = index.chunk_text(SAMPLE_TEXT)
    index.refit_embedder(chunks)
    vector_ids = index.add_chunks(doc.id, chunks)
    for i, (text, vid) in enumerate(zip(chunks, vector_ids)):
        db.add(Chunk(document_id=doc.id, chunk_index=i, text=text, vector_id=vid))

    # --- seed one sample Brief (Mode A) ---
    db.add(Brief(
        topic="Embedded insurance for gig economy platforms",
        market_size="Illustrative seed row — run a real brief from the /brief page to replace this.",
        competitors=["Sample Competitor A", "Sample Competitor B"],
        risks=["Regulatory approval timelines vary by state", "Thin underwriting margins at small scale"],
        open_questions=["Which state regulators move fastest on embedded insurance licensing?"],
        recommendation="Seed data — not a real recommendation.",
        confidence="low",
        source_document_ids=[doc.id],
    ))

    # --- seed one sample Signal (Mode B), matching the sample filing's actual content ---
    db.add(Signal(
        document_id=doc.id,
        company_ticker="DEMO",
        filing_date=datetime(2026, 3, 31),
        guidance_direction="lowered",
        margin_trend="declining",
        sentiment_score=-0.2,
        key_quote="the next couple of quarters will likely be choppier than we'd like",
        confidence="high",
    ))

    db.add(RunHistory(
        mode="monitor", input_summary="DEMO:seed", chunks_considered=len(chunks), chunks_used=len(chunks),
    ))

    db.commit()
    print(f"Seeded 1 document ({len(chunks)} chunks), 1 brief, 1 signal.")


if __name__ == "__main__":
    main()
