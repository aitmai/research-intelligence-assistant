"""
SQLAlchemy models.

Tables:
  documents    - one row per ingested filing / transcript / research doc
  chunks       - one row per chunk of a document (metadata only; vectors live in Chroma)
  briefs       - Mode A output: one-time "opportunity brief" for a new business idea
  signals      - Mode B output: one structured signal extracted from a filing
  run_history  - audit log of every extraction run (mode, inputs, cost/tokens, timestamp)
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    source_type = Column(String(32), nullable=False)      # "upload", "edgar_10k", "edgar_10q", "transcript"
    title = Column(String(255), nullable=False)
    company_ticker = Column(String(16), nullable=True, index=True)
    filing_date = Column(DateTime, nullable=True)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)  # dedupe / caching key
    raw_text_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    vector_id = Column(String(64), nullable=False)  # id used to look this chunk up in Chroma
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class Brief(Base):
    """Mode A: opportunity brief for a new business idea."""
    __tablename__ = "briefs"

    id = Column(Integer, primary_key=True)
    topic = Column(String(255), nullable=False)          # e.g. "Embedded insurance for gig platforms"
    market_size = Column(Text, nullable=True)
    competitors = Column(JSON, nullable=True)             # list[str]
    risks = Column(JSON, nullable=True)                   # list[str]
    open_questions = Column(JSON, nullable=True)           # list[str]
    recommendation = Column(Text, nullable=True)
    confidence = Column(String(16), nullable=True)         # "high" / "medium" / "low"
    source_document_ids = Column(JSON, nullable=True)      # list[int]
    created_at = Column(DateTime, default=datetime.utcnow)


class Signal(Base):
    """Mode B: one structured signal extracted from a company filing."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    company_ticker = Column(String(16), nullable=False, index=True)
    filing_date = Column(DateTime, nullable=True)
    guidance_direction = Column(String(16), nullable=True)  # "raised" / "lowered" / "maintained" / "unclear"
    margin_trend = Column(String(16), nullable=True)         # "improving" / "declining" / "stable" / "unclear"
    sentiment_score = Column(Float, nullable=True)           # -1.0 .. 1.0
    key_quote = Column(Text, nullable=True)
    confidence = Column(String(16), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document")


class RunHistory(Base):
    """Audit log of every extraction run, for cost tracking and caching."""
    __tablename__ = "run_history"

    id = Column(Integer, primary_key=True)
    mode = Column(String(16), nullable=False)   # "brief" or "monitor"
    input_summary = Column(String(255), nullable=True)
    tag_model = Column(String(64), nullable=True)
    synthesis_model = Column(String(64), nullable=True)
    chunks_considered = Column(Integer, nullable=True)
    chunks_used = Column(Integer, nullable=True)
    cache_hit = Column(Integer, default=0)  # 0/1 boolean
    created_at = Column(DateTime, default=datetime.utcnow)
