"""
Central configuration, loaded from environment variables.
Copy .env.example to .env and fill in real values before running.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Claude API ---
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_TAG_MODEL = os.getenv("CLAUDE_TAG_MODEL", "claude-haiku-4-5-20251001")
    CLAUDE_SYNTHESIS_MODEL = os.getenv("CLAUDE_SYNTHESIS_MODEL", "claude-sonnet-5")

    # --- Database (create_tables.py builds this) ---
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///via.db")

    # --- Vector store (Chroma, local persistent by default) ---
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")

    # --- Embeddings ---
    # "tfidf" = free, offline, deterministic (default, good for dev/demo).
    # "openai" = swap in a real embedding API for production quality.
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "tfidf")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # --- Chunking ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))

    # --- SEC EDGAR ---
    EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "ResearchIntelligenceAssistant admin@example.com")

    # --- Optional Slack alerting via n8n webhook ---
    N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")

    # --- Flask ---
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
