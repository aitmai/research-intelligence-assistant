import os
import sys
import tempfile
import shutil
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture()
def temp_chroma_path():
    path = tempfile.mkdtemp(prefix="chroma_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def temp_db_url():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield f"sqlite:///{path}"
    os.remove(path)


class FakeExtractor:
    """Stand-in for ClaudeExtractor in route tests — no network/API calls."""

    def filter_relevant_chunks(self, question, chunks):
        return chunks, len(chunks)

    def synthesize(self, mode, context_chunks, **prompt_vars):
        if mode == "brief":
            return {
                "market_size": "Fake market size for testing.",
                "competitors": ["Fake Competitor A"],
                "risks": ["Fake risk"],
                "open_questions": ["Fake open question"],
                "recommendation": "Fake recommendation.",
                "confidence": "medium",
            }
        if mode == "monitor":
            return {
                "guidance_direction": "lowered",
                "margin_trend": "declining",
                "sentiment_score": -0.25,
                "key_quote": "fake key quote",
                "confidence": "high",
            }
        raise ValueError(f"Unknown mode: {mode}")


@pytest.fixture()
def client(temp_chroma_path, temp_db_url, monkeypatch):
    """Flask test client wired to a temp SQLite DB, temp Chroma path, and a
    FakeExtractor so route tests never make a real network/API call."""
    from config import Config
    monkeypatch.setattr(Config, "CHROMA_PATH", temp_chroma_path)

    import app as app_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models import Base
    from core.indexer import VectorIndex

    engine = create_engine(temp_db_url)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(app_module, "SessionLocal", TestSession)

    real_index = VectorIndex(collection_name="test_routes")
    fake_extractor = FakeExtractor()
    monkeypatch.setattr(app_module, "get_index", lambda: real_index)
    monkeypatch.setattr(app_module, "get_extractor", lambda: fake_extractor)

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
