import pytest
from unittest.mock import patch

from config import Config
from core.indexer import VectorIndex, hash_text


@pytest.fixture()
def index(temp_chroma_path, monkeypatch):
    monkeypatch.setattr(Config, "CHROMA_PATH", temp_chroma_path)
    return VectorIndex(collection_name="test_collection")


def test_chunk_text_splits_long_document(index):
    long_text = "This is a sentence about revenue guidance. " * 200
    chunks = index.chunk_text(long_text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_chunk_text_short_document_single_chunk(index):
    chunks = index.chunk_text("Short filing text about margins.")
    assert len(chunks) == 1


def test_add_and_query_chunks_returns_relevant_first(index):
    chunks = [
        "Revenue guidance was lowered due to weak enterprise demand in Europe.",
        "Our office recently renovated the cafeteria on the third floor.",
        "Gross margin declined due to higher cloud infrastructure costs.",
    ]
    index.refit_embedder(chunks)
    index.add_chunks(document_id=1, chunks=chunks)

    results = index.query("revenue guidance", top_k=2)
    assert len(results) == 2
    top_chunk_text = results[0][0]
    assert "guidance" in top_chunk_text.lower() or "revenue" in top_chunk_text.lower()


def test_query_respects_document_id_filter(index):
    chunks_doc1 = ["Doc one: guidance was lowered."]
    chunks_doc2 = ["Doc two: guidance was raised."]
    index.refit_embedder(chunks_doc1 + chunks_doc2)
    index.add_chunks(document_id=1, chunks=chunks_doc1)
    index.add_chunks(document_id=2, chunks=chunks_doc2)

    results = index.query("guidance", top_k=5, where={"document_id": 2})
    assert len(results) == 1
    assert "Doc two" in results[0][0]


def test_reset_clears_collection(index):
    index.refit_embedder(["some chunk text"])
    index.add_chunks(document_id=1, chunks=["some chunk text"])
    index.reset()
    results = index.query("some chunk text", top_k=5)
    assert results == []


def test_hash_text_is_deterministic():
    assert hash_text("hello world") == hash_text("hello world")
    assert hash_text("hello world") != hash_text("hello worlds")
