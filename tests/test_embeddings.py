import pytest
from core.embeddings import HashingEmbeddingProvider, TfidfEmbeddingProvider, get_embedding_provider


def test_embed_documents_returns_fixed_dimension():
    provider = HashingEmbeddingProvider(n_features=512)
    vectors = provider.embed_documents(["revenue guidance was lowered", "margins improved this quarter"])
    assert len(vectors) == 2
    assert all(len(v) == 512 for v in vectors)


def test_embed_query_matches_document_dimensionality():
    provider = HashingEmbeddingProvider(n_features=512)
    doc_vec = provider.embed_documents(["revenue guidance was lowered"])[0]
    query_vec = provider.embed_query("guidance")
    assert len(doc_vec) == len(query_vec) == 512


def test_fit_is_a_safe_no_op():
    """fit() exists only for interface compatibility with VectorIndex.refit_embedder
    and must not change output dimension or raise, regardless of corpus size."""
    provider = HashingEmbeddingProvider(n_features=512)
    provider.fit(["a tiny corpus"])
    small_vec = provider.embed_documents(["some text"])[0]
    provider.fit(["a", "much", "much", "much", "larger", "corpus", "with", "many", "more", "unique", "words"] * 20)
    large_vec = provider.embed_documents(["some text"])[0]
    assert len(small_vec) == len(large_vec) == 512


def test_dimension_is_stable_across_differently_sized_batches():
    """
    Regression test for the exact production bug: ingesting a small-vocabulary
    document followed by a large-vocabulary document into the same vector
    store must produce same-dimension vectors, or Chroma raises
    InvalidArgumentError ('Collection expecting embedding with dimension of
    X, got Y') on the second add.
    """
    provider = HashingEmbeddingProvider(n_features=512)

    small_doc = ["Short filing text."]
    large_doc = [
        "NVIDIA Corporation SEC Filing Summary. Revenue for the quarter increased "
        "thirty four percent year over year driven by strong demand for data "
        "center accelerators across every major hyperscale customer segment, "
        "with gross margin improving substantially versus the prior year period "
        "and management commentary striking a highly confident forward looking tone."
    ]

    provider.fit(small_doc)
    small_vectors = provider.embed_documents(small_doc)

    provider.fit(large_doc)  # simulates a second, independent ingestion call
    large_vectors = provider.embed_documents(large_doc)

    assert len(small_vectors[0]) == len(large_vectors[0])


def test_get_embedding_provider_returns_hashing_provider():
    provider = get_embedding_provider("hashing")
    assert isinstance(provider, HashingEmbeddingProvider)


def test_get_embedding_provider_tfidf_alias_still_works():
    """Backward compatibility: existing EMBEDDING_PROVIDER=tfidf configs
    (e.g. already-deployed Render env vars) keep working unchanged."""
    provider = get_embedding_provider("tfidf")
    assert isinstance(provider, HashingEmbeddingProvider)
    assert isinstance(provider, TfidfEmbeddingProvider)  # class alias


def test_get_embedding_provider_raises_on_unknown():
    with pytest.raises(ValueError):
        get_embedding_provider("nonexistent")
