from core.embeddings import TfidfEmbeddingProvider, get_embedding_provider


def test_fit_and_embed_documents():
    provider = TfidfEmbeddingProvider()
    corpus = ["revenue guidance was lowered", "margins improved this quarter", "unrelated text about weather"]
    provider.fit(corpus)
    vectors = provider.embed_documents(corpus)
    assert len(vectors) == 3
    assert all(isinstance(v, list) for v in vectors)
    assert len(vectors[0]) == len(vectors[1])  # same dimensionality


def test_embed_query_matches_dimensionality():
    provider = TfidfEmbeddingProvider()
    provider.fit(["revenue guidance was lowered", "margins improved this quarter"])
    doc_vec = provider.embed_documents(["revenue guidance was lowered"])[0]
    query_vec = provider.embed_query("guidance")
    assert len(doc_vec) == len(query_vec)


def test_unfitted_provider_autofits_on_first_call():
    provider = TfidfEmbeddingProvider()
    vectors = provider.embed_documents(["some text here"])
    assert len(vectors) == 1


def test_get_embedding_provider_returns_tfidf():
    provider = get_embedding_provider("tfidf")
    assert isinstance(provider, TfidfEmbeddingProvider)


def test_get_embedding_provider_raises_on_unknown():
    import pytest
    with pytest.raises(ValueError):
        get_embedding_provider("nonexistent")
