"""
Pluggable embedding providers.

Default: HashingEmbeddingProvider — free, deterministic, fully offline,
and — critically — a FIXED output dimension regardless of input text.

--------------------------------------------------------------------------
Why hashing instead of TF-IDF (bug history, kept here on purpose)
--------------------------------------------------------------------------
The original default here used scikit-learn's TfidfVectorizer, refitting
its vocabulary on every ingestion batch (see VectorIndex.refit_embedder in
core/indexer.py). That's fine in isolation, but TF-IDF's output dimension
IS the size of whatever vocabulary it was just fit on — so ingesting one
small document (say, a 95-word vocabulary) and then a larger one (a
164-word vocabulary) produced vectors of two different lengths. Chroma
locks a collection to the dimension of the first vector it ever stores, so
the second batch failed at write time with:

    InvalidArgumentError: Collection expecting embedding with dimension of
    164, got 95

That's a structural mismatch between "refit-per-batch" and "one shared,
persistent vector store" — it was never a matter of if, only when, two
differently-sized documents would collide. A hashing vectorizer sidesteps
this entirely: it needs no fitting and every output vector has the same
fixed length by construction, no matter what text goes in.

For production semantic quality, swap in a real embedding model (OpenAI
text-embedding-3, Voyage, or a local sentence-transformers model) by
implementing the same `fit`/`embed_documents`/`embed_query` interface and
setting EMBEDDING_PROVIDER in .env — fit() can stay a no-op for API-based
providers too, since those also return a fixed dimension per model.
--------------------------------------------------------------------------
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import List

from sklearn.feature_extraction.text import HashingVectorizer


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Fit the embedding space on a corpus (no-op for stateless/API providers)."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """
    Offline embedding provider backed by scikit-learn's HashingVectorizer.

    Not as semantically rich as a neural embedding model, but requires no
    network access, no API key, AND — unlike TF-IDF — needs no fitting and
    always returns vectors of exactly `n_features` length. That fixed
    dimension is a hard requirement for any vector store (Chroma, pgvector,
    etc.) that persists vectors from independent ingestion calls into one
    shared collection.

    The interface is intentionally identical to a "real" embedding provider
    so swapping one in later is a one-line config change, not a rewrite.
    `fit()` is kept only so callers written against that interface (see
    VectorIndex.refit_embedder) don't need to change; it's a deliberate
    no-op here.
    """

    def __init__(self, n_features: int = 512):
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            stop_words="english",
            alternate_sign=False,  # non-negative features, closer to a term-count signal
            norm="l2",
        )

    def fit(self, corpus: List[str]) -> None:
        """No-op: HashingVectorizer needs no fitting, by design."""
        return

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned = [_clean(t) for t in texts]
        matrix = self.vectorizer.transform(cleaned)
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.vectorizer.transform([_clean(text)])
        return vec.toarray().astype(float)[0].tolist()


# Backward-compatible alias: earlier versions of this project called this
# class TfidfEmbeddingProvider. The public config value "tfidf" and any
# code importing that name both still work — see get_embedding_provider().
TfidfEmbeddingProvider = HashingEmbeddingProvider


def get_embedding_provider(name: str) -> BaseEmbeddingProvider:
    if name in ("hashing", "tfidf"):
        return HashingEmbeddingProvider()
    raise ValueError(
        f"Unknown embedding provider '{name}'. Add an implementation of "
        f"BaseEmbeddingProvider and register it in get_embedding_provider()."
    )
