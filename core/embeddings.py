"""
Pluggable embedding providers.

Default: TfidfEmbeddingProvider — free, deterministic, fully offline.
Good for local dev, tests, and demos with no external embedding API key.

For production quality, swap in a real embedding model (OpenAI text-embedding-3,
Voyage, or a local sentence-transformers model) by implementing the same
`fit`/`embed_documents`/`embed_query` interface and setting
EMBEDDING_PROVIDER in .env.
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Fit the embedding space on a corpus (no-op for API-based providers)."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class TfidfEmbeddingProvider(BaseEmbeddingProvider):
    """
    Offline embedding provider backed by scikit-learn's TfidfVectorizer.

    Not as semantically rich as a neural embedding model, but requires no
    network access and no API key — ideal for local dev, CI, and demos.
    The interface is intentionally identical to a "real" embedding provider
    so swapping one in later is a one-line config change, not a rewrite.
    """

    def __init__(self, max_features: int = 2048):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        cleaned = [_clean(t) for t in corpus] or ["placeholder"]
        self.vectorizer.fit(cleaned)
        self._fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        cleaned = [_clean(t) for t in texts]
        matrix = self.vectorizer.transform(cleaned)
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            self.fit([text])
        vec = self.vectorizer.transform([_clean(text)])
        return vec.toarray().astype(float)[0].tolist()


def get_embedding_provider(name: str) -> BaseEmbeddingProvider:
    if name == "tfidf":
        return TfidfEmbeddingProvider()
    raise ValueError(
        f"Unknown embedding provider '{name}'. Add an implementation of "
        f"BaseEmbeddingProvider and register it in get_embedding_provider()."
    )
