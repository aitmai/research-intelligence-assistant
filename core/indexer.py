"""
Indexing layer: chunk documents with LlamaIndex, embed with the configured
embedding provider, and persist vectors in a local Chroma collection.

This module is the shared spine for both product modes (opportunity briefs
and signal monitoring) — same chunking, same vector store, different prompt
templates layered on top in core/prompts.py.
"""
from __future__ import annotations
import hashlib
from typing import List, Tuple

import chromadb
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter

from config import Config
from core.embeddings import get_embedding_provider


def _simple_tokenizer(text: str) -> List[str]:
    """
    Word-based tokenizer passed to LlamaIndex's SentenceSplitter so chunking
    works fully offline (the default tokenizer pulls a tiktoken encoding
    file over the network, which we want to avoid in this environment).
    """
    return text.split()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VectorIndex:
    """Wraps a Chroma collection + embedding provider behind a small API."""

    def __init__(self, collection_name: str = "research_intel"):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        # embedding_function=None: we supply our own vectors so no network
        # call to a hosted default embedding model happens at query/add time.
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )
        self.embedder = get_embedding_provider(Config.EMBEDDING_PROVIDER)
        self.splitter = SentenceSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            tokenizer=_simple_tokenizer,
        )

    def chunk_text(self, text: str) -> List[str]:
        doc = LlamaDocument(text=text)
        nodes = self.splitter.get_nodes_from_documents([doc])
        return [n.text for n in nodes if n.text.strip()]

    def refit_embedder(self, corpus: List[str]) -> None:
        """TF-IDF needs a fitted vocabulary; call with all known chunk text."""
        self.embedder.fit(corpus)

    def add_chunks(self, document_id: int, chunks: List[str]) -> List[str]:
        """Embeds and stores chunks; returns the vector_id assigned to each chunk."""
        if not chunks:
            return []
        vectors = self.embedder.embed_documents(chunks)
        vector_ids = [f"doc{document_id}-chunk{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=vector_ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=[{"document_id": document_id, "chunk_index": i} for i in range(len(chunks))],
        )
        return vector_ids

    def query(self, query_text: str, top_k: int = None, where: dict = None) -> List[Tuple[str, dict, float]]:
        """Returns list of (chunk_text, metadata, distance) sorted by relevance."""
        top_k = top_k or Config.TOP_K_RETRIEVAL
        query_vector = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return list(zip(docs, metas, dists))

    def reset(self) -> None:
        """Wipes the collection (useful for tests / re-indexing demos)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="research_intel", embedding_function=None
        )
