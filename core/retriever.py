"""
Mode-aware retrieval + extraction orchestration.

This is the one function both product modes call:
    run_extraction(mode, query, document_ids, **prompt_vars)

Same retrieval layer (VectorIndex), same two-tier Claude pipeline
(ClaudeExtractor) — the only thing that changes between "opportunity brief"
and "signal monitor" is the query text, the metadata filter, and the
prompt_vars passed through to the synthesis prompt.
"""
from __future__ import annotations
from typing import List, Optional

from core.indexer import VectorIndex
from core.claude_client import ClaudeExtractor


def run_extraction(
    index: VectorIndex,
    extractor: ClaudeExtractor,
    mode: str,
    query: str,
    document_ids: Optional[List[int]] = None,
    top_k: Optional[int] = None,
    **prompt_vars,
) -> dict:
    """
    1. Retrieve top_k chunks relevant to `query`, optionally scoped to
       specific document_ids.
    2. Tier 1 (Haiku): filter to genuinely relevant chunks.
    3. Tier 2 (Sonnet): synthesize into the mode's structured JSON schema.

    Returns a dict with the schema fields plus '_meta' run stats
    (chunks_considered / chunks_used) for the run_history audit log.
    """
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    hits = index.query(query, top_k=top_k, where=where)
    candidate_chunks = [text for text, _meta, _dist in hits]

    relevant_chunks, considered = extractor.filter_relevant_chunks(query, candidate_chunks)
    result = extractor.synthesize(mode, relevant_chunks, **prompt_vars)

    result["_meta"] = {
        "chunks_considered": considered,
        "chunks_used": len(relevant_chunks),
    }
    return result
