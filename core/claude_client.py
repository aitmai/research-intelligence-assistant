"""
Two-tier Claude extraction pipeline, orchestrated with LangChain:

  Tier 1 (Haiku)  - cheap relevance tagging: filters retrieved chunks down
                    to the ones actually worth paying Sonnet to read.
  Tier 2 (Sonnet) - synthesis: reads the filtered chunks and returns a
                    JSON object matching the mode's schema.

This mirrors the Haiku-tag / Sonnet-synthesize pattern already used in the
Deep Tech IC Memo Generator, just wrapped in LangChain's LCEL (prompt | llm)
composition instead of hand-rolled API calls.
"""
from __future__ import annotations
import json
import logging
import re
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

from config import Config
from core.prompts import TAG_PROMPT, MODES

logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    """Claude sometimes wraps JSON in ```json fences despite instructions not to."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


class ClaudeExtractor:
    def __init__(self):
        self.tag_llm = ChatAnthropic(
            model=Config.CLAUDE_TAG_MODEL,
            api_key=Config.ANTHROPIC_API_KEY,
            max_tokens=10,
            temperature=0,
        )
        self.synthesis_llm = ChatAnthropic(
            model=Config.CLAUDE_SYNTHESIS_MODEL,
            api_key=Config.ANTHROPIC_API_KEY,
            max_tokens=1500,
            temperature=0,
        )
        self.tag_chain = ChatPromptTemplate.from_template(TAG_PROMPT) | self.tag_llm | StrOutputParser()

    def filter_relevant_chunks(self, question: str, chunks: List[str]) -> Tuple[List[str], int]:
        """
        Tier 1: cheap Haiku pass. Returns (relevant_chunks, chunks_considered).
        Falls back to keeping all chunks if tagging is inconclusive, so a
        flaky tag response never silently drops the analyst's only evidence.
        """
        relevant = []
        for chunk in chunks:
            try:
                verdict = self.tag_chain.invoke({"question": question, "passage": chunk[:1500]})
            except Exception:
                relevant.append(chunk)
                continue
            if "NOT_RELEVANT" not in verdict.upper():
                relevant.append(chunk)
        if not relevant:
            relevant = chunks
        return relevant, len(chunks)

    def synthesize(self, mode: str, context_chunks: List[str], **prompt_vars) -> dict:
        """
        Tier 2: Sonnet synthesis pass. Returns a dict matching the mode's
        JSON schema. Never raises: an API-level failure (bad/missing key,
        rate limit, invalid model name, network error) or a JSON parsing
        failure both degrade to a low-confidence error dict instead of
        crashing the request with an unhandled 500.
        """
        mode_config = MODES[mode]
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant excerpts found)"
        prompt = ChatPromptTemplate.from_template(mode_config["synthesis_prompt"])
        chain = prompt | self.synthesis_llm | StrOutputParser()

        try:
            raw = chain.invoke({
                **prompt_vars,
                "context": context,
                "schema": json.dumps(mode_config["schema"]),
            })
        except Exception as e:
            # This is the most common real-world failure point: bad/missing
            # ANTHROPIC_API_KEY, an invalid model string, hitting a rate
            # limit, or a transient network error talking to the Anthropic
            # API. Logged here so it shows up in `render logs` / stdout
            # instead of only surfacing as a generic 500 to the browser.
            logger.exception("Claude synthesis call failed (mode=%s)", mode)
            return {
                "error": f"Claude API call failed: {e}",
                "confidence": "low",
            }

        cleaned = _strip_json_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse structured output from model.",
                "raw_response": raw,
                "confidence": "low",
            }
