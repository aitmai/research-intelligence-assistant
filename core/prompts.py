"""
Mode-specific prompt templates and JSON schemas.

Adding a new mode later means adding one entry here — the retrieval and
extraction pipeline in core/retriever.py and core/claude_client.py is
mode-agnostic.
"""

TAG_PROMPT = """You are filtering research passages for relevance.

Question: {question}

Passage:
{passage}

Reply with ONLY one word: RELEVANT or NOT_RELEVANT."""


BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "market_size": {"type": "string"},
        "competitors": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["market_size", "competitors", "risks", "open_questions", "recommendation", "confidence"],
}

BRIEF_SYNTHESIS_PROMPT = """You are a research analyst evaluating a new business idea.

Business idea / topic: {topic}

Below are relevant excerpts retrieved from uploaded market research, competitor filings, and news.
Base your answer only on these excerpts. If something isn't covered, say so rather than guessing.

Excerpts:
{context}

Return ONLY a JSON object matching this schema (no markdown fences, no commentary):
{schema}

Guidance:
- "confidence" should be "low" if the excerpts are thin, contradictory, or largely off-topic.
- "open_questions" should list what a human should still dig into before a build/no-build decision.
"""


SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "guidance_direction": {"type": "string", "enum": ["raised", "lowered", "maintained", "unclear"]},
        "margin_trend": {"type": "string", "enum": ["improving", "declining", "stable", "unclear"]},
        "sentiment_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "key_quote": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["guidance_direction", "margin_trend", "sentiment_score", "key_quote", "confidence"],
}

SIGNAL_SYNTHESIS_PROMPT = """You are a research analyst extracting quantitative signals from a company filing
or earnings call transcript for {ticker}.

Below are relevant excerpts retrieved from the filing.

Excerpts:
{context}

Return ONLY a JSON object matching this schema (no markdown fences, no commentary):
{schema}

Guidance:
- "key_quote" must be a short excerpt (under 20 words) taken directly from the excerpts above, or empty if none is clearly supportive.
- "sentiment_score" ranges from -1.0 (very negative management tone) to 1.0 (very positive).
- "confidence" should be "low" if the excerpts don't clearly address guidance or margins.
"""


MODES = {
    "brief": {
        "schema": BRIEF_SCHEMA,
        "synthesis_prompt": BRIEF_SYNTHESIS_PROMPT,
    },
    "monitor": {
        "schema": SIGNAL_SCHEMA,
        "synthesis_prompt": SIGNAL_SYNTHESIS_PROMPT,
    },
}
