import json
import pytest
from unittest.mock import patch
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

from core.claude_client import ClaudeExtractor, _strip_json_fences


def _fake_chat_anthropic_factory(responses):
    """
    Returns a drop-in replacement for langchain_anthropic.ChatAnthropic that
    yields canned responses in order, one per .invoke() call, without any
    real network/API call.
    """
    call_count = {"n": 0}

    def _fake(*args, **kwargs):
        def _invoke(_input):
            i = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            return AIMessage(content=responses[i])
        return RunnableLambda(_invoke)

    return _fake


class TestStripJsonFences:
    def test_strips_json_code_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert _strip_json_fences(raw) == '{"a": 1}'

    def test_leaves_plain_json_untouched(self):
        raw = '{"a": 1}'
        assert _strip_json_fences(raw) == '{"a": 1}'


class TestFilterRelevantChunks:
    def test_keeps_only_relevant_chunks(self):
        with patch("core.claude_client.ChatAnthropic", side_effect=_fake_chat_anthropic_factory(
            ["RELEVANT", "NOT_RELEVANT", "RELEVANT"]
        )):
            extractor = ClaudeExtractor()
            chunks = ["about guidance", "about cafeteria renovation", "about margins"]
            relevant, considered = extractor.filter_relevant_chunks("revenue guidance", chunks)
            assert considered == 3
            assert relevant == ["about guidance", "about margins"]

    def test_falls_back_to_all_chunks_if_nothing_tagged_relevant(self):
        with patch("core.claude_client.ChatAnthropic", side_effect=_fake_chat_anthropic_factory(
            ["NOT_RELEVANT", "NOT_RELEVANT"]
        )):
            extractor = ClaudeExtractor()
            chunks = ["chunk one", "chunk two"]
            relevant, considered = extractor.filter_relevant_chunks("q", chunks)
            assert relevant == chunks  # fallback: never lose all evidence


class TestSynthesize:
    def test_synthesize_parses_valid_json(self):
        fake_json = json.dumps({
            "guidance_direction": "lowered",
            "margin_trend": "declining",
            "sentiment_score": -0.3,
            "key_quote": "choppier than we'd like",
            "confidence": "high",
        })
        with patch("core.claude_client.ChatAnthropic", side_effect=_fake_chat_anthropic_factory([fake_json])):
            extractor = ClaudeExtractor()
            result = extractor.synthesize("monitor", ["some excerpt"], ticker="DEMO")
            assert result["guidance_direction"] == "lowered"
            assert result["confidence"] == "high"

    def test_synthesize_handles_unparseable_output_gracefully(self):
        with patch("core.claude_client.ChatAnthropic", side_effect=_fake_chat_anthropic_factory(["not json at all"])):
            extractor = ClaudeExtractor()
            result = extractor.synthesize("monitor", ["some excerpt"], ticker="DEMO")
            assert result["confidence"] == "low"
            assert "error" in result
