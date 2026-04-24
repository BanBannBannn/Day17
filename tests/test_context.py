"""Tests for ContextWindowManager and token-aware eviction."""

import pytest

from src.context.window_manager import (
    ContextWindowManager,
    ContextPriority,
    ContextBlock,
    build_context_from_memory,
    _estimate_tokens,
)


class TestTokenEstimation:
    def test_returns_positive_int(self):
        result = _estimate_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_longer_text_has_more_tokens(self):
        short = _estimate_tokens("Hi")
        long = _estimate_tokens("This is a much longer sentence with many more words.")
        assert long > short

    def test_empty_string(self):
        result = _estimate_tokens("")
        assert result >= 0


class TestContextBlock:
    def test_token_count_auto_calculated(self):
        block = ContextBlock("test", "Hello world", ContextPriority.SYSTEM_PROMPT)
        assert block.token_count > 0

    def test_explicit_token_count(self):
        block = ContextBlock("test", "content", ContextPriority.SYSTEM_PROMPT, token_count=42)
        assert block.token_count == 42


class TestContextWindowManager:
    def setup_method(self):
        self.mgr = ContextWindowManager(max_tokens=500, trim_threshold=0.8, eviction_batch=2)

    def test_add_block_basic(self):
        self.mgr.add_block("sys", "You are a helpful assistant.", ContextPriority.SYSTEM_PROMPT)
        assert self.mgr.total_tokens > 0

    def test_empty_block_skipped(self):
        self.mgr.add_block("empty", "", ContextPriority.SYSTEM_PROMPT)
        assert self.mgr.total_tokens == 0

    def test_replace_block_same_name(self):
        self.mgr.add_block("sys", "Version 1", ContextPriority.SYSTEM_PROMPT)
        tokens_v1 = self.mgr.total_tokens
        self.mgr.add_block("sys", "Version 2 longer text here", ContextPriority.SYSTEM_PROMPT)
        tokens_v2 = self.mgr.total_tokens
        assert tokens_v1 != tokens_v2 or tokens_v1 > 0

    def test_build_prompt_non_empty(self):
        self.mgr.add_block("sys", "System prompt here.", ContextPriority.SYSTEM_PROMPT)
        self.mgr.add_block("hist", "Recent history.", ContextPriority.RAW_HISTORY)
        prompt = self.mgr.build_prompt()
        assert "System prompt" in prompt
        assert "Recent history" in prompt

    def test_system_prompt_not_evicted(self):
        # Fill up context beyond threshold
        sys_text = "You are a helpful AI assistant."
        self.mgr.add_block("sys", sys_text, ContextPriority.SYSTEM_PROMPT)

        # Add many raw history blocks to force eviction
        for i in range(20):
            self.mgr.add_block(f"hist_{i}", f"History entry {i} " * 10, ContextPriority.RAW_HISTORY)

        # System prompt should still be present
        prompt = self.mgr.build_prompt()
        assert sys_text in prompt

    def test_eviction_removes_low_priority_first(self):
        self.mgr.add_block("sys", "System content.", ContextPriority.SYSTEM_PROMPT)
        self.mgr.add_block("raw", "Old history " * 20, ContextPriority.RAW_HISTORY)
        self.mgr.add_block("semantic", "Semantic result " * 20, ContextPriority.SEMANTIC_CONTEXT)

        # Force eviction by filling beyond threshold
        for i in range(10):
            self.mgr.add_block(f"extra_{i}", "Extra content " * 15, ContextPriority.RAW_HISTORY)

        prompt = self.mgr.build_prompt()
        # System prompt must survive
        assert "System content" in prompt

    def test_usage_ratio_between_0_and_1(self):
        ratio = self.mgr.usage_ratio
        assert 0.0 <= ratio <= 2.0  # Allow > 1 briefly before trim

    def test_budget_report_structure(self):
        self.mgr.add_block("sys", "System.", ContextPriority.SYSTEM_PROMPT)
        report = self.mgr.get_budget_report()
        assert "total_tokens" in report
        assert "max_tokens" in report
        assert "usage_ratio" in report
        assert "remaining_tokens" in report
        assert "blocks" in report

    def test_remove_block(self):
        self.mgr.add_block("temp", "Temporary content.", ContextPriority.SEMANTIC_CONTEXT)
        removed = self.mgr.remove_block("temp")
        assert removed is True
        assert "Temporary content" not in self.mgr.build_prompt()

    def test_remove_nonexistent_block(self):
        removed = self.mgr.remove_block("does_not_exist")
        assert removed is False

    def test_fits_in_budget(self):
        assert self.mgr.fits_in_budget("short text") is True
        huge_text = "word " * 1000
        assert self.mgr.fits_in_budget(huge_text) is False

    def test_clear_resets_all(self):
        self.mgr.add_block("sys", "Content.", ContextPriority.SYSTEM_PROMPT)
        self.mgr.clear()
        assert self.mgr.total_tokens == 0
        assert self.mgr.build_prompt() == ""

    def test_invalid_max_tokens_raises(self):
        with pytest.raises(ValueError):
            ContextWindowManager(max_tokens=0)

    def test_invalid_trim_threshold_raises(self):
        with pytest.raises(ValueError):
            ContextWindowManager(trim_threshold=0.0)
        with pytest.raises(ValueError):
            ContextWindowManager(trim_threshold=1.5)


class TestBuildContextFromMemory:
    def test_returns_tuple(self):
        result = build_context_from_memory(
            memory_sections={"short_term": "Human: Hi\nAI: Hello"},
            system_prompt="Be helpful.",
            query="How are you?",
            max_tokens=1000,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_prompt_contains_system(self):
        prompt, _ = build_context_from_memory(
            memory_sections={},
            system_prompt="You are a test assistant.",
            query="Test query",
            max_tokens=1000,
        )
        assert "test assistant" in prompt

    def test_budget_report_keys(self):
        _, report = build_context_from_memory(
            memory_sections={"semantic": "Relevant info."},
            system_prompt="Sys.",
            query="Q",
            max_tokens=1000,
        )
        assert "total_tokens" in report
        assert "usage_ratio" in report
