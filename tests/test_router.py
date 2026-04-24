"""Tests for MemoryRouter and QueryIntent classification."""

import pytest

from src.router.memory_router import MemoryRouter, QueryIntent


class TestQueryIntentClassification:
    def setup_method(self):
        self.router = MemoryRouter(use_llm_fallback=False)

    def test_user_preference_intent(self):
        queries = [
            "Do I like dark mode?",
            "What is my preference for language?",
            "Tell me what I prefer",
            "I told you I like Python",
        ]
        for q in queries:
            intent = self.router.classify_intent(q)
            assert intent == QueryIntent.USER_PREFERENCE, (
                f"Expected USER_PREFERENCE for: '{q}', got {intent}"
            )

    def test_factual_recall_intent(self):
        queries = [
            "What is machine learning?",
            "What are neural networks?",
            "Explain transformers",
            "Define entropy",
            "Tell me about the Turing Test",
        ]
        for q in queries:
            intent = self.router.classify_intent(q)
            assert intent == QueryIntent.FACTUAL_RECALL, (
                f"Expected FACTUAL_RECALL for: '{q}', got {intent}"
            )

    def test_experience_recall_intent(self):
        queries = [
            "Remember when we discussed Python?",
            "What did we talk about last time?",
            "What happened in the previous session?",
            "Earlier you mentioned something important",
        ]
        for q in queries:
            intent = self.router.classify_intent(q)
            assert intent == QueryIntent.EXPERIENCE_RECALL, (
                f"Expected EXPERIENCE_RECALL for: '{q}', got {intent}"
            )

    def test_general_intent_fallback(self):
        query = "Hello!"
        intent = self.router.classify_intent(query)
        assert intent == QueryIntent.GENERAL

    def test_empty_query_returns_general(self):
        assert self.router.classify_intent("") == QueryIntent.GENERAL
        assert self.router.classify_intent("   ") == QueryIntent.GENERAL


class TestMemoryRouterRouting:
    def setup_method(self):
        self.router = MemoryRouter()

    def test_route_returns_required_keys(self):
        result = self.router.route("What is deep learning?")
        assert "intent" in result
        assert "backends" in result
        assert "primary" in result
        assert "explanation" in result

    def test_user_preference_uses_short_and_long_term(self):
        result = self.router.route("What do I prefer?")
        backends = result["backends"]
        assert "short_term" in backends
        assert "long_term" in backends

    def test_factual_recall_uses_semantic(self):
        result = self.router.route("What is entropy?")
        assert "semantic" in result["backends"]

    def test_experience_recall_uses_episodic(self):
        result = self.router.route("Remember when we talked about Python?")
        assert "episodic" in result["backends"]

    def test_general_uses_short_term_and_semantic(self):
        result = self.router.route("Hello there!")
        backends = result["backends"]
        assert "short_term" in backends
        assert "semantic" in backends

    def test_select_memory_sections_flags(self):
        flags = self.router.select_memory_sections("What is AI?")
        assert isinstance(flags["include_short_term"], bool)
        assert isinstance(flags["include_long_term"], bool)
        assert isinstance(flags["include_episodic"], bool)
        assert isinstance(flags["include_semantic"], bool)
        # Factual recall should activate semantic
        assert flags["include_semantic"] is True

    def test_select_memory_sections_respects_available(self):
        flags = self.router.select_memory_sections(
            "What is AI?",
            available_backends=["short_term"],
        )
        assert flags["include_long_term"] is False
        assert flags["include_episodic"] is False
        assert flags["include_semantic"] is False
