"""Tests for memory backends: ShortTermMemory, LongTermMemory, EpisodicMemory, SemanticMemory."""

import os
import tempfile
import pytest

# Use a temporary directory for episodic tests
os.environ.setdefault("EPISODES_DIR", tempfile.mkdtemp())
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp())


from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.episodic import EpisodicMemory
from src.memory.manager import MemoryManager


# ─── ShortTermMemory ─────────────────────────────────────────────────────────

class TestShortTermMemory:
    def setup_method(self):
        self.mem = ShortTermMemory(max_messages=4, session_id="test_st")

    def test_add_and_retrieve(self):
        self.mem.add_interaction("Hello", "Hi there!")
        history = self.mem.get_history()
        assert len(history) == 2  # HumanMessage + AIMessage

    def test_get_history_as_text(self):
        self.mem.add_interaction("What is Python?", "Python is a programming language.")
        text = self.mem.get_history_as_text()
        assert "Human:" in text
        assert "AI:" in text
        assert "Python" in text

    def test_message_count_increments(self):
        assert self.mem.message_count == 0
        self.mem.add_interaction("Q1", "A1")
        assert self.mem.message_count == 1

    def test_trim_on_max_exceeded(self):
        for i in range(6):
            self.mem.add_interaction(f"Question {i}", f"Answer {i}")
        # Should not exceed max_messages
        assert self.mem.message_count <= self.mem.max_messages

    def test_clear_resets_state(self):
        self.mem.add_interaction("Q", "A")
        self.mem.clear()
        assert self.mem.message_count == 0
        assert self.mem.get_history() == []

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            self.mem.add_interaction("", "Answer")
        with pytest.raises(ValueError):
            self.mem.add_interaction("Question", "")

    def test_get_recent(self):
        for i in range(5):
            self.mem.add_interaction(f"Q{i}", f"A{i}")
        recent = self.mem.get_recent(n=2)
        assert len(recent) <= 2

    def test_to_dict_structure(self):
        self.mem.add_interaction("Hello", "World")
        d = self.mem.to_dict()
        assert "session_id" in d
        assert "message_count" in d
        assert "messages" in d


# ─── LongTermMemory ───────────────────────────────────────────────────────────

class TestLongTermMemory:
    def setup_method(self):
        # Force fallback mode (no Redis required for tests)
        self.mem = LongTermMemory(session_id="test_lt_unique_xyz")
        self.mem._use_fallback = True

    def test_add_and_retrieve(self):
        self.mem.add_interaction("Hello LT", "Hi from LT!")
        history = self.mem.get_history()
        assert len(history) == 1
        assert history[0]["human"] == "Hello LT"
        assert history[0]["ai"] == "Hi from LT!"

    def test_get_history_as_text(self):
        self.mem.add_interaction("Q", "A")
        text = self.mem.get_history_as_text()
        assert "Human:" in text
        assert "AI:" in text

    def test_search_by_keyword(self):
        self.mem.add_interaction("Tell me about Python", "Python is great!")
        self.mem.add_interaction("What about Java?", "Java is also useful.")
        results = self.mem.search_by_keyword("Python")
        assert len(results) >= 1
        assert "Python" in results[0]["human"]

    def test_search_no_match(self):
        self.mem.add_interaction("Hello", "Hi")
        results = self.mem.search_by_keyword("XYZNOTFOUND")
        assert results == []

    def test_entry_count(self):
        assert self.mem.entry_count == 0
        self.mem.add_interaction("Q", "A")
        assert self.mem.entry_count == 1

    def test_clear_session(self):
        self.mem.add_interaction("Q", "A")
        self.mem.clear_session()
        assert self.mem.entry_count == 0

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            self.mem.add_interaction("", "Answer")


# ─── EpisodicMemory ───────────────────────────────────────────────────────────

class TestEpisodicMemory:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.mem = EpisodicMemory(session_id="test_ep", episodes_dir=self._tmpdir)

    def test_add_and_retrieve(self):
        ep_id = self.mem.add_episode("task_complete", "User completed setup", importance=0.8)
        assert isinstance(ep_id, str) and len(ep_id) > 0
        episodes = self.mem.get_episodes()
        assert any(e["id"] == ep_id for e in episodes)

    def test_filter_by_event_type(self):
        self.mem.add_episode("preference", "User likes dark mode")
        self.mem.add_episode("task_complete", "Finished coding task")
        prefs = self.mem.get_episodes(event_type="preference")
        assert all(e["event_type"] == "preference" for e in prefs)

    def test_filter_by_importance(self):
        self.mem.add_episode("low", "Minor event", importance=0.2)
        self.mem.add_episode("high", "Critical event", importance=0.9)
        high_priority = self.mem.get_episodes(min_importance=0.5)
        assert all(e["importance"] >= 0.5 for e in high_priority)

    def test_search_by_keyword(self):
        self.mem.add_episode("note", "User prefers Python over Java")
        results = self.mem.search("Python")
        assert len(results) >= 1

    def test_invalid_importance_raises(self):
        with pytest.raises(ValueError):
            self.mem.add_episode("test", "Summary", importance=1.5)

    def test_empty_event_type_raises(self):
        with pytest.raises(ValueError):
            self.mem.add_episode("", "Summary")

    def test_format_as_context(self):
        self.mem.add_episode("pref", "User likes concise answers", importance=0.7)
        ctx = self.mem.format_as_context()
        assert "[Episodic Memory]" in ctx

    def test_episode_count(self):
        assert self.mem.episode_count == 0
        self.mem.add_episode("test", "Test episode")
        assert self.mem.episode_count == 1

    def test_clear(self):
        self.mem.add_episode("test", "Test episode")
        self.mem.clear()
        assert self.mem.episode_count == 0


# ─── MemoryManager ────────────────────────────────────────────────────────────

class TestMemoryManager:
    def setup_method(self):
        tmpdir = tempfile.mkdtemp()
        chroma_dir = tempfile.mkdtemp()
        self.mgr = MemoryManager(
            session_id="test_mgr",
            episodes_dir=tmpdir,
            chroma_persist_dir=chroma_dir,
        )
        # Force fallback for Redis
        self.mgr.long_term._use_fallback = True

    def test_add_interaction_persists_all(self):
        self.mgr.add_interaction("Hello", "Hi there!", store_semantic=False)
        stats = self.mgr.stats()
        assert stats["short_term_messages"] == 1
        assert stats["long_term_entries"] == 1

    def test_build_context_returns_sections(self):
        self.mgr.add_interaction("What is AI?", "AI is artificial intelligence.", store_semantic=False)
        ctx = self.mgr.build_context("AI", include_semantic=False)
        assert "short_term" in ctx
        assert "long_term" in ctx

    def test_log_episode(self):
        ep_id = self.mgr.log_episode("test", "Test episode", importance=0.5)
        assert ep_id is not None
        assert self.mgr.stats()["episodic_episodes"] == 1

    def test_stats_structure(self):
        stats = self.mgr.stats()
        required_keys = ["session_id", "short_term_messages", "long_term_entries",
                         "episodic_episodes", "semantic_documents"]
        for key in required_keys:
            assert key in stats

    def test_clear_all(self):
        self.mgr.add_interaction("Q", "A", store_semantic=False)
        self.mgr.clear_all()
        stats = self.mgr.stats()
        assert stats["short_term_messages"] == 0
        assert stats["long_term_entries"] == 0
