"""Multi-tier memory manager orchestrating all four memory backends."""

import logging
from typing import Dict, List, Optional

from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified interface for all four memory tiers.

    Provides a single entry point to read from and write to:
      - ShortTermMemory  (in-session buffer)
      - LongTermMemory   (Redis persistent history)
      - EpisodicMemory   (JSON episodic log)
      - SemanticMemory   (ChromaDB vector store)

    Usage:
        manager = MemoryManager(session_id="user_123")
        manager.add_interaction("Hello", "Hi there!")
        ctx = manager.build_context("tell me about Python")
    """

    def __init__(
        self,
        session_id: str = "default",
        short_term_max: int = 20,
        long_term_ttl: int = 86400,
        episodes_dir: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        chroma_collection: Optional[str] = None,
    ):
        """Initialize all memory backends.

        Args:
            session_id: Shared session identifier across all backends.
            short_term_max: Maximum messages in the short-term buffer.
            long_term_ttl: Redis key TTL in seconds.
            episodes_dir: Directory for episodic JSON files.
            chroma_persist_dir: ChromaDB persistence directory.
            chroma_collection: ChromaDB collection name.
        """
        self.session_id = session_id

        self.short_term = ShortTermMemory(
            max_messages=short_term_max,
            session_id=session_id,
        )
        self.long_term = LongTermMemory(
            session_id=session_id,
            ttl=long_term_ttl,
        )
        self.episodic = EpisodicMemory(
            session_id=session_id,
            episodes_dir=episodes_dir,
        )
        self.semantic = SemanticMemory(
            persist_dir=chroma_persist_dir,
            collection_name=chroma_collection,
        )

        logger.info("MemoryManager initialized for session=%s", session_id)

    def add_interaction(
        self,
        human_input: str,
        ai_output: str,
        store_semantic: bool = True,
        episode_trigger: Optional[Dict] = None,
    ) -> None:
        """Persist an interaction across all relevant memory tiers.

        Args:
            human_input: The user's message.
            ai_output: The AI assistant's response.
            store_semantic: If True, also index this turn in semantic memory.
            episode_trigger: If provided, log an episode with these parameters
                (expects keys: event_type, summary, details, importance).

        Raises:
            ValueError: Propagated from individual backends on empty inputs.
        """
        self.short_term.add_interaction(human_input, ai_output)
        self.long_term.add_interaction(human_input, ai_output)

        if store_semantic:
            combined = f"Human: {human_input}\nAI: {ai_output}"
            self.semantic.add_memory(
                text=combined,
                metadata={"session_id": self.session_id, "type": "conversation"},
            )

        if episode_trigger:
            self.episodic.add_episode(**episode_trigger)

        logger.debug("MemoryManager: stored interaction (session=%s)", self.session_id)

    def build_context(
        self,
        query: str,
        include_short_term: bool = True,
        include_long_term: bool = True,
        include_episodic: bool = True,
        include_semantic: bool = True,
        semantic_top_k: int = 3,
        long_term_limit: int = 10,
        episodic_limit: int = 5,
    ) -> Dict[str, str]:
        """Assemble a structured context payload from all memory backends.

        Args:
            query: The user's current query (used for semantic search).
            include_short_term: Include buffer from short-term memory.
            include_long_term: Include recent history from long-term store.
            include_episodic: Include recent episodic entries.
            include_semantic: Include semantically similar fragments.
            semantic_top_k: Number of semantic results to retrieve.
            long_term_limit: Number of long-term interactions to include.
            episodic_limit: Number of episodes to include.

        Returns:
            Dict with keys 'short_term', 'long_term', 'episodic', 'semantic',
            each containing formatted text ready for prompt injection.
        """
        context: Dict[str, str] = {}

        if include_short_term:
            context["short_term"] = self.short_term.get_history_as_text()

        if include_long_term:
            context["long_term"] = self.long_term.get_history_as_text(limit=long_term_limit)

        if include_episodic:
            context["episodic"] = self.episodic.format_as_context(limit=episodic_limit)

        if include_semantic:
            context["semantic"] = self.semantic.format_as_context(query=query, top_k=semantic_top_k)

        return context

    def build_full_context_string(self, query: str, **kwargs) -> str:
        """Build and concatenate all context sections into a single string.

        Args:
            query: The user's current query.
            **kwargs: Forwarded to build_context().

        Returns:
            Single formatted string for direct LLM injection.
        """
        sections = self.build_context(query, **kwargs)
        parts: List[str] = []

        if sections.get("short_term"):
            parts.append(f"## Recent Conversation\n{sections['short_term']}")
        if sections.get("long_term"):
            parts.append(f"## Long-term History\n{sections['long_term']}")
        if sections.get("episodic"):
            parts.append(sections["episodic"])
        if sections.get("semantic"):
            parts.append(sections["semantic"])

        return "\n\n".join(parts) if parts else "No memory context available."

    def log_episode(
        self,
        event_type: str,
        summary: str,
        details: Optional[Dict] = None,
        importance: float = 0.5,
    ) -> str:
        """Convenience method to record an episode directly.

        Args:
            event_type: Category of the episode.
            summary: Short summary text.
            details: Optional structured payload.
            importance: Float 0-1 importance score.

        Returns:
            Episode ID.
        """
        return self.episodic.add_episode(event_type, summary, details, importance)

    def index_knowledge(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Add a static knowledge fragment to semantic memory.

        Args:
            text: Knowledge text to embed and index.
            metadata: Optional metadata dict.

        Returns:
            Document ID.
        """
        meta = {"type": "knowledge", "session_id": self.session_id, **(metadata or {})}
        return self.semantic.add_memory(text, metadata=meta)

    def clear_all(self) -> None:
        """Clear all memory backends for the current session."""
        self.short_term.clear()
        self.long_term.clear_session()
        self.episodic.clear()
        self.semantic.clear()
        logger.info("All memory backends cleared for session=%s", self.session_id)

    def stats(self) -> Dict:
        """Return storage statistics across all backends.

        Returns:
            Dict with counts per memory tier.
        """
        return {
            "session_id": self.session_id,
            "short_term_messages": self.short_term.message_count,
            "long_term_entries": self.long_term.entry_count,
            "episodic_episodes": self.episodic.episode_count,
            "semantic_documents": self.semantic.document_count,
        }
