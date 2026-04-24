"""Episodic memory: JSON-based log of important interaction episodes."""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = os.getenv("EPISODES_DIR", "./data/episodes")
_MAX_EPISODES = int(os.getenv("MAX_EPISODES", "1000"))


class EpisodicMemory:
    """Records significant conversation episodes to a JSON file store.

    Episodes are discrete, meaningful events (task completions, preference
    changes, important decisions) that can be retrieved by type or keyword.
    """

    def __init__(
        self,
        session_id: str = "default",
        episodes_dir: Optional[str] = None,
        max_episodes: int = _MAX_EPISODES,
    ):
        """Initialize episodic memory store.

        Args:
            session_id: Unique identifier for the current session.
            episodes_dir: Directory path for episode JSON files.
            max_episodes: Maximum number of episodes to retain per session.
        """
        self.session_id = session_id
        self.max_episodes = max_episodes
        self._dir = Path(episodes_dir or _DEFAULT_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{session_id}.json"
        self._episodes: List[Dict] = self._load()
        logger.info(
            "EpisodicMemory initialized: %d existing episodes (session=%s)",
            len(self._episodes), session_id,
        )

    def add_episode(
        self,
        event_type: str,
        summary: str,
        details: Optional[Dict] = None,
        importance: float = 0.5,
    ) -> str:
        """Record a new episode to the JSON log.

        Args:
            event_type: Category label (e.g., 'task_complete', 'preference').
            summary: Human-readable one-sentence summary of the episode.
            details: Optional structured payload attached to this episode.
            importance: Float 0-1 representing how important this episode is.

        Returns:
            The unique episode ID.

        Raises:
            ValueError: If event_type or summary is empty, or importance out of range.
        """
        if not event_type or not event_type.strip():
            raise ValueError("event_type must not be empty")
        if not summary or not summary.strip():
            raise ValueError("summary must not be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be between 0.0 and 1.0")

        episode_id = str(uuid.uuid4())
        episode = {
            "id": episode_id,
            "session_id": self.session_id,
            "event_type": event_type,
            "summary": summary,
            "details": details or {},
            "importance": importance,
            "timestamp": time.time(),
        }
        self._episodes.append(episode)

        if len(self._episodes) > self.max_episodes:
            self._evict_least_important()

        self._save()
        logger.debug(
            "Episode recorded: type=%s importance=%.2f id=%s",
            event_type, importance, episode_id,
        )
        return episode_id

    def get_episodes(
        self,
        event_type: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 20,
    ) -> List[Dict]:
        """Retrieve episodes filtered by type and importance.

        Args:
            event_type: Filter to only this event type if provided.
            min_importance: Minimum importance score to include.
            limit: Maximum number of episodes to return.

        Returns:
            List of episode dicts, sorted by recency.
        """
        results = self._episodes

        if event_type:
            results = [e for e in results if e["event_type"] == event_type]

        results = [e for e in results if e["importance"] >= min_importance]

        # Sort by timestamp descending (most recent first)
        results = sorted(results, key=lambda e: e["timestamp"], reverse=True)
        return results[:limit]

    def search(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Search episodes by keyword in summary or details.

        Args:
            keyword: Case-insensitive search term.
            limit: Maximum results to return.

        Returns:
            List of matching episodes sorted by importance descending.
        """
        keyword_lower = keyword.lower()
        matches = []

        for ep in self._episodes:
            summary_hit = keyword_lower in ep["summary"].lower()
            details_hit = keyword_lower in json.dumps(ep["details"]).lower()
            if summary_hit or details_hit:
                matches.append(ep)

        matches.sort(key=lambda e: e["importance"], reverse=True)
        return matches[:limit]

    def get_recent(self, n: int = 5) -> List[Dict]:
        """Return the N most recently recorded episodes.

        Args:
            n: Number of episodes to return.

        Returns:
            List of episodes sorted by timestamp descending.
        """
        sorted_eps = sorted(self._episodes, key=lambda e: e["timestamp"], reverse=True)
        return sorted_eps[:n]

    def format_as_context(self, episodes: Optional[List[Dict]] = None, limit: int = 5) -> str:
        """Format episodes as a text block suitable for LLM context injection.

        Args:
            episodes: Specific episodes to format; uses recent ones if None.
            limit: Max episodes to include when fetching automatically.

        Returns:
            Formatted text block.
        """
        if episodes is None:
            episodes = self.get_recent(n=limit)

        if not episodes:
            return "No episodic memory available."

        lines = ["[Episodic Memory]"]
        for ep in episodes:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(ep["timestamp"]))
            lines.append(f"- [{ts}] ({ep['event_type']}) {ep['summary']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Delete all episodes for the current session."""
        self._episodes = []
        if self._file.exists():
            self._file.unlink()
        logger.info("EpisodicMemory cleared for session=%s", self.session_id)

    def _evict_least_important(self) -> None:
        """Remove the least important episode when at capacity."""
        if not self._episodes:
            return
        least = min(self._episodes, key=lambda e: (e["importance"], e["timestamp"]))
        self._episodes.remove(least)
        logger.debug("Evicted episode id=%s (importance=%.2f)", least["id"], least["importance"])

    def _load(self) -> List[Dict]:
        """Load episodes from the JSON file on disk."""
        if not self._file.exists():
            return []
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load episodes from %s: %s", self._file, exc)
            return []

    def _save(self) -> None:
        """Persist current episodes list to the JSON file."""
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._episodes, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.error("Failed to save episodes to %s: %s", self._file, exc)

    @property
    def episode_count(self) -> int:
        """Number of stored episodes."""
        return len(self._episodes)
