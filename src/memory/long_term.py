"""Long-term memory backed by Redis for persistent conversation history."""

import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis package not installed; LongTermMemory will use in-memory fallback")


class LongTermMemory:
    """Persistent conversation history stored in Redis.

    Falls back to an in-process dictionary when Redis is unavailable,
    enabling development without a running Redis instance.
    """

    def __init__(
        self,
        session_id: str = "default",
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        db: int = 0,
        ttl: int = 86400,
    ):
        """Initialize Redis-backed long-term memory.

        Args:
            session_id: Unique identifier for this conversation session.
            host: Redis host (defaults to REDIS_HOST env or 'localhost').
            port: Redis port (defaults to REDIS_PORT env or 6379).
            password: Redis password (defaults to REDIS_PASSWORD env).
            db: Redis database index.
            ttl: Time-to-live in seconds for stored keys (default 24 h).
        """
        self.session_id = session_id
        self.ttl = ttl
        self._redis_key = f"memory:long_term:{session_id}"
        self._fallback: Dict[str, List] = {}
        self._use_fallback = False

        if not REDIS_AVAILABLE:
            self._use_fallback = True
            logger.info("Using in-memory fallback for LongTermMemory")
            return

        _host = host or os.getenv("REDIS_HOST", "localhost")
        _port = port or int(os.getenv("REDIS_PORT", "6379"))
        _password = password or os.getenv("REDIS_PASSWORD") or None
        _db = db or int(os.getenv("REDIS_DB", "0"))

        try:
            self._client = redis.Redis(
                host=_host,
                port=_port,
                password=_password,
                db=_db,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._client.ping()
            logger.info(
                "LongTermMemory connected to Redis at %s:%d (session=%s)",
                _host, _port, session_id,
            )
        except Exception as exc:
            logger.warning("Redis unavailable (%s); switching to fallback", exc)
            self._use_fallback = True

    def add_interaction(self, human_input: str, ai_output: str) -> None:
        """Persist a human-AI interaction to Redis or fallback store.

        Args:
            human_input: The user's message.
            ai_output: The AI assistant's response.

        Raises:
            ValueError: If either input is empty.
        """
        if not human_input or not human_input.strip():
            raise ValueError("human_input must not be empty")
        if not ai_output or not ai_output.strip():
            raise ValueError("ai_output must not be empty")

        entry = {
            "timestamp": time.time(),
            "human": human_input,
            "ai": ai_output,
        }
        serialized = json.dumps(entry, ensure_ascii=False)

        if self._use_fallback:
            self._fallback.setdefault(self._redis_key, []).append(serialized)
        else:
            try:
                self._client.rpush(self._redis_key, serialized)
                self._client.expire(self._redis_key, self.ttl)
            except Exception as exc:
                logger.error("Redis write failed: %s; using fallback", exc)
                self._fallback.setdefault(self._redis_key, []).append(serialized)

        logger.debug("Long-term: persisted interaction for session=%s", self.session_id)

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Retrieve the most recent interactions from persistent storage.

        Args:
            limit: Maximum number of recent interactions to return.

        Returns:
            List of interaction dicts with 'timestamp', 'human', 'ai' keys.
        """
        if self._use_fallback:
            raw_list = self._fallback.get(self._redis_key, [])
        else:
            try:
                raw_list = self._client.lrange(self._redis_key, -limit, -1)
            except Exception as exc:
                logger.error("Redis read failed: %s", exc)
                raw_list = self._fallback.get(self._redis_key, [])

        interactions = []
        for raw in raw_list:
            try:
                interactions.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed Redis entry: %s", exc)

        return interactions[-limit:]

    def get_history_as_text(self, limit: int = 20) -> str:
        """Return persistent history as formatted plain text.

        Args:
            limit: Maximum number of interactions to include.

        Returns:
            Formatted conversation string.
        """
        interactions = self.get_history(limit=limit)
        lines: List[str] = []
        for item in interactions:
            lines.append(f"Human: {item['human']}")
            lines.append(f"AI: {item['ai']}")
        return "\n".join(lines)

    def search_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Search history for interactions containing a keyword.

        Args:
            keyword: Case-insensitive keyword to search for.
            limit: Maximum number of matching interactions to return.

        Returns:
            List of matching interaction dicts.
        """
        keyword_lower = keyword.lower()
        history = self.get_history(limit=500)
        matches = [
            item for item in history
            if keyword_lower in item["human"].lower() or keyword_lower in item["ai"].lower()
        ]
        return matches[-limit:]

    def clear_session(self) -> None:
        """Delete all history for the current session."""
        if self._use_fallback:
            self._fallback.pop(self._redis_key, None)
        else:
            try:
                self._client.delete(self._redis_key)
            except Exception as exc:
                logger.error("Redis delete failed: %s", exc)
        logger.info("LongTermMemory cleared for session=%s", self.session_id)

    @property
    def entry_count(self) -> int:
        """Number of stored interactions."""
        if self._use_fallback:
            return len(self._fallback.get(self._redis_key, []))
        try:
            return self._client.llen(self._redis_key)
        except Exception:
            return 0
