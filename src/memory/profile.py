"""User profile memory: key-value facts with conflict-resolution update."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROFILES_DIR = os.getenv("PROFILES_DIR", "./data/profiles")

# Aliases used to detect which profile key a query refers to
_KEY_ALIASES: Dict[str, List[str]] = {
    "name": ["name", "called", "my name", "what's my name"],
    "language": ["language", "speak", "native", "prefer language"],
    "allergy": ["allergy", "allergic", "intolerant", "cannot eat", "food"],
    "job": ["job", "work", "role", "profession", "occupation"],
    "age": ["age", "old", "years old"],
    "preference": ["preference", "prefer", "like", "favorite", "favourite"],
    "goal": ["goal", "objective", "want to", "trying to"],
    "location": ["location", "city", "country", "from", "live in"],
}


class UserProfile:
    """Persistent key-value store for user profile facts.

    Implements conflict-aware update: setting a key always overwrites the
    previous value, preventing contradictory facts from accumulating.

    Example:
        profile = UserProfile(session_id="user_1")
        profile.update("allergy", "milk")
        profile.update("allergy", "soy")  # conflict resolved — milk removed
        profile.get("allergy")            # "soy"
    """

    def __init__(self, session_id: str = "default", profiles_dir: Optional[str] = None):
        """Initialize user profile store.

        Args:
            session_id: Unique user/session identifier.
            profiles_dir: Directory for profile JSON persistence.
        """
        self.session_id = session_id
        self._dir = Path(profiles_dir or _PROFILES_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{session_id}_profile.json"
        self._facts: Dict[str, str] = {}
        self._history: List[Dict] = []  # audit log of all updates
        self._load()
        logger.info("UserProfile loaded for session=%s (%d facts)", session_id, len(self._facts))

    def update(self, key: str, value: str, source: str = "user") -> Tuple[bool, Optional[str]]:
        """Set a profile fact, overwriting any previous value for that key.

        Args:
            key: Normalized fact key (e.g., 'allergy', 'name').
            value: New value for the fact.
            source: Who provided this update ('user', 'inferred', etc.).

        Returns:
            Tuple of (was_conflict: bool, old_value: Optional[str]).
            was_conflict is True when a different previous value was overwritten.

        Raises:
            ValueError: If key or value is empty.
        """
        if not key or not key.strip():
            raise ValueError("profile key must not be empty")
        if not value or not value.strip():
            raise ValueError("profile value must not be empty")

        key = key.strip().lower()
        value = value.strip()
        old_value = self._facts.get(key)
        was_conflict = old_value is not None and old_value.lower() != value.lower()

        self._facts[key] = value
        self._history.append({
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "source": source,
            "timestamp": time.time(),
            "conflict_resolved": was_conflict,
        })

        if was_conflict:
            logger.info(
                "Profile conflict resolved: key='%s' old='%s' new='%s'",
                key, old_value, value,
            )
        else:
            logger.debug("Profile updated: key='%s' value='%s'", key, value)

        self._save()
        return was_conflict, old_value

    def get(self, key: str) -> Optional[str]:
        """Retrieve a profile fact by key.

        Args:
            key: Fact key to look up.

        Returns:
            The stored value, or None if not set.
        """
        return self._facts.get(key.strip().lower())

    def get_all(self) -> Dict[str, str]:
        """Return a copy of all stored facts.

        Returns:
            Dict of all key-value profile facts.
        """
        return dict(self._facts)

    def find_key_for_query(self, query: str) -> Optional[str]:
        """Guess which profile key a natural-language query is asking about.

        Args:
            query: User query string.

        Returns:
            Matching profile key, or None if no match found.
        """
        query_lower = query.lower()
        for key, aliases in _KEY_ALIASES.items():
            if any(alias in query_lower for alias in aliases):
                if key in self._facts:
                    return key
        return None

    def extract_and_update(self, text: str) -> List[Tuple[str, str, bool]]:
        """Heuristically extract and save profile facts from free text.

        Looks for patterns like "I am allergic to X", "My name is X",
        "I prefer X", "I work as X", etc.

        Args:
            text: Free-form user input to scan for profile facts.

        Returns:
            List of (key, value, was_conflict) tuples for each fact found.
        """
        updates: List[Tuple[str, str, bool]] = []
        text_lower = text.lower()

        patterns = [
            ("name",       ["my name is ", "i am called ", "call me "]),
            ("allergy",    ["i am allergic to ", "i'm allergic to ", "allergic to ",
                            "i have an allergy to ", "not allergic to "]),
            ("language",   ["i speak ", "my language is ", "i prefer to speak "]),
            ("job",        ["i work as ", "i am a ", "my job is ", "my role is "]),
            ("goal",       ["i want to ", "my goal is ", "i am trying to "]),
            ("location",   ["i live in ", "i am from ", "i'm from "]),
        ]

        # Special case: "actually" / "correction" pattern for conflict
        # e.g. "actually, I'm allergic to soy, not milk"
        if "allergic" in text_lower:
            import re
            # "allergic to SOY, not milk" → new allergy = soy
            m = re.search(r"allergic to (\w+)[,.]?\s+not\b", text_lower)
            if not m:
                # "actually allergic to SOY"
                m = re.search(r"actually (?:i am |i'm )?allergic to (\w+)", text_lower)
            if m:
                new_val = m.group(1).strip()
                was_conflict, _ = self.update("allergy", new_val, source="inferred")
                updates.append(("allergy", new_val, was_conflict))
                return updates

        for key, triggers in patterns:
            for trigger in triggers:
                idx = text_lower.find(trigger)
                if idx != -1:
                    raw = text[idx + len(trigger):].split(".")[0].split(",")[0].strip()
                    if raw:
                        was_conflict, _ = self.update(key, raw, source="inferred")
                        updates.append((key, raw, was_conflict))
                        break  # one update per key per message

        return updates

    def to_context(self) -> str:
        """Format profile facts as a text block for LLM prompt injection.

        Returns:
            Formatted string, or empty string if no facts stored.
        """
        if not self._facts:
            return ""
        lines = ["[User Profile]"]
        for key, value in sorted(self._facts.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def conflict_history(self) -> List[Dict]:
        """Return all past conflict-resolution events.

        Returns:
            List of history entries where conflict_resolved=True.
        """
        return [h for h in self._history if h.get("conflict_resolved")]

    def delete(self, key: str) -> bool:
        """Remove a single profile fact.

        Args:
            key: Fact key to delete.

        Returns:
            True if the key existed and was removed.
        """
        key = key.strip().lower()
        if key in self._facts:
            del self._facts[key]
            self._save()
            logger.info("Profile fact deleted: key='%s'", key)
            return True
        return False

    def clear(self) -> None:
        """Delete all profile facts."""
        self._facts = {}
        self._history = []
        if self._file.exists():
            self._file.unlink()
        logger.info("UserProfile cleared for session=%s", self.session_id)

    def _load(self) -> None:
        """Load profile from JSON file if it exists."""
        if not self._file.exists():
            return
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._facts = data.get("facts", {})
            self._history = data.get("history", [])
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load profile from %s: %s", self._file, exc)

    def _save(self) -> None:
        """Persist current profile to JSON file."""
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(
                    {"facts": self._facts, "history": self._history},
                    f, ensure_ascii=False, indent=2,
                )
        except OSError as exc:
            logger.error("Failed to save profile to %s: %s", self._file, exc)

    @property
    def fact_count(self) -> int:
        """Number of facts currently stored."""
        return len(self._facts)
