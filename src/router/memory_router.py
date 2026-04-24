"""Memory Router: classifies query intent and selects memory backends."""

import logging
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Supported query intent categories."""

    USER_PREFERENCE = "user_preference"
    FACTUAL_RECALL = "factual_recall"
    EXPERIENCE_RECALL = "experience_recall"
    GENERAL = "general"


# Keyword patterns per intent (order matters: first match wins)
_INTENT_PATTERNS: List[Tuple[QueryIntent, List[str]]] = [
    (
        QueryIntent.USER_PREFERENCE,
        [
            r"\bdo i (like|prefer|want|enjoy|love|hate)\b",
            r"\bmy (preference|favorite|favourite|setting|style|choice)\b",
            r"\bwhat (do|did) i (like|prefer|want|choose|say i)\b",
            r"\b(remind me|tell me) (what|how) i (prefer|like|want)\b",
            r"\bi (told you|mentioned|said) i (like|prefer|want|don'?t)\b",
            r"\bmy (name|age|job|role|language|tone)\b",
        ],
    ),
    (
        QueryIntent.EXPERIENCE_RECALL,
        [
            r"\b(remember|recall) when\b",
            r"\b(last time|previously|before|earlier) (we|i|you)\b",
            r"\bwhat happened (when|after|before|during)\b",
            r"\bthe (last|previous|prior) (time|session|conversation|chat)\b",
            r"\bwhat did (we|i|you) (do|discuss|talk about|decide)\b",
            r"\b(episode|event|incident|experience|occasion)\b",
        ],
    ),
    (
        QueryIntent.FACTUAL_RECALL,
        [
            r"\bwhat is\b",
            r"\bwhat are\b",
            r"\bdefine\b",
            r"\bexplain\b",
            r"\bhow does\b",
            r"\bwhy (is|does|do|did|are)\b",
            r"\btell me about\b",
            r"\b(meaning|definition|concept|theory|principle)\b",
            r"\bfact(s)?\b",
        ],
    ),
]


class MemoryRouter:
    """Routes memory queries to appropriate backend(s) based on intent.

    Intent classification uses a two-stage strategy:
      1. Fast keyword/regex matching (no LLM call).
      2. Optional LLM fallback for ambiguous queries.

    Backend mapping:
      - USER_PREFERENCE   → short_term + long_term
      - FACTUAL_RECALL    → semantic
      - EXPERIENCE_RECALL → episodic
      - GENERAL           → short_term + semantic
    """

    def __init__(self, use_llm_fallback: bool = False, llm=None):
        """Initialize the memory router.

        Args:
            use_llm_fallback: If True and an LLM is provided, use it to
                classify ambiguous queries not matched by regex.
            llm: A LangChain chat model instance used for LLM fallback.
        """
        self._use_llm_fallback = use_llm_fallback and llm is not None
        self._llm = llm
        logger.info(
            "MemoryRouter initialized (llm_fallback=%s)", self._use_llm_fallback
        )

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify a user query into one of the QueryIntent categories.

        Args:
            query: The raw user query string.

        Returns:
            The most likely QueryIntent.
        """
        if not query or not query.strip():
            return QueryIntent.GENERAL

        query_lower = query.lower().strip()

        for intent, patterns in _INTENT_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.debug(
                        "Query matched intent=%s via pattern '%s'", intent.value, pattern
                    )
                    return intent

        if self._use_llm_fallback:
            return self._llm_classify(query)

        return QueryIntent.GENERAL

    def route(self, query: str) -> Dict:
        """Determine which memory backends to query for a given input.

        Args:
            query: The user's query string.

        Returns:
            Dict with keys:
              - 'intent'       : QueryIntent value
              - 'backends'     : list of backend names to query
              - 'primary'      : the highest-priority backend name
              - 'explanation'  : human-readable routing rationale
        """
        intent = self.classify_intent(query)
        routing = _ROUTING_TABLE[intent]
        result = {
            "intent": intent.value,
            "backends": routing["backends"],
            "primary": routing["primary"],
            "explanation": routing["explanation"],
        }
        logger.info(
            "Routed query to backends=%s (intent=%s)", routing["backends"], intent.value
        )
        return result

    def select_memory_sections(
        self,
        query: str,
        available_backends: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Return flags indicating which memory backends to activate.

        Args:
            query: User query string.
            available_backends: Whitelist of backends that can be queried.
                Defaults to all four backends.

        Returns:
            Dict with bool flags: include_short_term, include_long_term,
            include_episodic, include_semantic.
        """
        available = set(available_backends or ["short_term", "long_term", "episodic", "semantic"])
        routing = self.route(query)
        active = set(routing["backends"]) & available

        return {
            "include_short_term": "short_term" in active,
            "include_long_term": "long_term" in active,
            "include_episodic": "episodic" in active,
            "include_semantic": "semantic" in active,
        }

    def _llm_classify(self, query: str) -> QueryIntent:
        """Use an LLM to classify ambiguous queries.

        Args:
            query: The user query string.

        Returns:
            QueryIntent determined by the LLM, or GENERAL on failure.
        """
        prompt = (
            "Classify this user query into exactly one category:\n"
            "- user_preference: asks about user's personal preferences or settings\n"
            "- factual_recall: asks for facts, definitions, explanations\n"
            "- experience_recall: recalls past events, sessions, or experiences\n"
            "- general: does not fit the above categories\n\n"
            f"Query: {query}\n\n"
            "Reply with only the category name."
        )
        try:
            response = self._llm.invoke(prompt)
            content = response.content.strip().lower()
            for intent in QueryIntent:
                if intent.value in content:
                    return intent
        except Exception as exc:
            logger.error("LLM intent classification failed: %s", exc)

        return QueryIntent.GENERAL


# Static routing table mapping intents to backend configurations
_ROUTING_TABLE: Dict[QueryIntent, Dict] = {
    QueryIntent.USER_PREFERENCE: {
        "backends": ["short_term", "long_term"],
        "primary": "short_term",
        "explanation": "User preference queries rely on recent and persistent conversation history.",
    },
    QueryIntent.FACTUAL_RECALL: {
        "backends": ["semantic"],
        "primary": "semantic",
        "explanation": "Factual recall uses vector similarity search over indexed knowledge.",
    },
    QueryIntent.EXPERIENCE_RECALL: {
        "backends": ["episodic"],
        "primary": "episodic",
        "explanation": "Experience recall retrieves logged episodes from past interactions.",
    },
    QueryIntent.GENERAL: {
        "backends": ["short_term", "semantic"],
        "primary": "short_term",
        "explanation": "General queries use current context and semantic similarity.",
    },
}
