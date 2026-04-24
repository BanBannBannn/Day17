"""Main AI Agent: integrates MemoryGraph, UserProfile, and all memory backends."""

import logging
import os
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from ..memory.manager import MemoryManager
from ..memory.profile import UserProfile
from ..router.memory_router import MemoryRouter
from ..graph.state import MemoryGraph, MemoryState

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with a multi-tier memory system.\n"
    "You have access to:\n"
    "  - Short-term: recent conversation buffer\n"
    "  - Long-term: persistent user profile facts\n"
    "  - Episodic: log of important past events\n"
    "  - Semantic: vector-indexed knowledge fragments\n"
    "Always use the context provided to give personalized, accurate responses."
)


class MemoryAgent:
    """Memory-augmented AI agent powered by a LangGraph skeleton pipeline.

    Combines all four memory backends through a MemoryGraph that runs
    four nodes per turn: retrieve -> build_prompt -> generate -> save.

    Example:
        agent = MemoryAgent(session_id="alice")
        result = agent.chat("My name is Alice")
        result = agent.chat("What is my name?")
        # -> "Based on your profile, your name is: Alice."
    """

    def __init__(
        self,
        session_id: str = "default",
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_manager: Optional[MemoryManager] = None,
        profile: Optional[UserProfile] = None,
    ):
        """Initialize the memory agent with all sub-systems.

        Args:
            session_id: Unique session identifier for all backends.
            system_prompt: Base system instruction string.
            model: OpenAI model name (defaults to OPENAI_MODEL env or gpt-4o-mini).
            temperature: LLM sampling temperature.
            max_tokens: Token budget for the context window.
            memory_manager: Pre-configured MemoryManager (optional).
            profile: Pre-configured UserProfile (optional).
        """
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._turn_count = 0

        self.memory = memory_manager or MemoryManager(session_id=session_id)
        self.profile = profile or UserProfile(session_id=session_id)
        self.router = MemoryRouter()

        self.graph = MemoryGraph(
            manager=self.memory,
            profile=self.profile,
            router=self.router,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

        llm = self._init_llm(model, temperature)
        if llm is not None:
            self.graph.set_llm(self._make_llm_callable(llm))

        logger.info("MemoryAgent initialized (session=%s)", session_id)

    def _init_llm(self, model: Optional[str], temperature: float):
        """Create ChatOpenAI if an API key is available, otherwise None."""
        if not LANGCHAIN_AVAILABLE:
            return None
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-your"):
            logger.info("No valid OPENAI_API_KEY — running in smart mock mode")
            return None
        _model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            return ChatOpenAI(model=_model, temperature=temperature, openai_api_key=api_key)
        except Exception as exc:
            logger.error("ChatOpenAI init failed: %s", exc)
            return None

    @staticmethod
    def _make_llm_callable(llm):
        """Wrap a ChatOpenAI instance into a (context, query) -> str callable.

        Returns None on quota/auth errors so node_generate_response can fall
        back to the state-aware smart mock.
        """
        def call(context: str, query: str):
            messages = [SystemMessage(content=context), HumanMessage(content=query)]
            try:
                result = llm.invoke(messages)
                return result.content
            except Exception as exc:
                err = str(exc)
                if "429" in err or "401" in err or "quota" in err.lower() or "insufficient" in err.lower():
                    logger.warning("API quota/auth error — falling back to smart mock")
                    return None  # signal to caller to use smart mock
                return f"[LLM error: {exc}]"

        return call

    def chat(self, user_input: str) -> Dict:
        """Process one user message through the full memory graph pipeline.

        Args:
            user_input: The user's text input.

        Returns:
            Dict with keys:
              response, intent, backends_used, token_budget,
              profile_updates, turn, latency_ms.

        Raises:
            ValueError: If user_input is empty.
        """
        if not user_input or not user_input.strip():
            raise ValueError("user_input must not be empty")

        start = time.time()
        self._turn_count += 1

        state: MemoryState = self.graph.run(user_input)

        latency_ms = round((time.time() - start) * 1000, 1)
        budget = state.get("metadata", {}).get("token_budget", {})

        return {
            "response": state.get("response", ""),
            "intent": state.get("intent", "general"),
            "backends_used": state.get("metadata", {}).get("backends_used", []),
            "token_budget": budget,
            "profile": state.get("user_profile", {}),
            "episodes_count": len(state.get("episodes", [])),
            "semantic_hits": len(state.get("semantic_hits", [])),
            "turn": self._turn_count,
            "latency_ms": latency_ms,
        }

    def add_knowledge(self, text: str, source: Optional[str] = None) -> str:
        """Index a knowledge fragment into semantic memory.

        Args:
            text: Text to embed and store.
            source: Optional source label.

        Returns:
            Document ID.
        """
        meta = {"type": "knowledge", "source": source or "manual"}
        return self.memory.semantic.add_memory(text, metadata=meta)

    def reset_session(self) -> None:
        """Clear all memory and profile for a fresh start."""
        self.memory.clear_all()
        self.profile.clear()
        self._turn_count = 0
        logger.info("Session reset: %s", self.session_id)

    def get_stats(self) -> Dict:
        """Return runtime stats for this session.

        Returns:
            Dict with session info, memory counts, and profile facts.
        """
        return {
            "session_id": self.session_id,
            "turns_completed": self._turn_count,
            "profile_facts": self.profile.get_all(),
            "conflict_history": self.profile.conflict_history(),
            "memory": self.memory.stats(),
        }
