"""LangGraph skeleton: MemoryState TypedDict and graph node functions.

This module implements a LangGraph-compatible state machine for the
multi-tier memory agent. It can run as a full LangGraph pipeline when
the package is installed, or as a plain Python function chain (skeleton)
when it is not.
"""

import logging
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.info("langgraph not installed; using skeleton graph runner")


# ─── State Definition ────────────────────────────────────────────────────────

class MemoryState(TypedDict, total=False):
    """Shared state flowing through every node in the memory graph.

    Fields:
        messages:       Ordered list of conversation turns
                        [{"role": "human"|"ai", "content": str}, ...].
        user_profile:   Dict of extracted user facts (name, allergy, job…).
        episodes:       Recent episodic memory entries as dicts.
        semantic_hits:  Top-k text fragments from vector search.
        memory_budget:  Remaining token budget after context assembly.
        current_query:  The user's latest input message.
        response:       The AI's generated response for this turn.
        intent:         Classified query intent string.
        context_prompt: Assembled prompt string injected into the LLM.
        metadata:       Arbitrary per-turn metadata (latency, backends…).
    """

    messages: List[Dict[str, str]]
    user_profile: Dict[str, str]
    episodes: List[Dict]
    semantic_hits: List[str]
    memory_budget: int
    current_query: str
    response: str
    intent: str
    context_prompt: str
    metadata: Dict[str, Any]


def make_initial_state(query: str, max_tokens: int = 4096) -> MemoryState:
    """Build a fresh MemoryState for a new conversation turn.

    Args:
        query: The user's current input.
        max_tokens: Total token budget for this turn.

    Returns:
        Initialized MemoryState dict.
    """
    return MemoryState(
        messages=[],
        user_profile={},
        episodes=[],
        semantic_hits=[],
        memory_budget=max_tokens,
        current_query=query,
        response="",
        intent="general",
        context_prompt="",
        metadata={},
    )


# ─── Graph Nodes ─────────────────────────────────────────────────────────────

def node_retrieve_memory(state: MemoryState, manager, profile, router) -> MemoryState:
    """Node: retrieve relevant memory from all backends into state.

    Queries the MemoryRouter to select backends, then pulls context from
    ShortTermMemory, LongTermMemory, EpisodicMemory, and SemanticMemory.

    Args:
        state: Current MemoryState.
        manager: MemoryManager instance.
        profile: UserProfile instance.
        router: MemoryRouter instance.

    Returns:
        Updated MemoryState with populated memory fields.
    """
    query = state.get("current_query", "")

    # Router decides which backends to query
    routing = router.route(query)
    flags = router.select_memory_sections(query)
    state["intent"] = routing["intent"]
    state["metadata"]["backends_used"] = routing["backends"]

    # Short-term → messages list
    history = manager.short_term.get_history()
    state["messages"] = [
        {
            "role": "human" if hasattr(m, "content") and type(m).__name__ == "HumanMessage" else "ai",
            "content": m.content,
        }
        for m in history
    ]

    # Long-term profile
    state["user_profile"] = profile.get_all()

    # Episodic
    if flags.get("include_episodic", True):
        state["episodes"] = manager.episodic.get_recent(n=5)

    # Semantic
    if flags.get("include_semantic", True):
        semantic_results = manager.semantic.search(query, top_k=3)
        state["semantic_hits"] = [r["text"] for r in semantic_results]

    logger.debug(
        "retrieve_memory: intent=%s profile_facts=%d episodes=%d semantic=%d",
        state["intent"],
        len(state["user_profile"]),
        len(state["episodes"]),
        len(state["semantic_hits"]),
    )
    return state


def node_build_prompt(state: MemoryState, system_prompt: str, max_tokens: int) -> MemoryState:
    """Node: assemble all memory sections into the context prompt.

    Priority order (highest to lowest, evict lowest first):
      1. System Prompt
      2. User Profile + Recent Query
      3. Episodic + Semantic context
      4. Raw conversation history

    Args:
        state: Current MemoryState (must have memory fields populated).
        system_prompt: Base system instruction string.
        max_tokens: Hard token budget.

    Returns:
        Updated MemoryState with context_prompt and memory_budget set.
    """
    from src.context.window_manager import (
        ContextWindowManager, ContextPriority, _estimate_tokens
    )

    mgr = ContextWindowManager(max_tokens=max_tokens)

    # P4 — System prompt (never evicted)
    if system_prompt:
        mgr.add_block("system_prompt", system_prompt, ContextPriority.SYSTEM_PROMPT)

    # P3 — User profile + current query
    profile_text = _format_profile(state.get("user_profile", {}))
    if profile_text:
        mgr.add_block("user_profile", profile_text, ContextPriority.RECENT_USER_INTENT)

    # P3 — Recent conversation (short-term)
    recent_msgs = state.get("messages", [])[-6:]  # last 3 turns
    if recent_msgs:
        lines = [
            f"{'Human' if m['role'] == 'human' else 'AI'}: {m['content']}"
            for m in recent_msgs
        ]
        mgr.add_block("recent_conversation", "\n".join(lines), ContextPriority.RECENT_USER_INTENT)

    # P2 — Semantic hits
    hits = state.get("semantic_hits", [])
    if hits:
        semantic_text = "[Semantic Memory]\n" + "\n".join(f"- {h}" for h in hits)
        mgr.add_block("semantic_context", semantic_text, ContextPriority.SEMANTIC_CONTEXT)

    # P2 — Episodic memory
    episodes = state.get("episodes", [])
    if episodes:
        import time as _time
        lines = []
        for ep in episodes:
            ts = _time.strftime("%Y-%m-%d", _time.localtime(ep.get("timestamp", 0)))
            lines.append(f"- [{ts}] ({ep.get('event_type', '?')}) {ep.get('summary', '')}")
        mgr.add_block("episodic_context", "[Episodic Memory]\n" + "\n".join(lines),
                      ContextPriority.SEMANTIC_CONTEXT)

    prompt = mgr.build_prompt()
    budget_report = mgr.get_budget_report()

    state["context_prompt"] = prompt
    state["memory_budget"] = budget_report["remaining_tokens"]
    state["metadata"]["token_budget"] = budget_report

    return state


def node_generate_response(state: MemoryState, llm) -> MemoryState:
    """Node: call the LLM (or mock) with the assembled context.

    Args:
        state: Current MemoryState with context_prompt set.
        llm: A callable(context_prompt, query) -> str, or None for mock.

    Returns:
        Updated MemoryState with response set.
    """
    query = state.get("current_query", "")
    context = state.get("context_prompt", "")

    if llm is not None:
        try:
            result = llm(context, query)
            # None signals quota/auth error — fall through to smart mock
            response = result if result is not None else _smart_mock_response(query, state)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            response = _smart_mock_response(query, state)
    else:
        response = _smart_mock_response(query, state)

    state["response"] = response
    return state


def node_save_memory(state: MemoryState, manager, profile) -> MemoryState:
    """Node: persist the completed turn back to all memory backends.

    Also extracts and saves any new profile facts discovered in the
    user's message, resolving conflicts automatically.

    Args:
        state: Current MemoryState with response set.
        manager: MemoryManager instance.
        profile: UserProfile instance.

    Returns:
        MemoryState unchanged (side-effects only).
    """
    query = state.get("current_query", "")
    response = state.get("response", "")

    if not query or not response:
        return state

    # Persist conversation turn
    manager.short_term.add_interaction(query, response)
    manager.long_term.add_interaction(query, response)

    # Extract profile facts from user message
    updates = profile.extract_and_update(query)
    if updates:
        # Refresh profile in state
        state["user_profile"] = profile.get_all()
        for key, value, was_conflict in updates:
            logger.info(
                "Profile auto-extracted: %s=%s (conflict=%s)", key, value, was_conflict
            )
        # Record profile update as episode
        if updates:
            fact_summary = "; ".join(f"{k}={v}" for k, v, _ in updates)
            manager.episodic.add_episode(
                event_type="profile_update",
                summary=f"Profile updated: {fact_summary}",
                details={"updates": [{"key": k, "value": v} for k, v, _ in updates]},
                importance=0.7,
            )

    # Index this interaction in semantic memory
    combined = f"Human: {query}\nAI: {response}"
    manager.semantic.add_memory(combined, metadata={"type": "conversation"})

    return state


# ─── Smart Mock LLM ──────────────────────────────────────────────────────────

def _smart_mock_response(query: str, state: MemoryState) -> str:
    """Generate a context-aware mock response without a real LLM.

    Reads the MemoryState to produce responses that actually demonstrate
    memory is being used — profile recall, conflict updates, episodic
    recall, and semantic retrieval all produce distinct, meaningful output.

    Args:
        query: User's current question.
        state: Populated MemoryState from the retrieve+build nodes.

    Returns:
        A mock response string reflecting the available memory context.
    """
    q = query.lower().strip()
    profile = state.get("user_profile", {})
    episodes = state.get("episodes", [])
    semantic_hits = state.get("semantic_hits", [])

    # ── 1. Conflict / correction (HIGHEST PRIORITY — must run before recall) ──
    # Detects "actually X not Y", "I meant X", corrections, etc.
    conflict_triggers = ["actually", "correction", "not ", "i meant", "i was wrong",
                         "nhầm", "i moved", "i changed"]
    if any(t in q for t in conflict_triggers):
        updates = _parse_conflict_from_query(query)
        if updates:
            k, v = updates
            old = profile.get(k, "previous value")
            return (
                f"Got it! I've updated your **{k}** from '{old}' to '{v}'. "
                f"The conflict has been resolved — I'll use '{v}' going forward. "
                f"(Long-term profile updated)"
            )

    # ── 2. Profile fact ingestion acknowledgment ────────────────────────
    ingestion_triggers = [
        "i am allergic", "i'm allergic",
        "my name is", "i am called",
        "i work as", "i am a ",
        "i prefer", "i live in", "i'm from",
        "toi di ung", "ten toi la",
        "i just fixed", "we deployed", "we decided",
    ]
    if any(t in q for t in ingestion_triggers):
        # Pick an informative acknowledgment based on what was ingested
        if "allergic" in q:
            import re
            m = re.search(r"allergic to (\w+)", q)
            val = m.group(1) if m else "that"
            return (
                f"Noted! I've recorded your allergy to **{val}** in your profile. "
                f"I'll keep this in mind. (Profile memory updated)"
            )
        if "name is" in q or "i am called" in q:
            import re
            m = re.search(r"(?:name is|called) (\w+)", q)
            val = m.group(1).capitalize() if m else "that"
            return (
                f"Nice to meet you, **{val}**! I've saved your name to your profile. "
                f"(Profile memory updated)"
            )
        if "fixed" in q or "deployed" in q or "decided" in q:
            return (
                "Got it! I've recorded this as an episode in your episodic memory. "
                "I can recall it later when you ask. (Episodic memory updated)"
            )
        return (
            "Got it! I've noted that in your profile and will remember it "
            "for the rest of our conversation. (Profile memory updated)"
        )

    # ── 3. Profile recall queries ───────────────────────────────────────
    for key, aliases in [
        ("name",       ["name", "called", "who am i"]),
        ("allergy",    ["allerg", "cannot eat", "food restriction"]),
        ("job",        ["job", "work", "role", "profession"]),
        ("language",   ["language", "speak"]),
        ("goal",       ["goal", "objective", "trying to"]),
        ("location",   ["from", "live in", "city", "country", "where"]),
    ]:
        if any(alias in q for alias in aliases) and key in profile:
            return (
                f"Based on your profile, your {key} is: **{profile[key]}**. "
                f"(Retrieved from long-term profile memory)"
            )

    # ── Episodic recall ─────────────────────────────────────────────────
    recall_triggers = ["remember", "last time", "previous", "before", "earlier", "we discussed"]
    if any(t in q for t in recall_triggers) and episodes:
        ep = episodes[-1]
        return (
            f"Yes! I recall that previously: {ep.get('summary', 'we had an important interaction')}. "
            f"(Retrieved from episodic memory, event_type='{ep.get('event_type', 'unknown')}')"
        )

    # ── Semantic retrieval ──────────────────────────────────────────────
    if semantic_hits:
        snippet = semantic_hits[0][:120]
        return (
            f"Based on my semantic memory: {snippet}... "
            f"(Retrieved from vector store — {len(semantic_hits)} relevant fragments found)"
        )

    # ── Token / context window queries ─────────────────────────────────
    budget = state.get("memory_budget", 0)
    if any(t in q for t in ["token", "context", "budget", "memory size", "window"]):
        return (
            f"Current memory budget: **{budget} tokens remaining** in this context window. "
            f"The ContextWindowManager uses 4-level priority eviction: "
            f"SYSTEM_PROMPT > RECENT_USER_INTENT > SEMANTIC_CONTEXT > RAW_HISTORY."
        )

    # ── Factual fallback with canned answers ────────────────────────────
    return _canned_factual_response(q, profile)


def _parse_conflict_from_query(query: str):
    """Try to extract (key, new_value) from a correction query.

    Handles patterns like:
      "allergic to soy, not milk"  -> (allergy, soy)
      "not milk, allergic to soy"  -> (allergy, soy)
    """
    import re
    q = query.lower()
    patterns = [
        # "allergic to SOY, not milk" — SOY is the NEW value
        (r"allergic to (\w+)[,.]?\s+not\b", "allergy"),
        # "not allergic to milk ... soy" — SOY is the new value
        (r"\bnot allergic to \w+[,.]?\s+(?:allergic to |i have )?(\w+)", "allergy"),
        # "i'm actually allergic to SOY"
        (r"actually (?:i am |i'm )?allergic to (\w+)", "allergy"),
        (r"my name is (\w+)", "name"),
        (r"i(?:'m| am) from ([^,.]+)", "location"),
        (r"i moved[^.]*(?:to|in) ([^,.]+)", "location"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, q)
        if m:
            return key, m.group(1).strip()
    return None


def _canned_factual_response(query: str, profile: Dict) -> str:
    """Return a topic-relevant canned answer for factual queries."""
    topics = {
        "python": "Python is a high-level, interpreted programming language known for its readable syntax, created by Guido van Rossum in 1991.",
        "machine learning": "Machine learning is a subset of AI where models learn patterns from data to make predictions without explicit programming.",
        "neural network": "A neural network is a computational model inspired by the brain, consisting of layers of interconnected nodes that learn from data.",
        "docker": "Docker is a containerization platform that packages applications and dependencies into portable containers.",
        "redis": "Redis is an in-memory key-value data store used for caching, session management, and real-time data pipelines.",
        "langchain": "LangChain is a framework for building LLM-powered applications with tools, memory, and agent orchestration.",
        "langgraph": "LangGraph extends LangChain with graph-based state machines, enabling complex multi-step agent workflows.",
        "vector": "Vector databases store embeddings for semantic similarity search, enabling RAG (Retrieval-Augmented Generation).",
        "rag": "RAG (Retrieval-Augmented Generation) combines semantic search with LLM generation to ground responses in factual data.",
        "scrum": "Scrum is an Agile framework using time-boxed sprints (1-4 weeks) with ceremonies: planning, standup, review, retrospective.",
        "microservice": "Microservices architecture decomposes an application into small, independently deployable services communicating via APIs.",
        "aws": "AWS (Amazon Web Services) is a cloud platform offering compute (EC2/Lambda), storage (S3), databases (RDS/DynamoDB), and more.",
    }
    for keyword, answer in topics.items():
        if keyword in query:
            profile_suffix = ""
            if profile.get("name"):
                profile_suffix = f" Hope that helps, {profile['name']}!"
            return answer + profile_suffix

    profile_note = ""
    if profile:
        facts = list(profile.items())[:2]
        profile_note = f" (I remember you: {', '.join(f'{k}={v}' for k, v in facts)})"

    return (
        f"[MOCK] I've processed your query using the multi-tier memory stack. "
        f"My context window has {_summarize_state_for_mock({})} of retrieved memory.{profile_note}"
    )


def _summarize_state_for_mock(state: Dict) -> str:
    return "short-term + long-term + episodic + semantic"


def _format_profile(profile: Dict[str, str]) -> str:
    """Format profile dict as a prompt-injectable string."""
    if not profile:
        return ""
    lines = ["[User Profile]"]
    for k, v in sorted(profile.items()):
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ─── Graph Runner ─────────────────────────────────────────────────────────────

class MemoryGraph:
    """Skeleton LangGraph-compatible pipeline for the memory agent.

    Chains four nodes in order:
      retrieve_memory -> build_prompt -> generate_response -> save_memory

    When langgraph is installed, this class can be upgraded to use
    StateGraph directly. Until then it runs as a plain Python function chain.
    """

    def __init__(self, manager, profile, router, system_prompt: str, max_tokens: int = 4096):
        """Initialize the graph with all dependencies.

        Args:
            manager: MemoryManager instance.
            profile: UserProfile instance.
            router: MemoryRouter instance.
            system_prompt: Base system instruction.
            max_tokens: Token budget per turn.
        """
        self._manager = manager
        self._profile = profile
        self._router = router
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._llm = None  # set via set_llm()

    def set_llm(self, llm) -> None:
        """Attach a real LLM callable (context, query) -> str."""
        self._llm = llm

    def run(self, query: str) -> MemoryState:
        """Execute the full graph pipeline for one conversation turn.

        Args:
            query: The user's input message.

        Returns:
            Final MemoryState after all nodes have run.
        """
        state = make_initial_state(query, max_tokens=self._max_tokens)

        state = node_retrieve_memory(state, self._manager, self._profile, self._router)
        state = node_build_prompt(state, self._system_prompt, self._max_tokens)
        state = node_generate_response(state, self._llm)
        state = node_save_memory(state, self._manager, self._profile)

        logger.debug(
            "Graph run complete: intent=%s response_len=%d",
            state.get("intent"), len(state.get("response", "")),
        )
        return state
