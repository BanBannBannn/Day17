"""Context Window Manager: token-aware trimming and priority-based eviction."""

import logging
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
_TRIM_THRESHOLD = float(os.getenv("TRIM_THRESHOLD", "0.85"))
_EVICTION_BATCH = int(os.getenv("EVICTION_BATCH_SIZE", "5"))


class ContextPriority(IntEnum):
    """4-level priority hierarchy for context eviction.

    Higher numeric value = higher priority = evicted LAST.
    """

    RAW_HISTORY = 1        # Oldest, least critical — evicted first
    SEMANTIC_CONTEXT = 2   # Retrieved vector fragments
    RECENT_USER_INTENT = 3 # Recent turns directly relevant to the query
    SYSTEM_PROMPT = 4      # System instructions — never evicted


@dataclass
class ContextBlock:
    """A named section of context with priority and token count."""

    name: str
    content: str
    priority: ContextPriority
    token_count: int = field(default=0)

    def __post_init__(self):
        if self.token_count == 0 and self.content:
            self.token_count = _estimate_tokens(self.content)


def _estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken or a simple word-based heuristic.

    Args:
        text: Input text to count tokens for.

    Returns:
        Estimated token count.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Rough heuristic: ~1.3 tokens per word
        return max(1, int(len(text.split()) * 1.3))
    except Exception as exc:
        logger.warning("tiktoken encoding failed: %s; using word estimate", exc)
        return max(1, int(len(text.split()) * 1.3))


class ContextWindowManager:
    """Manages the LLM context window budget with priority-based eviction.

    Assembles context blocks from multiple memory sources, auto-trims when
    the combined token count exceeds the configured threshold, and evicts
    low-priority blocks first.

    Priority eviction order (lowest first):
      RAW_HISTORY → SEMANTIC_CONTEXT → RECENT_USER_INTENT → SYSTEM_PROMPT
    """

    def __init__(
        self,
        max_tokens: int = _MAX_TOKENS,
        trim_threshold: float = _TRIM_THRESHOLD,
        eviction_batch: int = _EVICTION_BATCH,
    ):
        """Initialize the context window manager.

        Args:
            max_tokens: Hard token limit for the assembled context.
            trim_threshold: Fraction of max_tokens that triggers trimming
                (e.g., 0.85 means trim when > 85% full).
            eviction_batch: Number of blocks to evict per trimming pass.

        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 < trim_threshold <= 1.0:
            raise ValueError("trim_threshold must be between 0 and 1")
        if eviction_batch <= 0:
            raise ValueError("eviction_batch must be positive")

        self.max_tokens = max_tokens
        self.trim_threshold = trim_threshold
        self.eviction_batch = eviction_batch
        self._blocks: List[ContextBlock] = []

        logger.info(
            "ContextWindowManager: max_tokens=%d threshold=%.0f%%",
            max_tokens, trim_threshold * 100,
        )

    def add_block(
        self,
        name: str,
        content: str,
        priority: ContextPriority,
    ) -> None:
        """Add a named content block to the context window.

        Automatically triggers trimming if the threshold is exceeded.

        Args:
            name: Unique label for this block (e.g., 'system_prompt').
            content: The text content of this block.
            priority: ContextPriority level for eviction ordering.
        """
        if not content or not content.strip():
            logger.debug("Skipping empty block '%s'", name)
            return

        # Replace existing block with same name
        self._blocks = [b for b in self._blocks if b.name != name]
        block = ContextBlock(name=name, content=content, priority=priority)
        self._blocks.append(block)

        logger.debug(
            "Added block '%s' (%d tokens, priority=%s)",
            name, block.token_count, priority.name,
        )

        if self.usage_ratio >= self.trim_threshold:
            self._trim()

    def build_prompt(self, separator: str = "\n\n") -> str:
        """Assemble all context blocks into a single ordered string.

        Blocks are sorted by priority (highest first) so that the most
        important context appears at the top of the prompt.

        Args:
            separator: String inserted between blocks.

        Returns:
            The assembled prompt context string.
        """
        sorted_blocks = sorted(self._blocks, key=lambda b: b.priority, reverse=True)
        return separator.join(b.content for b in sorted_blocks if b.content)

    def get_budget_report(self) -> Dict:
        """Return a breakdown of token usage per block and overall budget.

        Returns:
            Dict with 'total_tokens', 'max_tokens', 'usage_ratio',
            'remaining_tokens', and 'blocks' list.
        """
        total = self.total_tokens
        return {
            "total_tokens": total,
            "max_tokens": self.max_tokens,
            "usage_ratio": round(self.usage_ratio, 3),
            "remaining_tokens": max(0, self.max_tokens - total),
            "blocks": [
                {
                    "name": b.name,
                    "priority": b.priority.name,
                    "tokens": b.token_count,
                    "fraction": round(b.token_count / self.max_tokens, 3),
                }
                for b in sorted(self._blocks, key=lambda x: x.priority, reverse=True)
            ],
        }

    def clear(self) -> None:
        """Remove all context blocks."""
        self._blocks = []
        logger.debug("ContextWindowManager cleared")

    def remove_block(self, name: str) -> bool:
        """Remove a block by name.

        Args:
            name: Name of the block to remove.

        Returns:
            True if a block was removed, False if not found.
        """
        before = len(self._blocks)
        self._blocks = [b for b in self._blocks if b.name != name]
        removed = len(self._blocks) < before
        if removed:
            logger.debug("Removed block '%s'", name)
        return removed

    def fits_in_budget(self, text: str, reserved: int = 512) -> bool:
        """Check if adding text would stay within the token budget.

        Args:
            text: Candidate text to add.
            reserved: Token headroom to keep for LLM generation.

        Returns:
            True if adding text leaves at least 'reserved' tokens free.
        """
        candidate_tokens = _estimate_tokens(text)
        return (self.total_tokens + candidate_tokens) <= (self.max_tokens - reserved)

    def _trim(self) -> None:
        """Evict low-priority blocks until usage drops below threshold."""
        target = self.max_tokens * self.trim_threshold
        evicted = 0

        # Sort ascending by priority (evict lowest priority first),
        # break ties by token count (evict larger blocks first within same priority)
        candidates = sorted(
            [b for b in self._blocks if b.priority != ContextPriority.SYSTEM_PROMPT],
            key=lambda b: (b.priority, -b.token_count),
        )

        for block in candidates:
            if self.total_tokens <= target:
                break
            if evicted >= self.eviction_batch:
                break
            self._blocks.remove(block)
            evicted += 1
            logger.info(
                "Evicted block '%s' (%d tokens, priority=%s)",
                block.name, block.token_count, block.priority.name,
            )

        logger.info(
            "Context trim complete: evicted %d blocks, usage now %d/%d tokens",
            evicted, self.total_tokens, self.max_tokens,
        )

    @property
    def total_tokens(self) -> int:
        """Sum of token counts across all current blocks."""
        return sum(b.token_count for b in self._blocks)

    @property
    def usage_ratio(self) -> float:
        """Fraction of the token budget currently consumed (0.0 to 1.0+)."""
        return self.total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0


def build_context_from_memory(
    memory_sections: Dict[str, str],
    system_prompt: str = "",
    query: str = "",
    max_tokens: int = _MAX_TOKENS,
) -> Tuple[str, Dict]:
    """High-level helper to build a managed context from memory sections.

    Args:
        memory_sections: Dict from MemoryManager.build_context() with keys
            'short_term', 'long_term', 'episodic', 'semantic'.
        system_prompt: The system instruction string.
        query: Current user query (used as RECENT_USER_INTENT block).
        max_tokens: Token budget for the context window.

    Returns:
        Tuple of (assembled_prompt_string, budget_report_dict).
    """
    manager = ContextWindowManager(max_tokens=max_tokens)

    if system_prompt:
        manager.add_block("system_prompt", system_prompt, ContextPriority.SYSTEM_PROMPT)

    if query:
        manager.add_block("current_query", f"Current Query: {query}", ContextPriority.RECENT_USER_INTENT)

    short_term = memory_sections.get("short_term", "")
    if short_term:
        manager.add_block("recent_conversation", short_term, ContextPriority.RECENT_USER_INTENT)

    semantic = memory_sections.get("semantic", "")
    if semantic:
        manager.add_block("semantic_context", semantic, ContextPriority.SEMANTIC_CONTEXT)

    episodic = memory_sections.get("episodic", "")
    if episodic:
        manager.add_block("episodic_context", episodic, ContextPriority.SEMANTIC_CONTEXT)

    long_term = memory_sections.get("long_term", "")
    if long_term:
        manager.add_block("long_term_history", long_term, ContextPriority.RAW_HISTORY)

    return manager.build_prompt(), manager.get_budget_report()
