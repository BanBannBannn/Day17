"""Short-term memory: in-session conversation buffer using langchain_core messages."""

import logging
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """In-session conversation buffer memory.

    Stores the current conversation turn-by-turn in memory without
    persistence. Cleared on session end. Implemented as a plain list of
    langchain_core messages — no deprecated langchain.memory dependency.
    """

    def __init__(self, max_messages: int = 20, session_id: Optional[str] = None):
        """Initialize short-term memory buffer.

        Args:
            max_messages: Maximum number of interaction pairs (human+AI) to retain.
            session_id: Optional identifier for the current session.
        """
        self.max_messages = max_messages
        self.session_id = session_id or "default"
        self._messages: List[BaseMessage] = []
        self._message_count = 0  # counts interaction pairs, not individual messages
        logger.info("ShortTermMemory initialized (session=%s)", self.session_id)

    def add_interaction(self, human_input: str, ai_output: str) -> None:
        """Save a human-AI interaction pair to the buffer.

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

        self._messages.append(HumanMessage(content=human_input))
        self._messages.append(AIMessage(content=ai_output))
        self._message_count += 1
        logger.debug("Added interaction #%d to short-term buffer", self._message_count)

        if self._message_count > self.max_messages:
            self._trim_oldest()

    def get_history(self) -> List[BaseMessage]:
        """Return the full conversation history as message objects.

        Returns:
            List of LangChain BaseMessage instances (alternating Human/AI).
        """
        return list(self._messages)

    def get_history_as_text(self) -> str:
        """Return conversation history as a plain text string.

        Returns:
            Formatted conversation string with 'Human:' and 'AI:' prefixes.
        """
        lines: List[str] = []
        for msg in self._messages:
            prefix = "Human" if isinstance(msg, HumanMessage) else "AI"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def get_recent(self, n: int = 3) -> List[BaseMessage]:
        """Return the N most recent messages.

        Args:
            n: Number of individual messages (not pairs) to retrieve.

        Returns:
            List of the last N messages.
        """
        return self._messages[-n:] if len(self._messages) >= n else list(self._messages)

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        self._messages = []
        self._message_count = 0
        logger.info("ShortTermMemory cleared (session=%s)", self.session_id)

    def _trim_oldest(self) -> None:
        """Remove oldest interaction pairs when buffer exceeds max_messages."""
        # Each pair = 2 messages; keep the most recent max_messages pairs
        keep_msgs = self.max_messages * 2
        self._messages = self._messages[-keep_msgs:]
        self._message_count = len(self._messages) // 2
        logger.debug("Trimmed short-term buffer to %d pairs", self._message_count)

    def to_dict(self) -> Dict:
        """Serialize memory state to a dictionary.

        Returns:
            Dictionary with session metadata and message history.
        """
        return {
            "session_id": self.session_id,
            "message_count": self._message_count,
            "messages": [
                {
                    "role": "human" if isinstance(m, HumanMessage) else "ai",
                    "content": m.content,
                }
                for m in self._messages
            ],
        }

    @property
    def message_count(self) -> int:
        """Current number of interaction pairs stored."""
        return self._message_count
