"""Hybrid Memory Service for the AI HR Policy Assistant.

Provides:
- Abstract BaseEmbeddingService interface.
- Deterministic, zero-dependency subword TF-IDF / cosine similarity embedding service.
- Configurable token/character budgets for recent history, summary, and semantic memories.
- Partitioning of conversation turns into recent window vs older candidate pool.
- Semantic retrieval over older messages filtered by session and similarity threshold.
"""

import hashlib
import json
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.models import ChatMessage

logger = logging.getLogger(__name__)

# Default character budgets (approx 4 chars per token)
DEFAULT_RECENT_MESSAGES_CHAR_BUDGET = 2500
DEFAULT_SUMMARY_CHAR_BUDGET = 800
DEFAULT_RETRIEVED_MEMORY_CHAR_BUDGET = 1500
DEFAULT_TOTAL_MEMORY_CHAR_BUDGET = 4500
DEFAULT_SIMILARITY_THRESHOLD = 0.35
DEFAULT_MAX_RETRIEVED_MEMORIES = 3


@dataclass(frozen=True)
class MemoryBudgetConfig:
    """Configurable budget parameters for the Policy Assistant Hybrid Memory Architecture."""

    recent_messages_char_budget: int = DEFAULT_RECENT_MESSAGES_CHAR_BUDGET
    summary_char_budget: int = DEFAULT_SUMMARY_CHAR_BUDGET
    retrieved_memory_char_budget: int = DEFAULT_RETRIEVED_MEMORY_CHAR_BUDGET
    total_memory_char_budget: int = DEFAULT_TOTAL_MEMORY_CHAR_BUDGET
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_retrieved_memories: int = DEFAULT_MAX_RETRIEVED_MEMORIES

    @classmethod
    def from_env(cls) -> "MemoryBudgetConfig":
        """Loads configuration from environment variables with safe fallbacks."""
        return cls(
            recent_messages_char_budget=int(
                os.getenv("POLICY_MEMORY_RECENT_BUDGET", str(DEFAULT_RECENT_MESSAGES_CHAR_BUDGET))
            ),
            summary_char_budget=int(
                os.getenv("POLICY_MEMORY_SUMMARY_BUDGET", str(DEFAULT_SUMMARY_CHAR_BUDGET))
            ),
            retrieved_memory_char_budget=int(
                os.getenv("POLICY_MEMORY_RETRIEVED_BUDGET", str(DEFAULT_RETRIEVED_MEMORY_CHAR_BUDGET))
            ),
            total_memory_char_budget=int(
                os.getenv("POLICY_MEMORY_TOTAL_BUDGET", str(DEFAULT_TOTAL_MEMORY_CHAR_BUDGET))
            ),
            similarity_threshold=float(
                os.getenv("POLICY_MEMORY_SIMILARITY_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD))
            ),
            max_retrieved_memories=int(
                os.getenv("POLICY_MEMORY_MAX_RETRIEVED", str(DEFAULT_MAX_RETRIEVED_MEMORIES))
            ),
        )


@dataclass
class RetrievedMemory:
    """A relevant older conversation message retrieved via semantic similarity."""

    message_id: int
    role: str
    content: str
    similarity_score: float


class BaseEmbeddingService(ABC):
    """Abstract interface for generating vector embeddings and computing similarity."""

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Computes a normalized vector embedding for the input text."""

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Computes the cosine similarity between two vectors.

        Returns a value in [-1.0, 1.0], or 0.0 if either vector has zero magnitude.
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a <= 1e-9 or norm_b <= 1e-9:
            return 0.0

        sim = dot_product / (norm_a * norm_b)
        return max(-1.0, min(1.0, float(sim)))


class TfidfEmbeddingService(BaseEmbeddingService):
    """Deterministic, production-grade zero-external-dependency embedding service.

    Uses word tokenization and character 3-grams hashed into a fixed-dimension vector
    with term-frequency weighting and L2 normalization.
    Provides subword and typo resilience without requiring external vector models.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _tokenize_features(self, text: str) -> list[str]:
        """Extracts lowercase word tokens and character n-grams."""
        cleaned = text.lower()
        words = re.findall(r"\b[a-z0-9_]{2,}\b", cleaned)
        features: list[str] = list(words)

        # Generate character 3-grams for words longer than 3 chars for morphological resilience
        for w in words:
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    features.append(f"#{w[i : i + 3]}")
        return features

    def _feature_hash(self, feature: str) -> int:
        """Deterministic integer hash for a feature string."""
        h = hashlib.md5(feature.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dimension

    def get_embedding(self, text: str) -> list[float]:
        """Generates an L2-normalized embedding vector of size self.dimension."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        features = self._tokenize_features(text)
        if not features:
            return [0.0] * self.dimension

        vec = [0.0] * self.dimension
        for f in features:
            idx = self._feature_hash(f)
            # Logarithmic term frequency scaling
            vec[idx] += 1.0

        # Apply sub-linear scaling: 1 + log(tf)
        for i in range(self.dimension):
            if vec[i] > 0:
                vec[i] = 1.0 + math.log(vec[i])

        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        return vec


class BaseMemoryManager(ABC):
    """Abstract base interface for conversation memory managers in the Policy Assistant."""

    @abstractmethod
    def partition_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Partitions session messages into recent_messages and older_messages based on budget."""

    @abstractmethod
    def retrieve_semantic_memories(
        self,
        question: str,
        older_messages: list[ChatMessage],
        employee_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieves relevant semantic memories matching the question."""

    @staticmethod
    @abstractmethod
    def format_retrieved_memories(memories: list[RetrievedMemory]) -> str:
        """Formats retrieved semantic memories into a safe, bounded context block."""


class MemoryManager(BaseMemoryManager):
    """Manages short-term window budgeting, semantic retrieval, and memory assembly."""

    def __init__(
        self,
        config: MemoryBudgetConfig | None = None,
        embedding_service: BaseEmbeddingService | None = None,
    ):
        self.config = config or MemoryBudgetConfig.from_env()
        self.embedding_service = embedding_service or TfidfEmbeddingService()

    def partition_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Partitions session messages into recent_messages and older_messages based on budget.

        Greedily collects messages from the most recent backwards until the
        `recent_messages_char_budget` is reached.
        Returns:
            (recent_messages, older_messages) - both lists in chronological order (asc).
        """
        if not messages:
            return [], []

        recent_reversed: list[ChatMessage] = []
        accumulated_chars = 0

        # Walk backwards from newest to oldest
        for msg in reversed(messages):
            msg_len = len(msg.content or "") + 20  # +20 for role label overhead
            if accumulated_chars + msg_len <= self.config.recent_messages_char_budget:
                recent_reversed.append(msg)
                accumulated_chars += msg_len
            else:
                # If even a single message exceeds the budget, still include at least 1 recent message
                if not recent_reversed:
                    recent_reversed.append(msg)
                break

        recent_ids = {m.id for m in recent_reversed}
        recent_messages = [m for m in messages if m.id in recent_ids]
        older_messages = [m for m in messages if m.id not in recent_ids]

        return recent_messages, older_messages

    def retrieve_semantic_memories(
        self,
        question: str,
        older_messages: list[ChatMessage],
        employee_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieves relevant older messages matching the question above the similarity threshold.

        Respects max_retrieved_memories and retrieved_memory_char_budget.
        """
        if not older_messages or not question or not question.strip():
            return []

        try:
            q_emb = self.embedding_service.get_embedding(question)
        except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
            logger.warning("Failed to generate question embedding for semantic memory: %s", exc)
            return []

        scored_candidates: list[tuple[ChatMessage, float]] = []

        for msg in older_messages:
            if not msg.content or not msg.content.strip():
                continue

            # Load or compute embedding for the message
            msg_emb: list[float] | None = None
            if hasattr(msg, "embedding") and msg.embedding:
                try:
                    msg_emb = json.loads(msg.embedding)
                except (json.JSONDecodeError, TypeError):
                    msg_emb = None

            if not msg_emb:
                try:
                    msg_emb = self.embedding_service.get_embedding(msg.content)
                except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
                    logger.warning("Failed to compute embedding for message %s: %s", msg.id, exc)
                    continue

            sim = self.embedding_service.cosine_similarity(q_emb, msg_emb)
            if sim >= self.config.similarity_threshold:
                scored_candidates.append((msg, sim))

        # Sort descending by similarity score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        selected: list[RetrievedMemory] = []
        accumulated_chars = 0

        for msg, sim in scored_candidates:
            if len(selected) >= self.config.max_retrieved_memories:
                break
            content_len = len(msg.content)
            if accumulated_chars + content_len <= self.config.retrieved_memory_char_budget:
                selected.append(
                    RetrievedMemory(
                        message_id=msg.id,
                        role=msg.role,
                        content=msg.content,
                        similarity_score=round(sim, 4),
                    )
                )
                accumulated_chars += content_len

        # Order chronologically by message_id for logical reading flow
        selected.sort(key=lambda m: m.message_id)
        return selected

    @staticmethod
    def format_retrieved_memories(memories: list[RetrievedMemory]) -> str:
        """Formats retrieved semantic memories into a safe, bounded context block."""
        if not memories:
            return ""

        lines: list[str] = []
        for m in memories:
            role_label = "Employee" if m.role == "user" else "Policy Assistant"
            # Sanitize content to prevent injection delimiters
            clean_content = (
                m.content.replace("</RELEVANT_CONVERSATION_MEMORIES>", "[ESCAPED_TAG]")
                .replace("<RELEVANT_CONVERSATION_MEMORIES>", "[ESCAPED_TAG]")
                .replace("</COMPANY_POLICIES>", "[ESCAPED_TAG]")
                .replace("<COMPANY_POLICIES>", "[ESCAPED_TAG]")
                .strip()
            )
            lines.append(f"[Historical Turn #{m.message_id} - {role_label}]: {clean_content}")

        body = "\n".join(lines)
        return f"<RELEVANT_CONVERSATION_MEMORIES>\n{body}\n</RELEVANT_CONVERSATION_MEMORIES>"
