"""Mem0 Proof-of-Concept Memory Backend for the AI HR Policy Assistant.

Provides:
- Isolated Mem0 integration conforming to BaseMemoryManager.
- Deterministic zero-external-dependency local embedding integration (Mem0TfidfEmbedder).
- In-memory / local Qdrant vector storage (zero external container / service requirement).
- Strict employee isolation (scoped by user_id) and session scoping (run_id).
- Strict policy precedence: conversation memories are untrusted context, approved CompanyPolicy always wins.
- Delimiter injection sanitization on retrieved memories.
- Safe, graceful fallback if Mem0 operations fail.
"""

import logging
import os
import uuid
from typing import Any

from app.models import ChatMessage
from app.services.memory_service import (
    BaseEmbeddingService,
    BaseMemoryManager,
    MemoryBudgetConfig,
    RetrievedMemory,
    TfidfEmbeddingService,
)

logger = logging.getLogger(__name__)

# Ensure Mem0 telemetry is disabled by default to avoid external network calls
os.environ.setdefault("MEM0_TELEMETRY", "false")

try:
    from mem0 import Memory
    from mem0.configs.base import (
        EmbedderConfig,
        LlmConfig,
        MemoryConfig,
        VectorStoreConfig,
    )
    from mem0.embeddings.base import EmbeddingBase
    from mem0.utils.factory import EmbedderFactory

    MEM0_AVAILABLE = True
except ImportError:
    Memory = None  # type: ignore[assignment,misc]
    EmbeddingBase = object  # type: ignore[assignment,misc]
    MEM0_AVAILABLE = False


class Mem0TfidfEmbedder(EmbeddingBase):
    """Local, deterministic TF-IDF embedder for Mem0 using zero external dependencies."""

    def __init__(self, config: Any = None):
        if MEM0_AVAILABLE:
            super().__init__(config)
        dim = 256
        if config and hasattr(config, "model_dims") and config.model_dims:
            dim = config.model_dims
        self._service = TfidfEmbeddingService(dimension=dim)
        self.dimension = dim

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self._service.get_embedding(str(text or ""))

    def get_embedding(self, text: str) -> list[float]:
        """Compatibility alias with BaseEmbeddingService."""
        return self._service.get_embedding(str(text or ""))

    def embed_batch(self, texts: list[str], memory_action: str = "add") -> list[list[float]]:
        return [self.embed(t, memory_action) for t in texts]


# Register embedder into Mem0's factory if Mem0 is present
if MEM0_AVAILABLE:
    try:
        EmbedderFactory.provider_to_class["langchain"] = "app.services.mem0_service.Mem0TfidfEmbedder"
    except (AttributeError, KeyError) as _reg_exc:
        logger.debug("Could not register Mem0TfidfEmbedder in factory: %s", _reg_exc)


class Mem0MemoryManager(BaseMemoryManager):
    """Mem0-backed memory manager implementing the BaseMemoryManager interface.

    Allows the Policy Assistant to optionally use Mem0 for storing and retrieving
    conversation memories while maintaining strict employee isolation, session scoping,
    policy precedence, and delimiter injection protection.
    """

    def __init__(
        self,
        config: MemoryBudgetConfig | None = None,
        embedding_service: BaseEmbeddingService | None = None,
        mem0_client: Any | None = None,
        collection_name: str | None = None,
        infer: bool = False,
    ):
        self.config = config or MemoryBudgetConfig.from_env()
        self.embedding_service = embedding_service or TfidfEmbeddingService()
        self.infer = infer
        self.collection_name = collection_name or f"mem0_policy_{uuid.uuid4().hex[:8]}"

        if mem0_client is not None:
            self._client = mem0_client
        else:
            self._client = self._init_mem0_client()

    def _init_mem0_client(self) -> Any:
        """Initializes an isolated local Mem0 Memory instance with Qdrant in-memory."""
        if not MEM0_AVAILABLE:
            logger.warning("Mem0 package is not installed. Mem0MemoryManager will run in fallback mode.")
            return None

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

        try:
            mem_config = MemoryConfig(
                vector_store=VectorStoreConfig(
                    provider="qdrant",
                    config={
                        "collection_name": self.collection_name,
                        "embedding_model_dims": 256,
                        "path": ":memory:",
                        "on_disk": False,
                    },
                ),
                llm=LlmConfig(
                    provider="groq",
                    config={
                        "model": groq_model,
                        "api_key": groq_api_key,
                    },
                ),
                embedder=EmbedderConfig(
                    provider="langchain",
                    config={},
                ),
                history_db_path=":memory:",
            )
            return Memory(mem_config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize Mem0 Memory instance: %s. Using fallback mode.", exc)
            return None

    def partition_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Partitions session messages into recent_messages and older_messages based on budget."""
        if not messages:
            return [], []

        recent_reversed: list[ChatMessage] = []
        accumulated_chars = 0

        for msg in reversed(messages):
            msg_len = len(msg.content or "") + 20
            if accumulated_chars + msg_len <= self.config.recent_messages_char_budget:
                recent_reversed.append(msg)
                accumulated_chars += msg_len
            else:
                if not recent_reversed:
                    recent_reversed.append(msg)
                break

        recent_ids = {m.id for m in recent_reversed}
        recent_messages = [m for m in messages if m.id in recent_ids]
        older_messages = [m for m in messages if m.id not in recent_ids]

        return recent_messages, older_messages

    def add_turn(
        self,
        employee_id: str,
        session_id: str,
        user_content: str,
        assistant_content: str,
        infer: bool | None = None,
    ) -> bool:
        """Stores a completed turn into Mem0 scoped to employee_id and session_id."""
        if not self._client or not employee_id:
            return False

        use_infer = self.infer if infer is None else infer
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        try:
            self._client.add(
                messages=messages,
                user_id=employee_id,
                run_id=session_id,
                infer=use_infer,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mem0 add_turn failed for employee %s, session %s: %s", employee_id, session_id, exc)
            return False

    def on_chat_turn_recorded(
        self,
        session_id: str,
        employee_id: str | None,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Callback triggered after a chat turn is committed to MySQL database."""
        if employee_id:
            self.add_turn(
                employee_id=employee_id,
                session_id=session_id,
                user_content=user_content,
                assistant_content=assistant_content,
            )

    def search_memories(
        self,
        question: str,
        employee_id: str,
        session_id: str | None = None,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedMemory]:
        """Searches Mem0 memories with strict employee isolation and session scoping."""
        if not self._client or not question or not employee_id:
            return []

        max_k = limit or self.config.max_retrieved_memories
        sim_threshold = threshold if threshold is not None else self.config.similarity_threshold

        filters: dict[str, Any] = {"user_id": employee_id}
        if session_id:
            filters["run_id"] = session_id

        try:
            search_response = self._client.search(
                query=question,
                filters=filters,
                top_k=max_k * 2,
                threshold=0.05,
            )
            raw_results = search_response.get("results", []) if isinstance(search_response, dict) else []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mem0 search failed for employee %s: %s", employee_id, exc)
            return []

        retrieved: list[RetrievedMemory] = []
        accumulated_chars = 0

        for idx, item in enumerate(raw_results, start=1):
            if len(retrieved) >= max_k:
                break
            score = float(item.get("score", 0.0))
            if score < sim_threshold:
                continue

            content = item.get("memory", "")
            if not content:
                continue

            content_len = len(content)
            if accumulated_chars + content_len <= self.config.retrieved_memory_char_budget:
                role = item.get("role", "historical")
                mem_id = item.get("id") or idx
                retrieved.append(
                    RetrievedMemory(
                        message_id=hash(str(mem_id)) % 100000,
                        role=role,
                        content=content,
                        similarity_score=round(score, 4),
                    )
                )
                accumulated_chars += content_len

        return retrieved

    def retrieve_semantic_memories(
        self,
        question: str,
        older_messages: list[ChatMessage],
        employee_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieves semantic memories via Mem0.

        Falls back gracefully if employee_id is unavailable or Mem0 search encounters errors.
        """
        if not employee_id and older_messages and hasattr(older_messages[0], "session_id"):
            session_id = session_id or str(older_messages[0].session_id)

        if not employee_id:
            # Without employee_id, isolation cannot be guaranteed; return safe empty list
            return []

        return self.search_memories(
            question=question,
            employee_id=employee_id,
            session_id=session_id,
        )

    @staticmethod
    def format_retrieved_memories(memories: list[RetrievedMemory]) -> str:
        """Formats retrieved semantic memories into a safe, bounded, delimiter-sanitized context block.

        Memories are explicitly designated as UNTRUSTED dialogue history.
        Approved policies in <COMPANY_POLICIES> strictly take precedence.
        """
        if not memories:
            return ""

        lines: list[str] = []
        for m in memories:
            role_label = "Employee" if m.role == "user" else "Policy Assistant"
            # Strict delimiter sanitization
            clean_content = (
                m.content.replace("</RELEVANT_CONVERSATION_MEMORIES>", "[ESCAPED_TAG]")
                .replace("<RELEVANT_CONVERSATION_MEMORIES>", "[ESCAPED_TAG]")
                .replace("</COMPANY_POLICIES>", "[ESCAPED_TAG]")
                .replace("<COMPANY_POLICIES>", "[ESCAPED_TAG]")
                .replace("</EMPLOYEE_FACTS>", "[ESCAPED_TAG]")
                .replace("<EMPLOYEE_FACTS>", "[ESCAPED_TAG]")
                .strip()
            )
            lines.append(f"[Historical Turn #{m.message_id} - {role_label}]: {clean_content}")

        body = "\n".join(lines)
        return f"<RELEVANT_CONVERSATION_MEMORIES>\n{body}\n</RELEVANT_CONVERSATION_MEMORIES>"
