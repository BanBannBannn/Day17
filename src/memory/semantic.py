"""Semantic memory using ChromaDB for vector-based RAG retrieval."""

import logging
import os
import time
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "agent_memory")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed; SemanticMemory will use in-memory fallback")

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class SemanticMemory:
    """Vector-store memory for semantic similarity search (RAG).

    Uses ChromaDB with OpenAI or SentenceTransformer embeddings to store
    and retrieve memory fragments by semantic similarity.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_openai_embeddings: bool = True,
    ):
        """Initialize semantic memory with ChromaDB.

        Args:
            collection_name: ChromaDB collection name.
            persist_dir: Directory path for ChromaDB persistence.
            embedding_model: Model name for embeddings.
            use_openai_embeddings: Use OpenAI embeddings if True, else SentenceTransformer.
        """
        self._collection_name = collection_name or _COLLECTION
        self._persist_dir = persist_dir or _PERSIST_DIR
        self._embedding_fn = self._build_embedding_function(
            embedding_model, use_openai_embeddings
        )
        self._collection = None
        self._fallback_store: List[Dict] = []

        if CHROMA_AVAILABLE:
            self._init_chroma()
        else:
            logger.info("SemanticMemory using simple cosine fallback (no ChromaDB)")

    def _build_embedding_function(self, model_name: Optional[str], use_openai: bool):
        """Build the appropriate embedding callable."""
        api_key = os.getenv("OPENAI_API_KEY", "")

        if use_openai and OPENAI_EMBEDDINGS_AVAILABLE and api_key:
            model = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            logger.info("SemanticMemory: using OpenAI embeddings (%s)", model)
            return OpenAIEmbeddings(model=model, openai_api_key=api_key)

        if ST_AVAILABLE:
            model = model_name or "all-MiniLM-L6-v2"
            logger.info("SemanticMemory: using SentenceTransformer (%s)", model)
            return SentenceTransformer(model)

        logger.warning("No embedding model available; semantic search will be keyword-only")
        return None

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=self._persist_dir)

            if self._embedding_fn is None:
                self._collection = client.get_or_create_collection(
                    name=self._collection_name
                )
            elif isinstance(self._embedding_fn, SentenceTransformer if ST_AVAILABLE else type(None)):
                st_model = self._embedding_fn

                def st_embed(texts):
                    return st_model.encode(texts).tolist()

                ef = chromadb.EmbeddingFunction if hasattr(chromadb, "EmbeddingFunction") else None
                self._collection = client.get_or_create_collection(
                    name=self._collection_name,
                    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    ),
                )
            else:
                # OpenAI embeddings via custom function
                openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                )
                self._collection = client.get_or_create_collection(
                    name=self._collection_name,
                    embedding_function=openai_ef,
                )
            logger.info(
                "ChromaDB collection '%s' ready at %s",
                self._collection_name, self._persist_dir,
            )
        except Exception as exc:
            logger.error("ChromaDB init failed: %s; using fallback", exc)
            self._collection = None

    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Store a text fragment in the semantic memory store.

        Args:
            text: The text content to embed and store.
            metadata: Optional key-value metadata attached to this fragment.
            doc_id: Explicit ID; auto-generated if not provided.

        Returns:
            The document ID.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        doc_id = doc_id or str(uuid.uuid4())
        meta = {"timestamp": time.time(), **(metadata or {})}

        if self._collection is not None:
            try:
                self._collection.add(
                    documents=[text],
                    metadatas=[meta],
                    ids=[doc_id],
                )
                logger.debug("SemanticMemory: stored doc id=%s", doc_id)
                return doc_id
            except Exception as exc:
                logger.error("ChromaDB add failed: %s; using fallback", exc)

        # Keyword-only fallback
        self._fallback_store.append({"id": doc_id, "text": text, "metadata": meta})
        return doc_id

    def search(self, query: str, top_k: int = 5, where: Optional[Dict] = None) -> List[Dict]:
        """Retrieve top-k semantically similar documents for a query.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            where: Optional ChromaDB metadata filter dict.

        Returns:
            List of result dicts with 'id', 'text', 'metadata', 'distance' keys.
        """
        if not query or not query.strip():
            return []

        if self._collection is not None:
            try:
                kwargs = {"query_texts": [query], "n_results": min(top_k, self._collection.count())}
                if where:
                    kwargs["where"] = where
                if kwargs["n_results"] == 0:
                    return []

                results = self._collection.query(**kwargs)
                output = []
                for i, doc in enumerate(results["documents"][0]):
                    output.append({
                        "id": results["ids"][0][i],
                        "text": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    })
                return output
            except Exception as exc:
                logger.error("ChromaDB search failed: %s; using fallback", exc)

        # Keyword fallback: simple substring matching
        query_lower = query.lower()
        matches = [
            {"id": d["id"], "text": d["text"], "metadata": d["metadata"], "distance": 1.0}
            for d in self._fallback_store
            if query_lower in d["text"].lower()
        ]
        return matches[:top_k]

    def format_as_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve and format semantic matches as an LLM context block.

        Args:
            query: The search query.
            top_k: Number of results to include.

        Returns:
            Formatted text block for injection into the prompt.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return "No semantic memory found."

        lines = ["[Semantic Memory - Relevant Context]"]
        for r in results:
            dist_str = f"{r['distance']:.3f}" if r["distance"] != 1.0 else "N/A"
            lines.append(f"- (distance={dist_str}) {r['text']}")
        return "\n".join(lines)

    def delete(self, doc_id: str) -> None:
        """Remove a document by ID.

        Args:
            doc_id: The document ID to delete.
        """
        if self._collection is not None:
            try:
                self._collection.delete(ids=[doc_id])
                return
            except Exception as exc:
                logger.error("ChromaDB delete failed: %s", exc)

        self._fallback_store = [d for d in self._fallback_store if d["id"] != doc_id]

    def clear(self) -> None:
        """Delete all documents from the collection."""
        if self._collection is not None:
            try:
                self._collection.delete(ids=self._collection.get()["ids"])
            except Exception as exc:
                logger.error("ChromaDB clear failed: %s", exc)
        self._fallback_store = []
        logger.info("SemanticMemory cleared")

    @property
    def document_count(self) -> int:
        """Number of documents stored."""
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._fallback_store)
