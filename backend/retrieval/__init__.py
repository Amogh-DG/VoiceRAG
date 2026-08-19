"""
backend/retrieval/__init__.py
------------------------------
Exports vector database and retriever implementations and factory functions.
"""

from typing import Any, Dict, Optional
from backend.common.interfaces import IVectorDB, IRetriever
from backend.retrieval.in_memory import InMemoryVectorDB
from backend.retrieval.retriever import VectorRetriever
from backend.embeddings import get_embedder

_VECTOR_DB_REGISTRY: Dict[str, type] = {
    "in_memory": InMemoryVectorDB,
    "memory": InMemoryVectorDB,
}

def get_vector_db(name: str = "in_memory", **kwargs: Any) -> IVectorDB:
    """Factory to retrieve a vector database instance by name."""
    cls = _VECTOR_DB_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown vector DB '{name}'. Available: {list(_VECTOR_DB_REGISTRY.keys())}"
        )
    return cls(**kwargs)

def get_retriever(
    embedder_name: str = "mock",
    db_name: str = "in_memory",
    embedder_kwargs: Optional[Dict[str, Any]] = None,
    db_kwargs: Optional[Dict[str, Any]] = None,
) -> IRetriever:
    """Factory to construct a VectorRetriever with configured embedder and vector DB."""
    embedder = get_embedder(embedder_name, **(embedder_kwargs or {}))
    vector_db = get_vector_db(db_name, **(db_kwargs or {}))
    return VectorRetriever(embedder=embedder, vector_db=vector_db)

__all__ = [
    "IVectorDB",
    "IRetriever",
    "InMemoryVectorDB",
    "VectorRetriever",
    "get_vector_db",
    "get_retriever",
]
