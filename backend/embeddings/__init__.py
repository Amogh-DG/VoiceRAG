"""
backend/embeddings/__init__.py
------------------------------
Exports embedder implementations and factory function.
"""

from typing import Any, Dict
from backend.common.interfaces import IEmbedder
from backend.embeddings.mock import MockEmbedder
from backend.embeddings.sentence_transformer import SentenceTransformerEmbedder

_EMBEDDER_REGISTRY: Dict[str, type] = {
    "mock": MockEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
    "minilm": SentenceTransformerEmbedder,
}

def get_embedder(name: str = "mock", **kwargs: Any) -> IEmbedder:
    """
    Factory to retrieve an embedder instance by name.
    
    Args:
        name: "mock", "sentence-transformers", or "minilm"
        **kwargs: forwarded to constructor (e.g. model_name="all-MiniLM-L6-v2")
    """
    cls = _EMBEDDER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown embedder '{name}'. Available: {list(_EMBEDDER_REGISTRY.keys())}"
        )
    return cls(**kwargs)

__all__ = [
    "IEmbedder",
    "MockEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
]
