"""
backend/retrieval/retriever.py
------------------------------
VectorRetriever implementing the IRetriever public contract.

Reranking & Latency Rationale
──────────────────────────────
- `rerank` flag is **OFF (False) by default**.
- Heavy cross-encoder rerankers introduce 150-300ms additional latency per query,
  which risks violating our strict <200ms end-to-end RAG latency target.
- A lightweight BM25/keyword-frequency reranking adjustment is performed ONLY
  when `rerank=True` is explicitly passed.
"""

import math
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from backend.common.interfaces import IEmbedder, IRetriever, IVectorDB
from backend.common.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


def _compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _simple_keyword_rerank(
    query: str, chunks: List[RetrievedChunk]
) -> List[RetrievedChunk]:
    """
    Optional lightweight reranker: applies a keyword overlap boost to vector scores
    without calling heavy cross-encoder models.
    """
    if not query.strip() or not chunks:
        return chunks

    query_tokens = set(query.lower().split())
    if not query_tokens:
        return chunks

    reranked: List[RetrievedChunk] = []
    for chunk in chunks:
        chunk_tokens = chunk.text.lower().split()
        if not chunk_tokens:
            overlap_ratio = 0.0
        else:
            matches = sum(1 for t in query_tokens if t in chunk_tokens)
            overlap_ratio = matches / len(query_tokens)

        # Blend vector similarity score with keyword overlap boost (80/20 blend)
        new_score = 0.8 * chunk.score + 0.2 * overlap_ratio
        
        # Create new RetrievedChunk with updated score
        new_chunk = RetrievedChunk(
            text=chunk.text,
            score=round(new_score, 6),
            document_id=chunk.document_id,
            metadata=chunk.metadata,
        )
        reranked.append(new_chunk)

    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked


class VectorRetriever(IRetriever):
    """
    Concrete retriever implementation wrapping an IEmbedder and IVectorDB.
    """

    def __init__(self, embedder: IEmbedder, vector_db: IVectorDB):
        self.embedder = embedder
        self.vector_db = vector_db
        self.last_timings = {
            "embedding_ms": 0.0,
            "vector_db_ms": 0.0,
            "reranking_ms": 0.0
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
    ) -> List[RetrievedChunk]:
        """
        Retrieves top_k relevant chunks for a query string.

        Args:
            query: User's question or search query string.
            top_k: Maximum number of chunks to return (default 5).
            filters: Optional metadata filtering dictionary.
            rerank: If True, applies lightweight keyword reranking. Default False.

        Returns:
            List[RetrievedChunk]: Ranked list of RetrievedChunk objects matching contract.
        """
        if not query or not query.strip():
            logger.info("Empty query provided to retrieve(), returning empty list.")
            self.last_timings = {
                "embedding_ms": 0.0,
                "vector_db_ms": 0.0,
                "reranking_ms": 0.0
            }
            return []

        if top_k <= 0:
            self.last_timings = {
                "embedding_ms": 0.0,
                "vector_db_ms": 0.0,
                "reranking_ms": 0.0
            }
            return []

        import time

        # 1. Embed query
        start_embed = time.perf_counter()
        query_vector = self.embedder.embed(query)
        embed_time_ms = (time.perf_counter() - start_embed) * 1000.0

        # 2. Query Vector DB (fetch candidates; fetch extra if reranking is enabled)
        fetch_k = top_k * 2 if rerank else top_k
        start_db = time.perf_counter()
        raw_chunks: List[Chunk] = self.vector_db.query(
            vector=query_vector, top_k=fetch_k, filters=filters
        )
        db_time_ms = (time.perf_counter() - start_db) * 1000.0

        if not raw_chunks:
            self.last_timings = {
                "embedding_ms": embed_time_ms,
                "vector_db_ms": db_time_ms,
                "reranking_ms": 0.0
            }
            return []

        # 3. Convert Chunk objects to RetrievedChunk, computing similarity score
        retrieved_list: List[RetrievedChunk] = []
        for chunk in raw_chunks:
            if chunk.embedding is not None:
                score = _compute_cosine_similarity(query_vector, chunk.embedding)
            else:
                score = 0.0

            rc = RetrievedChunk(
                text=chunk.content,
                score=round(score, 6),
                document_id=chunk.parent_doc_id,
                metadata=chunk.metadata,
            )
            retrieved_list.append(rc)

        # Sort by initial score descending
        retrieved_list.sort(key=lambda c: c.score, reverse=True)

        # 4. Optional Reranking
        rerank_time_ms = 0.0
        if rerank:
            start_rerank = time.perf_counter()
            retrieved_list = _simple_keyword_rerank(query, retrieved_list)
            rerank_time_ms = (time.perf_counter() - start_rerank) * 1000.0

        self.last_timings = {
            "embedding_ms": embed_time_ms,
            "vector_db_ms": db_time_ms,
            "reranking_ms": rerank_time_ms
        }

        return retrieved_list[:top_k]
