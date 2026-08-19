"""
backend/retrieval/in_memory.py
-------------------------------
In-memory Vector DB implementation using numpy cosine similarity.

Latency & Practicality Rationale
────────────────────────────────
- Latency: <1ms per top-k search over thousands of chunks on CPU.
- Zero dependencies beyond standard Python / NumPy.
- Zero network or server setup overhead.
- Supports optional JSON persistence to disk for dev workflows.
- Fully satisfies <200ms end-to-end RAG latency constraint.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from backend.common.interfaces import IVectorDB
from backend.common.schemas import Chunk

logger = logging.getLogger(__name__)


class InMemoryVectorDB(IVectorDB):
    """
    In-memory vector database with optional disk save/load.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None  # shape: (N, D)
        self.persist_path = Path(persist_path) if persist_path else None

        if self.persist_path and self.persist_path.exists():
            self.load(str(self.persist_path))

    def upsert(self, chunks: List[Chunk]) -> None:
        """Upsert chunks into memory, updating vector matrix."""
        if not chunks:
            return

        new_vectors = []
        valid_chunks = []

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Chunk %s has no embedding, skipping upsert.", chunk.id)
                continue
            valid_chunks.append(chunk)
            # Normalize vector to unit length for cosine similarity via dot product
            vec = np.array(chunk.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            new_vectors.append(vec)

        if not valid_chunks:
            return

        new_matrix = np.vstack(new_vectors)

        if self.vectors is None or len(self.chunks) == 0:
            self.chunks = list(valid_chunks)
            self.vectors = new_matrix
        else:
            # Overwrite chunks with matching IDs, or append new ones
            existing_id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}
            
            for chunk, vec in zip(valid_chunks, new_vectors):
                if chunk.id in existing_id_to_idx:
                    idx = existing_id_to_idx[chunk.id]
                    self.chunks[idx] = chunk
                    self.vectors[idx] = vec
                else:
                    self.chunks.append(chunk)
                    self.vectors = np.vstack([self.vectors, vec])

        if self.persist_path:
            self.save(str(self.persist_path))

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Checks if chunk metadata satisfies equality filters."""
        for k, v in filters.items():
            if k not in metadata or metadata[k] != v:
                return False
        return True

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Query vector space for top_k most similar chunks."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Compute cosine similarity scores via matrix-vector product
        scores = np.dot(self.vectors, query_vec)

        # Sort indices by descending score
        sorted_indices = np.argsort(-scores)

        results: List[Chunk] = []
        for idx in sorted_indices:
            chunk = self.chunks[idx]
            if filters and not self._matches_filters(chunk.metadata, filters):
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: str) -> None:
        """Persists indexed chunks and vectors to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [chunk.model_dump() for chunk in self.chunks]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d chunks to %s", len(self.chunks), path)

    def load(self, filepath: str) -> None:
        """Loads indexed chunks and vectors from JSON file."""
        path = Path(filepath)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [Chunk(**item) for item in data]
        self.chunks = []
        self.vectors = None
        self.upsert(chunks)
        logger.info("Loaded %d chunks from %s", len(self.chunks), path)
