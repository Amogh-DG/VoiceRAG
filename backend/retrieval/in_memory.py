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

Performance design (deferred commit)
─────────────────────────────────────
upsert() runs in O(1) per chunk: it appends to a pending-vectors list.
The actual np.vstack (bulk matrix build) happens lazily, only when
query() or save() is called. This avoids the O(N²) copy pattern that
dominated previous indexing runs (1 full matrix copy per upsert call).

Auto-save has been removed from upsert() to allow high-throughput
batch indexing. Callers that need persistence must call save() explicitly.
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

    Internally maintains two layers:
      - committed: self.chunks / self.vectors (numpy matrix, None until first flush)
      - pending:   self._pending_chunks / self._pending_vecs (plain lists)

    _flush_pending() merges pending into committed via a single np.vstack.
    It is called automatically by query() and save() so callers never need
    to think about it.
    """

    def __init__(self, persist_path: Optional[str] = None):
        # Committed state (post-flush)
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None  # shape: (N, D)

        # Pending state (pre-flush) — O(1) appends
        self._pending_chunks: List[Chunk] = []
        self._pending_vecs: List[np.ndarray] = []

        self.persist_path = Path(persist_path) if persist_path else None

        if self.persist_path and self.persist_path.exists():
            self.load(str(self.persist_path))

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def upsert(self, chunks: List[Chunk]) -> None:
        """
        Stage chunks for indexing.

        O(k) where k = len(chunks). No matrix allocation happens here.
        The np.vstack only occurs when query() or save() is called.
        Auto-save has been intentionally removed — call save() explicitly.
        """
        if not chunks:
            return

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Chunk %s has no embedding, skipping.", chunk.id)
                continue
            vec = np.array(chunk.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._pending_chunks.append(chunk)
            self._pending_vecs.append(vec)

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Query vector space for top_k most similar chunks."""
        self._flush_pending()

        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores = np.dot(self.vectors, query_vec)
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
        """Flush pending chunks and persist index to JSON file."""
        self._flush_pending()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [chunk.model_dump() for chunk in self.chunks]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d chunks to %s", len(self.chunks), path)

    def load(self, filepath: str) -> None:
        """Load index from JSON file. Replaces current in-memory state."""
        path = Path(filepath)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Reset everything
        self.chunks = []
        self.vectors = None
        self._pending_chunks = []
        self._pending_vecs = []
        # Re-upsert without triggering save (persist_path not set yet)
        chunks = [Chunk(**item) for item in data]
        self.upsert(chunks)
        self._flush_pending()
        logger.info("Loaded %d chunks from %s", len(self.chunks), path)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _flush_pending(self) -> None:
        """
        Merge pending chunks/vectors into the committed numpy matrix.

        Uses a single np.vstack call over all pending vectors (O(k·D)),
        then concatenates to the existing matrix in one operation.
        Previously this was O(N) per chunk (one vstack per chunk), causing
        O(N²) total work over a full indexing run.
        """
        if not self._pending_chunks:
            return

        new_matrix = np.vstack(self._pending_vecs)  # shape: (k, D)

        if self.vectors is None or len(self.chunks) == 0:
            self.vectors = new_matrix
            self.chunks = list(self._pending_chunks)
        else:
            # Handle upserts (overwrite existing IDs) and appends
            existing_id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}
            new_chunks_list = []
            new_vecs_list = []

            for chunk, vec in zip(self._pending_chunks, self._pending_vecs):
                if chunk.id in existing_id_to_idx:
                    idx = existing_id_to_idx[chunk.id]
                    self.chunks[idx] = chunk
                    self.vectors[idx] = vec
                else:
                    new_chunks_list.append(chunk)
                    new_vecs_list.append(vec)

            if new_chunks_list:
                self.chunks.extend(new_chunks_list)
                self.vectors = np.vstack([self.vectors, np.vstack(new_vecs_list)])

        self._pending_chunks = []
        self._pending_vecs = []

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if chunk metadata satisfies equality filters."""
        for k, v in filters.items():
            if k not in metadata or metadata[k] != v:
                return False
        return True
