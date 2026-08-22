"""
backend/retrieval/faiss_retriever.py
-----------------------------------
High-Performance FAISS Vector Retriever for RAG pipeline.

Loads:
  1. FAISS Binary Index: `data/indices/hi_val_multilingual_sq_fp16.faiss`
     (scalar-quantised, QT_fp16 — exhaustive search, ~50% smaller than IndexFlatIP)
     Falls back to `data/indices/hi_val_multilingual.faiss` if the quantised file
     is absent, so old deployments continue to work without conversion.
  2. Metadata Backend (choose one):
       a. SQLite DB: `data/indices/hi_val_multilingual_metadata.db`
          Opened per-thread (threading.local) for asyncio.to_thread() safety.
          Only the rows returned by FAISS are fetched per query — no full load.
       b. JSONL fallback: `data/indices/hi_val_multilingual_metadata.jsonl`
          Legacy path: loads the entire file into self.metadata_list at init.
          Used when metadata_db_path is None or the .db file does not exist.
  3. Embedding Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
     on CUDA.

Runtime Characteristics
───────────────────────
- Load Time: All assets loaded exactly once during initialization.
# Search latency depends on hardware; benchmarked around ~154 ms
# on the current system for 1.07M × 384-dim IndexFlatIP search.
# IndexScalarQuantizer (QT_fp16) has comparable latency with ~50% less RAM.
- Zero Runtime JSON Dependency: Does NOT load or depend on the 12.2 GB JSON index.

SQLite Thread-Safety
────────────────────
sqlite3 Connection objects are NOT safe to share across OS threads.
When FAISSRetriever.search() is called via asyncio.to_thread(), the event loop
dispatches work to threads from the default ThreadPoolExecutor.  To handle this
safely, a threading.local() object holds one sqlite3.Connection per OS thread.
Each thread opens its own connection on first use and reuses it for all subsequent
queries on that thread.

check_same_thread=False is passed to sqlite3.connect() as a documented safety
valve, but the per-thread pattern makes shared-connection access impossible in
practice.  The net result is:
  - No locking overhead (connections are never shared)
  - No "ProgrammingError: SQLite objects created in a thread can only be used in
    that same thread" exceptions
  - Safe under any asyncio concurrency level
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.common.interfaces import IRetriever
from backend.common.schemas import RetrievedChunk
from backend.embeddings.sentence_transformer import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────
_DEFAULT_FAISS_NEW = "data/indices/hi_val_multilingual_sq_fp16.faiss"
_DEFAULT_FAISS_OLD = "data/indices/hi_val_multilingual.faiss"
_DEFAULT_METADATA_JSONL = "data/indices/hi_val_multilingual_metadata.jsonl"
_DEFAULT_METADATA_DB = "data/indices/hi_val_multilingual_metadata.db"

# SQLite query used to fetch metadata for a batch of positional indices
_SELECT_CHUNKS = (
    "SELECT pos, id, parent_doc_id, content, metadata "
    "FROM chunks WHERE pos IN ({placeholders})"
)


class FAISSRetriever(IRetriever):
    """
    Concrete IRetriever implementation backed by a FAISS index.

    Supports two index types (auto-detected at init):
      - IndexScalarQuantizer (QT_fp16)  — quantised, ~50% RAM savings, preferred.
      - IndexFlatIP                      — original float32, legacy fallback.

    Supports two metadata backends (chosen by metadata_db_path):
      - SQLite (preferred): per-thread connection, batched lookup per query.
      - JSONL list (fallback): entire file loaded at init into self.metadata_list.
    """

    def __init__(
        self,
        faiss_path: Optional[str] = None,
        metadata_path: str = _DEFAULT_METADATA_JSONL,
        metadata_db_path: Optional[str] = _DEFAULT_METADATA_DB,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cuda",
    ):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss package is required. Install via: pip install faiss-cpu"
            ) from exc

        # ── Resolve FAISS path (prefer quantised, fall back to original) ──────
        if faiss_path is not None:
            self.faiss_path = Path(faiss_path)
        else:
            sq_path = Path(_DEFAULT_FAISS_NEW)
            flat_path = Path(_DEFAULT_FAISS_OLD)
            if sq_path.exists():
                self.faiss_path = sq_path
                logger.info(
                    "Quantised FAISS index found (%s) — using QT_fp16 index.", sq_path
                )
            elif flat_path.exists():
                self.faiss_path = flat_path
                logger.info(
                    "Quantised index not found; falling back to original IndexFlatIP (%s).",
                    flat_path,
                )
            else:
                raise FileNotFoundError(
                    f"No FAISS index found at {_DEFAULT_FAISS_NEW} or {_DEFAULT_FAISS_OLD}."
                )

        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.device = device
        self.last_timings_ms: Dict[str, float] = {}

        if not self.faiss_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.faiss_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata JSONL file not found: {self.metadata_path}"
            )

        # ── 1. Load FAISS binary index ─────────────────────────────────────────
        logger.info("Loading FAISS index from %s...", self.faiss_path)
        self.faiss_index = faiss.read_index(str(self.faiss_path))
        logger.info(
            "FAISS index loaded! Type: %s, Total vectors: %d",
            type(self.faiss_index).__name__,
            self.faiss_index.ntotal,
        )

        # ── 2. Choose metadata backend ────────────────────────────────────────
        self._use_sqlite = False
        self.metadata_list: List[Dict[str, Any]] = []  # used only in JSONL mode

        if metadata_db_path is not None:
            db_path = Path(metadata_db_path)
            if db_path.exists():
                self._metadata_db_path = str(db_path)
                self._use_sqlite = True
                # threading.local() stores one sqlite3.Connection per OS thread.
                self._local = threading.local()
                logger.info(
                    "SQLite metadata backend: %s (per-thread connections).", db_path
                )
                # Validate row count against FAISS ntotal
                with self._get_sqlite_conn() as con:
                    (row_count,) = con.execute(
                        "SELECT COUNT(*) FROM chunks"
                    ).fetchone()
                logger.info("SQLite metadata row count: %d", row_count)
                if self.faiss_index.ntotal != row_count:
                    logger.warning(
                        "Mismatch: FAISS ntotal=%d vs SQLite row count=%d.",
                        self.faiss_index.ntotal,
                        row_count,
                    )
            else:
                logger.warning(
                    "metadata_db_path=%s specified but file not found; "
                    "falling back to JSONL list.",
                    db_path,
                )

        if not self._use_sqlite:
            # Fallback: load full JSONL into RAM
            logger.info(
                "Loading metadata JSONL from %s (legacy in-memory mode)...",
                self.metadata_path,
            )
            with self.metadata_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.metadata_list.append(json.loads(line))
            logger.info("Loaded %d metadata records.", len(self.metadata_list))

            if self.faiss_index.ntotal != len(self.metadata_list):
                logger.warning(
                    "Mismatch between FAISS index size (%d) and metadata record count (%d).",
                    self.faiss_index.ntotal,
                    len(self.metadata_list),
                )

        # ── 3. Load embedding model once on specified device (CUDA) ───────────
        logger.info(
            "Initializing SentenceTransformer model '%s' on %s...",
            self.model_name,
            self.device,
        )
        self.embedder = SentenceTransformerEmbedder(
            model_name=self.model_name, device=self.device
        )
        self.device = self.embedder.device

    # ── SQLite helpers ─────────────────────────────────────────────────────────

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        """
        Return the sqlite3.Connection for the current OS thread.

        Opens a new connection if the thread doesn't have one yet.
        check_same_thread=False is a safety valve; the per-thread pattern
        prevents any cross-thread connection sharing in practice.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._metadata_db_path, check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _sqlite_fetch_batch(
        self, positions: List[int]
    ) -> Dict[int, sqlite3.Row]:
        """
        Fetch metadata rows for the given positional indices in one SQL query.

        Returns a dict mapping pos → sqlite3.Row.
        """
        if not positions:
            return {}
        placeholders = ",".join("?" * len(positions))
        con = self._get_sqlite_conn()
        rows = con.execute(
            _SELECT_CHUNKS.format(placeholders=placeholders), positions
        ).fetchall()
        return {row["pos"]: row for row in rows}

    # ------------------------------------------------------------------ #
    # Public Search Methods                                              #
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes real FAISS search for a raw text query with sub-stage timing.

        The faiss_index.search() call signature and return shape are unchanged:
          scores, indices = self.faiss_index.search(q_matrix, top_k)
        where scores.shape == indices.shape == (1, top_k).

        Metadata lookup uses either:
          - SQLite batched SELECT (SQLite mode, preferred)
          - positional list lookup (JSONL fallback mode)

        Return shape is identical in both modes:
          List[Dict] with keys: score, chunk_id, parent_doc_id, text, metadata
        """
        if not query or not query.strip():
            return []

        t_search_start = time.perf_counter()

        # Sub-stage 4: Query embedding
        t_embed_start = time.perf_counter()
        q_vec = self.embedder.embed(query)
        q_arr = np.array(q_vec, dtype=np.float32)
        norm = np.linalg.norm(q_arr)
        if norm > 0:
            q_arr = q_arr / norm
        q_matrix = q_arr.reshape(1, -1)
        t_embed_end = time.perf_counter()

        # Sub-stage 5: FAISS search
        # Call signature and return shape are UNCHANGED from IndexFlatIP version.
        t_faiss_start = time.perf_counter()
        scores, indices = self.faiss_index.search(q_matrix, top_k)
        t_faiss_end = time.perf_counter()

        # Sub-stage 6: Metadata lookup & result construction
        t_meta_start = time.perf_counter()

        if self._use_sqlite:
            results = self._build_results_sqlite(scores, indices)
        else:
            results = self._build_results_list(scores, indices)

        t_meta_end = time.perf_counter()

        t_search_end = time.perf_counter()

        self.last_timings_ms = {
            "query_embedding_ms": (t_embed_end - t_embed_start) * 1000.0,
            "faiss_search_ms": (t_faiss_end - t_faiss_start) * 1000.0,
            "metadata_lookup_ms": (t_meta_end - t_meta_start) * 1000.0,
            "total_faiss_retriever_ms": (t_search_end - t_search_start) * 1000.0,
        }

        return results

    def _build_results_sqlite(
        self, scores: np.ndarray, indices: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Build result dicts using a single batched SQLite query."""
        valid_positions = [
            int(pos_idx)
            for pos_idx in indices[0]
            if pos_idx >= 0
        ]
        row_map = self._sqlite_fetch_batch(valid_positions)

        results: List[Dict[str, Any]] = []
        for score, pos_idx in zip(scores[0], indices[0]):
            if pos_idx < 0:
                continue
            pos_idx = int(pos_idx)
            row = row_map.get(pos_idx)
            if row is None:
                continue

            # Deserialise the metadata JSON blob stored in SQLite
            try:
                meta_payload = json.loads(row["metadata"]) if row["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                meta_payload = {}

            results.append({
                "score": float(score),
                "chunk_id": row["id"] if row["id"] else f"chunk_{pos_idx}",
                "parent_doc_id": row["parent_doc_id"],
                "text": row["content"] or "",
                "metadata": meta_payload,
            })
        return results

    def _build_results_list(
        self, scores: np.ndarray, indices: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Build result dicts using the in-memory metadata list (legacy path)."""
        results: List[Dict[str, Any]] = []
        for score, pos_idx in zip(scores[0], indices[0]):
            if pos_idx < 0 or pos_idx >= len(self.metadata_list):
                continue
            meta_payload = self.metadata_list[pos_idx]
            doc_id = (
                meta_payload.get("parent_doc_id")
                or meta_payload.get("document_id")
                or meta_payload.get("id")
            )
            content = meta_payload.get("content") or meta_payload.get("text") or ""

            results.append({
                "score": float(score),
                "chunk_id": meta_payload.get("id", f"chunk_{pos_idx}"),
                "parent_doc_id": doc_id,
                "text": content,
                "metadata": meta_payload.get("metadata", {}),
            })
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
    ) -> List[RetrievedChunk]:
        """
        IRetriever interface implementation returning standard `RetrievedChunk` items.
        """
        search_results = self.search(query=query, top_k=top_k)

        return [
            RetrievedChunk(
                text=res["text"],
                score=res["score"],
                document_id=res["parent_doc_id"],
                metadata={
                    "chunk_id": res["chunk_id"],
                    "similarity_score": res["score"],
                    **res["metadata"],
                },
            )
            for res in search_results
        ]
