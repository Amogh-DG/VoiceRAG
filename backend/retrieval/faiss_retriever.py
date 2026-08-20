"""
backend/retrieval/faiss_retriever.py
-----------------------------------
High-Performance FAISS Vector Retriever for RAG pipeline.

Loads:
  1. FAISS Binary Index: `data/indices/hi_val_multilingual.faiss`
  2. Metadata JSONL File: `data/indices/hi_val_multilingual_metadata.jsonl`
  3. Embedding Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on CUDA.

Runtime Characteristics
───────────────────────
- Load Time: All assets loaded exactly once during initialization.
# Search latency depends on hardware; benchmarked around ~154 ms
# on the current system for 1.07M × 384-dim IndexFlatIP search.
- Zero Runtime JSON Dependency: Does NOT load or depend on the 12.2 GB JSON index.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.common.interfaces import IRetriever
from backend.common.schemas import RetrievedChunk
from backend.embeddings.sentence_transformer import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


class FAISSRetriever(IRetriever):
    """
    Concrete IRetriever implementation backed by a FAISS IndexFlatIP.
    """

    def __init__(
        self,
        faiss_path: str = "data/indices/hi_val_multilingual.faiss",
        metadata_path: str = "data/indices/hi_val_multilingual_metadata.jsonl",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cuda",
    ):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss package is required. Install via: pip install faiss-cpu") from exc

        self.faiss_path = Path(faiss_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.device = device

        if not self.faiss_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.faiss_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata JSONL file not found: {self.metadata_path}")

        # 1. Load FAISS binary index
        logger.info("Loading FAISS index from %s...", self.faiss_path)
        self.faiss_index = faiss.read_index(str(self.faiss_path))
        logger.info("FAISS index loaded! Total vectors: %d", self.faiss_index.ntotal)

        # 2. Load metadata JSONL lines (1-to-1 index matching)
        logger.info("Loading metadata JSONL from %s...", self.metadata_path)
        self.metadata_list: List[Dict[str, Any]] = []
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

        # 3. Load embedding model once on specified device (CUDA)
        logger.info("Initializing SentenceTransformer model '%s' on %s...", self.model_name, self.device)
        self.embedder = SentenceTransformerEmbedder(model_name=self.model_name, device=self.device)

    # ------------------------------------------------------------------ #
    # Public Search Methods                                              #
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes real FAISS search for a raw text query.

        Returns structured dictionary objects with keys:
          - score:          float (cosine similarity)
          - chunk_id:       str
          - parent_doc_id:  str
          - text:           str (chunk content)
          - metadata:       dict
        """
        if not query or not query.strip():
            return []

        # Generate query embedding & normalize
        q_vec = self.embedder.embed(query)
        q_arr = np.array(q_vec, dtype=np.float32)
        norm = np.linalg.norm(q_arr)
        if norm > 0:
            q_arr = q_arr / norm
        q_matrix = q_arr.reshape(1, -1)

        # FAISS search
        scores, indices = self.faiss_index.search(q_matrix, top_k)

        results: List[Dict[str, Any]] = []
        for score, pos_idx in zip(scores[0], indices[0]):
            if pos_idx < 0 or pos_idx >= len(self.metadata_list):
                continue
            meta_payload = self.metadata_list[pos_idx]
            doc_id = meta_payload.get("parent_doc_id") or meta_payload.get("document_id") or meta_payload.get("id")
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
        
        retrieved: List[RetrievedChunk] = []
        for item in search_results:
            retrieved.append(
                RetrievedChunk(
                    text=item["text"],
                    score=item["score"],
                    document_id=item["parent_doc_id"],
                    metadata={
                        "chunk_id": item["chunk_id"],
                        **item["metadata"],
                    },
                )
            )

        return retrieved
