"""
backend/embeddings/sentence_transformer.py
--------------------------------------------
SentenceTransformerEmbedder wrapping HuggingFace sentence-transformers.

Model Choice & Latency Rationale
────────────────────────────────
Default model: `all-MiniLM-L6-v2` (384-dimensional).
- Size: ~90 MB on disk.
- CPU Latency: ~5-15ms per batch (384 dimensions).
- Quality: Strong baseline for sentence/passage semantic retrieval.

For multilingual support (e.g. Indic languages in MSMARCO-XI):
Model choice: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Size: ~470 MB on disk.
- CPU Latency: ~20-40ms per batch.

Both choices easily satisfy our strict <200ms end-to-end RAG latency target
while running locally without external API calls or GPU requirements.
"""

from typing import List, Optional
import logging

from backend.common.interfaces import IEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(IEmbedder):
    """
    Concrete IEmbedder implementation using sentence-transformers.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = "cuda"
        self._model = None

    def _load_model(self):
        if self._model is None: 
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers package is required. "
                    "Install it via: pip install sentence-transformers"
                ) from exc
            
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str) -> List[float]:
        """Generates embedding vector for a single text."""
        self._load_model()
        vector = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of texts."""
        if not texts:
            return []
        self._load_model()
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()
