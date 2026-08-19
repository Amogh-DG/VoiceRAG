import hashlib
import math
from typing import List

from backend.common.interfaces import IEmbedder


class EmbeddingModelWrapper(IEmbedder):
    """Wrapper around external or local embedding generation models.

    In stub/offline mode (no real model loaded), produces deterministic,
    normalised pseudo-random vectors derived from the SHA-256 hash of the
    input text.  This guarantees non-zero embeddings whose cosine similarity
    is positive for identical or related texts — critical for the grounding
    threshold check in the harness guardrails.

    TODO:
    - Integrate OpenAI embedder API, HuggingFace local models, or SentenceTransformers.
    - Implement caching for generated embeddings to avoid redundant remote API calls.
    - Add rate limiting and robust retry policies for APIs.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", dimension: int = 1536):
        self.model_name = model_name
        self.dimension = dimension

    # ── deterministic bag-of-words style vector ─────────────────────────────
    def _hash_to_vector(self, text: str) -> List[float]:
        """Produce a unit-length vector deterministically from *text*.

        Each word in *text* activates a handful of dimensions via hashing,
        so texts that share words will produce vectors with meaningful
        positive cosine similarity.  This is critical for the grounding
        threshold guardrail to work correctly in offline/stub mode.
        """
        import re
        vec = [0.0] * self.dimension
        words = re.findall(r"\w+", text.lower())
        for word in words:
            # Each word activates 4 dimensions deterministically
            for sub in range(4):
                h = hashlib.sha256(f"{word}:{sub}".encode("utf-8")).hexdigest()
                idx = int(h[:8], 16) % self.dimension
                val = ((int(h[8:16], 16) / 0xFFFFFFFF) * 2.0 - 1.0)
                vec[idx] += val
        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, text: str) -> List[float]:
        """Generates embedding vector for a single query string."""
        print(f"[EmbeddingModelWrapper] Stub embedding query with model '{self.model_name}'")
        return self._hash_to_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of documents."""
        print(f"[EmbeddingModelWrapper] Stub embedding batch of {len(texts)} documents with model '{self.model_name}'")
        return [self._hash_to_vector(t) for t in texts]

    async def embed_query(self, text: str) -> List[float]:
        """Generates embedding vector for a single query string (async wrapper)."""
        return self.embed(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of documents (async wrapper)."""
        return self.embed_batch(texts)
