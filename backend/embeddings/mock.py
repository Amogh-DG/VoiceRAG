"""
backend/embeddings/mock.py
---------------------------
Fast, deterministic MockEmbedder for unit tests and local testing.
Generates normalised pseudo-random embedding vectors of fixed dimension
without requiring model downloads or heavy PyTorch dependencies.
"""

import math
import hashlib
from typing import List

from backend.common.interfaces import IEmbedder


class MockEmbedder(IEmbedder):
    """
    Mock embedder that generates deterministic embedding vectors derived from
    the SHA-256 hash of the input text.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        vec = []
        # Generate pseudo-random numbers by hashing text + dimension index
        for i in range(self.dimension):
            h = hashlib.sha256(f"{text}:{i}".encode("utf-8")).hexdigest()
            # Scale to [-1.0, 1.0]
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)
        
        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for a single text."""
        return self._hash_to_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of texts."""
        return [self.embed(t) for t in texts]
