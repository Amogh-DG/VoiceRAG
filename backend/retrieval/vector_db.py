from typing import List, Optional, Dict, Any
from backend.common.interfaces import IVectorDB
from backend.common.schemas import Chunk

class MockVectorDB(IVectorDB):
    """An in-memory or external vector database client wrapper.
    
    TODO:
    - Integrate a production-grade Vector DB client (e.g. Qdrant, Chroma, Pinecone, PGVector).
    - Implement indexing logic that accepts embeddings from the embedder.
    - Implement distance metrics (e.g., Cosine Similarity, Dot Product, L2).
    - Implement filter parsing logic based on metadata filters.
    """

    def __init__(self, db_url: str = "memory://"):
        self.db_url = db_url
        # Keep an in-memory dictionary as a simple placeholder store
        self.store: Dict[str, Chunk] = {}

    def upsert(self, chunks: List[Chunk]) -> None:
        """Indexes document chunks into the database."""
        print(f"[MockVectorDB] Indexing {len(chunks)} chunks into vector store at: {self.db_url}")
        for chunk in chunks:
            self.store[chunk.id] = chunk

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Performs nearest-neighbor search for similarity."""
        print(f"[MockVectorDB] Performing similarity search (k={top_k}, filters={filters})")
        all_chunks = list(self.store.values())
        if filters:
            filtered = []
            for chunk in all_chunks:
                match = True
                for k, v in filters.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if match:
                    filtered.append(chunk)
            return filtered[:top_k]
        return all_chunks[:top_k]

    async def index_chunks(self, chunks: List[Chunk]) -> None:
        """Async wrapper for upsert."""
        self.upsert(chunks)

    async def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Async wrapper for query."""
        return self.query(vector=query_embedding, top_k=k, filters=filters)
