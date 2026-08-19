from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from backend.common.schemas import Document, Chunk, TextChunk, RetrievedChunk, GenerationResponse

class IDatasetLoader(ABC):
    """Abstract interface for loading and normalizing raw datasets."""
    
    @abstractmethod
    def load(self, source_path: str) -> List[Document]:
        """Loads files from source_path and normalizes them into Document objects."""
        pass


class IChunker(ABC):
    """Abstract interface for document chunking strategies."""
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[TextChunk]:
        """
        Chunk a single document into a list of TextChunk objects.

        Every returned TextChunk MUST have:
          - text          non-empty string
          - document_id   == document.id
          - metadata      dict containing at least:
                            chunk_strategy  (str)
                            chunk_index     (int, zero-based)
        """
        pass


class IEmbedder(ABC):
    """Abstract interface for generating vector embeddings."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generates embedding vector for a single text query."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of text documents."""
        pass

    async def embed_query(self, text: str) -> List[float]:
        """Async wrapper for embed(text)."""
        return self.embed(text)
        
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async wrapper for embed_batch(texts)."""
        return self.embed_batch(texts)


class IVectorDB(ABC):
    """Abstract interface for Vector Database storage and search."""
    
    @abstractmethod
    def upsert(self, chunks: List[Chunk]) -> None:
        """Upserts a list of Chunk objects into the vector DB store."""
        pass

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Queries vector space for chunks most similar to vector."""
        pass

    async def index_chunks(self, chunks: List[Chunk]) -> None:
        """Async wrapper for upsert(chunks)."""
        self.upsert(chunks)
        
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Async wrapper for query(query_embedding, top_k=k, filters=filters)."""
        return self.query(vector=query_embedding, top_k=k, filters=filters)


class IRetriever(ABC):
    """Abstract interface for the retriever component."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
    ) -> List[RetrievedChunk]:
        """
        Retrieves top_k relevant chunks for query, returning list of RetrievedChunk.
        """
        pass

    async def async_retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
    ) -> List[RetrievedChunk]:
        """Async wrapper for retrieve."""
        return self.retrieve(query, top_k=top_k, filters=filters, rerank=rerank)


from typing import List, Optional, Dict, Any, Union

class IGenerator(ABC):
    """Abstract interface for generation logic (LLM invocation, grounding, refusal)."""
    
    @abstractmethod
    def generate(
        self,
        question: str,
        retrieved_chunks: List[Union[RetrievedChunk, Chunk]],
    ) -> GenerationResponse:
        """Generates an answer grounded in retrieved context chunks, returning GenerationResponse."""
        pass

    async def async_generate(
        self,
        question: str,
        retrieved_chunks: List[Union[RetrievedChunk, Chunk]],
    ) -> GenerationResponse:
        """Async wrapper for generate."""
        return self.generate(question, retrieved_chunks)


class IPipelineOrchestrator(ABC):
    """Abstract interface for orchestrating the RAG execution harness."""
    
    @abstractmethod
    async def run_pipeline(self, query: str, k: int = 5) -> GenerationResponse:
        """Executes the end-to-end retrieval and generation pipeline for a query."""
        pass
