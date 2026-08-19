import os
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.common.schemas import QueryRequest, QueryResponse, ChunkReference, Chunk
from backend.embeddings.wrapper import EmbeddingModelWrapper
from backend.retrieval.vector_db import MockVectorDB
from backend.generation.generator import LLMGenerator
from backend.harness.orchestrator import PipelineOrchestrator

from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Instantiate core RAG modules using swappable interfaces
embedder = EmbeddingModelWrapper()
vector_db = MockVectorDB()
generator = LLMGenerator()
orchestrator = PipelineOrchestrator(embedder=embedder, vector_db=vector_db, generator=generator)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown lifecycles of the application."""
    print("[API Startup] Seeding mock vector database with initial chunk data...")
    stub_chunks = [
        Chunk(
            id="chunk_doc1_0",
            parent_doc_id="doc1",
            content="Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs).",
            index=0,
            embedding=embedder.embed("Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs)."),
            metadata={"source": "wikipedia_rag.txt"}
        ),
        Chunk(
            id="chunk_doc1_1",
            parent_doc_id="doc1",
            content="RAG backend architectures typically consist of ingestion pipelines, embedding generators, search indices, and LLM text generation systems.",
            index=1,
            embedding=embedder.embed("RAG backend architectures typically consist of ingestion pipelines, embedding generators, search indices, and LLM text generation systems."),
            metadata={"source": "wikipedia_rag.txt"}
        ),
        Chunk(
            id="chunk_doc2_0",
            parent_doc_id="doc2",
            content="FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.",
            index=0,
            embedding=embedder.embed("FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints."),
            metadata={"source": "fastapi_docs.txt"}
        )
    ]
    await vector_db.index_chunks(stub_chunks)
    print("[API Startup] Vector database seeded.")
    yield
    print("[API Shutdown] Tearing down application resources...")

app = FastAPI(
    title="RAG Backend API",
    description="Backend API exposing modular Retrieval-Augmented Generation services.",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Service status healthcheck endpoint."""
    return {"status": "healthy", "service": "RAG Backend Scaffolding"}

@app.post("/ask", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: QueryRequest):
    """Processes user query, retrieves relevant contexts, and generates an answer."""
    try:
        start_time = time.perf_counter()
        
        # 1. Fetch query embedding
        query_embedding = await embedder.embed_query(request.query)
        
        # 2. Query Vector DB using the embedding
        retrieved_chunks = await vector_db.similarity_search(
            query_embedding=query_embedding,
            k=request.k,
            filters=request.filters
        )
        
        # 3. Generate response using Orchestrator
        gen_response = await orchestrator.run_pipeline(query=request.query, k=request.k)
        
        # 4. Map internal chunks to public response metadata references
        context_refs = [
            ChunkReference(
                chunk_id=chunk.id,
                document_id=chunk.parent_doc_id,
                content=chunk.content,
                metadata=chunk.metadata
            )
            for chunk in retrieved_chunks
        ]
        
        elapsed_time = time.perf_counter() - start_time
        
        # 5. Build and return API QueryResponse
        return QueryResponse(
            query=request.query,
            answer=gen_response.answer,
            context=context_refs,
            is_grounded=gen_response.is_grounded,
            is_refused=gen_response.is_refused,
            metadata={
                "elapsed_seconds": elapsed_time,
                "generation_metadata": gen_response.metadata
            }
        )
        
    except Exception as e:
        print(f"Error processing RAG query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the RAG query pipeline: {str(e)}"
        )
