import os
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.common.schemas import AskRequest, AskResponse, SourceDocument, Chunk
from backend.embeddings.wrapper import EmbeddingModelWrapper
from backend.retrieval.vector_db import MockVectorDB
from backend.generation.generator import LLMGenerator
from backend.harness.orchestrator import PipelineOrchestrator

from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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

# Custom handler to return HTTP 400 Bad Request on input schema validation failure
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": "Input validation failed: malformed JSON or missing required fields."}
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

@app.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit user query to the RAG pipeline",
    description="Exposes end-to-end RAG question answering. Runs the user query through guardrails, context retrieval, LLM generation, and hallucination checks."
)
async def ask_question(request: AskRequest):
    """Processes user query, retrieves relevant contexts, and generates an answer."""
    # Custom validation: raise HTTP 400 for empty or whitespace-only query
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty or whitespace-only."
        )

    start_time = time.perf_counter()
    try:
        # Run user query through the pipeline orchestrator
        pipeline_result = await orchestrator.run_pipeline(query=request.query)

        # Map pipeline's internal context ChunkReferences to client-facing SourceDocuments
        sources = [
            SourceDocument(
                document_id=ref.document_id,
                text=ref.content,
                metadata=ref.metadata
            )
            for ref in pipeline_result.context
        ]

        # Calculate actual end-to-end wall-clock latency in milliseconds
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return AskResponse(
            answer=pipeline_result.answer,
            sources=sources,
            latency_ms=latency_ms
        )

    except Exception as e:
        print(f"Error processing RAG query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the RAG query pipeline: {str(e)}"
        )
