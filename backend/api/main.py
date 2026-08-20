import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from backend.common.schemas import AskRequest, AskResponse, SourceDocument
from backend.retrieval.faiss_retriever import FAISSRetriever
from backend.generation.generator import LLMGenerator
from backend.harness.orchestrator import PipelineOrchestrator

# Load environment variables
load_dotenv()

# Initialize FAISSRetriever ONCE when application starts
logger_prefix = "[API Initialization]"
print(f"{logger_prefix} Loading FAISSRetriever with CUDA model 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'...")

faiss_retriever = FAISSRetriever(
    faiss_path="data/indices/hi_val_multilingual.faiss",
    metadata_path="data/indices/hi_val_multilingual_metadata.jsonl",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda",
)

generator = LLMGenerator()
orchestrator = PipelineOrchestrator(retriever=faiss_retriever, generator=generator)
print(f"{logger_prefix} FAISSRetriever connected successfully to RAG Pipeline Orchestrator.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown lifecycles of the application."""
    print("[API Lifespan] RAG Backend active and ready for queries.")
    yield
    print("[API Lifespan] Tearing down application resources...")


app = FastAPI(
    title="RAG Backend API",
    description="Backend API exposing modular Retrieval-Augmented Generation services backed by FAISS vector index.",
    version="0.2.0",
    lifespan=lifespan,
)

# Custom handler to return HTTP 400 Bad Request on input schema validation failure
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": "Input validation failed: malformed JSON or missing required fields."},
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
    return {"status": "healthy", "service": "RAG Backend with FAISS Retriever"}


@app.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit user query to the RAG pipeline",
    description="Exposes end-to-end RAG question answering backed by FAISS vector retrieval.",
)
async def ask_question(request: AskRequest):
    """Processes user query, retrieves relevant contexts from FAISS, and returns structured RAG result."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty or whitespace-only.",
        )

    start_time = time.perf_counter()
    try:
        # Run query through orchestrator using pre-initialized FAISSRetriever
        pipeline_result = await orchestrator.run_pipeline(query=request.query)

        # Map pipeline context references to SourceDocument responses
        sources = [
            SourceDocument(
                document_id=ref.document_id,
                text=ref.content,
                metadata={
                    "chunk_id": ref.chunk_id,
                    **ref.metadata,
                },
            )
            for ref in pipeline_result.context
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return AskResponse(
            answer=pipeline_result.answer,
            sources=sources,
            latency_ms=latency_ms,
        )

    except Exception as e:
        print(f"Error processing RAG query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the RAG query pipeline: {str(e)}",
        )
