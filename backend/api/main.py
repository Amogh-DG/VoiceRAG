import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

from pydantic import BaseModel
from backend.common.schemas import AskRequest, AskResponse, SourceDocument
from backend.retrieval.faiss_retriever import FAISSRetriever
from backend.generation.llm_client import GeminiLLMClient, MockLLMClient
from backend.generation.generator import RAGGenerator
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

# Initialize Gemini LLM client if GEMINI_API_KEY is present, else fall back to MockLLMClient
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    print(f"{logger_prefix} Initializing Gemini 3.5 Flash-Lite LLM Client ('gemini-3.5-flash-lite')...")
    llm_client = GeminiLLMClient(api_key=gemini_key, model_name="gemini-3.5-flash-lite")
else:
    print(f"{logger_prefix} GEMINI_API_KEY not found; using MockLLMClient fallback.")
    llm_client = MockLLMClient()

generator = RAGGenerator(llm_client=llm_client)
orchestrator = PipelineOrchestrator(retriever=faiss_retriever, generator=generator)
print(f"{logger_prefix} Gemini RAG Pipeline initialized successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown lifecycles of the application."""
    print("[API Lifespan] RAG Backend active and ready for queries.")
    yield
    print("[API Lifespan] Tearing down application resources...")


app = FastAPI(
    title="RAG Backend API",
    description="Backend API exposing modular Retrieval-Augmented Generation services backed by FAISS vector index and Gemini 3.5 Flash-Lite LLM.",
    version="0.3.0",
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
    return {"status": "healthy", "service": "RAG Backend with FAISS Retriever and Gemini 3.5 Flash-Lite LLM"}


@app.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit user query to the RAG pipeline",
    description="Exposes end-to-end RAG question answering backed by FAISS vector retrieval and Gemini 3.5 Flash-Lite LLM.",
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
        # Run query through orchestrator using pre-initialized FAISSRetriever and Gemini LLM
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


@app.post(
    "/ask/stream",
    summary="Submit user query to the RAG pipeline with Server-Sent Events (SSE) streaming",
    description="Streams RAG tokens and source references as Server-Sent Events (SSE).",
)
async def ask_question_stream(request: AskRequest):
    """Processes user query and streams answer tokens and source documents as Server-Sent Events."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty or whitespace-only.",
        )

    async def event_generator():
        start_time = time.perf_counter()
        pipeline_result = await orchestrator.run_pipeline(query=request.query)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Stream sources event
        sources_payload = [
            {
                "document_id": ref.document_id,
                "text": ref.content,
                "metadata": {"chunk_id": ref.chunk_id, **ref.metadata},
            }
            for ref in pipeline_result.context
        ]
        yield f"event: sources\ndata: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

        # Stream answer tokens
        answer_text = pipeline_result.answer
        chunk_size = 12
        for i in range(0, len(answer_text), chunk_size):
            token_chunk = answer_text[i:i + chunk_size]
            yield f"event: token\ndata: {json.dumps({'content': token_chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        # Stream done event
        done_payload = {
            "is_grounded": pipeline_result.is_grounded,
            "is_refused": pipeline_result.is_refused,
            "refusal_reason": pipeline_result.refusal_reason,
            "latency_ms": latency_ms,
        }
        yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class TranslateRequest(BaseModel):
    text: str
    target_lang: str


class TranslateResponse(BaseModel):
    translated_text: str


@app.post(
    "/translate",
    response_model=TranslateResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate text to a target language",
    description="Translates the provided text into a target language using deep-translator (GoogleTranslator).",
)
async def translate_text(request: TranslateRequest):
    """Translates the provided text using deep-translator (GoogleTranslator). Fast, free, no LLM needed."""
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text to translate cannot be empty or whitespace-only.",
        )
    if not request.target_lang or not request.target_lang.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target language cannot be empty or whitespace-only.",
        )

    # Map human-readable language names to ISO 639-1 codes expected by GoogleTranslator
    LANG_NAME_TO_CODE = {
        "english": "en",
        "hindi": "hi",
        "bengali": "bn",
        "tamil": "ta",
        "telugu": "te",
        "marathi": "mr",
        "gujarati": "gu",
        "kannada": "kn",
        "malayalam": "ml",
        "punjabi": "pa",
        "urdu": "ur",
    }

    lang_key = request.target_lang.strip().lower()
    target_code = LANG_NAME_TO_CODE.get(lang_key, lang_key)  # fall back to raw value if already a code

    try:
        from deep_translator import GoogleTranslator

        def _translate():
            return GoogleTranslator(source="auto", target=target_code).translate(request.text)

        translated = await asyncio.to_thread(_translate)
        return TranslateResponse(translated_text=translated)
    except Exception as e:
        print(f"Error in translation endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while translating: {str(e)}",
        )
