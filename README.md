# RAG Backend Scaffolding

This project forms the backend layer for a Retrieval-Augmented Generation (RAG) system built around the [AI4Bharat MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset — the original MS MARCO QA dataset translated into 14 Indic languages.

The backend exposes a FastAPI API for querying the system and uses swappable interfaces for dataset ingestion, document chunking, embedding generation, vector DB storage/retrieval, generation, and orchestration.

## Project Structure

- `backend/`
  - `api/`         – FastAPI server (`POST /ask`, `GET /`)
  - `common/`      – Stable interfaces (ABCs) + shared Pydantic schemas
  - `ingestion/`   – Dataset loading, normalisation (`MSMarcoXILoader`)
  - `chunking/`    – Three chunking strategies + `get_chunker()` factory
  - `embeddings/`  – Embedding model wrappers
  - `retrieval/`   – Vector DB integrations
  - `generation/`  – LLM invocation, grounding, refusal
  - `harness/`     – Pipeline orchestration
  - `scripts/`     – `ingest.py`, `chunk.py`, `benchmark.py`
- `tests/`         – pytest unit + integration tests (no network calls)

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi`, `uvicorn` | HTTP server |
| `pydantic` | Request/response schemas |
| `datasets`, `pyarrow` | HF dataset metadata / local parquet loading |
| **`duckdb`** | Remote parquet reading for MSMARCO-XI (see note below) |
| `python-dotenv` | `.env` configuration |

### Why DuckDB for dataset loading?

The MSMARCO-XI parquet files store the `passages` column as a nested struct
containing list-typed sub-columns (`Translated_passages`, `English_passages`,
`is_selected`). The `datasets` library's native streaming reader uses Arrow's
C++ parquet backend, which raises:

```
ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs
```

This is an upstream limitation with no fix available in current pyarrow/datasets
versions. **DuckDB ships its own independent parquet reader** that handles nested
list columns correctly, so `MSMarcoXILoader._iter_hf_streaming()` uses DuckDB +
the `httpfs` extension to read remote parquet files over HTTPS. Local `.parquet`
and `.jsonl` files still use the standard Python path.

---

## Setup & Installation

### Prerequisites
- Python 3.10+

### Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### Install

```bash
pip install -e ".[dev]"
```

### Configure

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

---

## Running the Pipeline

### 1. Ingest dataset → Document JSONL

```bash
# Stream 500 rows of Hindi training data from HF Hub
python -m backend.scripts.ingest \
    --language hi \
    --split train \
    --max-rows 500 \
    --output data/processed/hi_train.jsonl

# All supported languages: as bn gu hi kn ml mr ne or pa sa ta te ur
```

### 2. Chunk documents → TextChunk JSONL

```bash
# Sentence-boundary chunking (recommended)
python -m backend.scripts.chunk \
    --input  data/processed/hi_train.jsonl \
    --output data/chunks/hi_train_sentence.jsonl \
    --strategy sentence \
    --max-chars 500

# Fixed-size + overlap (baseline)
python -m backend.scripts.chunk \
    --input  data/processed/hi_train.jsonl \
    --output data/chunks/hi_train_fixed.jsonl \
    --strategy fixed_overlap \
    --chunk-size 500 --overlap 50

# Metadata-aware (adapts by query_type and language)
python -m backend.scripts.chunk \
    --input  data/processed/hi_train.jsonl \
    --output data/chunks/hi_train_meta.jsonl \
    --strategy metadata_aware
```

### 3. Embed chunks → Index into Vector DB

```bash
# Embed text chunks and index into vector database
python -m backend.scripts.embed_and_index \
    --input data/chunks/hi_train_sentence.jsonl \
    --output-db data/vector_db/hi_index.json \
    --embedder mock \
    --batch-size 32
```

### 4. Run the API server

```bash
python -m uvicorn backend.api.main:app --reload
```

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Embeddings & Vector DB Choices (Latency Rationale)

To satisfy our hard **<200ms end-to-end `/ask` latency target**:

1. **Embeddings (`IEmbedder`)**:
   - **`SentenceTransformerEmbedder`** (`all-MiniLM-L6-v2` / `paraphrase-multilingual-MiniLM-L12-v2`):
     Runs locally on CPU with **~5–25ms** inference latency per query. Zero external API calls or network hops.
   - **`MockEmbedder`**: SHA-256 deterministic pseudo-random embedder for instant unit testing (<1ms).

2. **Vector DB (`IVectorDB`)**:
   - **`InMemoryVectorDB`**:
     NumPy dot product cosine similarity with JSON disk persistence.
     Provides sub-millisecond (**<1ms**) top-k similarity search over tens of thousands of vectors without requiring an external database process or native binary installation overhead.

3. **Retriever (`IRetriever`) & Reranking**:
   - **`retrieve(query, top_k=5, rerank=False)`**:
     Public contract returning `List[RetrievedChunk]`.
   - **Reranking Flag**: Gated behind `rerank=False` by default. Heavy cross-encoders add 150-300ms latency which risks exceeding our <200ms target. Setting `rerank=True` activates a lightweight lexical keyword overlap adjustment.

4. **Generation (`IGenerator`) & Refusal Guardrails**:
   - **`generate(question, retrieved_chunks)`**:
     Constructs a strictly grounded prompt instructing the LLM to answer only using context passages.
   - **Structured LLM Output**:
     Uses `LLMGenerationOutput` schema: `{ "answer": str, "grounded": bool, "insufficient": bool }`.
   - **Refusal / Insufficient-Information Path**:
     Triggered when context is empty, max score is below configurable `score_threshold` (default `0.15`), or the LLM output indicates `insufficient=True`. Returns `GenerationResponse(is_refused=True, is_grounded=False)`.

5. **Harness Orchestrator (`PipelineOrchestrator`) & Production Guardrails**:
   - **`run_pipeline(query, k=5)`**:
     Coordinates input validation, 5 guardrail evaluations, retrieval & generation timeouts/retries, and returns structured `AskResult`.
   - **Guardrail Taxonomy**:
     - `input_validation`: Sanitises query, rejects non-string, empty, or >2000 char queries (`invalid_input`).
     - `off_topic`: Detects prompt injections and off-topic requests (`off_topic`).
     - `query_safety` / `answer_safety`: Rejects unsafe/harmful content (`safety_violation`).
     - `insufficient_context`: Surfaces missing or low-scoring context (<threshold) (`insufficient_context`).
     - `hallucination`: NLI-style n-gram overlap check verifying answer claims against context (`hallucination_detected`).

---

## Retriever Public Contract

```python
retrieve(query: str, top_k: int = 5, filters: dict = None, rerank: bool = False) -> List[RetrievedChunk]
```

Every returned `RetrievedChunk` conforms strictly to this schema:
```json
{
  "text":        "verbatim text content",
  "score":       0.92345,
  "document_id": "doc123",
  "metadata":    { "chunk_strategy": "sentence", "..." }
}
```

## Chunk Schema (stable cross-team contract)

Every chunking strategy produces JSONL with this exact schema. **Do not change field names.**

```json
{
  "text":        "verbatim chunk text",
  "document_id": "parent document id",
  "metadata": {
    "chunk_strategy": "sentence",
    "chunk_index":    0,
    "..."
  }
}
```

## API Endpoint Design Decisions

The RAG backend exposes a single primary endpoint: `POST /ask`.

### Request Schema
```json
{
  "query": "What is the capital of India?"
}
```

### Response Schema
```json
{
  "answer": "...",
  "sources": [
    {
      "document_id": "abc",
      "text": "...",
      "metadata": {}
    }
  ],
  "latency_ms": 142.0
}
```

### HTTP Status Code Strategy
1. **Input Schema Validation (HTTP 400 Bad Request)**: 
   If the incoming request is malformed (e.g. invalid JSON) or does not contain the required `query` field, or if the `query` field is an empty string/whitespace-only, the API returns `HTTP 400 Bad Request`. This explicitly signals client integration errors.
   
2. **Content Policy & Guardrails Refusals (HTTP 200 OK)**:
   If a query is rejected due to active safety checks, off-topic detection, or low-scoring/insufficient context, the API returns a structured success payload (`HTTP 200 OK`) where the `answer` field contains the refusal/error message, `sources` is empty or low-scoring sources, and `latency_ms` is populated. This prevents the frontend/eval teams from having to implement custom exception handling or special-casing HTTP non-2xx status codes for content policy violations.

---

## Running Tests

```bash
pytest
```


