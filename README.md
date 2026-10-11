# VoiceRAG

**Ask questions out loud in your own language and get answers grounded in real sources.**

VoiceRAG is a voice-first, multilingual question-answering system for Indian languages. You speak a question, it is transcribed, relevant passages are retrieved from a vector index, and an LLM answers using only those passages, with citations. When the evidence is weak, it says so instead of guessing.

> **Built for:** HackerHouseGoa
---

## Features

- **Voice input** in the browser, with automatic language detection (Sarvam AI `saaras:v3`), plus a typed-question fallback
- **Retrieval-augmented answers** over AI4Bharat's MSMARCO-XI, the MS MARCO dataset translated into 14 Indic languages
- **Cited sources** shown with every answer, plus end-to-end latency
- **Refuses instead of hallucinating** when retrieved context is empty or low-scoring
- **Guardrails** for input validation, prompt injection and off-topic queries, unsafe content, insufficient context, and hallucination
- **Answer translation** into English and 10 Indian languages
- **Smart retry:** a failed transcription or failed question can be retried without re-recording
- **Fast retrieval path:** a scalar-quantised FAISS index with on-demand SQLite metadata lookups, aimed at a sub-200 ms `/ask` target

## How it works

```mermaid
flowchart LR
    A["Mic audio (MediaRecorder)"] --> B["Sarvam AI speech-to-text"]
    B --> C["POST /ask (FastAPI)"]
    C --> D["Input guardrails"]
    D --> E["Embed query (multilingual MiniLM)"]
    E --> F["FAISS top-k search"]
    F --> G["Gemini: grounded JSON answer"]
    G --> H["Output guardrails"]
    H --> I["Answer + cited sources"]
```

1. The React frontend records audio and sends it to Sarvam AI for transcription. Recordings are limited to 30 seconds by the Sarvam REST API.
2. The transcript is sent to the FastAPI backend (`POST /ask`).
3. The backend embeds the query with `paraphrase-multilingual-MiniLM-L12-v2` and searches the FAISS index.
4. Gemini is prompted to answer **only** from the retrieved passages and returns structured JSON: `{ answer, grounded, insufficient }`.
5. Guardrails check the result. If the context is empty, scores below the threshold (default `0.15`), or the model flags it as insufficient, the system returns a refusal.

## Tech stack

| Layer | Tools |
|-------|-------|
| Frontend | React 19, TypeScript, Vite, Vitest |
| Speech-to-text | Sarvam AI (`saaras:v3`) |
| Backend | Python, FastAPI, Pydantic |
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector search | FAISS (scalar-quantised fp16 index), SQLite for chunk metadata |
| Generation | Google Gemini (`gemini-3.5-flash-lite`) via `google-genai` |
| Data | AI4Bharat MSMARCO-XI, loaded with DuckDB |
| Translation | deep-translator (Google Translate) |

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `POST /ask` | Body `{ "query": "..." }`. Returns `{ answer, sources[], latency_ms }` |
| `POST /ask/stream` | Same pipeline, streamed as Server-Sent Events (`sources`, `token`, `done`) |
| `POST /translate` | Body `{ "text": "...", "target_lang": "Hindi" }`. Returns `{ translated_text }` |

Status codes: malformed or empty requests return `400`. Guardrail refusals (off-topic, unsafe, insufficient context) return `200` with the refusal message in `answer`, so clients don't need special handling for policy outcomes.

Interactive docs are available at `http://127.0.0.1:8000/docs` when the server is running.

## Project structure

```
backend/
  api/          FastAPI app (/ask, /ask/stream, /translate)
  common/       Interfaces (ABCs) and shared Pydantic schemas
  ingestion/    MSMARCO-XI loader and normaliser
  chunking/     Sentence, fixed-overlap and metadata-aware chunkers
  embeddings/   Sentence-transformer and mock embedders
  retrieval/    FAISS retriever, in-memory retriever
  generation/   Gemini client, grounded prompting, refusal logic
  harness/      Pipeline orchestrator and guardrails
  scripts/      Ingest, chunk, embed, index, benchmark and diagnostic tools
src/
  components/   VoiceQA, RecordButton, AnswerView, SourcesList, ...
  stt/          Sarvam speech-to-text client
  api/          Typed client for the backend
tests/          pytest unit and integration tests
```

## Getting started

### Prerequisites

- Python 3.10+ and Node.js 18+
- A [Gemini API key](https://aistudio.google.com/) (without one, the backend falls back to a mock LLM)
- A [Sarvam AI API key](https://dashboard.sarvam.ai/) for speech-to-text
- An NVIDIA GPU with CUDA, **or** change `device="cuda"` to `device="cpu"` in `backend/api/main.py`

### 1. Backend

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env             # then add GEMINI_API_KEY
```

### 2. Build the search index

The index files are generated artifacts and are not committed (`data/` and `*.faiss` are git-ignored). [Add a download link here if you host a prebuilt index.]

To build it yourself, run the pipeline in order. Each script supports `--help` for its full options.

```bash
# 1. Ingest a split of MSMARCO-XI into documents
python -m backend.scripts.ingest --language hi --split validation --output data/processed/hi_val.jsonl

# 2. Chunk documents (sentence-boundary chunking recommended)
python -m backend.scripts.chunk --input data/processed/hi_val.jsonl \
    --output data/chunks/hi_val_sentence.jsonl --strategy sentence --max-chars 500

# 3. Embed chunks and build the vector index
python -m backend.scripts.embed_and_index --input data/chunks/hi_val_sentence.jsonl \
    --output-db data/indices/hi_val_multilingual_index.json --embedder sentence_transformer

# 4. Convert to FAISS
python -m backend.scripts.convert_to_faiss \
    --input-index data/indices/hi_val_multilingual_index.json \
    --output-faiss data/indices/hi_val_multilingual.faiss \
    --output-metadata data/indices/hi_val_multilingual_metadata.jsonl

# 5. Optional: shrink RAM with fp16 quantisation and move metadata to SQLite
python -m backend.scripts.quantize_faiss_index --help
python -m backend.scripts.migrate_metadata_to_sqlite --help
```

The API loads `data/indices/hi_val_multilingual*.faiss` and its metadata at startup, so the shipped configuration searches the **Hindi validation split**. Other languages can be ingested with the same scripts (`as bn gu hi kn ml mr ne or pa sa ta te ur`).

### 3. Run the API

```bash
python -m uvicorn backend.api.main:app --reload
```

### 4. Frontend

```bash
npm install
cp .env.example .env     # set VITE_SARVAM_API_KEY (and VITE_API_BASE_URL if not localhost:8000)
npm run dev
```

Open the URL Vite prints, allow microphone access, and ask a question.

## Testing

```bash
pytest          # backend: ingestion, chunking, retrieval, generation, guardrails, API
npm test        # frontend: API client, STT client, VoiceQA component
npm run lint
```

## Performance notes

- Target: under 200 ms end-to-end for `/ask`.
- Exhaustive FAISS search over roughly 1.07M 384-dimension vectors was benchmarked at about 154 ms on the development machine. The fp16 scalar-quantised index has similar latency with about half the RAM. [Re-run `backend/scripts/benchmark_faiss_retrieval.py` and add your own numbers.]
- Cross-encoder reranking is off by default because it adds 150 to 300 ms. A lightweight keyword-overlap rerank is available with `rerank=True`.
- Benchmark and profiling tools live in `backend/scripts/` (`benchmark.py`, `benchmark_embedding_latency.py`, `profile_ask_path.py`, `measure_ram.py`).

## Known limitations

- Speech input is capped at 30 seconds by the Sarvam REST endpoint.
- The Sarvam key is read in the browser through a `VITE_` variable, so it is visible in the client bundle. For a public deployment, proxy transcription through the backend.
- The shipped index covers Hindi only, although ingestion supports 14 languages.
- The backend currently assumes a CUDA GPU unless you edit the device setting.

## Acknowledgements

[AI4Bharat](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) for MSMARCO-XI, [Sarvam AI](https://www.sarvam.ai/) for speech-to-text, Google for Gemini, and the FAISS and sentence-transformers communities.

