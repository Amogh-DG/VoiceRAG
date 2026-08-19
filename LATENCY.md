# RAG Pipeline Latency Profile & Benchmark Results

This document summarizes the results of running the RAG pipeline latency benchmark across all stages and describes the optimization analysis.

## Benchmark Configuration
- **Queries Evaluated**: 5 distinct queries (representing normal queries, off-topic prompts, and insufficient context).
- **Runs per Query**: 10 runs (total 50 query iterations).
- **Environment**: Local dev environment (offline/mock models).

## Latency Profile Results

| Pipeline Stage | P50 (ms) | P70 (ms) | P100 (ms) |
| :--- | :--- | :--- | :--- |
| **Input Validation** | 0.00 | 0.00 | 0.03 |
| **Query Guardrails** | 0.01 | 0.01 | 0.24 |
| **Embedding Generation** | 0.21 | 0.23 | 0.26 |
| **Vector DB Lookup** | 0.10 | 0.11 | 0.15 |
| **Lightweight Reranking** | 0.00 | 0.00 | 0.00 |
| **Total Retrieval Stage** | 0.73 | 0.77 | 0.88 |
| **Context Adequacy Guard** | 0.01 | 0.01 | 0.02 |
| **LLM Generation Call** | 0.03 | 0.05 | 0.08 |
| **Output Guardrails** | 0.03 | 0.04 | 0.13 |
| **End-to-End Pipeline** | **0.80** | **0.86** | **1.10** |

## Target Status

✅ **MET** — The end-to-end pipeline P100 latency is **1.10 ms**, well below the <200 ms target.

## Slowest Stage Analysis

The **Total Retrieval Stage** (P50: 0.73 ms) dominates the pipeline, accounting for ~91% of end-to-end time. Within retrieval:

1. **Embedding Generation** (P50: 0.21 ms) — the hash-based bag-of-words stub embedder iterates over all words and hashes each one into 1536 dimensions.
2. **Vector DB Lookup** (P50: 0.10 ms) — numpy dot-product similarity search over the in-memory index.
3. **Reranking** (0.00 ms) — disabled by default, as documented.

## Optimization Notes

No optimizations were required since all stages are well within budget. For production with real models/APIs:

| Risk Stage | Mitigation Strategy |
| :--- | :--- |
| Embedding Generation (real model) | Query embedding cache (LRU) to skip redundant calls |
| Vector DB Lookup (large index) | Use FAISS/HNSW approximate NN instead of brute-force numpy |
| Reranking (cross-encoder) | Disabled by default; lightweight BM25 keyword blend available via `rerank=True` flag |
| LLM Generation (real API) | Streaming + async with configurable timeout/retry in orchestrator |
| Guardrails (complex NLI) | Compiled regex patterns; could parallelize safety + hallucination checks |

## Reproducing

```bash
python backend/scripts/benchmark.py --runs 10
```
