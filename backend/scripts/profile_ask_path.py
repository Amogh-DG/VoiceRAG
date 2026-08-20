"""
backend/scripts/profile_ask_path.py
-------------------------------------
Precise latency profiling of the POST /ask retrieval path.

Instruments every stage between raw HTTP request and final response:
  1.  Request handling / start
  2.  Query preprocessing (input validation + query guardrails)
  3.  FAISSRetriever.retrieve() — total wall-clock time
  4.  │─ Query embedding (inside retriever)
  5.  │─ FAISS IndexFlatIP search (inside retriever)
  6.  └─ Metadata lookup & result construction (inside retriever)
  7.  Context adequacy guardrail
  8.  LLM generation
  9.  Output guardrails
  10. PipelineOrchestrator overhead (total pipeline − sum of instrumented stages)
  11. Response serialization (TestClient overhead − pipeline total)
  12. Total API latency

Does NOT modify retrieval behaviour, regenerate embeddings, or touch indexes.
"""

import asyncio
import sys
import time
import statistics
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path setup — run from repo root                                              #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

# ── 1. Import the already-initialized singletons from main.py ──────────────
from backend.api.main import app, orchestrator, faiss_retriever

QUERY = "निगम या कंपनी की परिभाषा क्या है?"
N_WARMUP = 1
N_MEASURE = 5


# --------------------------------------------------------------------------- #
# Monkey-patch FAISSRetriever.search() to capture per-stage timings           #
# --------------------------------------------------------------------------- #
import numpy as np
_original_search = faiss_retriever.search.__func__   # unbound method


def _instrumented_search(self, query: str, top_k: int = 5):
    if not query or not query.strip():
        return []

    t_retriever_start = time.perf_counter()

    t0 = time.perf_counter()
    q_vec = self.embedder.embed(query)
    q_arr = np.array(q_vec, dtype=np.float32)
    norm = np.linalg.norm(q_arr)
    if norm > 0:
        q_arr = q_arr / norm
    q_matrix = q_arr.reshape(1, -1)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    scores, indices = self.faiss_index.search(q_matrix, top_k)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    results = []
    for score, pos_idx in zip(scores[0], indices[0]):
        if pos_idx < 0 or pos_idx >= len(self.metadata_list):
            continue
        meta_payload = self.metadata_list[pos_idx]
        doc_id = meta_payload.get("parent_doc_id") or meta_payload.get("document_id") or meta_payload.get("id")
        content = meta_payload.get("content") or meta_payload.get("text") or ""
        results.append({
            "score": float(score),
            "chunk_id": meta_payload.get("id", f"chunk_{pos_idx}"),
            "parent_doc_id": doc_id,
            "text": content,
            "metadata": meta_payload.get("metadata", {}),
        })
    t5 = time.perf_counter()

    self.last_timings_ms = {
        "query_embedding_ms":     (t1 - t0) * 1000.0,
        "faiss_search_ms":        (t3 - t2) * 1000.0,
        "metadata_lookup_ms":     (t5 - t4) * 1000.0,
        "total_faiss_retriever_ms": (t5 - t_retriever_start) * 1000.0,
    }
    return results


import types
faiss_retriever.search = types.MethodType(_instrumented_search, faiss_retriever)


# --------------------------------------------------------------------------- #
# Monkey-patch PipelineOrchestrator.run_pipeline() to capture stage timings   #
# --------------------------------------------------------------------------- #
_original_run_pipeline = orchestrator.run_pipeline.__func__

_last_pipeline_timings: dict = {}

async def _instrumented_run_pipeline(self, query: str, k: int = 5):
    global _last_pipeline_timings
    t_pipe_start = time.perf_counter()
    result = await _original_run_pipeline(self, query=query, k=k)
    t_pipe_end = time.perf_counter()
    _last_pipeline_timings["pipeline_wall_ms"] = (t_pipe_end - t_pipe_start) * 1000.0
    # Pull already-captured orchestrator stage timings from result metadata
    if hasattr(result, "metadata") and "timings_ms" in (result.metadata or {}):
        _last_pipeline_timings.update(result.metadata["timings_ms"])
    return result

orchestrator.run_pipeline = types.MethodType(_instrumented_run_pipeline, orchestrator)


# --------------------------------------------------------------------------- #
# Run profiling                                                                #
# --------------------------------------------------------------------------- #
def run_profile():
    client = TestClient(app)

    print("\n" + "=" * 72)
    print("        /ask PATH PROFILER  —  FAISSRetriever + PipelineOrchestrator")
    print("=" * 72)
    print(f"  Query : {QUERY}")
    print(f"  Model : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (CUDA)")
    print(f"  Index : hi_val_multilingual.faiss  (1,072,572 × 384-dim)")
    print(f"  Runs  : {N_WARMUP} warm-up + {N_MEASURE} measured")
    print("=" * 72)

    # ── Warm-up ─────────────────────────────────────────────────────────────
    print(f"\n[Warm-up ×{N_WARMUP}]")
    for i in range(N_WARMUP):
        t0 = time.perf_counter()
        r = client.post("/ask", json={"query": QUERY})
        print(f"  warm-up {i+1}: {(time.perf_counter()-t0)*1000:.1f} ms  status={r.status_code}")

    # ── Measured runs ────────────────────────────────────────────────────────
    print(f"\n[Measured ×{N_MEASURE}]")
    header = (
        f"{'Run':>4} │ {'API':>7} │ {'Pipeline':>8} │ {'Guardrails':>10} │"
        f" {'Retrieval':>9} │ {'Embedding':>9} │ {'FAISS':>8} │ {'Meta':>7} │"
        f" {'CtxGuard':>8} │ {'Gen':>8} │ {'OutGuard':>8} │ {'SzOverhead':>10}"
    )
    print(header)
    print("─" * len(header))

    records = []
    for i in range(N_MEASURE):
        global _last_pipeline_timings
        _last_pipeline_timings = {}

        t_api_start = time.perf_counter()
        response = client.post("/ask", json={"query": QUERY})
        t_api_end = time.perf_counter()

        api_total_ms = (t_api_end - t_api_start) * 1000.0

        pt = _last_pipeline_timings
        retriever_t = faiss_retriever.last_timings_ms if hasattr(faiss_retriever, "last_timings_ms") else {}

        # Stage timings from orchestrator metadata
        input_val_ms      = pt.get("input_validation_ms",  0.0)
        query_guard_ms    = pt.get("query_guardrails_ms",   0.0)
        retrieval_ms      = pt.get("retrieval_ms",          0.0)
        ctx_guard_ms      = pt.get("context_adequacy_ms",   0.0)
        gen_ms            = pt.get("generation_ms",         0.0)
        out_guard_ms      = pt.get("output_guardrails_ms",  0.0)
        pipeline_wall_ms  = pt.get("pipeline_wall_ms",      0.0)

        # Sub-stage timings from FAISSRetriever
        embed_ms          = retriever_t.get("query_embedding_ms",       0.0)
        faiss_ms          = retriever_t.get("faiss_search_ms",          0.0)
        meta_ms           = retriever_t.get("metadata_lookup_ms",       0.0)

        # Derived overhead buckets
        guardrails_total_ms = input_val_ms + query_guard_ms + out_guard_ms
        serialization_ms    = api_total_ms - pipeline_wall_ms  # TestClient/FastAPI overhead

        print(
            f"{i+1:>4} │ {api_total_ms:>7.1f} │ {pipeline_wall_ms:>8.1f} │"
            f" {guardrails_total_ms:>10.1f} │ {retrieval_ms:>9.1f} │"
            f" {embed_ms:>9.1f} │ {faiss_ms:>8.1f} │ {meta_ms:>7.2f} │"
            f" {ctx_guard_ms:>8.2f} │ {gen_ms:>8.1f} │ {out_guard_ms:>8.2f} │"
            f" {serialization_ms:>10.1f}"
        )

        records.append({
            "api_total_ms": api_total_ms,
            "pipeline_wall_ms": pipeline_wall_ms,
            "guardrails_total_ms": guardrails_total_ms,
            "retrieval_ms": retrieval_ms,
            "embed_ms": embed_ms,
            "faiss_ms": faiss_ms,
            "meta_ms": meta_ms,
            "ctx_guard_ms": ctx_guard_ms,
            "gen_ms": gen_ms,
            "out_guard_ms": out_guard_ms,
            "serialization_ms": serialization_ms,
        })

    # ── Summary statistics ────────────────────────────────────────────────────
    def stat(key):
        vals = [r[key] for r in records]
        return dict(
            mean=statistics.mean(vals),
            median=statistics.median(vals),
            p95=sorted(vals)[int(0.95 * len(vals)) - 1] if len(vals) >= 2 else vals[-1],
            min=min(vals),
            max=max(vals),
        )

    print("\n" + "=" * 72)
    print("  SUMMARY  (all times in ms)")
    print("=" * 72)
    rows = [
        ("1. Request → API start",    "api_total_ms"),
        ("  2. Pipeline total",        "pipeline_wall_ms"),
        ("  3. Guardrails (in+q+out)", "guardrails_total_ms"),
        ("  4. FAISSRetriever.retrieve", "retrieval_ms"),
        ("    5. Query embedding",     "embed_ms"),
        ("    6. FAISS search",        "faiss_ms"),
        ("    7. Metadata lookup",     "meta_ms"),
        ("  8. Context guard",         "ctx_guard_ms"),
        ("  9. LLM generation",        "gen_ms"),
        ("  10. Output guard",         "out_guard_ms"),
        ("  11. Serialization overhead", "serialization_ms"),
    ]
    print(f"  {'Stage':<35} {'Mean':>8} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 77)
    for label, key in rows:
        s = stat(key)
        print(f"  {label:<35} {s['mean']:>8.1f} {s['median']:>8.1f} {s['p95']:>8.1f} {s['min']:>8.1f} {s['max']:>8.1f}")

    print("=" * 72)

    # ── Gap analysis ─────────────────────────────────────────────────────────
    first = records[0]
    retriever_internal = first["embed_ms"] + first["faiss_ms"] + first["meta_ms"]
    retriever_overhead = first["retrieval_ms"] - retriever_internal

    print("\n  GAP ANALYSIS (first measured run)")
    print(f"  Standalone FAISS benchmark (median):    ~177.9 ms total")
    print(f"  FAISSRetriever.retrieve() wall-clock:    {first['retrieval_ms']:.1f} ms")
    print(f"  ├─ Query embedding:                      {first['embed_ms']:.1f} ms")
    print(f"  ├─ FAISS IndexFlatIP search:             {first['faiss_ms']:.1f} ms")
    print(f"  ├─ Metadata lookup:                      {first['meta_ms']:.2f} ms")
    print(f"  └─ Retriever call overhead (asyncio):    {retriever_overhead:.1f} ms")
    print(f"  Guardrails total:                        {first['guardrails_total_ms']:.1f} ms")
    print(f"  Context adequacy guard:                  {first['ctx_guard_ms']:.2f} ms")
    print(f"  LLM generation (mock):                   {first['gen_ms']:.1f} ms")
    print(f"  Pipeline overhead (pipeline − stages):   {first['pipeline_wall_ms'] - first['retrieval_ms']:.1f} ms")
    print(f"  Serialization/TestClient overhead:       {first['serialization_ms']:.1f} ms")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total API latency:                       {first['api_total_ms']:.1f} ms")
    print(f"  Δ vs standalone benchmark:               +{first['api_total_ms'] - 177.9:.1f} ms")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run_profile()
