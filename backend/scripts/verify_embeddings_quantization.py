"""
backend/scripts/verify_embeddings_quantization.py
--------------------------------------------------
Verification script for Change 3: dynamic quantization of the embedding model.

Checks:
1. Cosine similarity of embedding vectors between original and quantized models.
2. Downstream FAISS top score threshold crossing (0.15) for 10 test queries.
3. Before/after process RSS at idle and after first query.
4. Before/after single-query embedding latency.
"""

import time
import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch
import psutil
from sentence_transformers import SentenceTransformer

# Reconfigure stdout for UTF-8 output to handle Hindi characters cleanly
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("verify_quantization")

TEST_QUERIES = [
    "निगम या कंपनी की परिभाषा क्या है?",
    "What is the definition of a corporation?",
    "संपत्ति हस्तांतरण अधिनियम के तहत पट्टे की परिभाषा क्या है?",
    "How is a partnership firm dissolved under Indian law?",
    "अनुबंध की वैधता के लिए क्या शर्तें आवश्यक हैं?",
    "What are the essential elements of a valid contract?",
    "भारत में company के director की responsibilities क्या हैं?",
    "Explain the concept of intellectual property rights.",
    "randomxyzquery12345 nonsense query",
    "भारतीय दंड संहिता में धारा 420 क्या है?",
]

def get_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def run_verification():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    faiss_path = Path("data/indices/hi_val_multilingual.faiss")
    
    import faiss
    if not faiss_path.exists():
        # Try new quantized index path if original is missing
        faiss_path = Path("data/indices/hi_val_multilingual_sq_fp16.faiss")
    
    if not faiss_path.exists():
        logger.error("No FAISS index found for downstream checking!")
        sys.exit(1)
        
    logger.info("Loading FAISS index from %s...", faiss_path)
    faiss_index = faiss.read_index(str(faiss_path))
    
    # ── Phase 1: Original Model (running on CPU to isolate quantization effect) ──
    logger.info("--- PHASE 1: Original Model (float32) ---")
    rss_start_orig = get_rss_mb()
    
    t_orig_load_start = time.perf_counter()
    orig_model = SentenceTransformer(model_name, device="cpu")
    orig_load_time = time.perf_counter() - t_orig_load_start
    
    rss_idle_orig = get_rss_mb()
    logger.info(f"Original model loaded in {orig_load_time:.2f}s (RSS: {rss_idle_orig:.1f} MB)")
    
    # Run queries and measure original times
    orig_embeddings = []
    orig_latencies = []
    
    # Warmup
    _ = orig_model.encode(TEST_QUERIES[0], convert_to_numpy=True, normalize_embeddings=True)
    
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        emb = orig_model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
        orig_latencies.append((time.perf_counter() - t0) * 1000.0)
        orig_embeddings.append(emb)
        
    rss_post_query_orig = get_rss_mb()
    
    # Free original model to isolate memory
    del orig_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # ── Phase 2: Quantized Model (int8, runs on CPU) ──
    logger.info("--- PHASE 2: Quantized Model (int8) ---")
    rss_start_quant = get_rss_mb()
    
    t_quant_load_start = time.perf_counter()
    quant_model_raw = SentenceTransformer(model_name, device="cpu")
    # Apply dynamic quantization targeting torch.nn.Linear layers, dtype=torch.qint8
    quant_model = torch.quantization.quantize_dynamic(
        quant_model_raw, {torch.nn.Linear}, dtype=torch.qint8
    )
    quant_load_time = time.perf_counter() - t_quant_load_start
    
    rss_idle_quant = get_rss_mb()
    logger.info(f"Quantized model loaded + quantized in {quant_load_time:.2f}s (RSS: {rss_idle_quant:.1f} MB)")
    
    # Run queries and measure quantized times
    quant_embeddings = []
    quant_latencies = []
    
    # Warmup
    _ = quant_model.encode(TEST_QUERIES[0], convert_to_numpy=True, normalize_embeddings=True)
    
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        emb = quant_model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
        quant_latencies.append((time.perf_counter() - t0) * 1000.0)
        quant_embeddings.append(emb)
        
    rss_post_query_quant = get_rss_mb()
    
    # ── Phase 3: Analysis and Comparison ──
    print("\n" + "=" * 105)
    print("                      EMBEDDING DYNAMIC QUANTIZATION AUDIT REPORT                      ")
    print("=" * 105)
    print(f"{'#':<3} {'Query (truncated)':<35} {'Cos Sim':<10} {'Old Top Score':<15} {'New Top Score':<15} {'Cross 0.15?':<12} {'Old Lat (ms)':<12} {'New Lat (ms)':<12}")
    print("-" * 105)
    
    threshold = 0.15
    any_crossing = False
    
    for idx, query in enumerate(TEST_QUERIES):
        v_orig = orig_embeddings[idx]
        v_quant = quant_embeddings[idx]
        
        # 1. Cosine similarity
        sim = cosine_similarity(v_orig, v_quant)
        sim_status = "✅" if sim >= 0.98 else "⚠️ BELOW 0.98"
        
        # 2. Downstream FAISS top score
        q_matrix_orig = v_orig.reshape(1, -1).astype(np.float32)
        q_matrix_quant = v_quant.reshape(1, -1).astype(np.float32)
        
        scores_orig, _ = faiss_index.search(q_matrix_orig, 1)
        scores_quant, _ = faiss_index.search(q_matrix_quant, 1)
        
        score_orig = float(scores_orig[0][0])
        score_quant = float(scores_quant[0][0])
        
        old_above = score_orig >= threshold
        new_above = score_quant >= threshold
        crossed = old_above != new_above
        
        cross_status = "✅ OK"
        if crossed:
            any_crossing = True
            direction = "answered→refused" if old_above and not new_above else "refused→answered"
            cross_status = f"⚠️ {direction}"
            
        q_preview = query[:32] + "..." if len(query) > 32 else query
        print(f"{idx+1:<3} {q_preview:<35} {sim:<10.5f} {score_orig:<15.4f} {score_quant:<15.4f} {cross_status:<12} {orig_latencies[idx]:<12.1f} {quant_latencies[idx]:<12.1f}")
        
    print("=" * 105)
    
    # Memory and Latency Summaries
    orig_mem_growth = rss_post_query_orig - rss_idle_orig
    quant_mem_growth = rss_post_query_quant - rss_idle_quant
    
    print("\nRESOURCE & LATENCY SUMMARY:")
    print(f"Original Model Load Time:      {orig_load_time:.2f} s")
    print(f"Quantized Model Load + Quant:  {quant_load_time:.2f} s")
    print("-" * 50)
    print(f"Original Process RSS at Idle:   {rss_idle_orig:.1f} MB")
    print(f"Quantized Process RSS at Idle:  {rss_idle_quant:.1f} MB")
    print(f"Memory Saved at Idle:          {rss_idle_orig - rss_idle_quant:.1f} MB")
    print("-" * 50)
    print(f"Original RSS after Query:      {rss_post_query_orig:.1f} MB (Growth: +{orig_mem_growth:.1f} MB)")
    print(f"Quantized RSS after Query:     {rss_post_query_quant:.1f} MB (Growth: +{quant_mem_growth:.1f} MB)")
    print(f"Memory Saved after Query:      {rss_post_query_orig - rss_post_query_quant:.1f} MB")
    print("-" * 50)
    print(f"Original Avg Latency:          {np.mean(orig_latencies):.1f} ms")
    print(f"Quantized Avg Latency:         {np.mean(quant_latencies):.1f} ms")
    print(f"Latency Difference:            {np.mean(quant_latencies) - np.mean(orig_latencies):+.1f} ms")
    print("=" * 105 + "\n")
    
    if any_crossing:
        print("⚠️  Warning: downstream FAISS query score threshold crossings detected!")
    else:
        print("✅  No downstream FAISS query score threshold crossings detected.")
        
    # Check if cosine similarity is high enough
    all_sim_high = all(cosine_similarity(orig_embeddings[i], quant_embeddings[i]) >= 0.98 for i in range(len(TEST_QUERIES)))
    if all_sim_high:
        print("✅  All embedding vectors show cosine similarity >= 0.98 between float32 and quantized int8.")
    else:
        print("⚠️  Some embedding vectors show cosine similarity below 0.98!")

if __name__ == "__main__":
    run_verification()
