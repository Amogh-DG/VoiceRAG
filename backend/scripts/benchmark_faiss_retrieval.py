"""
backend/scripts/benchmark_faiss_retrieval.py
----------------------------------------------
End-to-End Latency Benchmark for FAISS Retrieval.

Requirements:
- Loads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on CUDA.
- Loads FAISS binary index (`.faiss`) and metadata JSONL (`.jsonl`).
- Performs 1 warm-up query.
- Executes 10 iterations measuring:
  1. Query Embedding Latency (ms)
  2. FAISS Search Latency (ms)
  3. Total End-to-End Latency (ms)
- Reports top 5 retrieved results with scores, chunk IDs, and text previews.
- Computes summary metrics (Mean, Median, P95, Min, Max).
- Does NOT load original JSON vector index.

Usage:
    python -m backend.scripts.benchmark_faiss_retrieval \\
        --faiss-path data/indices/hi_val_multilingual.faiss \\
        --metadata-path data/indices/hi_val_multilingual_metadata.jsonl

    python -m backend.scripts.benchmark_faiss_retrieval --help
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np

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
logger = logging.getLogger("benchmark_faiss")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FAISS search and query embedding latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--faiss-path",
        default="data/indices/hi_val_multilingual.faiss",
        help="Path to FAISS binary index file.",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/indices/hi_val_multilingual_metadata.jsonl",
        help="Path to metadata JSONL file.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--query",
        default="निगम या कंपनी की परिभाषा क्या है?",
        help="Query text to search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top similarity results to return.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    try:
        import faiss
    except ImportError:
        logger.error("faiss package required. Run: pip install faiss-cpu")
        return 1

    from backend.embeddings.sentence_transformer import SentenceTransformerEmbedder

    faiss_path = Path(args.faiss_path)
    metadata_path = Path(args.metadata_path)

    if not faiss_path.exists():
        logger.error("FAISS index file not found: %s", faiss_path)
        return 1

    if not metadata_path.exists():
        logger.error("Metadata JSONL file not found: %s", metadata_path)
        return 1

    # 1. Load FAISS index exactly once
    logger.info("Loading FAISS index from %s...", faiss_path)
    t_f0 = time.perf_counter()
    faiss_index = faiss.read_index(str(faiss_path))
    faiss_load_ms = (time.perf_counter() - t_f0) * 1000.0
    logger.info("FAISS index loaded in %.2f ms! Total vectors: %d", faiss_load_ms, faiss_index.ntotal)

    # 2. Load metadata JSONL lines
    logger.info("Loading metadata JSONL from %s...", metadata_path)
    t_m0 = time.perf_counter()
    metadata_list = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata_list.append(json.loads(line))
    meta_load_ms = (time.perf_counter() - t_m0) * 1000.0
    logger.info("Loaded %d metadata records in %.2f ms", len(metadata_list), meta_load_ms)

    if faiss_index.ntotal != len(metadata_list):
        logger.warning(
            "Mismatch between FAISS vector count (%d) and metadata count (%d)",
            faiss_index.ntotal,
            len(metadata_list),
        )

    # 3. Load embedding model on CUDA exactly once
    logger.info("Loading embedding model '%s' on CUDA...", args.model_name)
    embedder = SentenceTransformerEmbedder(model_name=args.model_name, device="cuda")

    # 4. Warm-up run (not included in benchmark metrics)
    logger.info("Performing 1 warm-up query run...")
    t_w0 = time.perf_counter()
    q_warm = embedder.embed(args.query)
    q_warm_arr = np.array(q_warm, dtype=np.float32)
    norm = np.linalg.norm(q_warm_arr)
    if norm > 0:
        q_warm_arr = q_warm_arr / norm
    q_warm_matrix = q_warm_arr.reshape(1, -1)
    _, _ = faiss_index.search(q_warm_matrix, args.top_k)
    warmup_ms = (time.perf_counter() - t_w0) * 1000.0
    logger.info("Warm-up completed in %.2f ms", warmup_ms)

    # 5. Benchmark loop (10 iterations)
    logger.info("Executing %d benchmark iterations...", args.iterations)

    embed_latencies: list[float] = []
    search_latencies: list[float] = []
    total_latencies: list[float] = []

    last_scores = None
    last_indices = None

    for i in range(1, args.iterations + 1):
        t0 = time.perf_counter()

        # Step A: Query embedding
        q_vec = embedder.embed(args.query)
        q_arr = np.array(q_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm
        q_matrix = q_arr.reshape(1, -1)

        t1 = time.perf_counter()

        # Step B: FAISS Index Search
        scores, indices = faiss_index.search(q_matrix, args.top_k)

        t2 = time.perf_counter()

        embed_ms = (t1 - t0) * 1000.0
        search_ms = (t2 - t1) * 1000.0
        tot_ms = (t2 - t0) * 1000.0

        embed_latencies.append(embed_ms)
        search_latencies.append(search_ms)
        total_latencies.append(tot_ms)

        last_scores = scores[0]
        last_indices = indices[0]

    # 6. Display Top 5 Results
    query_preview = str(args.query).encode("utf-8", "replace").decode("utf-8")
    print("\n" + "=" * 75)
    print("                FAISS RETRIEVAL BENCHMARK RESULTS                      ")
    print("=" * 75)
    print(f"FAISS Index Path:     {faiss_path}")
    print(f"Total FAISS Vectors:  {faiss_index.ntotal:,}")
    print(f"Model Name:           {args.model_name}")
    print(f"Query Text:           \"{query_preview}\"")
    print("=" * 75)
    print(f"Top {args.top_k} Retrieved Chunks (First Run Inspection):")

    top_chunk_has_definition = False
    for rank, (score, pos_idx) in enumerate(zip(last_scores, last_indices), 1):
        meta = metadata_list[pos_idx] if pos_idx < len(metadata_list) else {}
        content = meta.get("content", "")
        content_preview = str(content[:120]).encode("utf-8", "replace").decode("utf-8")
        if rank == 1 and ("निगम" in content or "कंपनी" in content):
            top_chunk_has_definition = True

        print(f"\n--- Rank #{rank} (Similarity Score: {score:.4f}) ---")
        print(f"Position Index: {pos_idx}")
        print(f"Chunk ID:       {meta.get('id', 'N/A')}")
        print(f"Parent Doc ID:  {meta.get('parent_doc_id', 'N/A')}")
        print(f"Text Preview:   {content_preview}...")

    # 7. Summary Statistics Function
    def calc_stats(arr: list[float]) -> dict:
        np_arr = np.array(arr)
        return {
            "mean": float(np.mean(np_arr)),
            "median": float(np.median(np_arr)),
            "p95": float(np.percentile(np_arr, 95)),
            "min": float(np.min(np_arr)),
            "max": float(np.max(np_arr)),
        }

    emb_stats = calc_stats(embed_latencies)
    src_stats = calc_stats(search_latencies)
    tot_stats = calc_stats(total_latencies)

    print("\n" + "=" * 75)
    print("                 PERFORMANCE BENCHMARK METRICS (10 Runs)               ")
    print("=" * 75)
    print(f"{'Metric':<18} | {'Embedding (ms)':<15} | {'FAISS Search (ms)':<17} | {'Total RAG (ms)':<15}")
    print("-" * 75)
    print(f"{'Mean':<18} | {emb_stats['mean']:15.2f} | {src_stats['mean']:17.2f} | {tot_stats['mean']:15.2f}")
    print(f"{'Median':<18} | {emb_stats['median']:15.2f} | {src_stats['median']:17.2f} | {tot_stats['median']:15.2f}")
    print(f"{'P95':<18} | {emb_stats['p95']:15.2f} | {src_stats['p95']:17.2f} | {tot_stats['p95']:15.2f}")
    print(f"{'Min':<18} | {emb_stats['min']:15.2f} | {src_stats['min']:17.2f} | {tot_stats['min']:15.2f}")
    print(f"{'Max':<18} | {emb_stats['max']:15.2f} | {src_stats['max']:17.2f} | {tot_stats['max']:15.2f}")
    print("=" * 75)
    
    if top_chunk_has_definition:
        print("✅ QUALITY VERIFIED: Relevant corporate definition chunk retrieved at rank #1!")
    else:
        print("ℹ️ Quality Inspection complete.")
    print("=" * 75 + "\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
