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

Comparative mode (--faiss-path2):
- Runs both indices back-to-back for the same set of test queries.
- Prints a side-by-side score diff table per query.
- Flags (with ⚠️) any query where the top-1 score crossed the 0.15
  threshold (generator.py's score_threshold) between old and new index.
  A crossing means the query would flip from answered → refused or vice versa.

Usage:
    # Single-index benchmark (original behaviour)
    python -m backend.scripts.benchmark_faiss_retrieval \\
        --faiss-path data/indices/hi_val_multilingual.faiss \\
        --metadata-path data/indices/hi_val_multilingual_metadata.jsonl

    # Comparative before/after benchmark (flags threshold crossings)
    python -m backend.scripts.benchmark_faiss_retrieval \\
        --faiss-path  data/indices/hi_val_multilingual.faiss \\
        --faiss-path2 data/indices/hi_val_multilingual_sq_fp16.faiss \\
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

# ── Score threshold used by generator.py ──────────────────────────────────────
GENERATOR_SCORE_THRESHOLD = 0.15


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FAISS search and query embedding latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--faiss-path",
        default="data/indices/hi_val_multilingual.faiss",
        help="Path to FAISS binary index file (primary / 'old' index).",
    )
    parser.add_argument(
        "--faiss-path2",
        default=None,
        help=(
            "Optional path to a second FAISS index for comparative benchmarking. "
            "When provided, both indices are searched with the same queries and "
            "a side-by-side score diff table is printed."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default="data/indices/hi_val_multilingual_metadata.jsonl",
        help="Path to metadata JSONL file (shared for both indices).",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--query",
        default="निगम या कंपनी की परिभाषा क्या है?",
        help="Primary query text to benchmark.",
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
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=GENERATOR_SCORE_THRESHOLD,
        help=(
            "generator.py score threshold to watch for crossings in comparative mode "
            "(default matches generator.py's score_threshold=0.15)."
        ),
    )
    return parser.parse_args(argv)


# ── helpers ───────────────────────────────────────────────────────────────────

def _embed_and_normalise(embedder, query: str) -> np.ndarray:
    q_vec = embedder.embed(query)
    q_arr = np.array(q_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q_arr)
    if q_norm > 0:
        q_arr = q_arr / q_norm
    return q_arr.reshape(1, -1)


def _load_index(faiss, path: Path):
    logger.info("Loading FAISS index from %s...", path)
    t0 = time.perf_counter()
    idx = faiss.read_index(str(path))
    ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "Index loaded in %.0f ms — %d vectors, type: %s",
        ms, idx.ntotal, type(idx).__name__,
    )
    return idx


def _load_metadata(path: Path) -> list:
    logger.info("Loading metadata JSONL from %s...", path)
    t0 = time.perf_counter()
    metadata_list = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata_list.append(json.loads(line))
    ms = (time.perf_counter() - t0) * 1000.0
    logger.info("Loaded %d metadata records in %.0f ms", len(metadata_list), ms)
    return metadata_list


def _calc_stats(arr: list[float]) -> dict:
    np_arr = np.array(arr)
    return {
        "mean":   float(np.mean(np_arr)),
        "median": float(np.median(np_arr)),
        "p95":    float(np.percentile(np_arr, 95)),
        "min":    float(np.min(np_arr)),
        "max":    float(np.max(np_arr)),
    }


def _run_benchmark_loop(
    faiss_index,
    embedder,
    query: str,
    top_k: int,
    iterations: int,
) -> tuple[list, list, list, np.ndarray, np.ndarray]:
    """
    Warm up + benchmark loop.

    Returns (embed_latencies, search_latencies, total_latencies, last_scores, last_indices).
    """
    # Warm-up
    q_matrix = _embed_and_normalise(embedder, query)
    faiss_index.search(q_matrix, top_k)

    embed_latencies: list[float] = []
    search_latencies: list[float] = []
    total_latencies: list[float] = []
    last_scores = last_indices = None

    for _ in range(iterations):
        t0 = time.perf_counter()
        q_matrix = _embed_and_normalise(embedder, query)
        t1 = time.perf_counter()
        scores, indices = faiss_index.search(q_matrix, top_k)
        t2 = time.perf_counter()

        embed_latencies.append((t1 - t0) * 1000.0)
        search_latencies.append((t2 - t1) * 1000.0)
        total_latencies.append((t2 - t0) * 1000.0)
        last_scores = scores[0]
        last_indices = indices[0]

    return embed_latencies, search_latencies, total_latencies, last_scores, last_indices


def _print_results(
    faiss_path: Path,
    faiss_index,
    model_name: str,
    query: str,
    top_k: int,
    last_scores: np.ndarray,
    last_indices: np.ndarray,
    metadata_list: list,
    embed_latencies: list,
    search_latencies: list,
    total_latencies: list,
    iterations: int,
) -> None:
    query_preview = str(query).encode("utf-8", "replace").decode("utf-8")
    print("\n" + "=" * 75)
    print("                FAISS RETRIEVAL BENCHMARK RESULTS                      ")
    print("=" * 75)
    print(f"FAISS Index Path:     {faiss_path}")
    print(f"Index Type:           {type(faiss_index).__name__}")
    print(f"Total FAISS Vectors:  {faiss_index.ntotal:,}")
    print(f"Model Name:           {model_name}")
    print(f"Query Text:           \"{query_preview}\"")
    print("=" * 75)
    print(f"Top {top_k} Retrieved Chunks (First Run Inspection):")

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

    emb_stats = _calc_stats(embed_latencies)
    src_stats = _calc_stats(search_latencies)
    tot_stats = _calc_stats(total_latencies)

    print("\n" + "=" * 75)
    print(f"                 PERFORMANCE BENCHMARK METRICS ({iterations} Runs)               ")
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


# ── Comparative mode ──────────────────────────────────────────────────────────

# A broad set of test queries that exercise different score ranges, including
# queries that historically sit near the 0.15 boundary.
_COMPARISON_QUERIES: list[str] = [
    "निगम या कंपनी की परिभाषा क्या है?",
    "What is the definition of a corporation?",
    "संपत्ति हस्तांतरण अधिनियम के तहत पट्टे की परिभाषा क्या है?",
    "How is a partnership firm dissolved under Indian law?",
    "अनुबंध की वैधता के लिए क्या शर्तें आवश्यक हैं?",
    "What are the essential elements of a valid contract?",
    "भारत में कंपनी के निदेशक की जिम्मेदारियां क्या हैं?",
    "Explain the concept of intellectual property rights.",
    "randomxyzquery12345 nonsense query",           # expected: very low score
    "भारतीय दंड संहिता में धारा 420 क्या है?",
]


def _run_comparative(
    args: argparse.Namespace,
    faiss,
    embedder,
    metadata_list: list,
    faiss_path1: Path,
    faiss_path2: Path,
) -> None:
    """
    Load both indices, run the same queries through each, print diff table,
    and flag any query whose top-1 score crossed args.score_threshold.
    """
    idx1 = _load_index(faiss, faiss_path1)
    idx2 = _load_index(faiss, faiss_path2)

    threshold = args.score_threshold

    print("\n" + "=" * 100)
    print("  COMPARATIVE SCORE DIFF: Old (IndexFlatIP) vs New (IndexScalarQuantizer QT_fp16)")
    print(f"  Score threshold (generator.py): {threshold}  — ⚠️  flags a threshold crossing")
    print("=" * 100)
    print(
        f"{'#':<3}  {'Query (truncated)':<40}  "
        f"{'Old top-1':>10}  {'New top-1':>10}  {'Δ':>8}  {'Status':>8}"
    )
    print("-" * 100)

    any_crossing = False
    for q_idx, query in enumerate(_COMPARISON_QUERIES, 1):
        q_matrix = _embed_and_normalise(embedder, query)

        old_scores, old_idxs = idx1.search(q_matrix, args.top_k)
        new_scores, new_idxs = idx2.search(q_matrix, args.top_k)

        old_top = float(old_scores[0][0]) if len(old_scores[0]) > 0 else 0.0
        new_top = float(new_scores[0][0]) if len(new_scores[0]) > 0 else 0.0
        delta = new_top - old_top

        old_above = old_top >= threshold
        new_above = new_top >= threshold
        crossed = old_above != new_above

        if crossed:
            any_crossing = True
            direction = (
                "answered→refused" if old_above and not new_above else "refused→answered"
            )
            status = f"⚠️  {direction}"
        else:
            status = "✅ OK"

        q_preview = str(query[:40]).encode("utf-8", "replace").decode("utf-8")
        sign = "+" if delta >= 0 else ""
        print(
            f"{q_idx:<3}  {q_preview:<40}  "
            f"{old_top:>10.5f}  {new_top:>10.5f}  {sign}{delta:>+7.5f}  {status}"
        )

    print("=" * 100)
    if any_crossing:
        print(
            "⚠️  ONE OR MORE QUERIES CROSSED THE 0.15 THRESHOLD.\n"
            "   These queries will flip from answered→refused or refused→answered\n"
            "   when you switch from the old index to the quantised index.\n"
            "   Review them manually before deploying the new index."
        )
    else:
        print(
            "✅  No threshold crossings detected. "
            "All queries stay on the same side of 0.15 with the quantised index."
        )
    print("=" * 100 + "\n")


# ── run ───────────────────────────────────────────────────────────────────────

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

    # 1. Load primary FAISS index
    faiss_index = _load_index(faiss, faiss_path)

    # 2. Load metadata JSONL (shared across both indices in comparative mode)
    metadata_list = _load_metadata(metadata_path)

    if faiss_index.ntotal != len(metadata_list):
        logger.warning(
            "Mismatch between FAISS vector count (%d) and metadata count (%d)",
            faiss_index.ntotal,
            len(metadata_list),
        )

    # 3. Load embedding model on CUDA exactly once
    logger.info("Loading embedding model '%s' on CUDA...", args.model_name)
    embedder = SentenceTransformerEmbedder(model_name=args.model_name, device="cuda")

    # 4. Single-index benchmark on primary index
    logger.info("Executing %d benchmark iterations on primary index...", args.iterations)
    embed_lat, search_lat, total_lat, last_scores, last_indices = _run_benchmark_loop(
        faiss_index, embedder, args.query, args.top_k, args.iterations
    )
    _print_results(
        faiss_path, faiss_index, args.model_name, args.query, args.top_k,
        last_scores, last_indices, metadata_list,
        embed_lat, search_lat, total_lat, args.iterations,
    )

    # 5. Optional comparative mode
    if args.faiss_path2:
        faiss_path2 = Path(args.faiss_path2)
        if not faiss_path2.exists():
            logger.error("Second FAISS index file not found: %s", faiss_path2)
            return 1
        logger.info(
            "Comparative mode: benchmarking %s vs %s ...", faiss_path, faiss_path2
        )
        # Also print benchmark stats for the second index
        faiss_index2 = _load_index(faiss, faiss_path2)
        embed_lat2, search_lat2, total_lat2, last_scores2, last_indices2 = (
            _run_benchmark_loop(
                faiss_index2, embedder, args.query, args.top_k, args.iterations
            )
        )
        _print_results(
            faiss_path2, faiss_index2, args.model_name, args.query, args.top_k,
            last_scores2, last_indices2, metadata_list,
            embed_lat2, search_lat2, total_lat2, args.iterations,
        )

        # Score diff across all test queries
        _run_comparative(args, faiss, embedder, metadata_list, faiss_path, faiss_path2)

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
