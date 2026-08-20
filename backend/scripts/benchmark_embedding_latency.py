"""
backend/scripts/benchmark_embedding_latency.py
------------------------------------------------
Standalone latency benchmark for single-query embedding generation.

Requirements:
- Loads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` once on CUDA.
- Performs 1 warm-up embedding call.
- Executes 10 iterations of embedding for the target Hindi query.
- Reports per-iteration latencies and summary metrics (Mean, Median, P95, Min, Max).
- Does NOT load or touch any vector indexes.

Usage:
    python -m backend.scripts.benchmark_embedding_latency
"""

import argparse
import logging
import sys
import time
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
logger = logging.getLogger("benchmark_embedding_latency")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-query embedding latency on CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--query",
        default="निगम या कंपनी की परिभाषा क्या है?",
        help="Hindi query text to embed.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    from backend.embeddings.sentence_transformer import SentenceTransformerEmbedder

    logger.info("Initializing model '%s' on CUDA...", args.model_name)
    embedder = SentenceTransformerEmbedder(model_name=args.model_name, device="cuda")

    # 1. Warm-up run (not included in benchmark metrics)
    logger.info("Performing 1 warm-up embedding call...")
    warmup_start = time.perf_counter()
    _ = embedder.embed(args.query)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
    logger.info("Warm-up completed in %.2f ms", warmup_ms)

    # 2. Benchmark iterations
    logger.info("Running %d benchmark iterations...", args.iterations)
    latencies: list[float] = []

    for i in range(1, args.iterations + 1):
        t_start = time.perf_counter()
        vec = embedder.embed(args.query)
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(t_elapsed_ms)

    # 3. Calculate summary metrics
    lat_arr = np.array(latencies)
    mean_val = float(np.mean(lat_arr))
    median_val = float(np.median(lat_arr))
    p95_val = float(np.percentile(lat_arr, 95))
    min_val = float(np.min(lat_arr))
    max_val = float(np.max(lat_arr))

    query_preview = str(args.query).encode("utf-8", "replace").decode("utf-8")

    # 4. Print report
    print("\n" + "=" * 70)
    print("           SINGLE-QUERY EMBEDDING LATENCY BENCHMARK             ")
    print("=" * 70)
    print(f"Model Name:         {args.model_name}")
    print(f"Device:             CUDA")
    print(f"Query Text:         \"{query_preview}\"")
    print(f"Warm-up Latency:    {warmup_ms:.2f} ms")
    print("=" * 70)
    print("                 PER-ITERATION LATENCIES                        ")
    print("=" * 70)

    for idx, lat in enumerate(latencies, 1):
        print(f"Run #{idx:02d}:   {lat:8.2f} ms")

    print("=" * 70)
    print("                     SUMMARY METRICS                            ")
    print("=" * 70)
    print(f"Mean Latency:    {mean_val:8.2f} ms")
    print(f"Median Latency:  {median_val:8.2f} ms")
    print(f"P95 Latency:     {p95_val:8.2f} ms")
    print(f"Min Latency:     {min_val:8.2f} ms")
    print(f"Max Latency:     {max_val:8.2f} ms")
    print("=" * 70 + "\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
