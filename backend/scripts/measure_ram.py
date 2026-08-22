"""
backend/scripts/measure_ram.py
--------------------------------
Before/after RSS measurement for FAISS + metadata loading.

Captures process RSS (Resident Set Size) at two stages:
  1. ``idle``  — after the FAISSRetriever is fully initialised (index + metadata loaded,
                  embedding model warm), but before any query has been served.
  2. ``query`` — immediately after the first real FAISS search has completed.

Modes
─────
  --mode old   Load the original IndexFlatIP (.faiss) + JSONL metadata list
               (current production behaviour).
  --mode new   Load the quantised IndexScalarQuantizer (_sq_fp16.faiss) + SQLite
               metadata DB (post-migration behaviour).

Requirements
────────────
  pip install psutil

Usage
─────
    # Before migration
    python -m backend.scripts.measure_ram --mode old

    # After migration
    python -m backend.scripts.measure_ram --mode new

    # Full help
    python -m backend.scripts.measure_ram --help
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── stdout recoding for Windows consoles ──────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("measure_ram")


# ── helpers ───────────────────────────────────────────────────────────────────

def rss_mb() -> float:
    """Return current process RSS in megabytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:
        logger.error(
            "psutil is required for RAM measurement. "
            "Install with: pip install psutil"
        )
        sys.exit(1)


def fmt_mb(mb: float) -> str:
    return f"{mb:,.1f} MB"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure process RSS before/after FAISSRetriever initialisation and query.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["old", "new"],
        required=True,
        help=(
            "old = IndexFlatIP + JSONL list (current behaviour). "
            "new = IndexScalarQuantizer(QT_fp16) + SQLite metadata (post-migration)."
        ),
    )
    parser.add_argument(
        "--faiss-path-old",
        default="data/indices/hi_val_multilingual.faiss",
        help="Path to the original IndexFlatIP FAISS file (used in --mode old).",
    )
    parser.add_argument(
        "--faiss-path-new",
        default="data/indices/hi_val_multilingual_sq_fp16.faiss",
        help="Path to the quantised FAISS file (used in --mode new).",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/indices/hi_val_multilingual_metadata.jsonl",
        help="Path to metadata JSONL (used in --mode old).",
    )
    parser.add_argument(
        "--metadata-db",
        default="data/indices/hi_val_multilingual_metadata.db",
        help="Path to metadata SQLite DB (used in --mode new).",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for the embedding model (cuda or cpu).",
    )
    parser.add_argument(
        "--query",
        default="निगम या कंपनी की परिभाषा क्या है?",
        help="Query text used for the first-query RAM measurement.",
    )
    return parser.parse_args(argv)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    measurements: list[tuple[str, float]] = []

    baseline = rss_mb()
    measurements.append(("baseline (process start)", baseline))
    logger.info("Baseline RSS: %s", fmt_mb(baseline))

    # ── Choose paths based on mode ────────────────────────────────────────────
    if args.mode == "old":
        faiss_path = args.faiss_path_old
        metadata_db_path = None   # no SQLite — use JSONL list
        metadata_path = args.metadata_path
        mode_label = "OLD  (IndexFlatIP + JSONL list)"
    else:
        faiss_path = args.faiss_path_new
        metadata_db_path = args.metadata_db
        metadata_path = args.metadata_path  # still passed but won't be loaded
        mode_label = "NEW  (IndexScalarQuantizer QT_fp16 + SQLite)"

    logger.info("Mode: %s", mode_label)

    # ── Validate paths ────────────────────────────────────────────────────────
    if not Path(faiss_path).exists():
        logger.error("FAISS file not found: %s", faiss_path)
        return 1
    if args.mode == "old" and not Path(metadata_path).exists():
        logger.error("Metadata JSONL not found: %s", metadata_path)
        return 1
    if args.mode == "new" and not Path(str(metadata_db_path)).exists():
        logger.error(
            "SQLite DB not found: %s — run migrate_metadata_to_sqlite first.",
            metadata_db_path,
        )
        return 1

    # ── Initialise FAISSRetriever ─────────────────────────────────────────────
    logger.info("Initialising FAISSRetriever...")
    t_init = time.perf_counter()
    from backend.retrieval.faiss_retriever import FAISSRetriever

    retriever = FAISSRetriever(
        faiss_path=faiss_path,
        metadata_path=metadata_path,
        metadata_db_path=metadata_db_path,
        model_name=args.model_name,
        device=args.device,
    )
    init_ms = (time.perf_counter() - t_init) * 1000.0

    after_init = rss_mb()
    measurements.append(("after FAISSRetriever.__init__() [idle]", after_init))
    logger.info("Post-init RSS: %s (init took %.0f ms)", fmt_mb(after_init), init_ms)

    # ── Run first query ───────────────────────────────────────────────────────
    logger.info("Running first query: %r", args.query)
    t_q = time.perf_counter()
    results = retriever.search(args.query, top_k=5)
    query_ms = (time.perf_counter() - t_q) * 1000.0

    after_query = rss_mb()
    measurements.append(("after first query", after_query))
    logger.info(
        "Post-query RSS: %s (query took %.0f ms, got %d results)",
        fmt_mb(after_query),
        query_ms,
        len(results),
    )

    # ── Print summary table ───────────────────────────────────────────────────
    col_w = 48
    print("\n" + "=" * 70)
    print(f"  RAM MEASUREMENT  —  {mode_label}")
    print("=" * 70)
    print(f"{'Stage':<{col_w}} {'RSS (MB)':>10}  {'Δ from baseline':>16}")
    print("-" * 70)
    for label, mb in measurements:
        delta = mb - baseline
        sign = "+" if delta >= 0 else ""
        print(f"{label:<{col_w}} {mb:>10,.1f}  {sign}{delta:>14,.1f}")
    print("=" * 70)

    if len(measurements) >= 2:
        idle_mb = measurements[1][1] - baseline
        print(f"\n  Idle footprint above baseline:  {idle_mb:,.1f} MB")
    if len(measurements) >= 3:
        q_delta = measurements[2][1] - measurements[1][1]
        sign = "+" if q_delta >= 0 else ""
        print(f"  RSS change on first query:      {sign}{q_delta:,.1f} MB")

    print(f"\n  Init time:  {init_ms:,.0f} ms")
    print(f"  Query time: {query_ms:,.0f} ms")
    print("=" * 70 + "\n")

    # Print top result for sanity
    if results:
        r = results[0]
        print(f"Top result: score={r['score']:.4f}  chunk_id={r['chunk_id']}")
        text_preview = str(r.get("text", ""))[:100].encode("utf-8", "replace").decode("utf-8")
        print(f"Text preview: {text_preview}...\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
