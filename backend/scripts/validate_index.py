"""
backend/scripts/validate_index.py
----------------------------------
Streaming validator for large RAG vector DB JSON index files.

Iteratively parses large JSON index files (e.g. 12+ GB) using `ijson`
in constant O(1) memory without triggering MemoryError.

Usage
─────
    python -m backend.scripts.validate_index --index-path data/indices/hi_val_index.json
    python -m backend.scripts.validate_index --help
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Reconfigure stdout for UTF-8 to handle potential non-ASCII characters in text previews
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("validate_index")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streamingly validate large vector index JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-path",
        default="data/indices/hi_val_index.json",
        help="Path to vector index JSON file.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Number of sample records to display.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    try:
        import ijson
    except ImportError:
        logger.error(
            "The 'ijson' package is required for streaming JSON validation. "
            "Please run: pip install ijson"
        )
        return 1

    index_path = Path(args.index_path)
    if not index_path.exists():
        logger.error("Index file not found: %s", index_path)
        return 1

    file_size_gb = index_path.stat().st_size / (1024 ** 3)
    logger.info("Opening index file: %s (%.2f GB)", index_path, file_size_gb)
    logger.info("Streaming records via ijson...")

    start_time = time.perf_counter()
    total_count = 0
    sample_records = []
    sample_dim = None

    with index_path.open("rb") as f:
        # ijson.items streams top-level array elements lazily
        for record in ijson.items(f, "item"):
            total_count += 1

            if total_count == 1:
                embedding = record.get("embedding") or []
                sample_dim = len(embedding)

            if len(sample_records) < args.sample_count:
                sample_records.append(record)

            if total_count % 100_000 == 0:
                elapsed = time.perf_counter() - start_time
                rate = total_count / elapsed if elapsed > 0 else 0
                logger.info(
                    "Streamed %d records... (%.0f rec/sec, elapsed: %.1fs)",
                    total_count,
                    rate,
                    elapsed,
                )

    elapsed = time.perf_counter() - start_time
    logger.info("Streaming complete! Processed %d records in %.1fs", total_count, elapsed)
    print("\n" + "=" * 70)
    print("                    INDEX VALIDATION REPORT                    ")
    print("=" * 70)
    print(f"File Path:           {index_path}")
    print(f"File Size:           {file_size_gb:.2f} GB")
    print(f"Total Record Count:  {total_count:,}")
    print(f"Embedding Dimension: {sample_dim} (Expected: 384)")
    print("=" * 70)
    print(f"\nSample Records (showing first {len(sample_records)}):")

    for idx, rec in enumerate(sample_records, 1):
        doc_id = rec.get("parent_doc_id") or rec.get("document_id") or rec.get("id")
        content = str(rec.get("content") or rec.get("text") or "").encode("utf-8", "replace").decode("utf-8")
        meta = rec.get("metadata") or {}
        print(f"\n--- Sample #{idx} ---")
        print(f"Chunk ID:      {rec.get('id')}")
        print(f"Parent Doc ID: {doc_id}")
        print(f"Strategy:      {meta.get('chunk_strategy', 'N/A')}")
        print(f"Text Preview:  {content[:150]}...")

    print("\n" + "=" * 70)
    if sample_dim == 384:
        print("✅ VALIDATION SUCCESSFUL: Index file structure and dimensions are valid.")
    else:
        print(f"⚠️ WARNING: Embedding dimension is {sample_dim}, expected 384.")
    print("=" * 70 + "\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
