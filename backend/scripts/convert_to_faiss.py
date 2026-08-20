"""
backend/scripts/convert_to_faiss.py
-----------------------------------
Converts a large JSON vector index into a high-performance FAISS IndexFlatIP.

Design Decisions
────────────────
- FAISS Index Type: `faiss.IndexFlatIP` (Inner Product). Because vector embeddings
  are L2 unit-normalized, the inner product equals exact cosine similarity.
- Memory Efficiency: Streams JSON items via `ijson` in batches, and writes
  metadata directly to a streaming `.jsonl` file to avoid RAM accumulation.
- Output Artifacts:
  1. FAISS Index File: `.faiss` binary file containing optimized float32 vectors.
  2. Metadata File: `.jsonl` file preserving exact 1-to-1 ordering with FAISS positions.
- Non-Destructive: Does NOT modify or delete the source JSON index.

Usage
─────
    python -m backend.scripts.convert_to_faiss \\
        --input-index data/indices/hi_val_multilingual_index.json \\
        --output-faiss data/indices/hi_val_multilingual.faiss \\
        --output-metadata data/indices/hi_val_multilingual_metadata.jsonl

    python -m backend.scripts.convert_to_faiss --help
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
logger = logging.getLogger("convert_to_faiss")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSON vector index → FAISS IndexFlatIP + Streaming JSONL Metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-index",
        default="data/indices/hi_val_multilingual_index.json",
        help="Path to source JSON vector index file.",
    )
    parser.add_argument(
        "--output-faiss",
        default="data/indices/hi_val_multilingual.faiss",
        help="Path to output FAISS binary index file.",
    )
    parser.add_argument(
        "--output-metadata",
        default="data/indices/hi_val_multilingual_metadata.jsonl",
        help="Path to output metadata JSONL file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Batch size for adding vectors to FAISS.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    try:
        import ijson
    except ImportError:
        logger.error("ijson package required. Run: pip install ijson")
        return 1

    try:
        import faiss
    except ImportError:
        logger.error("faiss package required. Run: pip install faiss-cpu")
        return 1

    input_path = Path(args.input_index)
    if not input_path.exists():
        logger.error("Input JSON index file not found: %s", input_path)
        return 1

    output_faiss_path = Path(args.output_faiss)
    output_metadata_path = Path(args.output_metadata)

    output_faiss_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    file_size_gb = input_path.stat().st_size / (1024 ** 3)
    logger.info("Opening source index file: %s (%.2f GB)", input_path, file_size_gb)
    logger.info("Streaming records via ijson & writing metadata to %s...", output_metadata_path)

    start_time = time.perf_counter()

    faiss_index = None
    dimension = None
    vec_batch = []
    total_processed = 0

    def add_batch_to_faiss(batch: list[np.ndarray]):
        nonlocal faiss_index, dimension
        if not batch:
            return
        matrix = np.vstack(batch).astype(np.float32)
        if faiss_index is None:
            dimension = matrix.shape[1]
            logger.info("Initializing FAISS IndexFlatIP with dimension: %d", dimension)
            faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(matrix)

    # Open metadata JSONL file for streaming writes
    with output_metadata_path.open("w", encoding="utf-8") as f_meta:
        with input_path.open("rb") as f_in:
            for record in ijson.items(f_in, "item"):
                embedding = record.get("embedding")
                if embedding is None:
                    continue

                # L2 normalize vector
                arr = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                vec_batch.append(arr)

                # Construct metadata payload and write line immediately to JSONL
                doc_id = record.get("parent_doc_id") or record.get("document_id") or record.get("id")
                meta_entry = {
                    "id": record.get("id"),
                    "parent_doc_id": doc_id,
                    "content": record.get("content") or record.get("text") or "",
                    "metadata": record.get("metadata") or {},
                }
                f_meta.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")
                total_processed += 1

                if len(vec_batch) >= args.batch_size:
                    add_batch_to_faiss(vec_batch)
                    vec_batch = []
                    f_meta.flush()
                    elapsed = time.perf_counter() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Processed & streamed %d records... (%.0f rec/sec, elapsed: %.1fs)",
                        total_processed,
                        rate,
                        elapsed,
                    )

    # Process remaining partial batch
    if vec_batch:
        add_batch_to_faiss(vec_batch)
        vec_batch = []

    if faiss_index is None or total_processed == 0:
        logger.error("No valid vectors were found in %s", input_path)
        return 1

    # Save binary FAISS index
    logger.info("Saving FAISS index to %s...", output_faiss_path)
    faiss.write_index(faiss_index, str(output_faiss_path))
    faiss_size_mb = output_faiss_path.stat().st_size / (1024 ** 2)
    meta_size_mb = output_metadata_path.stat().st_size / (1024 ** 2)

    elapsed_total = time.perf_counter() - start_time
    avg_rate = total_processed / elapsed_total if elapsed_total > 0 else 0

    print("\n" + "=" * 70)
    print("                FAISS CONVERSION COMPLETE                              ")
    print("=" * 70)
    print(f"Source JSON Index:      {input_path}")
    print(f"Total Vectors Indexed:  {total_processed:,}")
    print(f"Vector Dimension:       {dimension}")
    print(f"FAISS Index Type:       IndexFlatIP (Cosine Similarity)")
    print(f"FAISS Binary File:      {output_faiss_path} ({faiss_size_mb:.1f} MB)")
    print(f"Metadata JSONL File:    {output_metadata_path} ({meta_size_mb:.1f} MB)")
    print(f"Conversion Time:        {elapsed_total:.1f} s (avg {avg_rate:.0f} rec/sec)")
    print("=" * 70)
    print("✅ FAISS conversion successful! Vector-to-metadata ordering preserved 1:1 in JSONL.")
    print("=" * 70 + "\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
