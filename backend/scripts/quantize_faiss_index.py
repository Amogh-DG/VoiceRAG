"""
backend/scripts/quantize_faiss_index.py
----------------------------------------
One-time offline conversion: IndexFlatIP  →  IndexScalarQuantizer (QT_fp16).

Why QT_fp16?
────────────
- Stores each float32 vector component as float16 (2 bytes vs 4 bytes).
- Halves the FAISS index RAM footprint (~1.57 GB → ~785 MB for 1.07M × 384-d).
- Search remains EXHAUSTIVE — no IVF, no clustering, no recall degradation.
  Dot-product arithmetic is promoted back to float32 at search time, so score
  values will differ from IndexFlatIP by at most ±0.001 per query.
- The quantised type is ``faiss.ScalarQuantizer.QT_fp16`` (NOT QT_8bit or QT_4bit).

Memory budget during conversion
────────────────────────────────
``index.reconstruct_n(0, ntotal)`` materialises the full float32 matrix into RAM
(~1.57 GB for this corpus) in addition to the IndexFlatIP that is already loaded
(~1.57 GB) → peak ~3 GB. This is a one-time operation and is done in configurable
batches to keep the incremental allocation smaller.

Output
──────
The original ``.faiss`` file is NEVER modified or deleted.
The new quantised index is written to a separate file (default suffix _sq_fp16).

Usage
─────
    python -m backend.scripts.quantize_faiss_index

    python -m backend.scripts.quantize_faiss_index \\
        --input-faiss  data/indices/hi_val_multilingual.faiss \\
        --output-faiss data/indices/hi_val_multilingual_sq_fp16.faiss \\
        --batch-size   200000

    python -m backend.scripts.quantize_faiss_index --help
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

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
logger = logging.getLogger("quantize_faiss_index")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FAISS IndexFlatIP → IndexScalarQuantizer (QT_fp16). "
            "The original file is never modified."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-faiss",
        default="data/indices/hi_val_multilingual.faiss",
        help="Path to the existing IndexFlatIP FAISS binary file.",
    )
    parser.add_argument(
        "--output-faiss",
        default="data/indices/hi_val_multilingual_sq_fp16.faiss",
        help="Destination path for the new quantised index (must not equal --input-faiss).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200_000,
        help=(
            "Number of vectors to reconstruct and add per batch. "
            "Larger batches are faster but use more peak RAM."
        ),
    )
    return parser.parse_args(argv)


# ── Core conversion ───────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    try:
        import faiss
    except ImportError:
        logger.error("faiss package required. Run: pip install faiss-cpu")
        return 1

    input_path = Path(args.input_faiss)
    output_path = Path(args.output_faiss)

    if not input_path.exists():
        logger.error("Input FAISS file not found: %s", input_path)
        return 1

    if input_path.resolve() == output_path.resolve():
        logger.error(
            "Input and output paths are the same (%s). "
            "This script must write to a different file to stay non-destructive.",
            input_path,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load source index ──────────────────────────────────────────────────
    input_size_mb = input_path.stat().st_size / (1024 ** 2)
    logger.info(
        "Loading source FAISS index from %s (%.1f MB)...", input_path, input_size_mb
    )
    t0 = time.perf_counter()
    src_index = faiss.read_index(str(input_path))
    load_ms = (time.perf_counter() - t0) * 1000.0
    ntotal = src_index.ntotal
    dimension = src_index.d
    logger.info(
        "Source index loaded in %.0f ms — %d vectors, dim=%d, type=%s",
        load_ms,
        ntotal,
        dimension,
        type(src_index).__name__,
    )

    if ntotal == 0:
        logger.error("Source index contains 0 vectors — nothing to convert.")
        return 1

    # ── 2. Create the quantised index and train it ────────────────────────────
    logger.info(
        "Creating IndexScalarQuantizer(dim=%d, QT_fp16) and training on first batch...",
        dimension,
    )
    sq_index = faiss.IndexScalarQuantizer(
        dimension, faiss.ScalarQuantizer.QT_fp16
    )

    # IndexScalarQuantizer.QT_fp16 does not require training (it's a codec, not
    # a learned quantiser), but calling train() is harmless and makes the index
    # ready for add() without the need to check is_trained.
    first_batch_size = min(args.batch_size, ntotal)
    logger.info(
        "Reconstructing first %d vectors from source index for training...",
        first_batch_size,
    )
    t_train = time.perf_counter()
    train_vecs = src_index.reconstruct_n(0, first_batch_size)
    sq_index.train(train_vecs)
    logger.info(
        "Training complete in %.1f s (is_trained=%s).",
        time.perf_counter() - t_train,
        sq_index.is_trained,
    )

    # Add the training batch immediately (don't reconstruct it twice).
    logger.info("Adding training batch (%d vectors) to quantised index...", first_batch_size)
    sq_index.add(train_vecs)
    del train_vecs  # free ~1.5 GB

    # ── 3. Add remaining vectors in batches ───────────────────────────────────
    t_add = time.perf_counter()
    cursor = first_batch_size
    while cursor < ntotal:
        end = min(cursor + args.batch_size, ntotal)
        count = end - cursor
        logger.info(
            "Reconstructing vectors %d–%d (%d vectors)...", cursor, end - 1, count
        )
        batch = src_index.reconstruct_n(cursor, count)
        sq_index.add(batch)
        del batch
        elapsed = time.perf_counter() - t_add
        rate = (end - first_batch_size) / elapsed if elapsed > 0 else 0
        logger.info(
            "Progress: %d / %d vectors added (%.0f vec/sec)", end, ntotal, rate
        )
        cursor = end

    add_elapsed = time.perf_counter() - t_add
    logger.info(
        "All %d vectors added to quantised index in %.1f s.", ntotal, add_elapsed
    )

    if sq_index.ntotal != ntotal:
        logger.error(
            "Vector count mismatch: source=%d, quantised=%d — aborting write.",
            ntotal,
            sq_index.ntotal,
        )
        return 1

    # ── 4. Verify a sample score against the original ─────────────────────────
    logger.info("Running sanity-check: comparing scores on 1 probe vector...")
    probe = np.random.default_rng(42).standard_normal((1, dimension)).astype(np.float32)
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)  # unit-normalise

    src_scores, src_idx = src_index.search(probe, 5)
    sq_scores,  sq_idx  = sq_index.search(probe,  5)

    print("\n── Sanity-check: top-5 score comparison (probe vector) ──")
    print(f"{'Rank':<6} {'Src score':>12} {'SQ score':>12} {'Δ score':>10} {'Same idx?':>10}")
    print("-" * 56)
    for rank, (ss, qs, si, qi) in enumerate(
        zip(src_scores[0], sq_scores[0], src_idx[0], sq_idx[0]), 1
    ):
        flag = "✅" if si == qi else "⚠️ "
        print(f"{rank:<6} {ss:>12.6f} {qs:>12.6f} {qs - ss:>+10.6f} {flag:>10}")

    max_delta = float(np.max(np.abs(sq_scores[0] - src_scores[0])))
    print(f"\nMax |Δ score| across top-5: {max_delta:.6f}")
    if max_delta > 0.01:
        print("⚠️  WARNING: score delta exceeds 0.01 — please investigate.")
    else:
        print("✅  Score delta within expected fp16 tolerance (<0.01).")

    # ── 5. Write output ───────────────────────────────────────────────────────
    logger.info("Writing quantised index to %s...", output_path)
    t_write = time.perf_counter()
    faiss.write_index(sq_index, str(output_path))
    write_ms = (time.perf_counter() - t_write) * 1000.0

    output_size_mb = output_path.stat().st_size / (1024 ** 2)
    savings_pct = (1.0 - output_size_mb / input_size_mb) * 100.0

    print("\n" + "=" * 70)
    print("            FAISS SCALAR QUANTIZATION COMPLETE                        ")
    print("=" * 70)
    print(f"Source Index:          {input_path}")
    print(f"Output Index:          {output_path}")
    print(f"Index Type:            IndexScalarQuantizer (QT_fp16, exhaustive)")
    print(f"Vector Count:          {ntotal:,}")
    print(f"Dimension:             {dimension}")
    print(f"Source Size:           {input_size_mb:.1f} MB")
    print(f"Quantised Size:        {output_size_mb:.1f} MB")
    print(f"Size Reduction:        {savings_pct:.1f}%")
    print(f"Write Time:            {write_ms:.0f} ms")
    print("=" * 70)
    print("✅  Original file untouched. Quantised index written successfully.")
    print("=" * 70 + "\n")

    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
