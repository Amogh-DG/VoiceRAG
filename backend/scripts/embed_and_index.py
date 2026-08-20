"""
backend/scripts/embed_and_index.py
-----------------------------------
CLI script: reads TextChunk JSONL → embeds text → indexes into Vector DB.

Usage
─────
    python -m backend.scripts.embed_and_index \\
        --input data/chunks/hi_val_sentence.jsonl \\
        --output-db data/indices/hi_val_index.json \\
        --embedder sentence-transformers \\
        --batch-size 512 \\
        --save-every 50000

    python -m backend.scripts.embed_and_index --help

Performance Notes
─────────────────
- --batch-size 512 is safe for a GTX 1650 4GB with 384-dim vectors.
  Try 256 if you see GPU OOM; try 1024 if you have spare VRAM.
- --save-every N: flush in-memory index to disk every N chunks (default 50000).
  0 = save only at the very end. InMemoryVectorDB.upsert() no longer
  auto-saves after every call, so disk I/O never blocks the hot loop.
- ~1M × 384 float32 vectors ≈ 1.5 GB RAM. Ensure ≥4 GB free system RAM.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator, List

from backend.common.schemas import Chunk, TextChunk
from backend.embeddings import get_embedder
from backend.retrieval import get_vector_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("embed_and_index")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed TextChunk JSONL → Index into Vector DB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to TextChunk JSONL file (output of chunk.py).",
    )
    parser.add_argument(
        "--output-db",
        default="data/vector_db/index.json",
        help="Output path to persist the vector database index.",
    )
    parser.add_argument(
        "--embedder",
        default="mock",
        choices=["mock", "sentence-transformers", "minilm"],
        help="Embedding model implementation to use.",
    )
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="Model name if using sentence-transformers.",
    )
    parser.add_argument(
        "--db-type",
        default="in_memory",
        choices=["in_memory", "memory"],
        help="Vector DB implementation to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help=(
            "GPU embedding batch size. 512 is safe for GTX 1650 4GB at 384-dim. "
            "Reduce to 256 if OOM, increase to 1024 if you have spare VRAM."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50_000,
        help=(
            "Flush in-memory index to disk every N chunks processed. "
            "0 = save only at the very end (fastest, but no checkpoints)."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Stop after indexing this many chunks (0 = process all).",
    )
    return parser.parse_args(argv)


def _iter_text_chunks(input_path: Path, max_chunks: int = 0) -> Iterator[TextChunk]:
    """Yield TextChunk objects from a JSONL file one at a time (streaming)."""
    count = 0
    with input_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield TextChunk(**obj)
                count += 1
                if max_chunks > 0 and count >= max_chunks:
                    return
            except Exception as exc:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", lineno, input_path, exc
                )


def _make_chunks(batch: List[TextChunk], embeddings: List[List[float]]) -> List[Chunk]:
    """Convert TextChunk+embedding pairs into Chunk objects for the vector DB."""
    out: List[Chunk] = []
    for i, (tc, vec) in enumerate(zip(batch, embeddings)):
        chunk_idx = tc.metadata.get("chunk_index", i)
        out.append(
            Chunk(
                id=f"{tc.document_id}_c{chunk_idx}",
                parent_doc_id=tc.document_id,
                content=tc.text,
                index=chunk_idx,
                embedding=vec,
                metadata=tc.metadata,
            )
        )
    return out


def run(args: argparse.Namespace) -> int:
    # --- CUDA check ---
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA available. Device: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("CUDA not available — embedder will run on CPU.")
    except ImportError:
        logger.warning("torch not importable; cannot verify CUDA.")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input chunks file not found: %s", input_path)
        return 1

    output_db_path = Path(args.output_db)
    output_db_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Embedder ---
    logger.info("Initializing embedder '%s'...", args.embedder)
    embedder_kwargs = {}
    if args.embedder in ("sentence-transformers", "minilm"):
        embedder_kwargs["model_name"] = args.model_name
    embedder = get_embedder(args.embedder, **embedder_kwargs)

    # --- Vector DB (fresh start) ---
    logger.info(
        "Initializing vector DB '%s' (persist: %s)...", args.db_type, output_db_path
    )
    if output_db_path.exists():
        logger.info("Removing existing index at %s", output_db_path)
        output_db_path.unlink()
    vector_db = get_vector_db(args.db_type, persist_path=str(output_db_path))

    # --- Warmup (triggers model download + CUDA transfer) ---
    logger.info("Warming up model...")
    _ = embedder.embed("warmup")
    model = getattr(embedder, "_model", None)
    if model is not None:
        device = getattr(model, "device", "unknown")
        logger.info("Model running on device: %s", device)

    # --- Main indexing loop ---
    logger.info(
        "Starting indexing: batch_size=%d, save_every=%d, max_chunks=%d",
        args.batch_size,
        args.save_every,
        args.max_chunks,
    )

    buffer: List[TextChunk] = []
    total_indexed = 0
    chunks_since_save = 0
    start_time = time.perf_counter()
    last_log_time = start_time

    def flush_buffer() -> None:
        nonlocal total_indexed, chunks_since_save, last_log_time
        if not buffer:
            return
        texts = [tc.text for tc in buffer]
        embeddings = embedder.embed_batch(texts)
        chunks = _make_chunks(buffer, embeddings)
        # upsert() is now O(1) per chunk — no vstack, no disk write
        vector_db.upsert(chunks)
        total_indexed += len(chunks)
        chunks_since_save += len(chunks)

        now = time.perf_counter()
        if now - last_log_time >= 10.0:
            elapsed = now - start_time
            rate = total_indexed / elapsed
            remaining = 0.0
            logger.info(
                "Progress: %d chunks | %.0f chunks/sec | elapsed %.0fs",
                total_indexed,
                rate,
                elapsed,
            )
            last_log_time = now

    for text_chunk in _iter_text_chunks(input_path, max_chunks=args.max_chunks):
        buffer.append(text_chunk)

        if len(buffer) < args.batch_size:
            continue

        flush_buffer()
        buffer = []

        # Checkpoint: explicit save() triggers the single vstack + disk write
        if args.save_every > 0 and chunks_since_save >= args.save_every:
            logger.info(
                "Checkpoint at %d chunks — flushing index to disk...", total_indexed
            )
            vector_db.save(str(output_db_path))
            chunks_since_save = 0

    # Final partial batch
    if buffer:
        flush_buffer()
        buffer = []

    # Final save — this is the only mandatory disk write if save_every=0
    logger.info("Final save...")
    vector_db.save(str(output_db_path))

    elapsed = time.perf_counter() - start_time
    avg_rate = total_indexed / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done. %d chunks indexed in %.1fs (avg %.1f chunks/sec) → %s",
        total_indexed,
        elapsed,
        avg_rate,
        output_db_path,
    )
    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
