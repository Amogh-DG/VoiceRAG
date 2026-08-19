"""
backend/scripts/embed_and_index.py
-----------------------------------
CLI script: reads TextChunk JSONL → embeds text → indexes into Vector DB.

Usage
─────
    python -m backend.scripts.embed_and_index \\
        --input data/chunks/hi_train_sentence.jsonl \\
        --output-db data/vector_db/hi_index.json \\
        --embedder mock \\
        --batch-size 32

    python -m backend.scripts.embed_and_index --help

Idempotency
───────────
The script reads all input chunks, computes embeddings, and persists the vector
database to the specified output directory/file.
"""

import argparse
import json
import logging
import os
import sys
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
        default=64,
        help="Batch size for generating embeddings.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Stop after embedding/indexing this many chunks (0 = process all).",
    )
    return parser.parse_args(argv)


def _iter_text_chunks(input_path: Path, max_chunks: int = 0) -> Iterator[TextChunk]:
    """Yields TextChunk objects from a JSONL file."""
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
                logger.warning("Skipping malformed line %d in %s: %s", lineno, input_path, exc)


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input chunks file not found: %s", input_path)
        return 1

    output_db_path = Path(args.output_db)
    output_db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate embedder and vector DB
    logger.info("Initializing embedder '%s'...", args.embedder)
    embedder_kwargs = {}
    if args.embedder in ("sentence-transformers", "minilm"):
        embedder_kwargs["model_name"] = args.model_name
    embedder = get_embedder(args.embedder, **embedder_kwargs)

    logger.info("Initializing vector DB '%s' (persist path: %s)...", args.db_type, output_db_path)
    # Start fresh for script indexing
    if output_db_path.exists():
        logger.info("Overwriting existing vector DB index at %s", output_db_path)
        output_db_path.unlink()
        
    vector_db = get_vector_db(args.db_type, persist_path=str(output_db_path))

    # 2. Read chunks and batch-embed
    batch_text_chunks: List[TextChunk] = []
    total_indexed = 0

    def process_batch(batch: List[TextChunk]) -> None:
        nonlocal total_indexed
        if not batch:
            return
        texts = [tc.text for tc in batch]
        embeddings = embedder.embed_batch(texts)

        chunks_to_index: List[Chunk] = []
        for i, (tc, vec) in enumerate(zip(batch, embeddings)):
            chunk_idx = tc.metadata.get("chunk_index", i)
            chunk_id = f"{tc.document_id}_c{chunk_idx}"
            chunk_obj = Chunk(
                id=chunk_id,
                parent_doc_id=tc.document_id,
                content=tc.text,
                index=chunk_idx,
                embedding=vec,
                metadata=tc.metadata,
            )
            chunks_to_index.append(chunk_obj)

        vector_db.upsert(chunks_to_index)
        total_indexed += len(chunks_to_index)
        logger.info("Indexed batch of %d chunks (total: %d)", len(chunks_to_index), total_indexed)

    logger.info("Embedding and indexing chunks from %s...", input_path)
    for text_chunk in _iter_text_chunks(input_path, max_chunks=args.max_chunks):
        batch_text_chunks.append(text_chunk)
        if len(batch_text_chunks) >= args.batch_size:
            process_batch(batch_text_chunks)
            batch_text_chunks = []

    # Process remaining final batch
    if batch_text_chunks:
        process_batch(batch_text_chunks)

    logger.info("Indexing complete! Total chunks indexed: %d → %s", total_indexed, output_db_path)
    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
