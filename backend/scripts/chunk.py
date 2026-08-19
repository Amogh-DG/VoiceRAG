"""
backend/scripts/chunk.py
--------------------------
CLI runner: reads normalised Document JSONL → applies a chunking strategy →
writes TextChunk JSONL.

Usage
─────
    python -m backend.scripts.chunk \\
        --input  data/processed/hi_train.jsonl \\
        --output data/chunks/hi_train_sentence.jsonl \\
        --strategy sentence \\
        --max-chars 500

    python -m backend.scripts.chunk --help

Strategies
──────────
    fixed_overlap    Character windows with configurable overlap (baseline).
    sentence         Sentence-boundary splitting (handles Indic stops too).
    metadata_aware   Adapts chunk size/overlap by query_type and language.

Output format
─────────────
One JSON object per line (JSONL), each a serialised TextChunk:
    {
        "text":        "...",
        "document_id": "1001_hi_p0",
        "metadata":    { "chunk_strategy": "sentence", "chunk_index": 0, ... }
    }

Idempotency
───────────
Output written atomically via temp file + rename (same as ingest.py).
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from backend.chunking.strategies import STRATEGY_NAMES, get_chunker
from backend.common.schemas import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("chunk")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk normalised Document JSONL → TextChunk JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a Document JSONL file (output of ingest.py).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output TextChunk JSONL path. "
            "Defaults to <input-stem>_<strategy>.jsonl in the same directory."
        ),
    )
    parser.add_argument(
        "--strategy",
        default="sentence",
        choices=STRATEGY_NAMES,
        help="Chunking strategy to apply.",
    )
    # --- FixedOverlapChunker options ---
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="[fixed_overlap] Max characters per chunk.")
    parser.add_argument("--overlap", type=int, default=50,
                        help="[fixed_overlap] Overlap characters between chunks.")
    # --- SentenceChunker options ---
    parser.add_argument("--max-chars", type=int, default=500,
                        help="[sentence / metadata_aware] Soft char limit per chunk.")
    parser.add_argument("--sentence-overlap", type=int, default=1,
                        help="[sentence / metadata_aware] Sentences of overlap.")
    parser.add_argument("--no-para-split", action="store_true",
                        help="[sentence] Disable paragraph-break chunking.")
    # --- MetadataAwareChunker options ---
    parser.add_argument("--description-max-chars", type=int, default=700,
                        help="[metadata_aware] Max chars for 'description' query_type.")
    parser.add_argument("--entity-max-chars", type=int, default=300,
                        help="[metadata_aware] Max chars for entity/numeric query_type.")
    return parser.parse_args(argv)


def _strategy_kwargs(args: argparse.Namespace) -> dict:
    """Build constructor kwargs for the selected strategy."""
    if args.strategy == "fixed_overlap":
        return {"chunk_size": args.chunk_size, "overlap": args.overlap}
    if args.strategy == "sentence":
        return {
            "max_chars": args.max_chars,
            "sentence_overlap": args.sentence_overlap,
            "split_on_paragraphs": not args.no_para_split,
        }
    if args.strategy == "metadata_aware":
        return {
            "default_max_chars": args.max_chars,
            "description_max_chars": args.description_max_chars,
            "entity_max_chars": args.entity_max_chars,
        }
    return {}


def _iter_documents(input_path: Path) -> Iterator[Document]:
    """Yield Document objects from a JSONL file."""
    with input_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield Document(**obj)
            except Exception as exc:
                logger.warning("Skipping malformed line %d: %s", lineno, exc)


def default_output_path(input_path: Path, strategy: str) -> Path:
    out_dir = input_path.parent
    stem = input_path.stem
    return out_dir / f"{stem}_{strategy}.jsonl"


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    output_path = (
        Path(args.output) if args.output
        else default_output_path(input_path, args.strategy)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunker = get_chunker(args.strategy, **_strategy_kwargs(args))
    logger.info(
        "Chunking %s → %s  (strategy=%s)",
        input_path, output_path, args.strategy,
    )

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent, prefix=".chunk_tmp_", suffix=".jsonl"
    )
    doc_count = chunk_count = skipped = 0

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            for doc in _iter_documents(input_path):
                doc_count += 1
                try:
                    chunks = chunker.chunk_document(doc)
                except Exception as exc:
                    logger.warning("Skipping document %s: %s", doc.id, exc)
                    skipped += 1
                    continue
                for chunk in chunks:
                    fh.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
                    chunk_count += 1
                if doc_count % 500 == 0:
                    logger.info("  …%d docs → %d chunks", doc_count, chunk_count)

        Path(tmp_path).replace(output_path)
        logger.info(
            "Done. %d docs → %d chunks (skipped %d) → %s",
            doc_count, chunk_count, skipped, output_path,
        )
    except Exception as exc:
        logger.error("Chunking failed: %s", exc, exc_info=True)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return 1

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
