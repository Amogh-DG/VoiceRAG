"""
backend/scripts/ingest.py
--------------------------
Reproducible CLI runner for ingesting the AI4Bharat MSMARCO-XI dataset.

Usage
─────
    python -m backend.scripts.ingest --language hi --split train --output data/processed/hi_train.jsonl
    python -m backend.scripts.ingest --language hi --split train --source-path /local/hintrain.parquet
    python -m backend.scripts.ingest --help

Idempotency
───────────
The output file is written atomically via a temp file + rename so a failed run
never leaves a partial file.  Re-running is safe: the existing output is
overwritten only once the new run succeeds.

Output format
─────────────
One JSON object per line (JSONL).  Each object is a serialised Document:
    {
        "id":       "<query_id>_<lang>_p<idx>",
        "content":  "<translated passage text>",
        "metadata": { ... }          # see normalizer.py for full field list
    }

The chunking stage reads this file directly — no coupling to ingestion code.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from backend.ingestion.loader import MSMarcoXILoader
from backend.ingestion.normalizer import SUPPORTED_LANGUAGES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ingest")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest AI4Bharat MSMARCO-XI dataset → normalised JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--language",
        default="hi",
        choices=SUPPORTED_LANGUAGES,
        help="ISO 639-1 language code to ingest.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help="Dataset split.",
    )
    parser.add_argument(
        "--source-path",
        default="auto",
        help=(
            "Path to a local .parquet or .jsonl file.  "
            "Use 'auto' (default) to stream directly from HF Hub."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL path.  Defaults to "
            "data/processed/<language>_<split>.jsonl"
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Stop after this many raw rows (0 = no limit).  Useful for smoke-tests.",
    )
    return parser.parse_args(argv)


def default_output_path(language: str, split: str) -> Path:
    """Returns the conventional output path for a given language + split."""
    out_dir = Path("data") / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{language}_{split}.jsonl"


def run(args: argparse.Namespace) -> int:
    """
    Main ingestion logic.  Returns 0 on success, non-zero on failure.
    """
    output_path = (
        Path(args.output) if args.output else default_output_path(args.language, args.split)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = args.source_path if args.source_path != "auto" else args.language

    logger.info(
        "Starting ingestion: language=%s split=%s source=%s → %s",
        args.language,
        args.split,
        source,
        output_path,
    )

    loader = MSMarcoXILoader(
        language=args.language,
        split=args.split,
        limit=args.max_rows,
    )

    # --- Write atomically via a temp file to ensure idempotency ---
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent, prefix=".ingest_tmp_", suffix=".jsonl"
    )
    doc_count = 0
    skipped_count = 0

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            for doc in loader._iter_documents(source):
                fh.write(json.dumps(doc.model_dump(), ensure_ascii=False) + "\n")
                doc_count += 1
                if doc_count % 1000 == 0:
                    logger.info("  …written %d documents", doc_count)

        # Atomic rename (idempotent — overwrites previous output if run again)
        Path(tmp_path).replace(output_path)
        logger.info(
            "Ingestion complete. Documents written: %d → %s", doc_count, output_path
        )

    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        # Clean up temp file on failure so we never leave partial output
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
