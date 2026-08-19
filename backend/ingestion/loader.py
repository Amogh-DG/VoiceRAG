"""
backend/ingestion/loader.py
-----------------------------
Concrete IDatasetLoader for the AI4Bharat MSMARCO-XI dataset.

Design decisions
────────────────
* Streaming via HuggingFace `datasets` streaming=True so the full ~3.7 GB
  per-language parquet is never fully downloaded; we only materialise rows
  as we iterate them.
* load() accepts a source_path that is either:
    - A language code string  ("hi", "bn", …)       → streams from HF Hub
    - A local file/directory path                    → loads from disk (parquet/JSONL)
* The limit kwarg caps rows consumed — used by the benchmark runner and tests.
* Malformed rows are logged + skipped (no crash).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from backend.common.interfaces import IDatasetLoader
from backend.common.schemas import Document
from backend.ingestion.normalizer import SUPPORTED_LANGUAGES, normalize_record

logger = logging.getLogger(__name__)

# HF Hub path pattern for parquet files
_HF_PARQUET_URL = (
    "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI"
    "/resolve/main/{split}/{lang_prefix}{split_suffix}.parquet"
)

# Maps split name → parquet filename suffix used in the repo
_SPLIT_SUFFIX = {
    "train": "train",
    "validation": "val",
}

# Maps 2-letter code → 3-letter prefix used in filenames (e.g. "hi" → "hin")
_LANG_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin",
    "kn": "kan", "ml": "mal", "mr": "mar", "ne": "nep",
    "or": "ori", "pa": "pan", "sa": "san", "ta": "tam",
    "te": "tel", "ur": "urd",
}


def _build_hf_url(language: str, split: str) -> str:
    prefix = _LANG_PREFIX.get(language, language[:3])
    suffix = _SPLIT_SUFFIX.get(split, split)
    return _HF_PARQUET_URL.format(
        split=split, lang_prefix=prefix, split_suffix=suffix
    )


class MSMarcoXILoader(IDatasetLoader):
    """
    Streams the AI4Bharat MSMARCO-XI dataset and normalises records into
    Document objects suitable for downstream chunking + indexing.

    Args:
        language:  ISO 639-1 code ("hi", "bn", …).  Ignored for local paths.
        split:     "train" or "validation".
        limit:     If > 0, stop after this many *raw rows* (pre-normalisation).
    """

    def __init__(
        self,
        language: str = "hi",
        split: str = "train",
        limit: int = 0,
    ):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not in supported set: {SUPPORTED_LANGUAGES}"
            )
        self.language = language
        self.split = split
        self.limit = limit

    # ------------------------------------------------------------------ #
    # IDatasetLoader interface                                             #
    # ------------------------------------------------------------------ #

    def load(self, source_path: str) -> List[Document]:
        """
        Load and normalise documents from *source_path*.

        source_path may be:
            - A 2-letter language code ("hi") → stream from HF Hub
            - A local file path (.parquet or .jsonl) → load from disk
            - An empty string / "auto" → same as language code

        Returns a flat list of Documents (one per passage).
        """
        return list(self._iter_documents(source_path))

    # ------------------------------------------------------------------ #
    # Streaming helpers                                                    #
    # ------------------------------------------------------------------ #

    def _iter_documents(self, source_path: str) -> Iterator[Document]:
        """Yield Documents one by one, suitable for large datasets."""
        raw_iter = self._iter_raw_rows(source_path)
        for raw_row in raw_iter:
            try:
                docs = normalize_record(raw_row, self.language, self.split)
                yield from docs
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed record: %s", exc)

    def _iter_raw_rows(self, source_path: str) -> Iterator[Dict[str, Any]]:
        """
        Yields raw dicts from either the HF Hub (streaming) or a local file.
        Respects self.limit if > 0.
        """
        # Determine whether source_path points to a local file
        path = Path(source_path)
        if path.exists():
            yield from self._iter_local(path)
        else:
            # Treat source_path as a language code / "auto" — stream from HF
            yield from self._iter_hf_streaming()

    def _iter_hf_streaming(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch rows from the remote HF parquet file using DuckDB + httpfs.

        Why DuckDB instead of datasets(streaming=True)?
        ─────────────────────────────────────────────────
        The MSMARCO-XI parquet files store the `passages` column as a nested
        struct containing list-typed sub-columns.  Arrow's C++ parquet reader
        (used internally by the `datasets` library) raises:

            ArrowNotImplementedError: Nested data conversions not implemented
            for chunked array outputs

        when it tries to reconstruct these columns from multi-chunk row groups.
        This is an upstream pyarrow/datasets limitation with no available fix.
        DuckDB ships its own independent parquet reader that handles nested
        list columns correctly, so we use it here for the remote case only.
        Local .parquet and .jsonl file loading is unchanged.
        """
        try:
            import duckdb  # deferred import
        except ImportError as exc:
            raise ImportError(
                "The 'duckdb' package is required for remote parquet reading. "
                "Run: pip install duckdb"
            ) from exc

        url = _build_hf_url(self.language, self.split)
        logger.info("Fetching MSMARCO-XI via DuckDB from: %s", url)

        con = duckdb.connect()
        try:
            # Enable the httpfs extension so DuckDB can read parquet over HTTPS
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")

            limit_clause = f"LIMIT {self.limit}" if self.limit > 0 else ""
            sql = f"SELECT * FROM read_parquet('{url}') {limit_clause}"
            logger.debug("DuckDB query: %s", sql)

            rel = con.execute(sql)
            col_names = [desc[0] for desc in rel.description]

            for raw_tuple in rel.fetchall():
                row: Dict[str, Any] = dict(zip(col_names, raw_tuple))
                # DuckDB returns nested structs as dicts already — no reshape needed
                yield row

        finally:
            con.close()


    def _iter_local(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Load rows from a local .parquet or .jsonl file."""
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            yield from self._iter_local_parquet(path)
        elif suffix in (".jsonl", ".json"):
            yield from self._iter_local_jsonl(path)
        else:
            raise ValueError(
                f"Unsupported local file format '{suffix}'. "
                "Use .parquet or .jsonl"
            )

    def _iter_local_parquet(self, path: Path) -> Iterator[Dict[str, Any]]:
        try:
            import pyarrow.parquet as pq  # deferred import
        except ImportError as exc:
            raise ImportError("pip install pyarrow") from exc

        logger.info("Loading local parquet: %s", path)
        table = pq.read_table(str(path))
        count = 0
        for batch in table.to_batches():
            for row in batch.to_pylist():
                yield row
                count += 1
                if self.limit > 0 and count >= self.limit:
                    return

    def _iter_local_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        import json

        logger.info("Loading local JSONL: %s", path)
        count = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line: %s", exc)
                    continue
                count += 1
                if self.limit > 0 and count >= self.limit:
                    return
