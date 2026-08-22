"""
backend/scripts/migrate_metadata_to_sqlite.py
----------------------------------------------
One-time migration: JSONL metadata file  →  SQLite database.

Schema
──────
Table: ``chunks``

    pos           INTEGER PRIMARY KEY   — 0-indexed row position in the JSONL file.
                                          This is the FAISS internal vector ordinal
                                          (i.e. what FAISS returns as ``indices[i]``).
                                          CRITICAL: this is NOT the "id" field in the JSON.

    id            TEXT                 — The "id" field from the JSON payload.
    parent_doc_id TEXT                 — Resolved parent document identifier.
    content       TEXT                 — "content" or "text" field from the JSON.
    metadata      TEXT                 — JSON-serialised metadata blob.

Index: ``idx_chunks_pos`` on ``pos`` (the PRIMARY KEY already implies this, but
the explicit index makes the intent clear and is a no-op overhead-wise).

Why positional primary key and NOT the "id" field?
───────────────────────────────────────────────────
FAISS returns raw integer offsets (``indices`` array) that match the insertion
order of vectors in the index.  The existing ``metadata_list[pos_idx]`` lookup
relies on this 1:1 positional correspondence.  SQLite ``pos`` preserves that
correspondence exactly.  The JSON "id" field is an opaque string that may be
non-integer or non-contiguous and MUST NOT be confused with the positional key.

Spot-check verification
────────────────────────
After migration the script selects 10 random positions from SQLite and compares
``id`` and ``content`` against the corresponding JSONL line to prove the
positional mapping is correct.  A verification table is printed to stdout.

Usage
─────
    python -m backend.scripts.migrate_metadata_to_sqlite

    python -m backend.scripts.migrate_metadata_to_sqlite \\
        --metadata-path data/indices/hi_val_multilingual_metadata.jsonl \\
        --output-db     data/indices/hi_val_multilingual_metadata.db \\
        --batch-size    50000

    python -m backend.scripts.migrate_metadata_to_sqlite --help
"""

import argparse
import json
import logging
import random
import sqlite3
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
logger = logging.getLogger("migrate_metadata_to_sqlite")

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    pos           INTEGER PRIMARY KEY,
    id            TEXT,
    parent_doc_id TEXT,
    content       TEXT,
    metadata      TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chunks_pos ON chunks(pos);
"""

_INSERT = """
INSERT OR REPLACE INTO chunks (pos, id, parent_doc_id, content, metadata)
VALUES (?, ?, ?, ?, ?);
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate FAISS metadata JSONL → SQLite. "
            "Uses positional (0-indexed) row number as PRIMARY KEY to match "
            "FAISS internal vector ordering."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata-path",
        default="data/indices/hi_val_multilingual_metadata.jsonl",
        help="Path to the source metadata JSONL file.",
    )
    parser.add_argument(
        "--output-db",
        default="data/indices/hi_val_multilingual_metadata.db",
        help="Destination SQLite database path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Number of rows per SQLite transaction (controls commit frequency).",
    )
    parser.add_argument(
        "--spot-check-count",
        type=int,
        default=10,
        help="Number of random positions to spot-check after migration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete and recreate the output DB if it already exists.",
    )
    return parser.parse_args(argv)


# ── Core migration ────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    metadata_path = Path(args.metadata_path)
    output_db = Path(args.output_db)

    if not metadata_path.exists():
        logger.error("Metadata JSONL file not found: %s", metadata_path)
        return 1

    if output_db.exists():
        if args.overwrite:
            logger.info("--overwrite specified; deleting existing DB: %s", output_db)
            output_db.unlink()
        else:
            logger.error(
                "Output DB already exists: %s. "
                "Use --overwrite to replace it or choose a different --output-db path.",
                output_db,
            )
            return 1

    output_db.parent.mkdir(parents=True, exist_ok=True)

    jsonl_size_mb = metadata_path.stat().st_size / (1024 ** 2)
    logger.info(
        "Migrating %s (%.0f MB) → %s ...", metadata_path, jsonl_size_mb, output_db
    )

    # ── Open SQLite and create schema ─────────────────────────────────────────
    con = sqlite3.connect(str(output_db), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute(_CREATE_TABLE)
    con.execute(_CREATE_INDEX)
    con.commit()

    # ── Stream JSONL and insert ───────────────────────────────────────────────
    t0 = time.perf_counter()
    total = 0
    batch: list = []

    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            pos = total  # 0-indexed positional key

            chunk_id = record.get("id")
            doc_id = (
                record.get("parent_doc_id")
                or record.get("document_id")
                or record.get("id")
            )
            content = record.get("content") or record.get("text") or ""
            meta_blob = json.dumps(record.get("metadata") or {}, ensure_ascii=False)

            batch.append((pos, chunk_id, doc_id, content, meta_blob))
            total += 1

            if len(batch) >= args.batch_size:
                con.executemany(_INSERT, batch)
                con.commit()
                batch = []
                elapsed = time.perf_counter() - t0
                rate = total / elapsed if elapsed > 0 else 0
                logger.info(
                    "Inserted %d rows... (%.0f rows/sec, elapsed: %.1fs)",
                    total,
                    rate,
                    elapsed,
                )

    if batch:
        con.executemany(_INSERT, batch)
        con.commit()
        batch = []

    elapsed_total = time.perf_counter() - t0
    avg_rate = total / elapsed_total if elapsed_total > 0 else 0
    logger.info(
        "Migration complete: %d rows inserted in %.1f s (avg %.0f rows/sec).",
        total,
        elapsed_total,
        avg_rate,
    )

    # ── Spot-check verification ───────────────────────────────────────────────
    logger.info(
        "Running %d-point positional spot-check...", args.spot_check_count
    )
    spot_positions = sorted(random.sample(range(total), min(args.spot_check_count, total)))

    # Re-read the JSONL to fetch lines at the sampled positions.
    jsonl_records: dict = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no in spot_positions:
                line = line.strip()
                if line:
                    jsonl_records[line_no] = json.loads(line)
            if len(jsonl_records) == len(spot_positions):
                break  # early exit once we've collected all needed lines

    # Fetch from SQLite
    placeholders = ",".join("?" * len(spot_positions))
    rows = con.execute(
        f"SELECT pos, id, parent_doc_id, content FROM chunks WHERE pos IN ({placeholders})",
        spot_positions,
    ).fetchall()
    sqlite_by_pos = {row[0]: row for row in rows}

    print("\n" + "=" * 90)
    print("  POSITIONAL SPOT-CHECK: SQLite row N vs. JSONL line N (0-indexed)")
    print("=" * 90)
    print(f"{'pos':>8}  {'jsonl_id':<30}  {'sqlite_id':<30}  {'match':>6}  {'content_match':>14}")
    print("-" * 90)

    all_match = True
    for pos in spot_positions:
        j_rec = jsonl_records.get(pos, {})
        s_row = sqlite_by_pos.get(pos)

        j_id = str(j_rec.get("id", ""))
        s_id = str(s_row[1]) if s_row else "MISSING"
        j_content = (j_rec.get("content") or j_rec.get("text") or "")[:30]
        s_content = (s_row[3] or "")[:30] if s_row else ""

        id_ok = j_id == s_id
        content_ok = j_content == s_content
        match = id_ok and content_ok
        if not match:
            all_match = False

        mark = "✅" if match else "❌"
        c_mark = "✅" if content_ok else "❌"
        print(f"{pos:>8}  {j_id[:30]:<30}  {s_id[:30]:<30}  {mark:>6}  {c_mark:>14}")

    print("=" * 90)
    if all_match:
        print("✅  All spot-checks passed — positional mapping is correct.")
    else:
        print("❌  SPOT-CHECK FAILURES DETECTED — positional mapping may be wrong!")
        print("    Please investigate before using the SQLite DB in production.")
    print("=" * 90)

    con.close()

    db_size_mb = output_db.stat().st_size / (1024 ** 2)
    print(f"\nSource JSONL:  {metadata_path} ({jsonl_size_mb:.0f} MB)")
    print(f"Output SQLite: {output_db} ({db_size_mb:.0f} MB)")
    print(f"Total rows:    {total:,}")
    print(f"Migration time:{elapsed_total:.1f} s\n")

    return 0 if all_match else 2


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
