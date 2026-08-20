"""
backend/scripts/diagnose_stored_vector.py
-------------------------------------------
Targeted diagnostic for checking stored vector integrity in `hi_val_multilingual_index.json`.

Steps:
1. Stream `data/indices/hi_val_multilingual_index.json` via `ijson` to locate chunk `1102432_hi_p0_c0`.
2. Extract stored embedding vector and chunk content.
3. Instantiate `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
4. Generate fresh query embedding for "निगम या कंपनी की परिभाषा क्या है?".
5. Generate fresh chunk embedding for the chunk's own content.
6. Calculate and print cosine similarities:
   - Stored Embedding vs Fresh Query Embedding
   - Stored Embedding vs Fresh Chunk Embedding

Usage:
    python -m backend.scripts.diagnose_stored_vector
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np

# Reconfigure stdout for UTF-8 output to handle Hindi text formatting cleanly
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
logger = logging.getLogger("diagnose_stored_vector")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose stored vector similarity against fresh embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-path",
        default="data/indices/hi_val_multilingual_index.json",
        help="Path to multilingual vector index JSON file.",
    )
    parser.add_argument(
        "--target-chunk-id",
        default="1102432_hi_p0_c0",
        help="Chunk ID to locate.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name.",
    )
    return parser.parse_args(argv)


def _cos_sim(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two float vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a / norm_a, b / norm_b))


def run(args: argparse.Namespace) -> int:
    try:
        import ijson
    except ImportError:
        logger.error("ijson package required. Run: pip install ijson")
        return 1

    from backend.embeddings.sentence_transformer import SentenceTransformerEmbedder

    index_path = Path(args.index_path)
    if not index_path.exists():
        logger.error("Index file not found: %s", index_path)
        return 1

    logger.info("Searching for target chunk '%s' in %s...", args.target_chunk_id, index_path)

    stored_record = None
    start_time = time.perf_counter()

    with index_path.open("rb") as f:
        for record in ijson.items(f, "item"):
            rec_id = record.get("id")
            parent_id = record.get("parent_doc_id")
            content = record.get("content") or record.get("text") or ""
            if rec_id == args.target_chunk_id or parent_id == "1102432_hi_p0" or "एक कंपनी एक विशिष्ट देश में निगमित होती है" in content:
                stored_record = record
                break

    if not stored_record:
        logger.error("Target chunk '%s' not found in index %s", args.target_chunk_id, index_path)
        return 1

    logger.info("Found target chunk in %.2fs!", time.perf_counter() - start_time)

    stored_vec = stored_record.get("embedding")
    chunk_content = stored_record.get("content") or stored_record.get("text") or ""

    if not stored_vec:
        logger.error("Stored chunk has no embedding field.")
        return 1

    # Load model once
    logger.info("Initializing model '%s'...", args.model_name)
    embedder = SentenceTransformerEmbedder(model_name=args.model_name)

    query_text = "निगम या कंपनी की परिभाषा क्या है?"
    logger.info("Generating fresh query embedding...")
    fresh_query_vec = embedder.embed(query_text)

    logger.info("Generating fresh chunk embedding...")
    fresh_chunk_vec = embedder.embed(chunk_content)

    # Compute similarities
    stored_vs_query_sim = _cos_sim(stored_vec, fresh_query_vec)
    stored_vs_fresh_chunk_sim = _cos_sim(stored_vec, fresh_chunk_vec)

    chunk_preview = str(chunk_content[:140]).encode("utf-8", "replace").decode("utf-8")
    query_preview = str(query_text).encode("utf-8", "replace").decode("utf-8")

    print("\n" + "=" * 75)
    print("            STORED VECTOR SIMILARITY DIAGNOSTIC REPORT                  ")
    print("=" * 75)
    print(f"Index File:               {index_path}")
    print(f"Target Chunk ID:          {stored_record.get('id')}")
    print(f"Model Name:               {args.model_name}")
    print("-" * 75)
    print(f"Chunk Text Preview:       \"{chunk_preview}...\"")
    print(f"Query Text:               \"{query_preview}\"")
    print("=" * 75)
    print("                       SIMILARITY SCORES                                ")
    print("=" * 75)
    print(f"1. Stored Vector vs. Fresh Query Embedding:       {stored_vs_query_sim:.4f}")
    print(f"2. Stored Vector vs. Fresh Chunk Embedding:       {stored_vs_fresh_chunk_sim:.4f}")
    print("=" * 75 + "\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
