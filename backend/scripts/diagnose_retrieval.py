"""
backend/scripts/diagnose_retrieval.py
---------------------------------------
Diagnostic script for RAG retrieval correctness on indexed JSON datasets.

Steps:
1. Stream `data/indices/multilingual_sample_index.json` via ijson.
2. Find the first chunk containing the exact term 'निगम' or 'कंपनी'.
3. Use that chunk's exact text content as the query string.
4. Embed the query with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
5. Compute cosine similarity against all 15,000 indexed chunk vectors.
6. Print original chunk details (ID, embedding norm, text) and top 5 results.
7. Verify whether Rank #1 matches the original chunk ID.

Usage:
    python -m backend.scripts.diagnose_retrieval --index-path data/indices/multilingual_sample_index.json
    python -m backend.scripts.diagnose_retrieval --help
"""

import argparse
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
logger = logging.getLogger("diagnose_retrieval")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose retrieval correctness by querying with an exact indexed chunk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-path",
        default="data/indices/multilingual_sample_index.json",
        help="Path to the vector index JSON file.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top similarity results to return.",
    )
    return parser.parse_args(argv)


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

    logger.info("Opening index file: %s", index_path)
    logger.info("Streaming records via ijson...")

    start_time = time.perf_counter()
    metadata_list = []
    vector_list = []
    target_chunk = None

    with index_path.open("rb") as f:
        for record in ijson.items(f, "item"):
            vec = record.get("embedding")
            if vec is None:
                continue

            raw_arr = np.array(vec, dtype=np.float32)
            raw_norm = float(np.linalg.norm(raw_arr))

            norm_arr = raw_arr / raw_norm if raw_norm > 0 else raw_arr
            vector_list.append(norm_arr)

            doc_id = record.get("parent_doc_id") or record.get("document_id") or record.get("id")
            content = record.get("content") or record.get("text") or ""
            chunk_data = {
                "id": record.get("id"),
                "parent_doc_id": doc_id,
                "content": content,
                "raw_norm": raw_norm,
                "metadata": record.get("metadata") or {},
            }
            metadata_list.append(chunk_data)

            # Look for target chunk containing 'निगम' or 'कंपनी'
            if target_chunk is None and ("निगम" in content or "कंपनी" in content):
                target_chunk = chunk_data

    if not vector_list:
        logger.error("No valid vectors loaded from index.")
        return 1

    if target_chunk is None:
        logger.warning("No chunk containing 'निगम' or 'कंपनी' was found. Falling back to chunk #0.")
        target_chunk = metadata_list[0]

    logger.info("Loaded %d records in %.2fs", len(vector_list), time.perf_counter() - start_time)
    vectors = np.vstack(vector_list)

    # Instantiate embedding model
    logger.info("Initializing model '%s'...", args.model_name)
    embedder = SentenceTransformerEmbedder(model_name=args.model_name)

    logger.info("Embedding exact target chunk content...")
    q_start = time.perf_counter()
    query_text = target_chunk["content"]
    q_vec = embedder.embed(query_text)
    q_arr = np.array(q_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_arr))
    if q_norm > 0:
        q_arr = q_arr / q_norm

    # Matrix multiplication dot product search
    scores = np.dot(vectors, q_arr)
    top_indices = np.argpartition(-scores, args.top_k)[:args.top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    search_time = (time.perf_counter() - q_start) * 1000.0

    print("\n" + "=" * 75)
    print("                 RETRIEVAL CORRECTNESS DIAGNOSTIC                      ")
    print("=" * 75)
    print(f"Target Chunk ID:      {target_chunk['id']}")
    print(f"Target Parent Doc ID: {target_chunk['parent_doc_id']}")
    print(f"Target Vector Norm:   {target_chunk['raw_norm']:.6f} (Expected: ~1.0 for normalized vectors)")
    print(f"Query Vector Norm:    {q_norm:.6f}")
    target_text_preview = str(query_text[:150]).encode("utf-8", "replace").decode("utf-8")
    print(f"Target Text Content:  \"{target_text_preview}...\"")
    print("=" * 75)
    print(f"\nTop {args.top_k} Search Results:")

    rank1_matches = False
    for rank, idx in enumerate(top_indices, 1):
        meta = metadata_list[idx]
        score = float(scores[idx])
        content_preview = str(meta["content"][:120]).encode("utf-8", "replace").decode("utf-8")
        is_target = meta["id"] == target_chunk["id"]
        if rank == 1 and is_target:
            rank1_matches = True

        flag = " [MATCH - TARGET CHUNK]" if is_target else ""
        print(f"\n--- Rank #{rank} (Score: {score:.6f}){flag} ---")
        print(f"Chunk ID:      {meta['id']}")
        print(f"Parent Doc ID: {meta['parent_doc_id']}")
        print(f"Text Preview:  {content_preview}...")

    print("\n" + "=" * 75)
    if rank1_matches:
        print("✅ DIAGNOSTIC PASSED: The exact target chunk ranked #1!")
    else:
        print("❌ DIAGNOSTIC WARNING: Target chunk did NOT rank #1.")
    print("=" * 75 + "\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
