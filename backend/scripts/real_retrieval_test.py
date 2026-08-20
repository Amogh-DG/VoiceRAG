"""
backend/scripts/real_retrieval_test.py
---------------------------------------
Real Retrieval Smoke Test on vector index datasets.

Memory Efficiency
─────────────────
Instead of json.load() which consumes >12 GB RAM, this script streams the index
via `ijson` into a compact float32 NumPy matrix (1.64 GB RAM) and lightweight
metadata list (~250 MB RAM).

Vector Search
─────────────
Uses matrix-vector dot product (cosine similarity) over the 384-dim vectors.

Usage
─────
    python -m backend.scripts.real_retrieval_test --index-path data/indices/multilingual_sample_index.json
    python -m backend.scripts.real_retrieval_test --help
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
logger = logging.getLogger("real_retrieval_test")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real retrieval query against vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-path",
        default="data/indices/multilingual_sample_index.json",
        help="Path to the vector index JSON file.",
    )
    parser.add_argument(
        "--query",
        default="निगम या कंपनी की परिभाषा क्या है?",
        help="Query string to search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top similarity results to return.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name.",
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

    file_size_gb = index_path.stat().st_size / (1024 ** 3)
    logger.info("Opening index file: %s (%.2f GB)", index_path, file_size_gb)
    logger.info("Streaming vectors & metadata into compact memory structures...")

    start_time = time.perf_counter()
    metadata_list = []
    vector_list = []

    with index_path.open("rb") as f:
        for record in ijson.items(f, "item"):
            vec = record.get("embedding")
            if vec is None:
                continue

            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            vector_list.append(arr)

            doc_id = record.get("parent_doc_id") or record.get("document_id") or record.get("id")
            metadata_list.append({
                "id": record.get("id"),
                "parent_doc_id": doc_id,
                "content": record.get("content") or record.get("text") or "",
                "metadata": record.get("metadata") or {},
            })

            if len(vector_list) % 200_000 == 0:
                elapsed = time.perf_counter() - start_time
                logger.info("Loaded %d records... (elapsed: %.1fs)", len(vector_list), elapsed)

    if not vector_list:
        logger.error("No valid vectors loaded from index.")
        return 1

    load_time = time.perf_counter() - start_time
    logger.info("Converting %d vectors to NumPy matrix...", len(vector_list))
    vectors = np.vstack(vector_list)
    logger.info("Index matrix ready! Shape: %s (RAM: %.2f GB). Total load time: %.1fs",
                vectors.shape, vectors.nbytes / (1024 ** 3), load_time)

    # Embed query using configured model
    logger.info("Initializing embedding model '%s'...", args.model_name)
    embedder = SentenceTransformerEmbedder(model_name=args.model_name)
    
    logger.info("Embedding query: '%s'...", args.query)
    q_start = time.perf_counter()
    q_vec = embedder.embed(args.query)
    q_arr = np.array(q_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q_arr)
    if q_norm > 0:
        q_arr = q_arr / q_norm

    # Perform similarity search via matrix product
    search_start = time.perf_counter()
    scores = np.dot(vectors, q_arr)
    
    top_indices = np.argpartition(-scores, args.top_k)[:args.top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    search_time = (time.perf_counter() - search_start) * 1000.0
    total_query_time = (time.perf_counter() - q_start) * 1000.0

    print("\n" + "=" * 75)
    print("                 REAL RETRIEVAL SMOKE TEST RESULTS                     ")
    print("=" * 75)
    print(f"Model Name:          {args.model_name}")
    print(f"Query Text:          \"{args.query}\"")
    print(f"Total Chunks Searched:{len(metadata_list):,}")
    print(f"Embedding Time:      {total_query_time - search_time:.2f} ms")
    print(f"Matrix Search Time:  {search_time:.2f} ms")
    print("=" * 75)

    for rank, idx in enumerate(top_indices, 1):
        meta = metadata_list[idx]
        score = float(scores[idx])
        content_preview = str(meta['content'][:150]).encode("utf-8", "replace").decode("utf-8")
        
        print(f"\n--- Rank #{rank} (Similarity Score: {score:.4f}) ---")
        print(f"Chunk ID:      {meta['id']}")
        print(f"Parent Doc ID: {meta['parent_doc_id']}")
        print(f"Strategy:      {meta['metadata'].get('chunk_strategy', 'N/A')}")
        print(f"Text Preview:  {content_preview}...")

    print("\n" + "=" * 75)
    print("✅ RETRIEVAL SUCCESSFUL: Real similarity search returned ranked chunks.")
    print("=" * 75 + "\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
