"""
backend/scripts/diagnose_semantic_similarity.py
-------------------------------------------------
Diagnostic tool for testing semantic vector similarity between queries and a known chunk.

Steps:
1. Load target chunk `1102432_hi_p0_c0` from source chunks (`data/chunks/hi_val_sentence.jsonl`).
2. Embed the target chunk using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. Embed Hindi query: "निगम या कंपनी की परिभाषा क्या है?"
4. Embed English query: "What is the definition of a corporation or company?"
5. Compute and print cosine similarity scores for all three comparisons.

Usage:
    python -m backend.scripts.diagnose_semantic_similarity
"""

import argparse
import json
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
logger = logging.getLogger("diagnose_semantic")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose semantic embedding similarity for a specific chunk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunks-path",
        default="data/chunks/hi_val_sentence.jsonl",
        help="Path to source TextChunk JSONL file.",
    )
    parser.add_argument(
        "--target-doc-id",
        default="1102432_hi_p0",
        help="Parent document ID of target chunk.",
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
    from backend.embeddings.sentence_transformer import SentenceTransformerEmbedder

    chunks_path = Path(args.chunks_path)
    target_text = None

    # Step 1: Find target chunk in source JSONL file
    if chunks_path.exists():
        logger.info("Searching for target chunk in %s...", chunks_path)
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    doc_id = obj.get("document_id")
                    text = obj.get("text", "")
                    if doc_id == args.target_doc_id or "एक कंपनी एक विशिष्ट देश में निगमित होती है" in text:
                        target_text = text
                        logger.info("Found target chunk! Document ID: %s", doc_id)
                        break
                except Exception:
                    continue

    # Fallback to known text snippet if file not found or record omitted
    if not target_text:
        logger.info("Using fallback target text for document ID: %s", args.target_doc_id)
        target_text = (
            "एक कंपनी एक विशिष्ट देश में निगमित होती है, अक्सर उस देश के एक छोटे उपसमूह, "
            "जैसे कि एक राज्य या प्रांत, की सीमाओं के भीतर। निगम तब उस राज्य में निगमन के लिए "
            "कानून लागू करता है।"
        )

    # Step 2: Initialize model once
    logger.info("Initializing model '%s'...", args.model_name)
    embedder = SentenceTransformerEmbedder(model_name=args.model_name)

    # Step 3: Define queries
    hindi_query = "निगम या कंपनी की परिभाषा क्या है?"
    english_query = "What is the definition of a corporation or company?"

    logger.info("Generating vector embeddings...")
    t0 = time.perf_counter()

    target_vec = embedder.embed(target_text)
    hindi_vec = embedder.embed(hindi_query)
    english_vec = embedder.embed(english_query)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Step 4: Calculate cosine similarities
    self_sim = _cos_sim(target_vec, target_vec)
    hindi_sim = _cos_sim(hindi_vec, target_vec)
    english_sim = _cos_sim(english_vec, target_vec)

    target_preview = str(target_text[:140]).encode("utf-8", "replace").decode("utf-8")
    hindi_preview = str(hindi_query).encode("utf-8", "replace").decode("utf-8")

    # Step 5: Report output
    print("\n" + "=" * 75)
    print("               SEMANTIC EMBEDDING SIMILARITY DIAGNOSTIC                 ")
    print("=" * 75)
    print(f"Model Name:           {args.model_name}")
    print(f"Embedding Time:       {elapsed_ms:.2f} ms")
    print("-" * 75)
    print(f"Target Chunk Text:    \"{target_preview}...\"")
    print(f"Hindi Query:          \"{hindi_preview}\"")
    print(f"English Query:        \"{english_query}\"")
    print("=" * 75)
    print("                       SIMILARITY SCORES                                ")
    print("=" * 75)
    print(f"1. Target Chunk vs. Self (Sanity Check):  {self_sim:.4f}")
    print(f"2. Hindi Query vs. Target Chunk:          {hindi_sim:.4f}")
    print(f"3. English Query vs. Target Chunk:        {english_sim:.4f}")
    print("=" * 75 + "\n")

    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
