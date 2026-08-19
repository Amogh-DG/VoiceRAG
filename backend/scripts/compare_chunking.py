"""
backend/scripts/compare_chunking.py
------------------------------------
CLI runner comparing multiple chunking strategies (fixed_overlap, sentence, metadata_aware)
on a normalized Document dataset.

Usage:
    python -m backend.scripts.compare_chunking --input data/processed/hi_train.jsonl
    python -m backend.scripts.compare_chunking --help
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

from backend.chunking.strategies import FixedOverlapChunker, SentenceChunker, MetadataAwareChunker
from backend.common.schemas import Document, TextChunk

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_chunking")


def compare_strategies(documents: List[Document]):
    strategies = {
        "fixed_overlap": FixedOverlapChunker(chunk_size=300, overlap=50),
        "sentence": SentenceChunker(max_chars=300, sentence_overlap=1, split_on_paragraphs=True),
        "metadata_aware": MetadataAwareChunker(default_max_chars=300, description_max_chars=500, entity_max_chars=200),
    }

    print("======================================================================")
    print("           Chunking Strategies Comparison & Evaluation                ")
    print("======================================================================")
    print(f"Total Input Documents: {len(documents)}\n")
    print(f"{'Strategy':<18} | {'Chunks':<8} | {'Avg Chars':<10} | {'Min Chars':<10} | {'Max Chars':<10}")
    print("----------------------------------------------------------------------")

    for name, chunker in strategies.items():
        all_chunks: List[TextChunk] = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            print(f"{name:<18} | {'0':<8} | {'0.0':<10} | {'0':<10} | {'0':<10}")
            continue

        lengths = [len(c.text) for c in all_chunks]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        print(f"{name:<18} | {len(all_chunks):<8} | {avg_len:<10.1f} | {min_len:<10} | {max_len:<10}")

    print("======================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Compare chunking strategies on Document JSONL.")
    parser.add_argument("--input", help="Path to Document JSONL input file.")
    args = parser.parse_args()

    documents: List[Document] = []
    if args.input and Path(args.input).exists():
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(Document(**json.loads(line)))
    else:
        # Fallback sample documents for demonstration
        print("[Notice] No input file specified or found. Using built-in sample documents.")
        documents = [
            Document(
                id="doc_sample_1",
                content="Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs). RAG backend architectures typically consist of ingestion pipelines, embedding generators, search indices, and LLM text generation systems.",
                metadata={"query_type": "description", "language": "en"}
            ),
            Document(
                id="doc_sample_2",
                content="FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.",
                metadata={"query_type": "entity", "language": "en"}
            )
        ]

    compare_strategies(documents)


if __name__ == "__main__":
    main()
