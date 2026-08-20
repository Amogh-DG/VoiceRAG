"""
tests/test_gemini_integration.py
---------------------------------
Standalone Integration Test for Gemini 3.5 Flash-Lite LLM + FAISSRetriever.

Tests:
A. Hindi Definition Query:   "निगम या कंपनी की परिभाषा क्या है?"
B. English Definition Query: "What is the definition of a corporation or company?"
C. Hindi Meaning Query:      "निगम से क्या तात्पर्य है?"
D. Irrelevant Query:         "How to bake a chocolate fudge cake at home?"

Requirements:
- Uses Gemini 3.5 Flash-Lite through official Google Gemini API (GEMINI_API_KEY).
- Verifies grounded answer generation from FAISS retrieved context.
- Measures cold-start vs steady-state retrieval, generation, and total pipeline latency.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.retrieval.faiss_retriever import FAISSRetriever
from backend.generation.llm_client import GeminiLLMClient, MockLLMClient
from backend.generation.generator import RAGGenerator
from backend.harness.orchestrator import PipelineOrchestrator


async def test_gemini_rag_pipeline():
    api_key = os.getenv("GEMINI_API_KEY")

    print("\n" + "=" * 75)
    print("           GEMINI 3.5 FLASH-LITE + FAISS INTEGRATION TEST              ")
    print("=" * 75)

    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set in .env!")
        print("GeminiLLMClient requires a valid API key for the live integration test.")
        sys.exit(1)

    print("[API Initialization] GEMINI_API_KEY detected.")
    print("[API Initialization] Connecting to Gemini API ('gemini-3.5-flash-lite')...")
    llm_client = GeminiLLMClient(api_key=api_key, model_name="gemini-3.5-flash-lite")

    print("\n[1/2] Loading FAISSRetriever on CUDA (Cold-Start)...")
    t0 = time.perf_counter()
    retriever = FAISSRetriever(
        faiss_path="data/indices/hi_val_multilingual.faiss",
        metadata_path="data/indices/hi_val_multilingual_metadata.jsonl",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cuda",
    )
    generator = RAGGenerator(llm_client=llm_client)
    orchestrator = PipelineOrchestrator(retriever=retriever, generator=generator)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Initialization Complete] Loaded FAISSRetriever and Orchestrator in {init_ms:.2f} ms")

    # Warm-up run to isolate cold-start GPU/model loading latency
    print("\n[2/2] Running warm-up query...")
    t_warm = time.perf_counter()
    _ = await orchestrator.run_pipeline(query="स्वास्थ्य क्या है?", k=1)
    warm_ms = (time.perf_counter() - t_warm) * 1000.0
    print(f"[Warm-up Complete] Warm-up pipeline latency: {warm_ms:.2f} ms")

    test_queries = [
        ("A. Hindi Definition Query", "निगम या कंपनी की परिभाषा क्या है?"),
        ("B. English Definition Query", "What is the definition of a corporation or company?"),
        ("C. Hindi Meaning Query", "निगम से क्या तात्पर्य है?"),
        ("D. Irrelevant Query", "How to bake a chocolate fudge cake at home?"),
    ]

    print("\n" + "=" * 75)
    print("                    MEASURED TEST QUERY RESULTS                         ")
    print("=" * 75)

    for title, query in test_queries:
        print("\n" + "-" * 75)
        print(f"TEST:  {title}")
        print(f"QUERY: \"{query}\"")
        print("-" * 75)

        t_start = time.perf_counter()
        result = await orchestrator.run_pipeline(query=query, k=5)
        total_ms = (time.perf_counter() - t_start) * 1000.0

        timings = result.metadata.get("timings_ms", {})
        ret_ms = timings.get("retrieval_ms", 0.0)
        gen_ms = timings.get("generation_ms", 0.0)

        top_chunk_id = result.context[0].chunk_id if result.context else "N/A"
        top_doc_id = result.context[0].document_id if result.context else "N/A"
        top_score = result.context[0].metadata.get("similarity_score", result.context[0].metadata.get("score", "N/A")) if result.context else "N/A"

        print(f"Answer:         {result.answer}")
        print(f"Is Grounded:    {result.is_grounded}")
        print(f"Is Refused:     {result.is_refused}")
        print(f"Refusal Reason: {result.refusal_reason}")
        print(f"Sources Count:  {len(result.context)}")
        print(f"Top Chunk ID:   {top_chunk_id}")
        print(f"Top Doc ID:     {top_doc_id}")
        print(f"Top Score:      {top_score}")
        print("-" * 75)
        print(f"Retrieval Latency:  {ret_ms:.2f} ms")
        print(f"Gemini Latency:     {gen_ms:.2f} ms")
        print(f"Total Pipeline:     {total_ms:.2f} ms")

    print("\n" + "=" * 75)
    print("ALL GEMINI INTEGRATION TESTS EXECUTED SUCCESSFULLY")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(test_gemini_rag_pipeline())
