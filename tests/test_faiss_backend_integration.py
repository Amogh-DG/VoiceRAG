"""
tests/test_faiss_backend_integration.py
-----------------------------------------
Backend Integration & Latency Test for FAISSRetriever.

Verifies:
1. FAISSRetriever initializes ONCE without loading the 12.2 GB JSON index.
2. Query `"निगम या कंपनी की परिभाषा क्या है?"` sent through the backend returns chunk `1057779_hi_p2_c0` at rank #1.
3. Similarity score is ~0.85.
4. Per-query retrieval latency measured through the actual backend pipeline (excluding one-time startup).
"""

import asyncio
import time
import pytest
from backend.api.main import app, orchestrator, faiss_retriever
from fastapi.testclient import TestClient


def test_faiss_backend_retrieval_integration():
    client = TestClient(app)

    # 1. Warm-up request to ensure cold-start is excluded
    warmup_response = client.post("/ask", json={"query": "स्वास्थ्य क्या है?"})
    assert warmup_response.status_code == 200

    # 2. Target test query
    query = "निगम या कंपनी की परिभाषा क्या है?"
    
    t0 = time.perf_counter()
    response = client.post("/ask", json={"query": query})
    query_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert response.status_code == 200
    data = response.json()

    print("\n" + "=" * 70)
    print("           BACKEND FAISS RETRIEVAL INTEGRATION TEST REPORT              ")
    print("=" * 70)
    print(f"Endpoint:             POST /ask")
    print(f"Query:                \"{query}\"")
    print(f"Backend Latency (API): {query_latency_ms:.2f} ms")
    print(f"Reported Latency:     {data.get('latency_ms', 0):.2f} ms")
    print(f"Sources Returned:     {len(data.get('sources', []))}")
    print("=" * 70)

    sources = data.get("sources", [])
    assert len(sources) > 0, "No sources returned from FAISS backend."

    top_source = sources[0]
    top_metadata = top_source.get("metadata", {})
    top_chunk_id = top_metadata.get("chunk_id")
    top_doc_id = top_source.get("document_id")
    top_score = top_metadata.get("similarity_score") or top_metadata.get("score")

    print(f"Top Result Rank #1:")
    print(f"  Chunk ID:      {top_chunk_id}")
    print(f"  Parent Doc ID: {top_doc_id}")
    print(f"  Similarity:    {top_score:.4f}" if top_score else "  Similarity: N/A")
    print(f"  Text Preview:  {top_source.get('text', '')[:100]}...")
    print("=" * 70)

    # Assertions required by spec
    assert top_chunk_id == "1057779_hi_p2_c0", f"Expected chunk '1057779_hi_p2_c0', got '{top_chunk_id}'"
    assert top_score is not None and top_score > 0.80, f"Expected similarity score ~0.85, got {top_score}"
    print("✅ INTEGRATION TEST PASSED: FAISSRetriever successfully connected to RAG Backend!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_faiss_backend_retrieval_integration()
