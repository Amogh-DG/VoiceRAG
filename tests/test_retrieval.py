"""
tests/test_retrieval.py
-----------------------
Smoke tests and unit tests for the vector database storage and query layer.
"""

import json
import pytest
from pathlib import Path
from backend.common.schemas import Chunk, TextChunk
from backend.embeddings.mock import MockEmbedder
from backend.retrieval.in_memory import InMemoryVectorDB
from backend.retrieval import get_vector_db


def create_sample_chunks(embedder: MockEmbedder) -> list[Chunk]:
    """Helper to create sample Chunk objects with embeddings."""
    texts = [
        ("doc1_c0", "doc1", "FastAPI is a modern, fast web framework for building APIs with Python.", {"chunk_strategy": "sentence", "chunk_index": 0}),
        ("doc2_c0", "doc2", "Retrieval-Augmented Generation (RAG) enhances LLMs with contextual retrieval.", {"chunk_strategy": "fixed_overlap", "chunk_index": 0}),
        ("doc3_c0", "doc3", "Python is an interpreted, high-level, general-purpose programming language.", {"chunk_strategy": "metadata_aware", "chunk_index": 0}),
    ]

    chunks = []
    for chunk_id, parent_id, text, meta in texts:
        vec = embedder.embed(text)
        chunk = Chunk(
            id=chunk_id,
            parent_doc_id=parent_id,
            content=text,
            index=meta["chunk_index"],
            embedding=vec,
            metadata=meta,
        )
        chunks.append(chunk)
    return chunks


class TestInMemoryVectorDB:

    def test_smoke_test_index_and_query(self):
        """
        Smoke test: Index a handful of known chunks, query with a related string,
        and assert the expected chunk comes back in top_k.
        """
        embedder = MockEmbedder(dimension=64)
        chunks = create_sample_chunks(embedder)
        db = InMemoryVectorDB()

        # 1. Index chunks
        db.upsert(chunks)

        # 2. Query with string related to doc2 ("Retrieval-Augmented Generation (RAG)")
        query_text = "Retrieval-Augmented Generation (RAG) enhances LLMs with contextual retrieval."
        query_vec = embedder.embed(query_text)

        results = db.query(query_vec, top_k=1)

        # 3. Assert top result is doc2_c0
        assert len(results) == 1
        top_chunk = results[0]
        assert top_chunk.id == "doc2_c0"
        assert top_chunk.parent_doc_id == "doc2"
        assert top_chunk.metadata["chunk_strategy"] == "fixed_overlap"
        assert "RAG" in top_chunk.content

    def test_metadata_filtering(self):
        embedder = MockEmbedder(dimension=64)
        chunks = create_sample_chunks(embedder)
        db = InMemoryVectorDB()
        db.upsert(chunks)

        query_vec = embedder.embed("Python programming")

        # Filter specifically for sentence strategy
        results = db.query(query_vec, top_k=5, filters={"chunk_strategy": "sentence"})
        assert len(results) == 1
        assert results[0].id == "doc1_c0"

    def test_persistence_save_and_load(self, tmp_path):
        embedder = MockEmbedder(dimension=64)
        chunks = create_sample_chunks(embedder)
        save_file = tmp_path / "test_index.json"

        # Create DB, insert, save
        db1 = InMemoryVectorDB(persist_path=str(save_file))
        db1.upsert(chunks)
        assert save_file.exists()

        # Create new DB instance loading from file
        db2 = InMemoryVectorDB(persist_path=str(save_file))
        assert len(db2.chunks) == 3

        query_vec = embedder.embed(chunks[0].content)
        results = db2.query(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0].id == "doc1_c0"

    @pytest.mark.anyio
    async def test_async_interface_wrappers(self):
        embedder = MockEmbedder(dimension=64)
        chunks = create_sample_chunks(embedder)
        db = InMemoryVectorDB()

        await db.index_chunks(chunks)
        query_vec = embedder.embed(chunks[0].content)
        results = await db.similarity_search(query_vec, k=2)
        assert len(results) == 2


class TestEmbedAndIndexScript:

    def test_script_end_to_end(self, tmp_path):
        """Integration test for scripts/embed_and_index.py."""
        from backend.scripts.embed_and_index import run, parse_args

        # 1. Prepare temporary TextChunk JSONL input file
        input_file = tmp_path / "text_chunks.jsonl"
        output_db_file = tmp_path / "vector_db.json"

        text_chunks = [
            TextChunk(text="Chunk 1 content for testing", document_id="doc1", metadata={"chunk_strategy": "sentence", "chunk_index": 0}),
            TextChunk(text="Chunk 2 content for testing", document_id="doc2", metadata={"chunk_strategy": "sentence", "chunk_index": 0}),
        ]

        with input_file.open("w", encoding="utf-8") as f:
            for tc in text_chunks:
                f.write(json.dumps(tc.model_dump(), ensure_ascii=False) + "\n")

        # 2. Run embed_and_index CLI
        args = parse_args([
            "--input", str(input_file),
            "--output-db", str(output_db_file),
            "--embedder", "mock",
            "--batch-size", "2",
        ])

        exit_code = run(args)
        assert exit_code == 0
        assert output_db_file.exists()

        # 3. Load DB and query to verify contents
        db = get_vector_db("in_memory", persist_path=str(output_db_file))
        assert len(db.chunks) == 2
        assert db.chunks[0].metadata["chunk_strategy"] == "sentence"
