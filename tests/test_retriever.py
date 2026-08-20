import pytest
from backend.common.schemas import Chunk, RetrievedChunk
from backend.embeddings.mock import MockEmbedder
from backend.retrieval.in_memory import InMemoryVectorDB
from backend.retrieval.retriever import VectorRetriever


def setup_sample_retriever() -> VectorRetriever:
    embedder = MockEmbedder(dimension=64)
    vector_db = InMemoryVectorDB()

    sample_texts = [
        ("doc1_c0", "doc1", "FastAPI is a modern, fast web framework for building APIs with Python.", {"chunk_strategy": "sentence", "chunk_index": 0}),
        ("doc2_c0", "doc2", "Retrieval-Augmented Generation (RAG) enhances LLMs with contextual retrieval.", {"chunk_strategy": "fixed_overlap", "chunk_index": 0}),
        ("doc3_c0", "doc3", "Python is an interpreted, high-level, general-purpose programming language.", {"chunk_strategy": "metadata_aware", "chunk_index": 0}),
    ]

    chunks = []
    for chunk_id, parent_id, text, meta in sample_texts:
        vec = embedder.embed(text)
        chunks.append(
            Chunk(
                id=chunk_id,
                parent_doc_id=parent_id,
                content=text,
                index=meta["chunk_index"],
                embedding=vec,
                metadata=meta,
            )
        )

    vector_db.upsert(chunks)
    return VectorRetriever(embedder=embedder, vector_db=vector_db)


class TestVectorRetriever:

    def test_retrieve_returns_correctly_shaped_retrieved_chunks(self):
        retriever = setup_sample_retriever()
        results = retriever.retrieve("FastAPI web framework", top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        for rc in results:
            assert isinstance(rc, RetrievedChunk)
            assert isinstance(rc.text, str)
            assert isinstance(rc.score, float)
            assert isinstance(rc.document_id, str)
            assert isinstance(rc.metadata, dict)
            assert "chunk_strategy" in rc.metadata

    def test_retrieve_ranking(self):
        retriever = setup_sample_retriever()
        exact_text = "Retrieval-Augmented Generation (RAG) enhances LLMs with contextual retrieval."
        results = retriever.retrieve(exact_text, top_k=3)

        assert len(results) == 3
    
        top = results[0]
        assert top.document_id == "doc2"
        assert top.text == exact_text
        assert top.score > 0.9  

    def test_empty_query_returns_empty_list(self):
        retriever = setup_sample_retriever()
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []
        assert retriever.retrieve(None) == []

    def test_no_match_query_on_empty_db(self):
        embedder = MockEmbedder(dimension=64)
        db = InMemoryVectorDB()
        retriever = VectorRetriever(embedder=embedder, vector_db=db)

        results = retriever.retrieve("Any query text")
        assert results == []

    def test_top_k_parameter_respected(self):
        retriever = setup_sample_retriever()
        results = retriever.retrieve("Python", top_k=1)
        assert len(results) == 1

        results_all = retriever.retrieve("Python", top_k=10)
        assert len(results_all) == 3

    def test_rerank_toggle_flag(self):
        retriever = setup_sample_retriever()
        query = "Python web API"
        
        results_no_rerank = retriever.retrieve(query, top_k=2, rerank=False)
        results_reranked = retriever.retrieve(query, top_k=2, rerank=True)

        assert len(results_no_rerank) == 2
        assert len(results_reranked) == 2

    @pytest.mark.anyio
    async def test_async_retrieve(self):
        retriever = setup_sample_retriever()
        results = await retriever.async_retrieve("FastAPI", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
