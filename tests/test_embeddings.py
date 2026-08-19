"""
tests/test_embeddings.py
------------------------
Unit tests for embedding models (MockEmbedder & SentenceTransformerEmbedder).
"""

import pytest
from backend.embeddings import get_embedder, MockEmbedder
from backend.common.interfaces import IEmbedder


class TestMockEmbedder:

    def test_mock_embedder_contract(self):
        embedder = MockEmbedder(dimension=128)
        assert isinstance(embedder, IEmbedder)

    def test_embed_returns_vector_of_expected_dimension(self):
        embedder = MockEmbedder(dimension=64)
        vec = embedder.embed("Hello world")
        assert isinstance(vec, list)
        assert len(vec) == 64
        assert all(isinstance(x, float) for x in vec)

    def test_embed_batch_returns_list_of_vectors(self):
        embedder = MockEmbedder(dimension=32)
        texts = ["Text one", "Text two", "Text three"]
        vectors = embedder.embed_batch(texts)
        assert isinstance(vectors, list)
        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 32

    def test_deterministic_outputs(self):
        embedder = MockEmbedder(dimension=64)
        v1 = embedder.embed("Same query string")
        v2 = embedder.embed("Same query string")
        assert v1 == v2

    def test_different_texts_produce_different_vectors(self):
        embedder = MockEmbedder(dimension=64)
        v1 = embedder.embed("Text A")
        v2 = embedder.embed("Text B")
        assert v1 != v2

    @pytest.mark.anyio
    async def test_async_interface_wrappers(self):
        embedder = MockEmbedder(dimension=64)
        q_vec = await embedder.embed_query("Query string")
        assert len(q_vec) == 64

        docs_vecs = await embedder.embed_documents(["Doc 1", "Doc 2"])
        assert len(docs_vecs) == 2


class TestEmbedderFactory:

    def test_factory_returns_mock_embedder(self):
        embedder = get_embedder("mock", dimension=32)
        assert isinstance(embedder, MockEmbedder)
        assert embedder.dimension == 32

    def test_unknown_embedder_raises(self):
        with pytest.raises(ValueError, match="Unknown embedder"):
            get_embedder("nonexistent_embedder")
