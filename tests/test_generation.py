"""
tests/test_generation.py
------------------------
Unit tests for the generation module: prompt construction, grounding checks,
score threshold guardrails, structured output handling, and refusal paths.

All tests use MockLLMClient and make ZERO external API calls.
"""

import pytest
from backend.common.schemas import Chunk, GenerationResponse, LLMGenerationOutput, RetrievedChunk
from backend.generation.generator import RAGGenerator
from backend.generation.llm_client import MockLLMClient


def make_retrieved_chunk(
    text: str = "FastAPI is a modern web framework.",
    score: float = 0.85,
    doc_id: str = "doc101",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        document_id=doc_id,
        metadata={"chunk_strategy": "sentence"},
    )


class TestRAGGenerator:

    def test_sufficient_context_returns_grounded_answer(self):
        """(a) Sufficient context + high score -> grounded answer."""
        mock_output = LLMGenerationOutput(
            answer="FastAPI is a Python web framework.",
            grounded=True,
            insufficient=False,
        )
        client = MockLLMClient(override_output=mock_output)
        generator = RAGGenerator(llm_client=client, score_threshold=0.15)

        chunks = [make_retrieved_chunk(score=0.85)]
        response = generator.generate("What is FastAPI?", chunks)

        assert isinstance(response, GenerationResponse)
        assert response.is_grounded is True
        assert response.is_refused is False
        assert response.answer == "FastAPI is a Python web framework."
        assert "citations" in response.metadata
        assert "doc101" in response.metadata["citations"]

    def test_empty_context_triggers_refusal_path(self):
        """(b) Empty context -> refusal path without calling LLM."""
        client = MockLLMClient()
        generator = RAGGenerator(llm_client=client, score_threshold=0.15)

        response = generator.generate("What is FastAPI?", [])

        assert response.is_grounded is False
        assert response.is_refused is True
        assert "no relevant context passages were retrieved" in response.answer.lower()
        assert response.metadata["refusal_reason"] == "empty_context"

    def test_low_scoring_context_below_threshold_triggers_refusal_path(self):
        """(c) Low-scoring context below threshold -> refusal path."""
        client = MockLLMClient()
        generator = RAGGenerator(llm_client=client, score_threshold=0.50)

        # Score is 0.10, which is below 0.50 threshold
        chunks = [make_retrieved_chunk(score=0.10)]
        response = generator.generate("What is FastAPI?", chunks)

        assert response.is_grounded is False
        assert response.is_refused is True
        assert "relevance threshold" in response.answer.lower()
        assert response.metadata["refusal_reason"] == "score_below_threshold"

    def test_configurable_score_threshold(self):
        """Configurable score threshold behavior check."""
        chunks = [make_retrieved_chunk(score=0.30)]
        mock_output = LLMGenerationOutput(answer="Answer text", grounded=True, insufficient=False)

        # Strict generator (threshold 0.40) -> Refused
        strict_gen = RAGGenerator(llm_client=MockLLMClient(mock_output), score_threshold=0.40)
        resp_strict = strict_gen.generate("Question", chunks)
        assert resp_strict.is_refused is True

        # Permissive generator (threshold 0.20) -> Grounded
        permissive_gen = RAGGenerator(llm_client=MockLLMClient(mock_output), score_threshold=0.20)
        resp_perm = permissive_gen.generate("Question", chunks)
        assert resp_perm.is_refused is False
        assert resp_perm.is_grounded is True

    def test_llm_insufficient_signal_triggers_refusal_path(self):
        """(d) LLM structured output signals insufficient=True -> refusal path."""
        mock_output = LLMGenerationOutput(
            answer="The context does not state the author.",
            grounded=False,
            insufficient=True,
        )
        client = MockLLMClient(override_output=mock_output)
        generator = RAGGenerator(llm_client=client, score_threshold=0.15)

        chunks = [make_retrieved_chunk(score=0.90)]
        response = generator.generate("Who authored the text?", chunks)

        assert response.is_grounded is False
        assert response.is_refused is True
        assert response.answer == "The context does not state the author."
        assert response.metadata["refusal_reason"] == "llm_insufficient_signal"

    def test_accepts_standard_chunk_schema(self):
        """Accepts internal Chunk model as well as RetrievedChunk."""
        mock_output = LLMGenerationOutput(answer="Extracted fact", grounded=True, insufficient=False)
        generator = RAGGenerator(llm_client=MockLLMClient(mock_output))

        chunk = Chunk(
            id="c1",
            parent_doc_id="d1",
            content="Context content here",
            index=0,
            metadata={"score": 0.88},
        )

        response = generator.generate("Question?", [chunk])
        assert response.is_grounded is True
        assert response.is_refused is False

    @pytest.mark.anyio
    async def test_async_generate_wrapper(self):
        mock_output = LLMGenerationOutput(answer="Async answer", grounded=True, insufficient=False)
        generator = RAGGenerator(llm_client=MockLLMClient(mock_output))

        chunks = [make_retrieved_chunk(score=0.80)]
        response = await generator.async_generate("Question?", chunks)
        assert response.is_grounded is True
        assert response.answer == "Async answer"
