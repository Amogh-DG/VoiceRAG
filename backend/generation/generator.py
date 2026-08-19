"""
backend/generation/generator.py
--------------------------------
RAG Generation logic using structured LLM outputs, grounding prompts,
and configurable threshold guardrails for refusal/insufficient-information handling.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from backend.common.interfaces import IGenerator
from backend.common.schemas import Chunk, GenerationResponse, LLMGenerationOutput, RetrievedChunk
from backend.generation.llm_client import ILLMClient, MockLLMClient

logger = logging.getLogger(__name__)

# Default system prompt enforcing strict grounding in retrieved context
DEFAULT_SYSTEM_PROMPT = (
    "You are a factual Retrieval-Augmented Generation (RAG) assistant.\n"
    "Your primary duty is to answer the user's question strictly using only the provided context passages.\n"
    "Do NOT use outside knowledge, speculate, or fabricate answers.\n"
    "If the provided context does not contain sufficient facts to answer the question, or if the context is irrelevant, "
    "set 'insufficient': true, 'grounded': false, and state clearly in 'answer' that the context is insufficient."
)


def _to_retrieved_chunk(chunk: Union[RetrievedChunk, Chunk]) -> RetrievedChunk:
    """Normalize Chunk or RetrievedChunk into a RetrievedChunk."""
    if isinstance(chunk, RetrievedChunk):
        return chunk
    # Handle standard Chunk objects
    return RetrievedChunk(
        text=chunk.content,
        score=float(chunk.metadata.get("score", 1.0)),
        document_id=chunk.parent_doc_id,
        metadata=chunk.metadata,
    )


class RAGGenerator(IGenerator):
    """
    RAG Generator component enforcing grounding and insufficient-information refusal policies.

    Args:
        llm_client:      Client for structured LLM completions (defaults to MockLLMClient).
        score_threshold: Minimum relevance score required for retrieved chunks (default 0.15).
        system_prompt:   Custom system prompt override.
    """

    def __init__(
        self,
        llm_client: Optional[ILLMClient] = None,
        score_threshold: float = 0.15,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.llm_client = llm_client or MockLLMClient()
        self.score_threshold = score_threshold
        self.system_prompt = system_prompt

    def _build_user_prompt(
        self, question: str, chunks: List[RetrievedChunk]
    ) -> str:
        """Formats question and context passages with chunk citations."""
        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            doc_id = chunk.document_id
            score = chunk.score
            context_blocks.append(
                f"[Passage {idx}] (doc_id: {doc_id}, score: {score:.4f})\n{chunk.text}"
            )

        context_str = "\n\n".join(context_blocks)
        return (
            f"Question: {question}\n\n"
            f"Context Passages:\n{context_str}\n\n"
            f"Provide a structured response with 'answer', 'grounded', and 'insufficient'."
        )

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Union[RetrievedChunk, Chunk]],
    ) -> GenerationResponse:
        """
        Generates a grounded answer from retrieved context or triggers the refusal path.

        Args:
            question: Original user question.
            retrieved_chunks: List of retrieved context chunks.

        Returns:
            GenerationResponse: Structured response containing answer, grounding status, refusal flag.
        """
        norm_chunks = [_to_retrieved_chunk(c) for c in retrieved_chunks]

        # ── Refusal Path 1: Empty context list ─────────────────────────────────
        if not norm_chunks:
            logger.info("Refusal path triggered: Empty context list.")
            return GenerationResponse(
                answer="I am sorry, but no relevant context passages were retrieved to answer your question.",
                is_grounded=False,
                is_refused=True,
                metadata={
                    "refusal_reason": "empty_context",
                    "score_threshold": self.score_threshold,
                    "max_retrieved_score": 0.0,
                },
            )

        # ── Refusal Path 2: All chunks below relevance score threshold ──────────
        max_score = max(c.score for c in norm_chunks)
        if max_score < self.score_threshold:
            logger.info(
                "Refusal path triggered: Max score %.4f below threshold %.4f.",
                max_score,
                self.score_threshold,
            )
            return GenerationResponse(
                answer="I am sorry, but the retrieved context passages are below the relevance threshold required to provide a confident answer.",
                is_grounded=False,
                is_refused=True,
                metadata={
                    "refusal_reason": "score_below_threshold",
                    "score_threshold": self.score_threshold,
                    "max_retrieved_score": max_score,
                },
            )

        # ── Step 3: Format prompt and call structured LLM client ───────────────
        user_prompt = self._build_user_prompt(question, norm_chunks)
        llm_out: LLMGenerationOutput = self.llm_client.complete_structured(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )

        # ── Refusal Path 3: Model signals insufficient context or ungrounded ────
        if llm_out.insufficient or not llm_out.grounded:
            logger.info("Refusal path triggered by LLM structured signal (insufficient=%s, grounded=%s).", llm_out.insufficient, llm_out.grounded)
            return GenerationResponse(
                answer=llm_out.answer,
                is_grounded=False,
                is_refused=True,
                metadata={
                    "refusal_reason": "llm_insufficient_signal",
                    "score_threshold": self.score_threshold,
                    "max_retrieved_score": max_score,
                    "raw_llm_output": llm_out.model_dump(),
                },
            )

        # ── Grounded Success Path ─────────────────────────────────────────────
        citations = [c.document_id for c in norm_chunks]
        return GenerationResponse(
            answer=llm_out.answer,
            is_grounded=True,
            is_refused=False,
            metadata={
                "score_threshold": self.score_threshold,
                "max_retrieved_score": max_score,
                "citations": citations,
                "num_context_chunks": len(norm_chunks),
            },
        )


# Retain alias for backwards compatibility
LLMGenerator = RAGGenerator
