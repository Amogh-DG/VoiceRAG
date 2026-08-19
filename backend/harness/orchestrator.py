"""
backend/harness/orchestrator.py
--------------------------------
Structured orchestration harness combining input sanitisation, vector retrieval,
LLM generation, guardrails evaluation, timeout/retry resilience, and structured AskResult outputs.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union

from backend.common.interfaces import IEmbedder, IGenerator, IPipelineOrchestrator, IRetriever, IVectorDB
from backend.common.schemas import (
    AskResult,
    Chunk,
    ChunkReference,
    GenerationResponse,
    GuardrailCheckResult,
    RefusalReason,
    RetrievedChunk,
)
from backend.generation.generator import RAGGenerator
from backend.harness.guardrails import (
    HallucinationGuardrail,
    InputValidationGuardrail,
    InsufficientContextGuardrail,
    OffTopicGuardrail,
    SafetyGuardrail,
)
from backend.retrieval.retriever import VectorRetriever

logger = logging.getLogger(__name__)


def _to_chunk_reference(chunk: Union[RetrievedChunk, Chunk]) -> ChunkReference:
    """Convert RetrievedChunk or Chunk to public ChunkReference schema."""
    if isinstance(chunk, RetrievedChunk):
        return ChunkReference(
            chunk_id=f"{chunk.document_id}_retrieved",
            document_id=chunk.document_id,
            content=chunk.text,
            metadata=chunk.metadata,
        )
    return ChunkReference(
        chunk_id=chunk.id,
        document_id=chunk.parent_doc_id,
        content=chunk.content,
        metadata=chunk.metadata,
    )


class PipelineOrchestrator(IPipelineOrchestrator):
    """
    Production-ready RAG Orchestration Harness.

    Coordinates:
      1. Input validation & sanitisation.
      2. Query guardrails (off-topic & safety checks).
      3. Vector DB retrieval with timeout & retry handling.
      4. Context adequacy evaluation.
      5. LLM generation with timeout & retry handling.
      6. Output guardrails (answer safety & hallucination checks).
      7. Structured AskResult construction with explicit refusal reasons.
    """

    def __init__(
        self,
        embedder: Optional[IEmbedder] = None,
        vector_db: Optional[IVectorDB] = None,
        generator: Optional[IGenerator] = None,
        retriever: Optional[IRetriever] = None,
        timeout_seconds: float = 5.0,
        max_retries: int = 2,
    ):
        self.embedder = embedder
        self.vector_db = vector_db
        
        # If retriever is not explicitly provided, build one from embedder and vector_db if present
        if retriever is None and embedder is not None and vector_db is not None:
            self.retriever: Optional[IRetriever] = VectorRetriever(embedder=embedder, vector_db=vector_db)
        else:
            self.retriever = retriever

        self.generator: Optional[IGenerator] = generator or RAGGenerator()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Guardrail instances
        self.input_validator = InputValidationGuardrail()
        self.off_topic_guard = OffTopicGuardrail()
        self.safety_guard = SafetyGuardrail()
        self.context_guard = InsufficientContextGuardrail()
        self.hallucination_guard = HallucinationGuardrail()

    async def _execute_with_retry(self, coro_func, action_name: str) -> Any:
        """Executes an async call with timeout and retry resilience."""
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await asyncio.wait_for(coro_func(), timeout=self.timeout_seconds)
            except (asyncio.TimeoutError, Exception) as exc:
                last_exception = exc
                logger.warning(
                    "%s attempt %d/%d failed: %s",
                    action_name,
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.1 * attempt)

        raise RuntimeError(f"{action_name} failed after {self.max_retries} attempts: {last_exception}")

    async def run_pipeline(self, query: str, k: int = 5) -> AskResult:
        """
        Executes the end-to-end RAG harness pipeline for a user query.

        Args:
            query: Raw user query input string.
            k: Number of context chunks to retrieve (default 5).

        Returns:
            AskResult: Fully structured result containing answer, context, and guardrail reports.
        """
        start_time = time.perf_counter()
        guardrails_report: Dict[str, GuardrailCheckResult] = {}
        
        # Initialize dictionary to track latency across stages in milliseconds
        timings_ms = {
            "input_validation_ms": 0.0,
            "query_guardrails_ms": 0.0,
            "embedding_ms": 0.0,
            "vector_db_ms": 0.0,
            "reranking_ms": 0.0,
            "retrieval_ms": 0.0,
            "context_adequacy_ms": 0.0,
            "generation_ms": 0.0,
            "output_guardrails_ms": 0.0,
            "total_ms": 0.0
        }

        # ── 1. Input Validation Guardrail ───────────────────────────────────────
        start_val = time.perf_counter()
        val_res = self.input_validator.check(query)
        timings_ms["input_validation_ms"] = (time.perf_counter() - start_val) * 1000.0
        guardrails_report["input_validation"] = val_res
        if not val_res.passed:
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=str(query) if query is not None else "",
                reason=val_res.reason or RefusalReason.INVALID_INPUT.value,
                message=val_res.message or "Invalid input query.",
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms
            )

        sanitized_query = val_res.details.get("sanitized_query", str(query).strip())

        # ── 2. Off-Topic & Safety Guardrails on Query ───────────────────────────
        start_q_guards = time.perf_counter()
        off_topic_res = self.off_topic_guard.check(sanitized_query)
        guardrails_report["off_topic"] = off_topic_res
        if not off_topic_res.passed:
            timings_ms["query_guardrails_ms"] = (time.perf_counter() - start_q_guards) * 1000.0
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=sanitized_query,
                reason=off_topic_res.reason or RefusalReason.OFF_TOPIC.value,
                message=off_topic_res.message or "Query is off-topic or contains prompt injections.",
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms,
            )

        query_safety_res = self.safety_guard.check_text(sanitized_query, target_name="Query")
        guardrails_report["query_safety"] = query_safety_res
        if not query_safety_res.passed:
            timings_ms["query_guardrails_ms"] = (time.perf_counter() - start_q_guards) * 1000.0
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=sanitized_query,
                reason=query_safety_res.reason or RefusalReason.SAFETY_VIOLATION.value,
                message=query_safety_res.message or "Query violates safety policies.",
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms,
            )
        timings_ms["query_guardrails_ms"] = (time.perf_counter() - start_q_guards) * 1000.0

        # ── 3. Vector Retrieval with Timeout & Retry Resilience ────────────────
        retrieved_chunks: List[RetrievedChunk] = []
        start_retrieval = time.perf_counter()
        try:
            if self.retriever:
                async def _retrieve_op():
                    return self.retriever.retrieve(sanitized_query, top_k=k)
                retrieved_chunks = await self._execute_with_retry(_retrieve_op, "Retrieval")
                
                # Fetch detailed sub-stage timings from retriever
                if hasattr(self.retriever, "last_timings"):
                    timings_ms["embedding_ms"] = self.retriever.last_timings.get("embedding_ms", 0.0)
                    timings_ms["vector_db_ms"] = self.retriever.last_timings.get("vector_db_ms", 0.0)
                    timings_ms["reranking_ms"] = self.retriever.last_timings.get("reranking_ms", 0.0)
            elif self.embedder and self.vector_db:
                async def _legacy_retrieve_op():
                    start_emb = time.perf_counter()
                    q_vec = await self.embedder.embed_query(sanitized_query)
                    emb_time_ms = (time.perf_counter() - start_emb) * 1000.0
                    
                    start_db = time.perf_counter()
                    raw_chunks = await self.vector_db.similarity_search(q_vec, k=k)
                    db_time_ms = (time.perf_counter() - start_db) * 1000.0
                    
                    _legacy_retrieve_op.emb_time_ms = emb_time_ms
                    _legacy_retrieve_op.db_time_ms = db_time_ms
                    
                    return [
                        RetrievedChunk(
                            text=c.content,
                            score=float(c.metadata.get("score", 1.0)),
                            document_id=c.parent_doc_id,
                            metadata=c.metadata,
                        )
                        for c in raw_chunks
                    ]
                _legacy_retrieve_op.emb_time_ms = 0.0
                _legacy_retrieve_op.db_time_ms = 0.0
                retrieved_chunks = await self._execute_with_retry(_legacy_retrieve_op, "Legacy Retrieval")
                
                timings_ms["embedding_ms"] = _legacy_retrieve_op.emb_time_ms
                timings_ms["vector_db_ms"] = _legacy_retrieve_op.db_time_ms
                timings_ms["reranking_ms"] = 0.0
                
            timings_ms["retrieval_ms"] = (time.perf_counter() - start_retrieval) * 1000.0
        except Exception as exc:
            logger.error("Retrieval failed in harness: %s", exc)
            timings_ms["retrieval_ms"] = (time.perf_counter() - start_retrieval) * 1000.0
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=sanitized_query,
                reason=RefusalReason.SYSTEM_ERROR.value,
                message="Retrieval system failed or timed out.",
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms,
            )

        # ── 4. Insufficient Context Guardrail ──────────────────────────────────
        start_ctx_guard = time.perf_counter()
        context_res = self.context_guard.check(retrieved_chunks)
        timings_ms["context_adequacy_ms"] = (time.perf_counter() - start_ctx_guard) * 1000.0
        guardrails_report["insufficient_context"] = context_res
        if not context_res.passed:
            context_refs = [_to_chunk_reference(c) for c in retrieved_chunks]
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=sanitized_query,
                reason=context_res.reason or RefusalReason.INSUFFICIENT_CONTEXT.value,
                message=context_res.message or "Retrieved context is insufficient to answer the query.",
                context=context_refs,
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms,
            )

        # ── 5. LLM Generation with Timeout & Retry Resilience ─────────────────
        gen_response: Optional[GenerationResponse] = None
        start_gen = time.perf_counter()
        try:
            if self.generator:
                async def _generate_op():
                    return await self.generator.async_generate(sanitized_query, retrieved_chunks)
                gen_response = await self._execute_with_retry(_generate_op, "Generation")
            timings_ms["generation_ms"] = (time.perf_counter() - start_gen) * 1000.0
        except Exception as exc:
            logger.error("Generation failed in harness: %s", exc)
            timings_ms["generation_ms"] = (time.perf_counter() - start_gen) * 1000.0
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=sanitized_query,
                reason=RefusalReason.SYSTEM_ERROR.value,
                message="Generation system failed or timed out.",
                context=[_to_chunk_reference(c) for c in retrieved_chunks],
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms,
            )

        if not gen_response or gen_response.is_refused or not gen_response.is_grounded:
            refusal_reason = (
                gen_response.metadata.get("refusal_reason") if gen_response else RefusalReason.INSUFFICIENT_CONTEXT.value
            ) or RefusalReason.INSUFFICIENT_CONTEXT.value
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=sanitized_query,
                reason=refusal_reason,
                message=gen_response.answer if gen_response else "Generator refused query.",
                context=[_to_chunk_reference(c) for c in retrieved_chunks],
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms,
            )

        # ── 6. Answer Safety & Hallucination Guardrails ────────────────────────
        start_out_guards = time.perf_counter()
        answer_safety_res = self.safety_guard.check_text(gen_response.answer, target_name="Answer")
        guardrails_report["answer_safety"] = answer_safety_res
        if not answer_safety_res.passed:
            timings_ms["output_guardrails_ms"] = (time.perf_counter() - start_out_guards) * 1000.0
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=sanitized_query,
                reason=answer_safety_res.reason or RefusalReason.SAFETY_VIOLATION.value,
                message=answer_safety_res.message or "Generated answer violates safety policies.",
                context=[_to_chunk_reference(c) for c in retrieved_chunks],
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms,
            )

        hallucination_res = self.hallucination_guard.check(gen_response.answer, retrieved_chunks)
        guardrails_report["hallucination"] = hallucination_res
        if not hallucination_res.passed:
            timings_ms["output_guardrails_ms"] = (time.perf_counter() - start_out_guards) * 1000.0
            timings_ms["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            return self._build_refusal_result(
                query=sanitized_query,
                reason=hallucination_res.reason or RefusalReason.HALLUCINATION_DETECTED.value,
                message=hallucination_res.message or "Generated answer contains unsupported claims.",
                context=[_to_chunk_reference(c) for c in retrieved_chunks],
                guardrails=guardrails_report,
                start_time=start_time,
                timings_ms=timings_ms,
            )
        timings_ms["output_guardrails_ms"] = (time.perf_counter() - start_out_guards) * 1000.0

        # ── 7. Grounded Success Outcome ─────────────────────────────────────────
        elapsed_seconds = round(time.perf_counter() - start_time, 4)
        timings_ms["total_ms"] = elapsed_seconds * 1000.0
        context_refs = [_to_chunk_reference(c) for c in retrieved_chunks]

        return AskResult(
            query=sanitized_query,
            answer=gen_response.answer,
            context=context_refs,
            is_grounded=True,
            is_refused=False,
            refusal_reason=None,
            guardrails=guardrails_report,
            metadata={
                "elapsed_seconds": elapsed_seconds,
                "retrieved_chunk_count": len(retrieved_chunks),
                "generation_metadata": gen_response.metadata,
                "timings_ms": timings_ms,
            },
        )

    def _build_refusal_result(
        self,
        query: str,
        reason: str,
        message: str,
        guardrails: Dict[str, GuardrailCheckResult],
        start_time: float,
        context: Optional[List[ChunkReference]] = None,
        timings_ms: Optional[Dict[str, float]] = None,
    ) -> AskResult:
        """Helper to construct a consistent refusal AskResult."""
        elapsed = round(time.perf_counter() - start_time, 4)
        metadata = {"elapsed_seconds": elapsed, "refusal_message": message}
        if timings_ms:
            metadata["timings_ms"] = timings_ms
        return AskResult(
            query=query,
            answer=message,
            context=context or [],
            is_grounded=False,
            is_refused=True,
            refusal_reason=reason,
            guardrails=guardrails,
            metadata=metadata,
        )
