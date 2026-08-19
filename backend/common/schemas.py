from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RefusalReason(str, Enum):
    INVALID_INPUT = "invalid_input"
    OFF_TOPIC = "off_topic"
    SAFETY_VIOLATION = "safety_violation"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    HALLUCINATION_DETECTED = "hallucination_detected"
    SYSTEM_ERROR = "system_error"


class GuardrailCheckResult(BaseModel):
    """Result from an individual guardrail evaluation."""
    passed: bool = Field(..., description="Whether the check passed")
    reason: Optional[str] = Field(default=None, description="Refusal reason code if failed")
    message: Optional[str] = Field(default=None, description="Human-readable explanation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Debug/diagnostic attributes")


class Document(BaseModel):
    """Represents a raw document loaded from a dataset."""
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="The main text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata attributes")


class TextChunk(BaseModel):
    """
    Stable output contract from the chunking layer.

    This is the schema ALL chunking strategies must produce.  It is consumed
    directly by the embedder, evaluation harness, and evaluation team — do NOT
    rename or remove any of the three fields.

    text        – the verbatim chunk text (no truncation, no ellipsis).
    document_id – the id of the parent Document this chunk came from.
    metadata    – arbitrary key/value bag; MUST include at minimum:
                    chunk_strategy  (str)  which IChunker produced this
                    chunk_index     (int)  zero-based position within the doc
    """
    text: str = Field(..., description="Verbatim chunk text")
    document_id: str = Field(..., description="Parent document ID")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata including chunk_strategy and chunk_index",
    )


class Chunk(BaseModel):
    """
    Internal retrieval-layer model (extends TextChunk with embedding + id).

    Created by the embedding module from a TextChunk.  NOT the direct output
    of chunking strategies — use TextChunk for that stable contract.
    """
    id: str = Field(..., description="Unique identifier for the chunk")
    parent_doc_id: str = Field(..., description="Reference to the parent document ID")
    content: str = Field(..., description="The text content of the chunk")
    index: int = Field(..., description="The sequential index of this chunk within the parent document")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding values for the chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Inherited or chunk-specific metadata")


class RetrievedChunk(BaseModel):
    """
    Public contract for a retrieved chunk returned by IRetriever.retrieve().

    This is a stable contract — do not change field names:
      text        – Verbatim chunk text
      score       – Similarity / relevance score
      document_id – Parent document identity
      metadata    – Metadata bag (chunk_strategy, split, etc.)
    """
    text: str = Field(..., description="Verbatim chunk text")
    score: float = Field(..., description="Relevance or similarity score")
    document_id: str = Field(..., description="Source document identity")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata bag")

class QueryRequest(BaseModel):
    """Client request schema for the /ask endpoint."""
    query: str = Field(..., min_length=1, description="The user query text")
    k: int = Field(default=5, ge=1, le=50, description="Number of source chunks to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters for retrieval")

class ChunkReference(BaseModel):
    """A lightweight chunk reference returned in the final API response."""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]


class AskResult(BaseModel):
    """Structured result returned by the harness pipeline orchestrator."""
    query: str = Field(..., description="The original or sanitized query text")
    answer: str = Field(..., description="Generated answer or structured refusal message")
    context: List[ChunkReference] = Field(default_factory=list, description="Source context chunk references used")
    is_grounded: bool = Field(default=True, description="Whether answer is grounded in context")
    is_refused: bool = Field(default=False, description="Whether request was refused by guardrails or thresholds")
    refusal_reason: Optional[str] = Field(default=None, description="Specific refusal reason code if refused")
    guardrails: Dict[str, GuardrailCheckResult] = Field(default_factory=dict, description="Results from each guardrail check")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline timing, token counts, or debug info")

class QueryResponse(BaseModel):
    """Client response schema for the /ask endpoint."""
    query: str = Field(..., description="The original query text")
    answer: str = Field(..., description="The answer generated by the model")
    context: List[ChunkReference] = Field(default_factory=list, description="List of source chunks used to answer the query")
    is_grounded: bool = Field(default=True, description="Whether the answer is grounded in the retrieved context")
    is_refused: bool = Field(default=False, description="Whether the request was refused due to guardrails or safety reasons")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution timing, token usage, or debug information")

class GenerationResponse(BaseModel):
    """Internal model for generated responses from the generator module."""
    answer: str
    is_grounded: bool
    is_refused: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMGenerationOutput(BaseModel):
    """
    Structured output schema returned by the LLM generation call.

    answer       – The answer text grounded in context, or a refusal message
    grounded     – True if the answer is derived strictly from the provided context
    insufficient – True if retrieved context was missing/insufficient to answer
    """
    answer: str = Field(..., description="Factual answer or refusal message")
    grounded: bool = Field(..., description="Whether answer is grounded in provided context")
    insufficient: bool = Field(..., description="Whether context was insufficient to answer")
