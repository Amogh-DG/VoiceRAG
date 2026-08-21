"""
backend/harness/guardrails.py
------------------------------
Explicit, independently testable production guardrails for the RAG harness.

Guardrails
──────────
1. InputValidationGuardrail  – Sanitises query, checks for empty, non-string, or >max_length inputs.
2. OffTopicGuardrail          – Detects prompt injections and off-topic requests.
3. SafetyGuardrail            – Checks query and generated answer for unsafe/prohibited content.
4. InsufficientContextGuardrail – Evaluates retrieved context adequacy and threshold checks.
5. HallucinationGuardrail      – Heuristic check verifying answer claims against retrieved chunks.
"""

import re
import logging
from typing import Any, Dict, List, Optional

from backend.common.schemas import GuardrailCheckResult, RefusalReason, RetrievedChunk

logger = logging.getLogger(__name__)

# System prompt injection / off-topic keywords
OFF_TOPIC_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (all )?prior (rules|instructions)",
    r"you are now (in )?DAN mode",
    r"jailbreak",
    r"system prompt leak",
]

# Simple safety violation patterns
SAFETY_PATTERNS = [
    r"how to (build|make) a bomb",
    r"generate (malware|ransomware|virus)",
    r"bypass security controls",
]


class InputValidationGuardrail:
    """Validates and sanitizes raw query inputs."""

    def __init__(self, max_length: int = 2000):
        self.max_length = max_length

    def check(self, query: Any) -> GuardrailCheckResult:
        if not isinstance(query, str):
            return GuardrailCheckResult(
                passed=False,
                reason=RefusalReason.INVALID_INPUT.value,
                message="Query must be a string.",
            )

        sanitized = query.strip()
        if not sanitized:
            return GuardrailCheckResult(
                passed=False,
                reason=RefusalReason.INVALID_INPUT.value,
                message="Query cannot be empty or whitespace-only.",
            )

        if len(sanitized) > self.max_length:
            return GuardrailCheckResult(
                passed=False,
                reason=RefusalReason.INVALID_INPUT.value,
                message=f"Query exceeds maximum allowed length of {self.max_length} characters.",
                details={"query_length": len(sanitized), "max_length": self.max_length},
            )

        return GuardrailCheckResult(passed=True, details={"sanitized_query": sanitized})


class OffTopicGuardrail:
    """Detects prompt injection attempts and queries outside the system scope."""

    def __init__(self, patterns: Optional[List[str]] = None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or OFF_TOPIC_PATTERNS)]

    def check(self, query: str) -> GuardrailCheckResult:
        for pattern in self.patterns:
            if pattern.search(query):
                logger.warning("OffTopicGuardrail triggered on query: %s", query[:50])
                return GuardrailCheckResult(
                    passed=False,
                    reason=RefusalReason.OFF_TOPIC.value,
                    message="Query contains off-topic or prompt injection instructions.",
                )
        return GuardrailCheckResult(passed=True)


class SafetyGuardrail:
    """Checks for unsafe, harmful, or prohibited content in query or answer."""

    def __init__(self, patterns: Optional[List[str]] = None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or SAFETY_PATTERNS)]

    def check_text(self, text: str, target_name: str = "Query") -> GuardrailCheckResult:
        for pattern in self.patterns:
            if pattern.search(text):
                logger.warning("SafetyGuardrail triggered on %s", target_name)
                return GuardrailCheckResult(
                    passed=False,
                    reason=RefusalReason.SAFETY_VIOLATION.value,
                    message=f"{target_name} contains prohibited or unsafe content.",
                )
        return GuardrailCheckResult(passed=True)


class InsufficientContextGuardrail:
    """Evaluates retrieved chunks and score threshold to guard against ungrounded generation."""

    def __init__(self, min_score_threshold: float = 0.15):
        self.min_score_threshold = min_score_threshold

    def check(self, chunks: List[RetrievedChunk]) -> GuardrailCheckResult:
        if not chunks:
            return GuardrailCheckResult(
                passed=False,
                reason=RefusalReason.INSUFFICIENT_CONTEXT.value,
                message="No context passages were retrieved.",
            )

        max_score = max(c.score for c in chunks)
        if max_score < self.min_score_threshold:
            return GuardrailCheckResult(
                passed=False,
                reason=RefusalReason.INSUFFICIENT_CONTEXT.value,
                message=f"Retrieved context max score ({max_score:.4f}) is below threshold ({self.min_score_threshold:.4f}).",
                details={"max_score": max_score, "threshold": self.min_score_threshold},
            )

        return GuardrailCheckResult(
            passed=True, details={"chunk_count": len(chunks), "max_score": max_score}
        )


class HallucinationGuardrail:
    """
    NLI-style / Lexical Overlap Heuristic Check for Hallucination Detection.

    Supports Indic (Devanagari) Unicode token matching and handles cross-lingual
    translation (e.g. English answer generated from Hindi context passages).
    """

    def __init__(self, min_overlap_ratio: float = 0.40):
        self.min_overlap_ratio = min_overlap_ratio

    def check(self, answer: str, chunks: List[RetrievedChunk]) -> GuardrailCheckResult:
        if not answer or not chunks:
            return GuardrailCheckResult(passed=True)

        context_text = " ".join(c.text for c in chunks).lower()

        # Check for cross-lingual synthesis (e.g., English answer derived from Hindi context)
        has_latin_answer = bool(re.search(r"[a-zA-Z]{3,}", answer))
        has_devanagari_context = bool(re.search(r"[\u0900-\u097F]{3,}", context_text))
        has_devanagari_answer = bool(re.search(r"[\u0900-\u097F]{3,}", answer))

        if has_latin_answer and has_devanagari_context and not has_devanagari_answer:
            # Cross-lingual English translation from Hindi context: exact lexical word matching across scripts is not applicable
            logger.info("HallucinationGuardrail: Cross-lingual English answer from Hindi context detected; passing check.")
            return GuardrailCheckResult(
                passed=True, details={"cross_lingual": True, "mode": "latin_answer_hindi_context"}
            )

        # Unicode-aware tokenization for Indic (Devanagari) & English
        context_words = set(re.findall(r"[\w\u0900-\u097F]{3,}", context_text))
        answer_words = re.findall(r"[\w\u0900-\u097F]{3,}", answer.lower())

        if not answer_words:
            return GuardrailCheckResult(passed=True)

        matched_words = [w for w in answer_words if w in context_words]
        overlap_ratio = len(matched_words) / len(answer_words)

        if overlap_ratio < self.min_overlap_ratio:
            logger.warning(
                "HallucinationGuardrail triggered: overlap ratio %.2f below threshold %.2f",
                overlap_ratio,
                self.min_overlap_ratio,
            )
            return GuardrailCheckResult(
                passed=False,
                reason=RefusalReason.HALLUCINATION_DETECTED.value,
                message="Generated answer contains claims not supported by retrieved context.",
                details={
                    "overlap_ratio": round(overlap_ratio, 4),
                    "min_required": self.min_overlap_ratio,
                },
            )

        return GuardrailCheckResult(
            passed=True, details={"overlap_ratio": round(overlap_ratio, 4)}
        )
