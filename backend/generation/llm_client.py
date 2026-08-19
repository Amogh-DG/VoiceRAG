"""
backend/generation/llm_client.py
--------------------------------
Structured LLM completion wrapper and mock implementation for testing.

The LLM client enforces structured JSON output conforming to LLMGenerationOutput:
    { "answer": "...", "grounded": bool, "insufficient": bool }
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from backend.common.schemas import LLMGenerationOutput

logger = logging.getLogger(__name__)


class ILLMClient(ABC):
    """Abstract client interface for structured LLM completions."""

    @abstractmethod
    def complete_structured(
        self, system_prompt: str, user_prompt: str
    ) -> LLMGenerationOutput:
        """
        Executes an LLM completion returning a structured LLMGenerationOutput schema.
        """
        pass


class MockLLMClient(ILLMClient):
    """
    Mock LLM Client for unit testing and offline runs.
    Allows injecting mock responses or deterministically simulating grounded/refusal outputs.
    """

    def __init__(self, override_output: Optional[LLMGenerationOutput] = None):
        self.override_output = override_output

    def complete_structured(
        self, system_prompt: str, user_prompt: str
    ) -> LLMGenerationOutput:
        """Simulates structured LLM output without making external network calls."""
        if self.override_output is not None:
            return self.override_output

        # Check if the prompt explicitly signals refusal or empty context via test tokens
        if "NO_CONTEXT" in user_prompt or "REFUSE_QUERY" in user_prompt:
            return LLMGenerationOutput(
                answer="I am sorry, but the provided context does not contain sufficient information to answer your question.",
                grounded=False,
                insufficient=True,
            )

        # Extract passage content text to avoid template header words in mock answer
        if "Context Passages:" in user_prompt:
            raw_ctx = user_prompt.split("Context Passages:", 1)[1]
            # Strip passage headers if present
            lines = [l for l in raw_ctx.splitlines() if l.strip() and not l.strip().startswith(("[Passage", "Provide"))]
            ctx_text = " ".join(lines)[:120].strip()
        else:
            ctx_text = user_prompt[:120].strip()

        return LLMGenerationOutput(
            answer=ctx_text,
            grounded=True,
            insufficient=False,
        )
