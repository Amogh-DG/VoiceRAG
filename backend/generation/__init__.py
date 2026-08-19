"""
backend/generation/__init__.py
------------------------------
Exports generation components and factory function.
"""

from typing import Any, Dict, Optional
from backend.common.interfaces import IGenerator
from backend.generation.generator import RAGGenerator, LLMGenerator
from backend.generation.llm_client import ILLMClient, MockLLMClient

def get_generator(
    score_threshold: float = 0.15,
    llm_client: Optional[ILLMClient] = None,
    **kwargs: Any,
) -> IGenerator:
    """Factory to instantiate a configured RAGGenerator instance."""
    return RAGGenerator(
        llm_client=llm_client or MockLLMClient(),
        score_threshold=score_threshold,
        **kwargs,
    )

__all__ = [
    "IGenerator",
    "RAGGenerator",
    "LLMGenerator",
    "ILLMClient",
    "MockLLMClient",
    "get_generator",
]
