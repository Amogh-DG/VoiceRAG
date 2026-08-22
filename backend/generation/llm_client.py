"""
backend/generation/llm_client.py
--------------------------------
Structured LLM completion wrapper and implementations for RAG answer generation.

Classes:
  - ILLMClient: Abstract base class for structured LLM completion interfaces.
  - MockLLMClient: Fast, deterministic offline testing client.
  - GeminiLLMClient: Production client calling Google Gemini 3.5 Flash-Lite API
    with strict JSON output matching LLMGenerationOutput schema.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, Generator, Optional

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

    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        """
        Translates the provided text into the target language.
        """
        pass

    def complete_structured_stream(
        self, system_prompt: str, user_prompt: str
    ) -> Generator[str, None, LLMGenerationOutput]:
        """
        Optional streaming generator yielding answer tokens/chunks as they arrive.
        Returns final LLMGenerationOutput.
        """
        result = self.complete_structured(system_prompt, user_prompt)
        yield result.answer
        return result


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
            lines = [l for l in raw_ctx.splitlines() if l.strip() and not l.strip().startswith(("[Passage", "Provide"))]
            ctx_text = " ".join(lines)[:120].strip()
        else:
            ctx_text = user_prompt[:120].strip()

        return LLMGenerationOutput(
            answer=ctx_text,
            grounded=True,
            insufficient=False,
        )

    def translate(self, text: str, target_lang: str) -> str:
        """Simulates translation for offline testing."""
        return f"[Translated to {target_lang}]: {text}"


class GeminiLLMClient(ILLMClient):
    """
    Production Gemini 3.5 Flash-Lite LLM Client via official Google Gemini API.

    Requirements:
      - Reads API key from GEMINI_API_KEY environment variable.
      - Uses official google-genai SDK or direct REST API endpoint.
      - Default Model: `gemini-3.5-flash-lite`.
      - Enforces structured JSON output matching LLMGenerationOutput schema:
          { "answer": "...", "grounded": bool, "insufficient": bool }
      - Never exposes internal reasoning.
      - Refuses gracefully when context is insufficient or API key is missing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature

        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY environment variable not set. "
                "GeminiLLMClient requires GEMINI_API_KEY for API calls."
            )

    def complete_structured(
        self, system_prompt: str, user_prompt: str
    ) -> LLMGenerationOutput:
        """
        Executes structured JSON generation using Gemini 3.5 Flash-Lite API.
        """
        if not self.api_key:
            logger.warning("GEMINI_API_KEY missing during generation request. Falling back to refusal.")
            return LLMGenerationOutput(
                answer="GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable to use the Gemini LLM.",
                grounded=False,
                insufficient=True,
            )

        # 1. Attempt using official google-genai SDK
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=LLMGenerationOutput,
                temperature=self.temperature,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )

            if response.text:
                parsed = json.loads(response.text)
                return LLMGenerationOutput(
                    answer=parsed.get("answer", "").strip(),
                    grounded=bool(parsed.get("grounded", True)),
                    insufficient=bool(parsed.get("insufficient", False)),
                )
        except ImportError:
            logger.debug("google-genai SDK not imported; using REST API fallback.")
        except Exception as exc:
            logger.warning("google-genai SDK call failed: %s. Trying REST API fallback.", exc)

        # 2. Direct REST API fallback
        return self._complete_via_rest_api(system_prompt, user_prompt)

    def _complete_via_rest_api(
        self, system_prompt: str, user_prompt: str
    ) -> LLMGenerationOutput:
        """
        Direct HTTP REST API call to Google Gemini endpoint.
        """
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        )

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "answer": {"type": "STRING"},
                        "grounded": {"type": "BOOLEAN"},
                        "insufficient": {"type": "BOOLEAN"}
                    },
                    "required": ["answer", "grounded", "insufficient"]
                },
                "temperature": self.temperature
            }
        }

        headers = {"Content-Type": "application/json"}
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                candidates = resp_data.get("candidates", [])
                if not candidates:
                    raise ValueError("Empty candidate list returned by Gemini API.")
                
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError("No text content returned in candidate.")

                raw_json = parts[0].get("text", "").strip()
                parsed = json.loads(raw_json)

                return LLMGenerationOutput(
                    answer=parsed.get("answer", "").strip(),
                    grounded=bool(parsed.get("grounded", True)),
                    insufficient=bool(parsed.get("insufficient", False)),
                )
        except Exception as exc:
            logger.error("Gemini REST API completion failed for model '%s': %s", self.model_name, exc)
            return LLMGenerationOutput(
                answer=f"Gemini API request failed: {str(exc)}",
                grounded=False,
                insufficient=True,
            )

    def translate(self, text: str, target_lang: str) -> str:
        """Translates the given text into the target language using Gemini."""
        if not self.api_key:
            logger.warning("GEMINI_API_KEY missing during translation request. Returning untranslated text.")
            return f"[API Key Missing] {text}"

        system_prompt = (
            "You are a professional, high-fidelity translator.\n"
            "Translate the input text precisely into the requested target language.\n"
            "Maintain the original meaning, tone, and formatting.\n"
            "Return ONLY the translated text, with no preamble, explanations, or quotes."
        )
        user_prompt = f"Translate the following text to target language '{target_lang}':\n\n{text}"

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )
            if response.text:
                return response.text.strip()
        except ImportError:
            logger.debug("google-genai SDK not imported; using REST API fallback.")
        except Exception as exc:
            logger.warning("SDK translation failed: %s. Using REST API fallback.", exc)

        return self._translate_via_rest_api(system_prompt, user_prompt)

    def _translate_via_rest_api(self, system_prompt: str, user_prompt: str) -> str:
        """Direct HTTP REST API call to Google Gemini endpoint for translation."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3
            }
        }
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                candidates = res_body.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
        except Exception as exc:
            logger.error("REST API translation failed: %s", exc)
        return "[Translation failed due to API error]"
