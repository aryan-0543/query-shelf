"""Provider-isolated client for generating answers from prepared prompts."""

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from app.prompt_builder import PromptResult


class LLMClientError(Exception):
    """Raised when LLM configuration, communication, or output is invalid."""


@dataclass(frozen=True)
class LLMResponse:
    """A generated response returned by the configured LLM."""

    text: str
    model: str


class LLMClient:
    """Generate responses through the Google GenAI SDK boundary."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        load_dotenv()
        configured_key = api_key if api_key is not None else os.getenv("LLM_API_KEY")
        configured_model = model if model is not None else os.getenv("LLM_MODEL")
        if not configured_key or not configured_key.strip():
            raise LLMClientError("LLM_API_KEY is not configured")
        if not configured_model or not configured_model.strip():
            raise LLMClientError("LLM_MODEL is not configured")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise LLMClientError("timeout_ms must be a positive integer")

        self._model = configured_model.strip()
        if client is not None:
            self._client = client
            return

        try:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=configured_key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
        except Exception as error:
            raise LLMClientError("Unable to initialize the LLM client") from error

    def generate(self, prompt: PromptResult) -> LLMResponse:
        """Generate one response using separate system and user content."""

        if not isinstance(prompt, PromptResult):
            raise LLMClientError("Prompt must be a PromptResult")
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt.user_message,
                config=types.GenerateContentConfig(
                    system_instruction=prompt.system_instructions,
                ),
            )
        except Exception as error:
            raise LLMClientError("LLM generation failed") from error

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMClientError("LLM returned an empty response")
        return LLMResponse(text=text.strip(), model=self._model)
