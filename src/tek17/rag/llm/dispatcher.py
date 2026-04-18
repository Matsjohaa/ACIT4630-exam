from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tek17.rag.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

from tek17.rag.llm.providers.ollama import ollama_chat_result
from tek17.rag.llm.providers.openai import openai_chat_result

Provider = Literal["ollama", "openai"]


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None
    usage: dict[str, Any] | None


def chat(
    messages: list[dict[str, str]],
    provider: str = LLM_PROVIDER,
    model: str = LLM_MODEL,
    base_url: str | None = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int | None = LLM_MAX_TOKENS,
) -> str:
    """Return only the generated text from the selected LLM provider."""
    return chat_result(
        messages=messages,
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    ).content


def chat_result(
    messages: list[dict[str, str]],
    provider: str = LLM_PROVIDER,
    model: str = LLM_MODEL,
    base_url: str | None = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int | None = LLM_MAX_TOKENS,
) -> ChatResult:
    """Return normalized content and metadata from the selected LLM provider."""
    normalized_provider = (provider or "").strip().lower()

    if normalized_provider == "ollama":
        resolved_base_url = (base_url or LLM_BASE_URL or "").strip()
        result = ollama_chat_result(
            messages=messages,
            model=model,
            base_url=resolved_base_url,
            temperature=temperature,
        )
        return ChatResult(
            content=str(result.get("content", "")),
            finish_reason=result.get("finish_reason"),
            usage=result.get("usage"),
        )

    if normalized_provider == "openai":
        result = openai_chat_result(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
        return ChatResult(
            content=str(result.get("content", "")),
            finish_reason=result.get("finish_reason"),
            usage=result.get("usage"),
        )

    raise ValueError(
        f"Unknown LLM provider: {provider}. Supported providers are 'ollama' and 'openai'."
    )