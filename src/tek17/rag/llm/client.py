from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

from tek17.rag.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

from tek17.rag.llm.providers.ollama import ollama_chat, ollama_chat_result
from tek17.rag.llm.providers.openai import openai_chat, openai_chat_result

Provider = Literal["ollama", "openai"]


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None
    usage: dict[str, Any] | None


def chat(
    messages: list[dict[str, str]],
    provider: Provider = "ollama",
    model: str = LLM_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int | None = LLM_MAX_TOKENS,
) -> str:
    """Generic chat interface for LLM providers."""

    return chat_result(
        messages,
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    ).content


def chat_result(
    messages: list[dict[str, str]],
    provider: Provider = "ollama",
    model: str = LLM_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int | None = LLM_MAX_TOKENS,
) -> ChatResult:
    """Chat interface that also returns metadata (usage/finish reason)."""

    if provider == "ollama":
        r = ollama_chat_result(messages, model=model, base_url=base_url, temperature=temperature)
        return ChatResult(
            content=str(r.get("content", "")),
            finish_reason=r.get("finish_reason"),
            usage=r.get("usage"),
        )

    if provider == "openai":
        r = openai_chat_result(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        return ChatResult(
            content=str(r.get("content", "")),
            finish_reason=r.get("finish_reason"),
            usage=r.get("usage"),
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
