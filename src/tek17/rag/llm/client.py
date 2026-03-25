from __future__ import annotations

from typing import Literal

from tek17.rag.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

from tek17.rag.llm.providers.ollama import ollama_chat
from tek17.rag.llm.providers.openai import openai_chat

Provider = Literal["ollama", "openai"]


def chat(
    messages: list[dict[str, str]],
    provider: Provider = "ollama",
    model: str = LLM_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int | None = LLM_MAX_TOKENS,
) -> str:
    """Generic chat interface for LLM providers."""

    if provider == "ollama":
        return ollama_chat(messages, model=model, base_url=base_url, temperature=temperature)

    if provider == "openai":
        return openai_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)

    raise ValueError(f"Unknown LLM provider: {provider}")
