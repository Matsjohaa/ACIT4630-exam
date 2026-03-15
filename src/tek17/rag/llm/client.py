from __future__ import annotations

from typing import Literal

import os
import requests

from tek17.rag.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

Provider = Literal["ollama", "openai"]


def chat(
    messages: list[dict[str, str]],
    provider: Provider = "ollama",
    model: str = LLM_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """Generic chat interface for LLM providers."""

    if provider == "ollama":
        return _ollama_chat(messages, model=model, base_url=base_url, temperature=temperature)

    if provider == "openai":
        return _openai_chat(messages, model=model, temperature=temperature)

    raise ValueError(f"Unknown LLM provider: {provider}")


def _ollama_chat(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    temperature: float,
) -> str:
    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def _openai_chat(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    """Chat completion via OpenAI.

    Expects an API key in OPENAI_API_KEY or OPEN_AI_API_KEY.
    """

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI chat requested but no API key found. "
            "Set OPENAI_API_KEY or OPEN_AI_API_KEY in the environment/.env."
        )

    try:
        from openai import OpenAI  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - defensive import
        raise RuntimeError(
            "The 'openai' package is required for OpenAI chat. "
            "Install it via pip and try again."
        ) from exc

    base_url = OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    # Take the first choice's content as the answer
    return resp.choices[0].message.content or ""
