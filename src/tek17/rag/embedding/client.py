from __future__ import annotations

from typing import Literal

import os
import requests

from tek17.rag.config import EMBED_MODEL, OLLAMA_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL

Provider = Literal["ollama", "openai"]


def embed_texts(
    texts: list[str],
    provider: Provider = "ollama",
    model: str = EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> list[list[float]]:
    """Embed a batch of texts using the selected provider."""

    if provider == "ollama":
        return _embed_ollama(texts, model=model, base_url=base_url)

    if provider == "openai":
        return _embed_openai(texts, model=model)

    raise ValueError(f"Unknown embedding provider: {provider}")


def embed_query(
    text: str,
    provider: Provider = "ollama",
    model: str = EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> list[float]:
    """Convenience wrapper for embedding a single query string."""

    embeddings = embed_texts([text], provider=provider, model=model, base_url=base_url)
    return embeddings[0]


def _embed_ollama(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings.extend(data["embeddings"])

    return embeddings


def _embed_openai(texts: list[str], model: str) -> list[list[float]]:
    """Embed texts using OpenAI's embeddings API.

    The API key is taken from either OPENAI_API_KEY or OPEN_AI_API_KEY.
    """

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI embeddings requested but no API key found. "
            "Set OPENAI_API_KEY or OPEN_AI_API_KEY in the environment/.env."
        )

    try:
        from openai import OpenAI  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - defensive import
        raise RuntimeError(
            "The 'openai' package is required for OpenAI embeddings. "
            "Install it via pip and try again."
        ) from exc

    base_url = OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]
