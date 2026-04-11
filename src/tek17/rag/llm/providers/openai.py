from __future__ import annotations

from typing import Any

from tek17.rag.config import OPENAI_API_KEY, OPENAI_BASE_URL


def openai_chat_result(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int | None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run a chat completion via OpenAI and return normalized metadata."""

    api_key = OPENAI_API_KEY
    if not api_key:
        raise RuntimeError(
            "OpenAI chat requested but no API key found. "
            "Set OPENAI_API_KEY or OPEN_AI_API_KEY in the environment."
        )

    try:
        from openai import OpenAI  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "The 'openai' package is required for OpenAI chat. "
            "Install it via pip and try again."
        ) from exc

    resolved_base_url = base_url or OPENAI_BASE_URL
    client = (
        OpenAI(api_key=api_key, base_url=resolved_base_url)
        if resolved_base_url
        else OpenAI(api_key=api_key)
    )

    request_kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request_kwargs)

    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = str(getattr(message, "content", "") or "")
    finish_reason = getattr(choice, "finish_reason", None)

    usage: dict[str, int] | None = None
    response_usage = getattr(response, "usage", None)
    if response_usage is not None:
        usage = {
            "prompt_tokens": int(getattr(response_usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(response_usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(response_usage, "total_tokens", 0) or 0),
        }

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
    }