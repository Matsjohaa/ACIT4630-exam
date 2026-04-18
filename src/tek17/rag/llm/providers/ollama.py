from __future__ import annotations

from typing import Any

import requests

from tek17.rag.config import REQUEST_TIMEOUT_S


def ollama_chat_result(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    temperature: float,
) -> dict[str, Any]:
    """Chat completion via Ollama, returning content + metadata.

    Ollama's response typically includes fields like:
    - done (bool)
    - done_reason (str)
    - prompt_eval_count / eval_count (token-ish counters)
    """

    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()

    message = (data or {}).get("message") or {}
    content = str(message.get("content") or "")

    finish_reason = data.get("done_reason")
    usage: dict[str, int] | None = None

    # Ollama exposes token-ish counters (naming may vary by version).
    prompt_eval = data.get("prompt_eval_count")
    eval_count = data.get("eval_count")
    if isinstance(prompt_eval, int) or isinstance(eval_count, int):
        usage = {
            "prompt_eval_count": int(prompt_eval or 0),
            "eval_count": int(eval_count or 0),
        }

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
    }