from __future__ import annotations

import requests


def ollama_chat(
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
