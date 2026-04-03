from __future__ import annotations

import os
from typing import Any

from tek17.rag.config import OPENAI_API_KEY, OPENAI_BASE_URL


def openai_chat_result(
	messages: list[dict[str, str]],
	model: str,
	temperature: float,
	max_tokens: int | None,
) -> dict[str, Any]:
	"""Chat completion via OpenAI, returning content + metadata.

	Returns a dict containing:
	- content: str
	- finish_reason: str | None
	- usage: dict | None (prompt_tokens/completion_tokens/total_tokens)
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

	kwargs: dict[str, object] = {
		"model": model,
		"messages": messages,
		"temperature": temperature,
	}
	if max_tokens is not None:
		kwargs["max_tokens"] = max_tokens

	resp = client.chat.completions.create(**kwargs)
	choice = resp.choices[0]
	content = (choice.message.content or "") if choice.message else ""
	finish_reason = getattr(choice, "finish_reason", None)

	usage: dict[str, int] | None = None
	if getattr(resp, "usage", None) is not None:
		u = resp.usage
		usage = {
			"prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
			"completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
			"total_tokens": int(getattr(u, "total_tokens", 0) or 0),
		}

	return {
		"content": content,
		"finish_reason": finish_reason,
		"usage": usage,
	}


def openai_chat(
	messages: list[dict[str, str]],
	model: str,
	temperature: float,
	max_tokens: int | None,
) -> str:
	"""Chat completion via OpenAI.

	Expects an API key in OPENAI_API_KEY or OPEN_AI_API_KEY.
	If OPENAI_BASE_URL is set, it will be used (Azure/custom endpoints).
	"""

	return str(openai_chat_result(messages, model=model, temperature=temperature, max_tokens=max_tokens)["content"])

