from __future__ import annotations

import os

from tek17.rag.config import OPENAI_API_KEY, OPENAI_BASE_URL


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
		max_tokens=max_tokens,
	)
	return resp.choices[0].message.content or ""

