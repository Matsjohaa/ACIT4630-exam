from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[0-9A-Za-zÆØÅæøå]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
	return _TOKEN_RE.findall((text or "").lower())


@dataclass(frozen=True)
class _SparseIndex:
	documents: list[str]
	metadatas: list[dict[str, Any]]
	doc_tokens: list[list[str]]
	df: dict[str, int]
	avgdl: float


_INDEX_CACHE: dict[Path, _SparseIndex] = {}


def _load_chunks_jsonl(chunks_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
	documents: list[str] = []
	metadatas: list[dict[str, Any]] = []

	with chunks_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			row = json.loads(line)
			documents.append(str(row.get("text", "")))
			meta = row.get("metadata") or {}
			metadatas.append(dict(meta))

	return documents, metadatas


def _get_index(chunks_path: Path) -> _SparseIndex:
	chunks_path = chunks_path.resolve()
	if chunks_path in _INDEX_CACHE:
		return _INDEX_CACHE[chunks_path]

	if not chunks_path.exists():
		raise RuntimeError(
			f"Chunks file not found at {chunks_path}. "
			"Run `python -m tek17 ingest` (or chunk step) first."
		)

	documents, metadatas = _load_chunks_jsonl(chunks_path)
	doc_tokens: list[list[str]] = []
	df: dict[str, int] = {}

	total_len = 0
	for doc in documents:
		toks = _tokenize(doc)
		doc_tokens.append(toks)
		total_len += len(toks)
		for t in set(toks):
			df[t] = df.get(t, 0) + 1

	avgdl = (total_len / len(documents)) if documents else 0.0

	idx = _SparseIndex(
		documents=documents,
		metadatas=metadatas,
		doc_tokens=doc_tokens,
		df=df,
		avgdl=avgdl,
	)
	_INDEX_CACHE[chunks_path] = idx
	return idx


def _bm25_scores(
	query_tokens: list[str],
	idx: _SparseIndex,
	k1: float = 1.5,
	b: float = 0.75,
) -> list[float]:
	# BM25 Okapi
	N = len(idx.documents)
	if N == 0:
		return []

	scores = [0.0] * N
	if not query_tokens:
		return scores

	q_terms = list(dict.fromkeys(query_tokens))

	for term in q_terms:
		df = idx.df.get(term, 0)
		if df <= 0:
			continue
		idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
		for i, toks in enumerate(idx.doc_tokens):
			if not toks:
				continue
			tf = toks.count(term)
			if tf <= 0:
				continue
			dl = len(toks)
			denom = tf + k1 * (1.0 - b + b * (dl / idx.avgdl if idx.avgdl else 0.0))
			scores[i] += idf * (tf * (k1 + 1.0)) / (denom if denom else 1.0)

	return scores


def retrieve_sparce(
	query_text: str,
	top_k: int,
	chunks_path: Path,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
	"""Sparse retrieval over chunks via a lightweight BM25 implementation.

	Returns distances in [0, 1] where lower is better (like Chroma distances).
	"""

	idx = _get_index(chunks_path)
	q_toks = _tokenize(query_text)
	scores = _bm25_scores(q_toks, idx)

	if not scores:
		return [], [], []

	# Take top_k by score
	ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
	ranked = [i for i in ranked if scores[i] > 0.0][:top_k]

	if not ranked:
		return [], [], []

	max_score = max(scores[i] for i in ranked) or 1.0
	documents = [idx.documents[i] for i in ranked]
	metadatas = [idx.metadatas[i] for i in ranked]
	# Convert to distance: higher score => smaller distance
	distances = [1.0 - (scores[i] / max_score) for i in ranked]

	return documents, metadatas, distances

