from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tek17.rag.config import BM25_B, BM25_K1

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

    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            documents.append(str(row.get("text", "")))
            metadata = row.get("metadata") or {}
            metadatas.append(dict(metadata))

    return documents, metadatas


def _get_index(chunks_path: Path) -> _SparseIndex:
    resolved_chunks_path = chunks_path.resolve()

    if resolved_chunks_path in _INDEX_CACHE:
        return _INDEX_CACHE[resolved_chunks_path]

    if not resolved_chunks_path.exists():
        raise RuntimeError(
            f"Chunks file not found at {resolved_chunks_path}. "
            "Run `python -m tek17 chunk` (and ingest if needed) first."
        )

    documents, metadatas = _load_chunks_jsonl(resolved_chunks_path)

    doc_tokens: list[list[str]] = []
    df: dict[str, int] = {}
    total_length = 0

    for document in documents:
        tokens = _tokenize(document)
        doc_tokens.append(tokens)
        total_length += len(tokens)

        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    avgdl = (total_length / len(documents)) if documents else 0.0

    index = _SparseIndex(
        documents=documents,
        metadatas=metadatas,
        doc_tokens=doc_tokens,
        df=df,
        avgdl=avgdl,
    )
    _INDEX_CACHE[resolved_chunks_path] = index

    return index


def _bm25_scores(
    query_tokens: list[str],
    index: _SparseIndex,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    """Return BM25 scores for all indexed documents."""
    document_count = len(index.documents)
    if document_count == 0:
        return []

    scores = [0.0] * document_count
    if not query_tokens:
        return scores

    unique_query_terms = list(dict.fromkeys(query_tokens))

    for term in unique_query_terms:
        document_frequency = index.df.get(term, 0)
        if document_frequency <= 0:
            continue

        idf = math.log(
            (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
            + 1.0
        )

        for i, tokens in enumerate(index.doc_tokens):
            if not tokens:
                continue

            term_frequency = tokens.count(term)
            if term_frequency <= 0:
                continue

            document_length = len(tokens)
            denominator = term_frequency + k1 * (
                1.0 - b + b * (document_length / index.avgdl if index.avgdl else 0.0)
            )

            scores[i] += idf * (term_frequency * (k1 + 1.0)) / (denominator or 1.0)

    return scores


def retrieve_sparse(
    query_text: str,
    top_k: int,
    chunks_path: Path,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Sparse retrieval over chunks using a lightweight BM25 implementation.

    Returns distances in [0, 1], where lower is better.
    """
    index = _get_index(chunks_path)
    query_tokens = _tokenize(query_text)
    scores = _bm25_scores(query_tokens, index)

    if not scores:
        return [], [], []

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )
    ranked_indices = [i for i in ranked_indices if scores[i] > 0.0][:top_k]

    if not ranked_indices:
        return [], [], []

    max_score = max(scores[i] for i in ranked_indices) or 1.0

    documents = [index.documents[i] for i in ranked_indices]
    metadatas = [index.metadatas[i] for i in ranked_indices]
    distances = [1.0 - (scores[i] / max_score) for i in ranked_indices]

    return documents, metadatas, distances