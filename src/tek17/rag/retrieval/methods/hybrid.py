from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb

from tek17.rag.config import HYBRID_CANDIDATE_MULTIPLIER
from tek17.rag.retrieval.methods.dense import retrieve_dense
from tek17.rag.retrieval.methods.sparse import retrieve_sparse


def _chunk_key(doc: str, metadata: dict[str, Any]) -> str:
    """Return a stable identifier for a retrieved chunk."""
    section_id = str(metadata.get("section_id", ""))
    text_type = str(metadata.get("text_type", ""))
    para_start = str(metadata.get("para_start", ""))
    para_end = str(metadata.get("para_end", ""))
    title = str(metadata.get("title", ""))

    base = "|".join([section_id, title, text_type, para_start, para_end])
    if base.strip("|"):
        return base

    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


def retrieve_hybrid(
    collection: chromadb.Collection,
    query_text: str,
    query_embedding: list[float],
    top_k: int,
    *,
    alpha: float = 0.5,
    dense_candidates: int | None = None,
    sparse_candidates: int | None = None,
    chunks_path: Path | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Combine dense and sparse retrieval into one ranked result set.

    Alpha controls the dense-vs-sparse weighting:
    - alpha = 1.0 -> dense only
    - alpha = 0.0 -> sparse only

    Returns distances in [0, 1], where lower is better.
    """
    if chunks_path is None:
        raise ValueError("chunks_path is required for hybrid retrieval")

    dense_k = dense_candidates or max(top_k * HYBRID_CANDIDATE_MULTIPLIER, top_k)
    sparse_k = sparse_candidates or max(top_k * HYBRID_CANDIDATE_MULTIPLIER, top_k)

    dense_docs, dense_metas, dense_dists = retrieve_dense(
        collection,
        query_embedding,
        dense_k,
    )
    sparse_docs, sparse_metas, sparse_dists = retrieve_sparse(
        query_text,
        sparse_k,
        chunks_path,
    )

    dense_scores: dict[str, float] = {}
    dense_payload: dict[str, tuple[str, dict[str, Any], float]] = {}

    for document, metadata, distance in zip(dense_docs, dense_metas, dense_dists):
        key = _chunk_key(document, metadata)
        score = 1.0 / (1.0 + float(distance))
        if key not in dense_scores or score > dense_scores[key]:
            dense_scores[key] = score
            dense_payload[key] = (document, metadata, float(distance))

    sparse_scores: dict[str, float] = {}
    sparse_payload: dict[str, tuple[str, dict[str, Any], float]] = {}

    for document, metadata, distance in zip(sparse_docs, sparse_metas, sparse_dists):
        key = _chunk_key(document, metadata)
        score = 1.0 - float(distance)
        if key not in sparse_scores or score > sparse_scores[key]:
            sparse_scores[key] = score
            sparse_payload[key] = (document, metadata, float(distance))

    keys = set(dense_scores) | set(sparse_scores)
    if not keys:
        return [], [], []

    max_dense = max(dense_scores.values(), default=0.0) or 1.0
    max_sparse = max(sparse_scores.values(), default=0.0) or 1.0

    combined_scores: list[tuple[str, float]] = []
    for key in keys:
        dense_score = dense_scores.get(key, 0.0) / max_dense
        sparse_score = sparse_scores.get(key, 0.0) / max_sparse
        combined_score = (alpha * dense_score) + ((1.0 - alpha) * sparse_score)
        combined_scores.append((key, combined_score))

    combined_scores.sort(key=lambda item: item[1], reverse=True)
    combined_scores = combined_scores[:top_k]

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    distances: list[float] = []

    for key, combined_score in combined_scores:
        if key in dense_payload:
            document, metadata, _ = dense_payload[key]
        else:
            document, metadata, _ = sparse_payload[key]

        documents.append(document)
        metadatas.append(metadata)
        distances.append(1.0 - combined_score)

    return documents, metadatas, distances