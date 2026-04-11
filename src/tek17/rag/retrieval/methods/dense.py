from __future__ import annotations

from typing import Any

import chromadb


def retrieve_dense(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Dense retrieval via Chroma vector similarity."""

    if not query_embedding:
        raise ValueError("query_embedding must be a non-empty vector")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return documents, metadatas, distances