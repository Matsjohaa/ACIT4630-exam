from __future__ import annotations

from typing import Any

import chromadb


def retrieve_dense(
	collection: chromadb.Collection,
	query_embedding: list[float],
	top_k: int,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
	"""Dense retrieval via Chroma vector similarity."""

	results = collection.query(
		query_embeddings=[query_embedding],
		n_results=top_k,
		include=["documents", "metadatas", "distances"],
	)

	documents = results["documents"][0] if results.get("documents") else []
	metadatas = results["metadatas"][0] if results.get("metadatas") else []
	distances = results["distances"][0] if results.get("distances") else []

	return documents, metadatas, distances

