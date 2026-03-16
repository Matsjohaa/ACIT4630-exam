from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


# Chroma's Posthog telemetry integration can be noisy if the installed
# `posthog` package has an incompatible API. Telemetry is non-essential
# for this project, so we silence the telemetry logger.
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


_collection_cache: dict[tuple[Path, str], chromadb.Collection] = {}


def get_collection(chroma_dir: Path, collection_name: str) -> chromadb.Collection:
    """Get (and cache) a ChromaDB collection for retrieval.

    This keeps the server code simple and centralises how we connect
    to the persistent vector store.
    """

    key = (chroma_dir.resolve(), collection_name)
    if key in _collection_cache:
        return _collection_cache[key]

    if not chroma_dir.exists():
        raise RuntimeError(
            f"ChromaDB vector store not found at {chroma_dir}. "
            "Run `python -m tek17 ingest` first."
        )

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(collection_name)
    _collection_cache[key] = collection
    return collection


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Query a Chroma collection and return docs, metadatas, distances."""

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    return documents, metadatas, distances
