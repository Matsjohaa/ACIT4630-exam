from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Literal

import chromadb
from chromadb.config import Settings


# Chroma's Posthog telemetry integration can be noisy if the installed
# `posthog` package has an incompatible API. Telemetry is non-essential
# for this project, so we silence the telemetry logger.
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


_collection_cache: dict[tuple[Path, str], chromadb.Collection] = {}


RetrievalMethod = Literal["dense", "hybrid", "sparse", "sparce"]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=16)
def vectorstore_snapshot(chroma_dir: Path, collection_name: str) -> dict[str, Any]:
    """Return a lightweight snapshot of the vector store for reproducibility.

    Includes the collection count and a stable fingerprint of Chroma's
    `chroma.sqlite3` file (when present).
    """

    col = get_collection(chroma_dir, collection_name)
    count = col.count()

    sqlite_path = chroma_dir / "chroma.sqlite3"
    sqlite_sha256: str | None = None
    sqlite_size: int | None = None

    if sqlite_path.exists() and sqlite_path.is_file():
        sqlite_sha256 = _sha256_file(sqlite_path)
        sqlite_size = sqlite_path.stat().st_size

    return {
        "chroma_dir": str(chroma_dir.resolve()),
        "collection": collection_name,
        "count": int(count),
        "sqlite_sha256": sqlite_sha256,
        "sqlite_size": sqlite_size,
    }


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


def retrieve(
    *,
    collection: chromadb.Collection,
    query_text: str,
    query_embedding: list[float] | None,
    top_k: int,
    method: RetrievalMethod = "dense",
    chunks_path: Path | None = None,
    hybrid_alpha: float = 0.5,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Retrieve documents using the selected retrieval method.

    Returns (documents, metadatas, distances) where lower distance is better.
    """

    m = (method or "dense").strip().lower()
    # Backwards-compatible alias (older code used 'sparce')
    if m == "sparce":
        m = "sparse"

    if m == "dense":
        if query_embedding is None:
            raise ValueError("query_embedding is required for dense retrieval")
        return query_collection(collection=collection, query_embedding=query_embedding, top_k=top_k)

    if m == "sparse":
        if chunks_path is None:
            raise ValueError("chunks_path is required for sparse retrieval")
        from tek17.rag.retrieval.methods.sparse import retrieve_sparce

        return retrieve_sparce(query_text=query_text, top_k=top_k, chunks_path=chunks_path)

    if m == "hybrid":
        if query_embedding is None:
            raise ValueError("query_embedding is required for hybrid retrieval")
        if chunks_path is None:
            raise ValueError("chunks_path is required for hybrid retrieval")
        from tek17.rag.retrieval.methods.hybrid import retrieve_hybrid

        return retrieve_hybrid(
            collection=collection,
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            alpha=hybrid_alpha,
            chunks_path=chunks_path,
        )

    raise ValueError(f"Unknown retrieval method: {method}")
