from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import chromadb
from chromadb.config import Settings

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

_COLLECTION_CACHE: dict[tuple[Path, str], chromadb.Collection] = {}

RetrievalMethod = Literal["dense", "hybrid", "sparse"]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 checksum of a file."""
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


@lru_cache(maxsize=16)
def vectorstore_snapshot(chroma_dir: Path, collection_name: str) -> dict[str, Any]:
    """Return a lightweight snapshot of the vector store for reproducibility."""
    collection = get_collection(chroma_dir, collection_name)
    count = collection.count()

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
    """Return a cached ChromaDB collection handle."""
    resolved_chroma_dir = chroma_dir.resolve()
    cache_key = (resolved_chroma_dir, collection_name)

    if cache_key in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[cache_key]

    if not resolved_chroma_dir.exists():
        raise RuntimeError(
            f"ChromaDB vector store not found at {resolved_chroma_dir}. "
            "Run `python -m tek17 ingest` first."
        )

    client = chromadb.PersistentClient(
        path=str(resolved_chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(collection_name)
    _COLLECTION_CACHE[cache_key] = collection

    return collection


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Query a Chroma collection and return documents, metadatas, and distances."""
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
    method: str = "dense",
    chunks_path: Path | None = None,
    hybrid_alpha: float = 0.5,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Retrieve documents using the selected retrieval method."""
    normalized_method = (method or "dense").strip().lower()
    if normalized_method == "sparce":
        normalized_method = "sparse"

    if normalized_method == "dense":
        if query_embedding is None:
            raise ValueError("query_embedding is required for dense retrieval")
        return query_collection(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    if normalized_method == "sparse":
        if chunks_path is None:
            raise ValueError("chunks_path is required for sparse retrieval")
        from tek17.rag.retrieval.methods.sparse import retrieve_sparse

        return retrieve_sparse(
            query_text=query_text,
            top_k=top_k,
            chunks_path=chunks_path,
        )

    if normalized_method == "hybrid":
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

    raise ValueError(
        f"Unknown retrieval method: {method}. Supported methods are 'dense', 'sparse', and 'hybrid'."
    )