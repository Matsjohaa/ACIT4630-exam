from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Literal

import chromadb
import requests
from chromadb.config import Settings

from tek17.rag.config import (
    CHUNKS_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBED_PROVIDER,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

Provider = Literal["ollama", "openai"]


def _stable_id(text: str, metadata: dict) -> str:
    """Deterministic chunk ID for reproducible ingestion."""
    section_id = metadata.get("section_id", "")
    text_type = metadata.get("text_type", "")
    para_start = metadata.get("para_start", "")
    para_end = metadata.get("para_end", "")

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    key = f"{section_id}::{text_type}::{para_start}-{para_end}::{text_hash}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _load_chunks(path: Path) -> list[dict]:
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def embed_texts(
    texts: list[str],
    provider: Provider = EMBED_PROVIDER,
    model: str = EMBED_MODEL,
    base_url: str | None = None,
) -> list[list[float]]:
    """Embed a batch of texts using the configured provider."""
    if not texts:
        return []

    if provider == "ollama":
        return _embed_ollama(
            texts=texts,
            model=model,
            base_url=base_url or OLLAMA_BASE_URL,
        )

    if provider == "openai":
        return _embed_openai(
            texts=texts,
            model=model,
            base_url=base_url or OPENAI_BASE_URL,
        )

    raise ValueError(f"Unknown embedding provider: {provider}")


def embed_query(
    text: str,
    provider: Provider = EMBED_PROVIDER,
    model: str = EMBED_MODEL,
    base_url: str | None = None,
) -> list[float]:
    """Embed a single query string."""
    embeddings = embed_texts(
        texts=[text],
        provider=provider,
        model=model,
        base_url=base_url,
    )
    return embeddings[0]


def _embed_ollama(
    texts: list[str],
    model: str,
    base_url: str,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = 32

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": batch},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embeddings.extend(data["embeddings"])

    return embeddings


def _embed_openai(
    texts: list[str],
    model: str,
    base_url: str | None = None,
) -> list[list[float]]:
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI embeddings requested but no API key found. "
            "Set OPENAI_API_KEY or OPEN_AI_API_KEY."
        )

    try:
        from openai import OpenAI  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "The 'openai' package is required for OpenAI embeddings."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    embeddings: list[list[float]] = []
    batch_size = 128

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def ingest_chunks_to_chroma(
    chunks_path: Path = CHUNKS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    embed_provider: Provider = EMBED_PROVIDER,
    embed_model: str = EMBED_MODEL,
    base_url: str | None = None,
) -> None:
    """Read prebuilt chunks, embed them, and write them into ChromaDB."""
    resolved_chunks_path = chunks_path.resolve()
    resolved_chroma_dir = chroma_dir.resolve()

    if not resolved_chunks_path.exists():
        raise FileNotFoundError(f"Chunk corpus not found: {resolved_chunks_path}")

    chunks = _load_chunks(resolved_chunks_path)
    print(f"Loaded {len(chunks)} chunks from {resolved_chunks_path}")

    texts: list[str] = []
    metadatas: list[dict] = []
    for chunk in chunks:
        if "text" not in chunk or "metadata" not in chunk:
            raise ValueError("Invalid chunk record: expected 'text' and 'metadata'")
        texts.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    print(f"Embedding {len(texts)} chunks with {embed_provider}:{embed_model} ...")
    embeddings = embed_texts(
        texts=texts,
        provider=embed_provider,
        model=embed_model,
        base_url=base_url,
    )

    resolved_chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(resolved_chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(collection_name)
    except Exception:
        # Ignore missing collection on first ingest.
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for start in range(0, len(texts), batch_size):
        text_batch = texts[start:start + batch_size]
        metadata_batch = metadatas[start:start + batch_size]
        embedding_batch = embeddings[start:start + batch_size]
        ids = [_stable_id(text, metadata) for text, metadata in zip(text_batch, metadata_batch)]

        collection.upsert(
            ids=ids,
            embeddings=embedding_batch,
            documents=text_batch,
            metadatas=metadata_batch,
        )

    total = collection.count()
    print(f"ChromaDB collection '{collection_name}' now has {total} chunks.")
    print(f"Vector store persisted at: {resolved_chroma_dir}")


def run_ingest(
    chunks_path: Path = CHUNKS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    embed_provider: Provider = EMBED_PROVIDER,
    embed_model: str = EMBED_MODEL,
    base_url: str | None = None,
) -> None:
    """Embed prebuilt TEK17 chunks and ingest them into ChromaDB.
    Default setup uses Ollama embeddings. OpenAI embeddings are supported
    for optional experiments, but are not required for normal ingestion.
    """
    ingest_chunks_to_chroma(
        chunks_path=chunks_path,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embed_provider=embed_provider,
        embed_model=embed_model,
        base_url=base_url,
    )