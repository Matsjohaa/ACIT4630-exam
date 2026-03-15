from __future__ import annotations

import json
import hashlib
from pathlib import Path

import chromadb

from .client import embed_texts
from tek17.rag.config import (
    CHUNKS_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBED_PROVIDER,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
)


def _stable_id(text: str, meta: dict) -> str:
    """Deterministic ID so re-ingestion is idempotent."""
    key = f"{meta.get('section_id', '')}::{text[:200]}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _load_chunks(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def ingest_chunks_to_chroma(
    chunks_path: Path = CHUNKS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    embed_provider: str = EMBED_PROVIDER,
    embed_model: str = EMBED_MODEL,
    ollama_url: str = OLLAMA_BASE_URL,
) -> None:
    """Read pre-built chunks, embed them and upsert into ChromaDB."""
    chunks_path = chunks_path.resolve()
    chroma_dir = chroma_dir.resolve()

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk corpus not found: {chunks_path}\n"
            "Run the chunking step first to create tek17_chunks.jsonl."
        )

    chunks = _load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(texts)} chunks with {embed_provider}:{embed_model} …")
    embeddings = embed_texts(
        texts,
        provider=embed_provider,  # type: ignore[arg-type]
        model=embed_model,
        base_url=ollama_url,
    )

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        client.delete_collection(collection_name)
    except Exception:
        # ignore if it does not exist yet
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        t_batch = texts[i : i + batch_size]
        m_batch = metadatas[i : i + batch_size]
        e_batch = embeddings[i : i + batch_size]

        ids = [_stable_id(text, meta) for text, meta in zip(t_batch, m_batch)]

        collection.upsert(
            ids=ids,
            embeddings=e_batch,
            documents=t_batch,
            metadatas=m_batch,
        )

    total = collection.count()
    print(f"ChromaDB collection '{collection_name}' now has {total} chunks.")
    print(f"Vector store persisted at: {chroma_dir}")
