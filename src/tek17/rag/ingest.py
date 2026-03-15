"""RAG ingestion orchestration.

This module coordinates two explicit steps:

1. Chunking the TEK17 JSONL corpus into `tek17_chunks.jsonl`.
2. Embedding those chunks and ingesting them into ChromaDB.

The actual chunking and embedding logic lives in:
- ``tek17.corpus.chunks``
- ``tek17.rag.embedding.chroma_ingest``
"""

from __future__ import annotations

from pathlib import Path

from tek17.corpus.chunks import build_and_save_chunks
from tek17.rag.embedding.chroma_ingest import ingest_chunks_to_chroma
from tek17.rag.config import (
    JSONL_PATH,
    CHUNKS_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBED_PROVIDER,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def run_ingest(
    jsonl_path: Path = JSONL_PATH,
    chunks_path: Path = CHUNKS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_provider: str = EMBED_PROVIDER,
    embed_model: str = EMBED_MODEL,
    ollama_url: str = OLLAMA_BASE_URL,
) -> None:
    """Full ingestion pipeline used by the CLI ``tek17 ingest``.

    Keeps the original behaviour but delegates to dedicated modules for:
    - building chunks from the JSONL corpus, and
    - embedding + ingesting chunks into ChromaDB.
    """

    # 1. Build chunked corpus from the canonical TEK17 JSONL.
    build_and_save_chunks(
        jsonl_path=jsonl_path,
        chunks_path=chunks_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 2. Embed chunks and write them into ChromaDB.
    ingest_chunks_to_chroma(
        chunks_path=chunks_path,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embed_provider=embed_provider,
        embed_model=embed_model,
        ollama_url=ollama_url,
    )
