"""
Ingest TEK17 JSONL into a ChromaDB vector store using Ollama embeddings.

Reads the parsed TEK17 provisions from data/processed/tek17_dibk.jsonl,
chunks them, and stores embeddings in a persistent ChromaDB collection.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_JSONL_PATH = Path("data/processed/tek17_dibk.jsonl")
DEFAULT_CHROMA_DIR = Path("data/vectorstore/chroma")
COLLECTION_NAME = "tek17"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_id(text: str, meta: dict) -> str:
    """Deterministic ID so re-ingestion is idempotent."""
    key = f"{meta.get('section_id', '')}::{text[:200]}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_documents(records: list[dict]) -> list[tuple[str, dict]]:
    """
    Turn provision records into (text, metadata) pairs ready for chunking.

    Each provision produces up to TWO documents:
    1. Regulation text   (tag: reg)
    2. Guidance text     (tag: guidance)

    This separation lets downstream retrieval experiments ablate reg-only
    vs reg+guidance.
    """
    docs: list[tuple[str, dict]] = []
    for rec in records:
        section_id = rec.get("section_id", "unknown") or "unknown"
        title = rec.get("title", "") or ""
        chapter = rec.get("chapter", "") or ""

        base_meta = {
            "source": "dibk",
            "section_id": section_id,
            "title": title,
            "chapter": chapter,
        }

        reg = rec.get("reg_text", "").strip()
        if reg:
            meta = {**base_meta, "text_type": "regulation"}
            header = f"{section_id} – {title}\n(Forskriftstekst)\n\n"
            docs.append((header + reg, meta))

        guidance = rec.get("guidance_text", "").strip()
        if guidance:
            meta = {**base_meta, "text_type": "guidance"}
            header = f"{section_id} – {title}\n(Veiledning)\n\n"
            docs.append((header + guidance, meta))

    return docs


# ---------------------------------------------------------------------------
# Embedding helper using Ollama REST API directly
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str], model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL) -> list[list[float]]:
    """Call Ollama /api/embed endpoint to get embeddings."""
    import requests

    embeddings = []
    # Batch in groups of 32 to avoid overwhelming the API
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings.extend(data["embeddings"])
    return embeddings


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def run_ingest(
    jsonl_path: Path = DEFAULT_JSONL_PATH,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_model: str = EMBED_MODEL,
    ollama_url: str = OLLAMA_BASE_URL,
) -> None:
    """
    Full ingestion pipeline:
    1. Load JSONL provision records
    2. Build text documents (reg + guidance separately)
    3. Chunk with RecursiveCharacterTextSplitter
    4. Embed with Ollama
    5. Upsert into ChromaDB
    """
    jsonl_path = jsonl_path.resolve()
    chroma_dir = chroma_dir.resolve()

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"JSONL corpus not found: {jsonl_path}\n"
            "Run `python -m tek17 download-dibk` then `python -m tek17 parse-dibk` first."
        )

    print(f"Loading provisions from {jsonl_path} …")
    records = _load_jsonl(jsonl_path)
    print(f"  {len(records)} provision records loaded.")

    raw_docs = _build_documents(records)
    print(f"  {len(raw_docs)} documents (reg + guidance) before chunking.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[tuple[str, dict]] = []
    for text, meta in raw_docs:
        for chunk_text in splitter.split_text(text):
            chunks.append((chunk_text, meta))

    print(f"  {len(chunks)} chunks after splitting.")

    # Embed all chunks
    print(f"Embedding {len(chunks)} chunks with {embed_model} via {ollama_url} …")
    all_texts = [c[0] for c in chunks]
    all_embeddings = _embed_texts(all_texts, model=embed_model, base_url=ollama_url)

    # Prepare ChromaDB
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete old collection if it exists, for a clean re-index
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeds = all_embeddings[i : i + batch_size]

        ids = [_stable_id(text, meta) for text, meta in batch_chunks]
        documents = [text for text, _ in batch_chunks]
        metadatas = [meta for _, meta in batch_chunks]

        collection.upsert(
            ids=ids,
            embeddings=batch_embeds,
            documents=documents,
            metadatas=metadatas,
        )

    total = collection.count()
    print(f"ChromaDB collection '{collection_name}' now has {total} chunks.")
    print(f"Vector store persisted at: {chroma_dir}")
