"""
FastAPI RAG server for TEK17.

Retrieves relevant provision chunks from ChromaDB and generates answers
using an Ollama LLM.

Run with:
    uvicorn tek17.rag.server:app --reload --port 8000
Or via the CLI:
    python -m tek17 serve
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import chromadb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CHROMA_DIR = Path("data/vectorstore/chroma")
COLLECTION_NAME = "tek17"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

TOP_K = 6

SYSTEM_PROMPT = textwrap.dedent("""\
    Du er en ekspert på norske byggeforskrifter, spesielt TEK17 \
    (Byggteknisk forskrift). Svar alltid på norsk med mindre brukeren \
    skriver på engelsk. Baser svaret ditt på konteksten som er gitt. \
    Hvis konteksten ikke inneholder nok informasjon til å svare, si fra \
    om det. Referer til relevante paragrafer (§) når du svarer.\
""")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="TEK17 RAG Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: load ChromaDB collection
# ---------------------------------------------------------------------------
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        chroma_dir = DEFAULT_CHROMA_DIR.resolve()
        if not chroma_dir.exists():
            raise RuntimeError(
                f"ChromaDB vector store not found at {chroma_dir}. "
                "Run `python -m tek17 ingest` first."
            )
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_query(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ---------------------------------------------------------------------------
# Ollama chat helper
# ---------------------------------------------------------------------------

def _ollama_chat(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.3,
) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about TEK17")
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    model: str = Field(default=LLM_MODEL, description="Ollama model to use")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class SourceChunk(BaseModel):
    section_id: str
    title: str
    chapter: str
    text_type: str
    text: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str
    question: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        collection = _get_collection()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Embed the query
    q_embedding = _embed_query(req.question)

    # Retrieve top-k chunks
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Build context
    context_parts: list[str] = []
    sources: list[SourceChunk] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        context_parts.append(doc)
        sources.append(
            SourceChunk(
                section_id=meta.get("section_id", ""),
                title=meta.get("title", ""),
                chapter=meta.get("chapter", ""),
                text_type=meta.get("text_type", ""),
                text=doc,
                distance=dist,
            )
        )

    context_block = "\n\n---\n\n".join(context_parts)

    user_msg = (
        f"Kontekst fra TEK17:\n\n{context_block}\n\n"
        f"---\n\nSpørsmål: {req.question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    answer = _ollama_chat(messages, model=req.model, temperature=req.temperature)

    return QueryResponse(
        answer=answer,
        sources=sources,
        model=req.model,
        question=req.question,
    )


@app.get("/models")
def list_models():
    """Proxy to Ollama's model list so the Streamlit client can show available models."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return {"models": [m["name"] for m in models]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {e}")


@app.get("/collection/stats")
def collection_stats():
    """Return basic stats about the vector store collection."""
    try:
        col = _get_collection()
        return {"collection": COLLECTION_NAME, "count": col.count()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
