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

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tek17.rag.embedding.client import embed_query
from tek17.rag.llm.client import chat
from tek17.rag.retrieval.client import get_collection, query_collection
from tek17.rag.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    LOG_DIR,
    QUERY_LOG_PATH,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    TOP_K,
)

SYSTEM_PROMPT = textwrap.dedent("""\
    Du er en ekspert på norske byggeforskrifter, spesielt TEK17 \
    (Byggteknisk forskrift). Svar alltid på norsk med mindre brukeren \
    skriver på engelsk. Baser svaret ditt utelukkende på konteksten som er gitt \
    (RAG/vektordatabasen). Du har ikke lov til å bruke egen kunnskap eller antakelser. \
    Hvis konteksten ikke inneholder grunnlag for et konkret svar, start svaret med: \
    "KAN_IKKE_SVARE:" og si at du ikke har nok informasjon i databasen/konteksten. \
    Referer til relevante paragrafer (§) kun når de faktisk finnes i konteksten.\
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
# Logging helpers
# ---------------------------------------------------------------------------

def _log_query_event(event: dict) -> None:
    """Append a single query event as JSONL for offline analysis.

    Logging failures should never break the API, so errors are swallowed.
    """

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with QUERY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort logging only
        return


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        collection = get_collection(CHROMA_DIR, CHROMA_COLLECTION)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Embed the query
    q_embedding = embed_query(
        req.question,
        provider=EMBED_PROVIDER,
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Retrieve top-k chunks
    documents, metadatas, distances = query_collection(
        collection=collection,
        query_embedding=q_embedding,
        top_k=req.top_k,
    )

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

    if not context_block.strip():
        answer = (
            'KAN_IKKE_SVARE: Jeg har ikke nok informasjon i RAG-databasen/konteksten '
            'til å gi et konkret svar på dette spørsmålet.'
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            model=req.model,
            question=req.question,
        )

    user_msg = (
        f"Kontekst fra TEK17:\n\n{context_block}\n\n"
        f"---\n\nSpørsmål: {req.question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        answer = chat(
            messages,
            provider=LLM_PROVIDER,
            model=req.model,
            base_url=OLLAMA_BASE_URL,
            temperature=req.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e}")

    # Best-effort structured logging for refusal / retrieval analysis
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": req.question,
        "top_k": req.top_k,
        "model": req.model,
        "temperature": req.temperature,
        "retrieved": [
            {
                "section_id": s.section_id,
                "title": s.title,
                "chapter": s.chapter,
                "text_type": s.text_type,
                "distance": s.distance,
            }
            for s in sources
        ],
        "context": context_block,
        "answer": answer,
    }
    _log_query_event(event)

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
        import requests

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
        col = get_collection(CHROMA_DIR, CHROMA_COLLECTION)
        return {"collection": CHROMA_COLLECTION, "count": col.count()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
