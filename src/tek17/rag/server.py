"""
FastAPI RAG server for TEK17.

Retrieves relevant provision chunks from ChromaDB and generates answers
using a configurable LLM provider (Ollama by default, OpenAI optional).

Run with:
    uvicorn tek17.rag.server:app --reload --port 8000
Or via the CLI:
    python -m tek17 serve
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tek17.rag.embedding.client import embed_query
from tek17.rag.llm.client import chat
from tek17.rag.prompts import SYSTEM_PROMPT
from tek17.rag.retrieval.client import get_collection, retrieve
from tek17.rag.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    LOG_DIR,
    QUERY_LOG_PATH,
    CHUNKS_PATH,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_MAX_TOKENS,
    TOP_K,
    RETRIEVAL_METHOD,
    HYBRID_ALPHA,
)

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

    retrieval_method: str = Field(
        default=RETRIEVAL_METHOD,
        description="Retrieval method: 'dense', 'sparse' (or legacy 'sparce'), or 'hybrid'",
    )
    hybrid_alpha: float = Field(
        default=HYBRID_ALPHA,
        ge=0.0,
        le=1.0,
        description="Hybrid weighting: alpha*dense + (1-alpha)*sparse",
    )
    provider: Optional[str] = Field(
        default=None,
        description="LLM provider override: 'ollama' or 'openai' (defaults to TEK17_LLM_PROVIDER)",
    )
    model: str = Field(default=LLM_MODEL, description="LLM model name (Ollama or OpenAI)")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Optional output token cap (defaults to TEK17_LLM_MAX_TOKENS if set)",
    )


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

    retrieval_method = (req.retrieval_method or RETRIEVAL_METHOD).strip().lower()
    if retrieval_method == "sparce":
        retrieval_method = "sparse"

    q_embedding: list[float] | None = None
    if retrieval_method in {"dense", "hybrid"}:
        q_embedding = embed_query(
            req.question,
            provider=EMBED_PROVIDER,
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

    documents, metadatas, distances = retrieve(
        collection=collection,
        query_text=req.question,
        query_embedding=q_embedding,
        top_k=req.top_k,
        method=retrieval_method,
        chunks_path=CHUNKS_PATH,
        hybrid_alpha=req.hybrid_alpha,
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
        provider = (req.provider or LLM_PROVIDER).strip().lower()
        if provider not in {"ollama", "openai"}:
            raise HTTPException(
                status_code=400,
                detail="Invalid provider. Must be 'ollama' or 'openai'.",
            )

        max_tokens = req.max_tokens if req.max_tokens is not None else LLM_MAX_TOKENS
        answer = chat(
            messages,
            provider=provider,  # type: ignore[arg-type]
            model=req.model,
            base_url=OLLAMA_BASE_URL,
            temperature=req.temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e}")

    # Best-effort structured logging for refusal / retrieval analysis
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": req.question,
        "top_k": req.top_k,
        "retrieval_method": retrieval_method,
        "hybrid_alpha": req.hybrid_alpha,
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
    if (LLM_PROVIDER or "").strip().lower() != "ollama":
        return {
            "models": [],
            "note": "Model listing is only available for the Ollama provider.",
        }
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
