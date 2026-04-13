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
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tek17.rag.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNKS_PATH,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
    HYBRID_ALPHA,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LOG_DIR,
    PROMPT_VERSION,
    QUERY_LOG_PATH,
    RETRIEVAL_METHOD,
    TOP_K,
)
from tek17.rag.ingest import embed_query
from tek17.rag.llm.dispatcher import chat_result
from tek17.rag.prompts import get_system_prompt, get_system_prompt_sha256
from tek17.rag.retrieval.client import get_collection, retrieve, vectorstore_snapshot


app = FastAPI(title="TEK17 RAG Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    prompt_version: str = Field(
        default=PROMPT_VERSION,
        description="Prompt variant: baseline, relaxed, or strict",
    )
    provider: Optional[str] = Field(
        default=None,
        description="LLM provider override: 'ollama' or 'openai' (defaults to TEK17_LLM_PROVIDER)",
    )
    model: str = Field(default=LLM_MODEL, description="LLM model name (Ollama or OpenAI)")
    temperature: float = Field(default=LLM_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Optional output token cap (defaults to TEK17_LLM_MAX_TOKENS if set)",
    )
    requires_qualification: bool = Field(
        default=False,
        description="Whether the question requires explicit qualification due to missing project/context facts",
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


def _log_query_event(event: dict) -> None:
    """Append a single query event as JSONL for offline analysis."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with QUERY_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        return


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        collection = get_collection(CHROMA_DIR, CHROMA_COLLECTION)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        vs_snapshot = vectorstore_snapshot(CHROMA_DIR, CHROMA_COLLECTION)
    except Exception:
        vs_snapshot = None

    retrieval_method = (req.retrieval_method or RETRIEVAL_METHOD).strip().lower()
    if retrieval_method == "sparce":
        retrieval_method = "sparse"

    prompt_version = (req.prompt_version or PROMPT_VERSION).strip().lower()
    system_prompt = get_system_prompt(
        prompt_version,
        requires_qualification=req.requires_qualification,
    )
    system_prompt_sha256 = get_system_prompt_sha256(
        prompt_version,
        requires_qualification=req.requires_qualification,
    )

    query_embedding: list[float] | None = None
    if retrieval_method in {"dense", "hybrid"}:
        query_embedding = embed_query(
            req.question,
            provider=EMBED_PROVIDER,
            model=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
        )

    documents, metadatas, distances = retrieve(
        collection=collection,
        query_text=req.question,
        query_embedding=query_embedding,
        top_k=req.top_k,
        method=retrieval_method,
        chunks_path=CHUNKS_PATH,
        hybrid_alpha=req.hybrid_alpha,
    )

    context_parts: list[str] = []
    sources: list[SourceChunk] = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        context_parts.append(document)
        sources.append(
            SourceChunk(
                section_id=metadata.get("section_id", ""),
                title=metadata.get("title", ""),
                chapter=metadata.get("chapter", ""),
                text_type=metadata.get("text_type", ""),
                text=document,
                distance=distance,
            )
        )

    context_block = "\n\n---\n\n".join(context_parts)

    provider = (req.provider or LLM_PROVIDER).strip().lower()
    if provider not in {"ollama", "openai"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid provider. Must be 'ollama' or 'openai'.",
        )

    max_tokens = req.max_tokens if req.max_tokens is not None else LLM_MAX_TOKENS

    if not context_block.strip():
        answer = (
            "KAN_IKKE_SVARE: Jeg har ikke nok informasjon i RAG-databasen/konteksten "
            "til å gi et konkret svar på dette spørsmålet."
        )

        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prompt_version": prompt_version,
            "system_prompt_sha256": system_prompt_sha256,
            "question": req.question,
            "top_k": req.top_k,
            "retrieval_method": retrieval_method,
            "hybrid_alpha": req.hybrid_alpha,
            "embed_provider": EMBED_PROVIDER,
            "embed_model": EMBED_MODEL,
            "vectorstore": vs_snapshot,
            "provider": provider,
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": max_tokens,
            "retrieved": [
                {
                    "section_id": source.section_id,
                    "title": source.title,
                    "chapter": source.chapter,
                    "text_type": source.text_type,
                    "distance": source.distance,
                }
                for source in sources
            ],
            "context": context_block,
            "answer": answer,
            "llm_finish_reason": None,
            "llm_usage": None,
            "requires_qualification": req.requires_qualification,
        }
        _log_query_event(event)

        return QueryResponse(
            answer=answer,
            sources=sources,
            model=req.model,
            question=req.question,
        )

    user_message = (
        f"Kontekst fra TEK17:\n\n{context_block}\n\n"
        f"---\n\nSpørsmål: {req.question}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        result = chat_result(
            messages=messages,
            provider=provider,  # type: ignore[arg-type]
            model=req.model,
            base_url=LLM_BASE_URL,
            temperature=req.temperature,
            max_tokens=max_tokens,
        )
        answer = result.content
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}")

    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt_version": prompt_version,
        "system_prompt_sha256": system_prompt_sha256,
        "question": req.question,
        "top_k": req.top_k,
        "retrieval_method": retrieval_method,
        "hybrid_alpha": req.hybrid_alpha,
        "embed_provider": EMBED_PROVIDER,
        "embed_model": EMBED_MODEL,
        "vectorstore": vs_snapshot,
        "provider": provider,
        "model": req.model,
        "temperature": req.temperature,
        "max_tokens": max_tokens,
        "retrieved": [
            {
                "section_id": source.section_id,
                "title": source.title,
                "chapter": source.chapter,
                "text_type": source.text_type,
                "distance": source.distance,
            }
            for source in sources
        ],
        "context": context_block,
        "answer": answer,
        "llm_finish_reason": result.finish_reason,
        "llm_usage": result.usage,
        "requires_qualification": req.requires_qualification,
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
    if LLM_PROVIDER != "ollama":
        return {
            "models": [],
            "note": "Model listing is only available for the Ollama provider.",
        }

    try:
        import requests

        response = requests.get(f"{LLM_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        return {"models": [model["name"] for model in models]}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {exc}")


@app.get("/collection/stats")
def collection_stats():
    """Return basic stats about the vector store collection."""
    try:
        collection = get_collection(CHROMA_DIR, CHROMA_COLLECTION)
        return {"collection": CHROMA_COLLECTION, "count": collection.count()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))