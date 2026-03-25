from __future__ import annotations

"""Central configuration for the TEK17 RAG stack.

This module defines all paths and defaults used by the corpus pipeline,
vector store, embeddings, and LLM. Every setting can be overridden with
environment variables so you can change models or data locations without
touching code.

Typical usage from the shell:

    # Change embedding model and LLM model
    export TEK17_EMBED_MODEL="nomic-embed-text"
    export TEK17_LLM_MODEL="llama3.2"

    # Point to a different Ollama instance
    export TEK17_OLLAMA_BASE_URL="http://localhost:11434"

    # Adjust retrieval behaviour
    export TEK17_TOP_K="8"

    # Then run the normal commands
    python -m tek17 ingest
    python -m tek17 serve

Defaults are chosen to work out-of-the-box for local development, so you
only need to set env vars when you want to experiment with alternatives.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(os.getenv("TEK17_BASE_DIR", ".")).resolve()

DATA_DIR = BASE_DIR / os.getenv("TEK17_DATA_SUBDIR", "data")
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
CHROMA_DIR = VECTORSTORE_DIR / "chroma"

ANALYSIS_DIR = BASE_DIR / os.getenv("TEK17_ANALYSIS_SUBDIR", "analysis")
LOG_DIR = ANALYSIS_DIR / "logging"
QUERY_LOG_PATH = LOG_DIR / "rag_queries.jsonl"


# Canonical corpus files
JSONL_PATH = PROCESSED_DIR / os.getenv(
    "TEK17_JSONL_FILENAME", "tek17_dibk.jsonl"
)
CHUNKS_PATH = PROCESSED_DIR / os.getenv(
    "TEK17_CHUNKS_FILENAME", "tek17_chunks.jsonl"
)


# ---------------------------------------------------------------------------
# Vector store / retrieval
# ---------------------------------------------------------------------------

CHROMA_COLLECTION = os.getenv("TEK17_CHROMA_COLLECTION", "tek17")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBED_PROVIDER = os.getenv("TEK17_EMBED_PROVIDER", "ollama")
EMBED_MODEL = os.getenv("TEK17_EMBED_MODEL", "nomic-embed-text")

# Ollama configuration (default local provider)
OLLAMA_BASE_URL = os.getenv("TEK17_OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI configuration (optional alternative provider)
#
# API key can be provided as either OPENAI_API_KEY (standard) or
# OPEN_AI_API_KEY (as in the .env example). If both are set, the
# standard OPENAI_API_KEY wins.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # for Azure / custom endpoints


# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("TEK17_LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("TEK17_LLM_MODEL", "llama3.2")

# Optional output cap for hosted LLMs (e.g., OpenAI) to control cost.
# If unset/invalid, no explicit cap is sent to the provider.
try:
    _LLM_MAX_TOKENS_RAW = os.getenv("TEK17_LLM_MAX_TOKENS", "").strip()
    LLM_MAX_TOKENS = int(_LLM_MAX_TOKENS_RAW) if _LLM_MAX_TOKENS_RAW else None
except ValueError:
    LLM_MAX_TOKENS = None

try:
    LLM_TEMPERATURE = float(os.getenv("TEK17_LLM_TEMPERATURE", "0.3"))
except ValueError:
    LLM_TEMPERATURE = 0.3


# ---------------------------------------------------------------------------
# RAG behaviour
# ---------------------------------------------------------------------------

# Retrieval method selection
RETRIEVAL_METHOD = os.getenv("TEK17_RETRIEVAL_METHOD", "dense").strip().lower()

try:
    HYBRID_ALPHA = float(os.getenv("TEK17_HYBRID_ALPHA", "0.5"))
except ValueError:
    HYBRID_ALPHA = 0.5

try:
    TOP_K = int(os.getenv("TEK17_TOP_K", "6"))
except ValueError:
    TOP_K = 6


try:
    CHUNK_SIZE = int(os.getenv("TEK17_CHUNK_SIZE", "800"))
except ValueError:
    CHUNK_SIZE = 800


try:
    CHUNK_OVERLAP = int(os.getenv("TEK17_CHUNK_OVERLAP", "200"))
except ValueError:
    CHUNK_OVERLAP = 200
