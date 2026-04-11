from __future__ import annotations

"""Central configuration for the TEK17 corpus, retrieval, and generation pipeline."""

import os
from pathlib import Path


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_optional_int_env(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


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

JSONL_PATH = PROCESSED_DIR / os.getenv("TEK17_JSONL_FILENAME", "tek17_dibk.jsonl")
CHUNKS_PATH = PROCESSED_DIR / os.getenv("TEK17_CHUNKS_FILENAME", "tek17_chunks.jsonl")


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

CHROMA_COLLECTION = os.getenv("TEK17_CHROMA_COLLECTION", "tek17")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBED_PROVIDER = os.getenv("TEK17_EMBED_PROVIDER", "ollama").strip().lower()
EMBED_MODEL = os.getenv("TEK17_EMBED_MODEL", "nomic-embed-text").strip()
OLLAMA_BASE_URL = os.getenv("TEK17_OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_BASE_URL = os.getenv("TEK17_EMBED_BASE_URL", OLLAMA_BASE_URL)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

PROMPT_VERSION = os.getenv("TEK17_PROMPT_VERSION", "baseline").strip().lower()
LLM_BASE_URL = os.getenv("TEK17_LLM_BASE_URL", OLLAMA_BASE_URL)
LLM_PROVIDER = os.getenv("TEK17_LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.getenv("TEK17_LLM_MODEL", "llama3.2").strip()
LLM_MAX_TOKENS = _get_optional_int_env("TEK17_LLM_MAX_TOKENS")
LLM_TEMPERATURE = _get_float_env("TEK17_LLM_TEMPERATURE", 0.3)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = _get_int_env("TEK17_CHUNK_SIZE", 800)
CHUNK_OVERLAP = _get_int_env("TEK17_CHUNK_OVERLAP", 200)


# ---------------------------------------------------------------------------
# Evaluation / test runner defaults
# ---------------------------------------------------------------------------

SERVER_URL = os.getenv("TEK17_SERVER_URL", "http://localhost:8000")
TEST_MODE = os.getenv("TEK17_TEST_MODE", "local").strip().lower()

GROUNDEDNESS_THRESHOLD = _get_float_env("TEK17_GROUNDEDNESS_THRESHOLD", 0.25)
SOURCE_PREVIEW_LIMIT = _get_int_env("TEK17_SOURCE_PREVIEW_LIMIT", 6)

# ---------------------------------------------------------------------------
# Evaluation / analysis defaults
# ---------------------------------------------------------------------------

EVAL_MODE = os.getenv("TEK17_EVAL_MODE", "local")
REQUEST_TIMEOUT_S = _get_int_env("TEK17_REQUEST_TIMEOUT_S", 300)
COLLECTION_STATS_TIMEOUT_S = _get_int_env("TEK17_COLLECTION_STATS_TIMEOUT_S", 10)

# Refusal detection
REFUSAL_TAG = os.getenv("TEK17_REFUSAL_TAG", "KAN_IKKE_SVARE")
REFUSAL_PATTERNS = [
    "kan ikke svare",
    "kan ikke gi et sikkert svar",
    "kan ikke gi et konkret svar",
    "har ikke nok informasjon",
    "finner ikke nok informasjon",
    "ikke nok informasjon",
    "mangler informasjon",
    "har ikke grunnlag",
    "finner ikke grunnlag",
    "utenfor det som dekkes av tek17",
    "utenfor tek17",
    "i can't answer",
    "i cannot answer",
    "i don't have enough information",
    "not enough information",
    "outside the scope",
]

# Groundedness / lexical overlap
CONTENT_WORD_MIN_LEN = _get_int_env("TEK17_CONTENT_WORD_MIN_LEN", 4)
WILSON_Z = _get_float_env("TEK17_WILSON_Z", 1.96)

CONTENT_STOPWORDS = {
    "og",
    "eller",
    "som",
    "med",
    "for",
    "til",
    "av",
    "på",
    "i",
    "jf",
    "kapittel",
    "paragraf",
    "ledd",
    "bokstav",
    "gjelder",
    "skal",
    "kan",
    "må",
    "ikke",
    "det",
    "den",
    "de",
    "et",
    "en",
    "er",
    "å",
    "når",
    "hva",
    "hvordan",
    "hvilke",
    "hvilken",
    "hvor",
    "jeg",
    "du",
    "vi",
    "man",
    "tek17",
}
# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_METHOD = os.getenv("TEK17_RETRIEVAL_METHOD", "dense").strip().lower()
HYBRID_ALPHA = _get_float_env("TEK17_HYBRID_ALPHA", 0.5)
TOP_K = _get_int_env("TEK17_TOP_K", 6)
BM25_K1 = _get_float_env("TEK17_BM25_K1", 1.5)
BM25_B = _get_float_env("TEK17_BM25_B", 0.75)
HYBRID_CANDIDATE_MULTIPLIER = _get_int_env("TEK17_HYBRID_CANDIDATE_MULTIPLIER", 3)

# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------

BENCHMARK_OUT_DIR = ANALYSIS_DIR / "logging"
BENCHMARK_OPENAI_MODELS = os.getenv(
    "TEK17_BENCHMARK_OPENAI_MODELS",
    "gpt-4.1-mini,gpt-5.2",
)
BENCHMARK_COMPARE_SHOW = _get_int_env("TEK17_BENCHMARK_COMPARE_SHOW", 50)

# ---------------------------------------------------------------------------
# Sweep / orchestration defaults
# ---------------------------------------------------------------------------

SWEEP_REPEAT = _get_int_env("TEK17_SWEEP_REPEAT", 1)