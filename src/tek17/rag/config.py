"""Central configuration for the TEK17 RAG pipeline.

Resolution order (highest priority first):
  1. Environment variables  (TEK17_LLM_MODEL, OPENAI_API_KEY, etc.)
  2. tek17.conf             (INI file in the project root)
  3. Built-in defaults      (defined in this module)
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path


def _find_conf() -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    candidates = [
        Path(os.getenv("TEK17_CONF", "")).expanduser() if os.getenv("TEK17_CONF") else None,
        Path.cwd() / "tek17.conf",
        Path(__file__).resolve().parents[3] / "tek17.conf",
    ]
    for p in candidates:
        if p and p.is_file():
            cp.read(p, encoding="utf-8")
            break
    return cp

_CONF = _find_conf()

def _conf(section: str, key: str, fallback: str = "") -> str:
    return _CONF.get(section, key, fallback=fallback).strip()

def _env(name: str, fallback: str = "") -> str:
    return os.getenv(name, fallback).strip()

def _env_or_conf(env_name: str, section: str, key: str, default: str = "") -> str:
    val = os.getenv(env_name, "").strip()
    if val:
        return val
    return _conf(section, key, default)

def _int(val: str, default: int) -> int:
    try: return int(val)
    except (ValueError, TypeError): return default

def _float(val: str, default: float) -> float:
    try: return float(val)
    except (ValueError, TypeError): return default

def _optional_int(val: str) -> int | None:
    try: return int(val) if val else None
    except (ValueError, TypeError): return None

# --- paths ------------------------------------------------------------------

BASE_DIR = Path(_env("TEK17_BASE_DIR", ".")).resolve()
DATA_DIR = BASE_DIR / _env("TEK17_DATA_SUBDIR", "data")
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
CHROMA_DIR = VECTORSTORE_DIR / "chroma"

ANALYSIS_DIR = BASE_DIR / _env("TEK17_ANALYSIS_SUBDIR", "analysis")
LOG_DIR = ANALYSIS_DIR / "logging"
QUERY_LOG_PATH = LOG_DIR / "rag_queries.jsonl"

JSONL_PATH = PROCESSED_DIR / _env("TEK17_JSONL_FILENAME", "tek17_dibk.jsonl")
CHUNKS_PATH = PROCESSED_DIR / _env("TEK17_CHUNKS_FILENAME", "tek17_chunks.jsonl")

# --- vector store -----------------------------------------------------------

CHROMA_COLLECTION = _env_or_conf("TEK17_CHROMA_COLLECTION", "retrieval", "collection", "tek17")

# --- embeddings -------------------------------------------------------------

EMBED_PROVIDER = _env_or_conf("TEK17_EMBED_PROVIDER", "embeddings", "provider", "ollama").lower()
EMBED_MODEL = _env_or_conf("TEK17_EMBED_MODEL", "embeddings", "model", "nomic-embed-text")
OLLAMA_BASE_URL = _env_or_conf("TEK17_OLLAMA_BASE_URL", "embeddings", "base_url", "http://localhost:11434")
EMBED_BASE_URL = _env_or_conf("TEK17_EMBED_BASE_URL", "embeddings", "base_url", OLLAMA_BASE_URL)

OPENAI_API_KEY = _env("OPENAI_API_KEY") or _env("OPEN_AI_API_KEY") or _conf("openai", "api_key") or None
OPENAI_BASE_URL = _env("OPENAI_BASE_URL") or _conf("openai", "base_url") or None

# --- LLM --------------------------------------------------------------------

PROMPT_VERSION = _env_or_conf("TEK17_PROMPT_VERSION", "prompt", "version", "baseline").lower()
PROMPT_VERSIONS = {"baseline", "relaxed", "strict"}
LLM_BASE_URL = _env_or_conf("TEK17_LLM_BASE_URL", "llm", "base_url", OLLAMA_BASE_URL)
LLM_PROVIDER = _env_or_conf("TEK17_LLM_PROVIDER", "llm", "provider", "ollama").lower()
LLM_MODEL = _env_or_conf("TEK17_LLM_MODEL", "llm", "model", "llama3.2")
LLM_MAX_TOKENS = _optional_int(_env_or_conf("TEK17_LLM_MAX_TOKENS", "llm", "max_tokens", ""))
LLM_TEMPERATURE = _float(_env_or_conf("TEK17_LLM_TEMPERATURE", "llm", "temperature", "0.3"), 0.3)

# --- chunking ---------------------------------------------------------------

CHUNK_SIZE = _int(_env_or_conf("TEK17_CHUNK_SIZE", "chunking", "chunk_size", "800"), 800)
CHUNK_OVERLAP = _int(_env_or_conf("TEK17_CHUNK_OVERLAP", "chunking", "chunk_overlap", "200"), 200)

# --- evaluation -------------------------------------------------------------

SERVER_URL = _env("TEK17_SERVER_URL", "http://localhost:8000")
TEST_MODE = _env_or_conf("TEK17_TEST_MODE", "evaluation", "mode", "local").lower()
EVAL_MODE = _env_or_conf("TEK17_EVAL_MODE", "evaluation", "mode", "local")
REQUEST_TIMEOUT_S = _int(_env_or_conf("TEK17_REQUEST_TIMEOUT_S", "evaluation", "request_timeout_s", "300"), 300)
COLLECTION_STATS_TIMEOUT_S = _int(_env("TEK17_COLLECTION_STATS_TIMEOUT_S", "10"), 10)
GROUNDEDNESS_THRESHOLD = _float(
    _env_or_conf("TEK17_GROUNDEDNESS_THRESHOLD", "evaluation", "groundedness_threshold", "0.25"), 0.25
)
SOURCE_PREVIEW_LIMIT = _int(_env("TEK17_SOURCE_PREVIEW_LIMIT", "6"), 6)

# --- refusal detection ------------------------------------------------------

REFUSAL_TAG = _env("TEK17_REFUSAL_TAG", "KAN_IKKE_SVARE")

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

CONTENT_WORD_MIN_LEN = _int(_env("TEK17_CONTENT_WORD_MIN_LEN", "4"), 4)
WILSON_Z = _float(_env("TEK17_WILSON_Z", "1.96"), 1.96)

CONTENT_STOPWORDS = {
    "og", "eller", "som", "med", "for", "til", "av", "på", "i", "jf",
    "kapittel", "paragraf", "ledd", "bokstav", "gjelder", "skal", "kan",
    "må", "ikke", "det", "den", "de", "et", "en", "er", "å", "når",
    "hva", "hvordan", "hvilke", "hvilken", "hvor", "jeg", "du", "vi",
    "man", "tek17",
}

# --- retrieval --------------------------------------------------------------

RETRIEVAL_METHOD = _env_or_conf("TEK17_RETRIEVAL_METHOD", "retrieval", "method", "dense").lower()
HYBRID_ALPHA = _float(_env_or_conf("TEK17_HYBRID_ALPHA", "retrieval", "hybrid_alpha", "0.5"), 0.5)
TOP_K = _int(_env_or_conf("TEK17_TOP_K", "retrieval", "top_k", "6"), 6)
BM25_K1 = _float(_env_or_conf("TEK17_BM25_K1", "retrieval", "bm25_k1", "1.5"), 1.5)
BM25_B = _float(_env_or_conf("TEK17_BM25_B", "retrieval", "bm25_b", "0.75"), 0.75)
HYBRID_CANDIDATE_MULTIPLIER = _int(
    _env_or_conf("TEK17_HYBRID_CANDIDATE_MULTIPLIER", "retrieval", "hybrid_candidate_multiplier", "3"), 3
)

# --- benchmark / sweep ------------------------------------------------------

BENCHMARK_OUT_DIR = ANALYSIS_DIR / "logging"
BENCHMARK_OPENAI_MODELS = _env_or_conf(
    "TEK17_BENCHMARK_OPENAI_MODELS", "benchmark", "openai_models", "gpt-4.1-mini,gpt-5.2"
)
BENCHMARK_COMPARE_SHOW = _int(_env("TEK17_BENCHMARK_COMPARE_SHOW", "50"), 50)
SWEEP_REPEAT = _int(_env_or_conf("TEK17_SWEEP_REPEAT", "evaluation", "sweep_repeat", "1"), 1)

# --- taxonomy categories ----------------------------------------------------

CONDITIONAL_REFUSAL_CATEGORY_NAMES = [
    "retrieval_miss_correct_refusal",
    "over_refusal_with_partial_evidence",
    "over_refusal_with_full_evidence",
    "answer_without_evidence",
    "partial_support_answer",
    "correct_answer",
    "correct_qualified_answer",
    "missing_qualification_warning",
    "correct_refusal_with_partial_context",
    "correct_refusal_with_full_context",
    "under_refusal_with_partial_context",
    "under_refusal_with_full_context",
    "unsafe_answer_no_evidence",
    "other",
]

# --- qualification patterns -------------------------------------------------

QUALIFICATION_PATTERNS = [
    "trenger mer informasjon",
    "trenger man mer informasjon",
    "trenger jeg mer informasjon",
    "trenger vi mer informasjon",
    "uten mer informasjon",
    "mangler informasjon",
    "mangler opplysninger",
    "krever tilleggskontekst",
    "krever prosjektspesifikke opplysninger",
    "krever mer informasjon",
    "krever flere opplysninger",
    "avhenger av",
    "vil variere",
    "kan variere",
    "kan være tilpasset",
    "må vurderes konkret",
    "må vurderes ut fra",
    "for en fullstendig vurdering",
    "for en endelig vurdering",
    "prosjektets spesifikke forhold",
    "byggverkets spesifikke forhold",
    "rom eller bygning",
]

QUALIFICATION_TOP_K = 10