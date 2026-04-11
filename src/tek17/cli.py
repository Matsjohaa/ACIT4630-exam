import typer
from pathlib import Path

from tek17.rag.config import EMBED_MODEL as _CONFIG_EMBED_MODEL
from tek17.rag.config import EMBED_PROVIDER as _CONFIG_EMBED_PROVIDER
from tek17.rag.config import OLLAMA_BASE_URL as _CONFIG_OLLAMA_BASE_URL
from tek17.rag.config import CHUNK_SIZE as _CONFIG_CHUNK_SIZE
from tek17.rag.config import CHUNK_OVERLAP as _CONFIG_CHUNK_OVERLAP
from tek17.rag.config import CHUNKS_PATH as _CONFIG_CHUNKS_PATH
from tek17.rag.config import JSONL_PATH as _CONFIG_JSONL_PATH
from tek17.rag.config import CHROMA_DIR as _CONFIG_CHROMA_DIR
from tek17.rag.config import CHROMA_COLLECTION as _CONFIG_CHROMA_COLLECTION

# ---------------------------------------------------------------------------
# Lazy-import constants for CLI defaults.
# Heavy dependencies (chromadb, etc.) are imported inside commands to keep
# CLI startup fast and avoid compatibility issues on newer Python versions.
# ---------------------------------------------------------------------------
_DEFAULT_ROOT_PRINT_URL = (
    "https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17"
    "?subtype=root&print=true"
)

_DEFAULT_JSONL_PATH = _CONFIG_JSONL_PATH
_DEFAULT_CHUNKS_PATH = _CONFIG_CHUNKS_PATH
_DEFAULT_CHROMA_DIR = _CONFIG_CHROMA_DIR
_COLLECTION_NAME = _CONFIG_CHROMA_COLLECTION
_EMBED_MODEL = _CONFIG_EMBED_MODEL
_EMBED_PROVIDER = _CONFIG_EMBED_PROVIDER
_BASE_URL = _CONFIG_OLLAMA_BASE_URL
_CHUNK_SIZE = _CONFIG_CHUNK_SIZE
_CHUNK_OVERLAP = _CONFIG_CHUNK_OVERLAP

app = typer.Typer(
    help="TEK17 (DiBK) extraction, RAG and chat pipeline.",
    no_args_is_help=True,
)


# ── Corpus commands ────────────────────────────────────────────────────────


@app.command()
def hello() -> None:
    """Sanity check that the CLI wiring works."""
    typer.echo("CLI is alive.")


@app.command("download-dibk")
def download_dibk(
    url: str = typer.Option(
        _DEFAULT_ROOT_PRINT_URL,
        "--url",
        help="DiBK TEK17 root-print URL (full TEK17 with guidance).",
    ),
    out_dir: Path = typer.Option(
        Path("data/raw/dibk_root_print"),
        "--out-dir",
        help="Folder to store the downloaded root-print HTML snapshot.",
    ),
    manifest: Path = typer.Option(
        Path("data/raw/dibk_root_print_manifest.jsonl"),
        "--manifest",
        help="Manifest JSONL for root-print snapshots.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if URL is already present in manifest.",
    ),
) -> None:
    """
    Download a single authoritative TEK17 snapshot from DiBK.
    The downloaded snapshot is recorded in a manifest for reproducibility.
    """
    from .corpus.download import run_download_root_print

    run_download_root_print(
        url=url,
        out_dir=out_dir,
        manifest_path=manifest,
        force=force,
    )


@app.command("parse-dibk")
def parse_dibk(
    manifest: Path = typer.Option(
        Path("data/raw/dibk_root_print_manifest.jsonl"),
        "--manifest",
        help="Manifest JSONL produced by download-dibk.",
    ),
    out: Path = typer.Option(
        _DEFAULT_JSONL_PATH,
        "--out",
        help="Output JSONL: one record per provision (§ x-y).",
    ),
) -> None:
    """
    Parse the root-print snapshot into per-provision records.
    The parser:
    - identifies each § provision using structural headings
    - separates regulation text and guidance text
    - produces a deterministic JSONL dataset
    - guarantees one canonical record per §
    This output is used for downstream chunking and retrieval experiments.
    """
    from .corpus.parse import run_parse_root_print

    run_parse_root_print(
        manifest_path=manifest,
        out_path=out,
    )


@app.command("chunk")
def chunk(
    jsonl: Path = typer.Option(
        _DEFAULT_JSONL_PATH,
        "--jsonl",
        help="Path to parsed TEK17 JSONL corpus.",
    ),
    out: Path = typer.Option(
        _DEFAULT_CHUNKS_PATH,
        "--out",
        help="Output JSONL with chunked TEK17 records.",
    ),
    chunk_size: int = typer.Option(
        _CHUNK_SIZE,
        "--chunk-size",
        help="Max characters per chunk.",
    ),
    chunk_overlap: int = typer.Option(
        _CHUNK_OVERLAP,
        "--chunk-overlap",
        help="Overlap between chunks.",
    ),
) -> None:
    """
    Build TEK17 chunks from the parsed JSONL corpus.
    """
    from .corpus.chunks import build_and_save_chunks

    build_and_save_chunks(
        jsonl_path=jsonl,
        chunks_path=out,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# ── RAG commands ───────────────────────────────────────────────────────────


@app.command("ingest")
def ingest(
    chunks: Path = typer.Option(
        _DEFAULT_CHUNKS_PATH,
        "--chunks",
        help="Path to chunked TEK17 JSONL corpus.",
    ),
    chroma_dir: Path = typer.Option(
        _DEFAULT_CHROMA_DIR,
        "--chroma-dir",
        help="Directory for persistent ChromaDB vector store.",
    ),
    collection: str = typer.Option(
        _COLLECTION_NAME,
        "--collection",
        help="ChromaDB collection name.",
    ),
    embed_provider: str = typer.Option(
        _EMBED_PROVIDER,
        "--embed-provider",
        help="Embedding provider: 'ollama' (default) or 'openai'.",
    ),
    embed_model: str = typer.Option(
        _EMBED_MODEL,
        "--embed-model",
        help="Embedding model name (Ollama or OpenAI).",
    ),
    base_url: str = typer.Option(
        _BASE_URL,
        "--base-url",
        help="Base URL for the embedding provider when applicable.",
    ),
) -> None:
    """
    Embed TEK17 chunks and ingest them into ChromaDB.

    This command:
    - reads precomputed chunks from JSONL
    - generates embeddings
    - stores them in a persistent ChromaDB collection

    Prerequisites:
    - Chunked corpus exists (run `tek17 chunk` first)
    - Embedding backend available (Ollama running or OpenAI key set)
    """
    from .rag.ingest import run_ingest

    run_ingest(
        chunks_path=chunks,
        chroma_dir=chroma_dir,
        collection_name=collection,
        embed_provider=embed_provider,
        embed_model=embed_model,
        base_url=base_url,
    )


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """
    Start the TEK17 RAG FastAPI server.

    The server exposes:
    - POST /query  – ask a question, get RAG-augmented answer
    - GET  /health – health check
    - GET  /models – list available Ollama models
    - GET  /collection/stats – vector store stats

    Prerequisites:
    - Ollama running with LLM + embedding model pulled
    - Vector store populated (run `ingest` first)
    """
    import uvicorn

    uvicorn.run(
        "tek17.rag.server:app",
        host=host,
        port=port,
        reload=reload,
    )


# ── UI command ─────────────────────────────────────────────────────────────


@app.command("ui")
def ui(
    port: int = typer.Option(8501, "--port", help="Streamlit port."),
) -> None:
    """
    Launch the Streamlit chat UI.

    Prerequisites:
    - RAG server running (run `serve` first)
    """
    import subprocess
    import sys
    from importlib.resources import files

    ui_path = files("tek17.app").joinpath("ui.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            "--server.port",
            str(port),
        ],
    )