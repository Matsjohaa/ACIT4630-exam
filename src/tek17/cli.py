"""TEK17 CLI – corpus extraction, chunking, and vector store ingestion."""

import typer
from pathlib import Path

from tek17.rag.config import (
    EMBED_MODEL,
    EMBED_PROVIDER,
    OLLAMA_BASE_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKS_PATH,
    JSONL_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION,
)

_DEFAULT_ROOT_PRINT_URL = (
    "https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17"
    "?subtype=root&print=true"
)

app = typer.Typer(
    help="TEK17 (DiBK) extraction, chunking, and RAG ingestion.",
    no_args_is_help=True,
)


@app.command("download-dibk")
def download_dibk(
    url: str = typer.Option(_DEFAULT_ROOT_PRINT_URL, "--url", help="DiBK TEK17 root-print URL."),
    out_dir: Path = typer.Option(Path("data/raw/dibk_root_print"), "--out-dir"),
    manifest: Path = typer.Option(Path("data/raw/dibk_root_print_manifest.jsonl"), "--manifest"),
    force: bool = typer.Option(False, "--force", help="Re-download even if already in manifest."),
) -> None:
    """Download a TEK17 snapshot from DiBK."""
    from .corpus.download import run_download_root_print
    run_download_root_print(url=url, out_dir=out_dir, manifest_path=manifest, force=force)


@app.command("parse-dibk")
def parse_dibk(
    manifest: Path = typer.Option(Path("data/raw/dibk_root_print_manifest.jsonl"), "--manifest"),
    out: Path = typer.Option(JSONL_PATH, "--out"),
) -> None:
    """Parse the TEK17 root-print snapshot into per-provision JSONL."""
    from .corpus.parse import run_parse_root_print
    run_parse_root_print(manifest_path=manifest, out_path=out)


@app.command("chunk")
def chunk(
    jsonl: Path = typer.Option(JSONL_PATH, "--jsonl"),
    out: Path = typer.Option(CHUNKS_PATH, "--out"),
    chunk_size: int = typer.Option(CHUNK_SIZE, "--chunk-size"),
    chunk_overlap: int = typer.Option(CHUNK_OVERLAP, "--chunk-overlap"),
) -> None:
    """Build TEK17 chunks from the parsed JSONL corpus."""
    from .corpus.chunks import build_and_save_chunks
    build_and_save_chunks(jsonl_path=jsonl, chunks_path=out, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@app.command("ingest")
def ingest(
    chunks: Path = typer.Option(CHUNKS_PATH, "--chunks"),
    chroma_dir: Path = typer.Option(CHROMA_DIR, "--chroma-dir"),
    collection: str = typer.Option(CHROMA_COLLECTION, "--collection"),
    embed_provider: str = typer.Option(EMBED_PROVIDER, "--embed-provider"),
    embed_model: str = typer.Option(EMBED_MODEL, "--embed-model"),
    base_url: str = typer.Option(OLLAMA_BASE_URL, "--base-url"),
) -> None:
    """Embed TEK17 chunks and ingest them into ChromaDB."""
    from .rag.ingest import run_ingest
    run_ingest(
        chunks_path=chunks,
        chroma_dir=chroma_dir,
        collection_name=collection,
        embed_provider=embed_provider,
        embed_model=embed_model,
        base_url=base_url,
    )