import typer
from pathlib import Path

# Root-print based pipeline:
# Instead of crawling each § page individually,
# we download one authoritative "root print" snapshot of TEK17 from DiBK.
# That snapshot contains the entire regulation + guidance in a stable layout.
#
# Parsing is then performed deterministically from that single document,
# splitting into one record per § provision.

from .download_dibk import DEFAULT_ROOT_PRINT_URL, run_download_root_print
from .parse_dibk import run_parse_root_print

app = typer.Typer(
    help="TEK17 (DiBK) extraction and processing pipeline.",
    no_args_is_help=True,
)

@app.command()
def hello() -> None:
    """Sanity check that the CLI wiring works."""
    typer.echo("CLI is alive.")

@app.command("download-dibk")
def download_dibk(
    url: str = typer.Option(
        DEFAULT_ROOT_PRINT_URL,
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
        Path("data/processed/tek17_dibk.jsonl"),
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
    run_parse_root_print(
        manifest_path=manifest,
        out_path=out,
    )