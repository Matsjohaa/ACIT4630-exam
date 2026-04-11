from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup, Tag

SECTION_ID_RE = re.compile(r"§\s*(\d{1,2}-\d{1,2}[a-zA-Z]?)")
SECTION_TITLE_PREFIX_RE = re.compile(r"^§\s*\d{1,2}-\d{1,2}[a-zA-Z]?\s*[\.\-–:]?\s*")


@dataclass(frozen=True)
class ManifestRow:
    """One downloaded TEK17 root-print snapshot recorded in the manifest."""

    url: str
    final_url: str
    status: int
    downloaded_at: str
    path: str
    sha256: str
    content_type: str | None


def _repo_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    """Resolve a path relative to the repo root unless already absolute."""
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_snapshot_path(path_str: str, repo_root: Path) -> Path:
    """
    Resolve a snapshot path stored in the manifest.

    Supported formats:
    - repo-relative paths
    - absolute paths on the current machine
    - absolute paths from another machine, if the suffix from /data/... exists
      inside the current repository
    """
    path = Path(path_str)

    if not path.is_absolute():
        return (repo_root / path).resolve()

    if path.exists():
        return path

    parts = path.parts
    if "data" in parts:
        data_index = parts.index("data")
        relative_from_data = Path(*parts[data_index:])
        return (repo_root / relative_from_data).resolve()

    return path


def canonicalize_url(url: str) -> str:
    """Remove query parameters and fragments for deterministic URL comparison."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def iter_manifest_rows(manifest_path: Path):
    """Yield valid manifest rows from JSONL. Malformed lines are ignored."""
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            yield ManifestRow(
                url=row.get("url", ""),
                final_url=row.get("final_url", row.get("url", "")),
                status=int(row.get("status", 0) or 0),
                downloaded_at=row.get("downloaded_at", ""),
                path=row.get("path", ""),
                sha256=row.get("sha256", ""),
                content_type=row.get("content_type"),
            )


def select_latest_valid_snapshot(manifest_path: Path) -> ManifestRow:
    """Return the latest valid HTML snapshot recorded in the manifest."""
    chosen: ManifestRow | None = None

    for row in iter_manifest_rows(manifest_path):
        if (
            row.status == 200
            and row.path
            and row.content_type
            and "text/html" in row.content_type.lower()
        ):
            chosen = row

    if chosen is None:
        raise RuntimeError("No valid HTML snapshot found in manifest.")

    return chosen


def clean_text(text: str) -> str:
    """
    Normalize extracted text by:
    - stripping whitespace
    - removing empty lines
    - collapsing consecutive duplicate lines
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    cleaned_lines: list[str] = []
    previous_line: str | None = None

    for line in lines:
        if line != previous_line:
            cleaned_lines.append(line)
        previous_line = line

    return "\n".join(cleaned_lines)


def parse_section_heading(heading_text: str) -> tuple[str, str]:
    """
    Parse a TEK17 provision heading.

    Example:
        "§ 1-2. Forskriftens anvendelse på særskilte tiltak"
        -> ("§ 1-2", "Forskriftens anvendelse på særskilte tiltak")
    """
    normalized = heading_text.strip()
    match = SECTION_ID_RE.search(normalized)
    if match is None:
        raise ValueError(f"Could not parse section id from heading: {heading_text}")

    section_id = f"§ {match.group(1)}"
    title = SECTION_TITLE_PREFIX_RE.sub("", normalized).strip()

    return section_id, title


def extract_guidance_text(section_node: Tag) -> str:
    """Extract veiledning text from one provision node."""
    guidance_parts: list[str] = []

    for item in section_node.select("ul.guidance-text li.accordion-navigation"):
        label = item.select_one(".accordion-title")
        label_text = label.get_text(" ", strip=True) if label else "Veiledning"

        body = item.select_one("div.content")
        body_text = clean_text(body.get_text("\n", strip=True)) if body else ""

        if body_text:
            guidance_parts.append(f"{label_text}\n{body_text}")

    return "\n\n".join(guidance_parts).strip()


def extract_regulation_text(section_node: Tag) -> str:
    """Extract regulation text from one provision node, excluding guidance blocks."""
    temp_tree = BeautifulSoup(str(section_node), "lxml")
    for guidance_block in temp_tree.select("ul.guidance-text"):
        guidance_block.decompose()

    return clean_text(temp_tree.get_text("\n", strip=True))


def split_reg_and_guidance(section_node: Tag) -> tuple[str, str]:
    """Split one provision into regulation text and guidance text."""
    reg_text = extract_regulation_text(section_node)
    guidance_text = extract_guidance_text(section_node)
    return reg_text, guidance_text


def build_full_text(reg_text: str, guidance_text: str) -> str:
    """Join regulation and guidance text into one convenience field."""
    parts = []
    if reg_text:
        parts.append(reg_text)
    if guidance_text:
        parts.append(guidance_text)
    return "\n\n".join(parts).strip()


def find_section_body(heading: Tag) -> Tag | None:
    """
    Find the provision body associated with one section heading.

    In the DiBK print layout, the body is usually the next sibling
    <section class="section-big ...">.
    """
    sibling = heading.find_next_sibling()

    while sibling is not None:
        sibling_name = getattr(sibling, "name", None)
        sibling_classes = sibling.get("class") or []

        if sibling_name == "section" and "section-big" in sibling_classes:
            return sibling

        if sibling_name == "h2" and "big-header" in sibling_classes:
            return None

        sibling = sibling.find_next_sibling()

    return None


def iter_provision_records(doc: Tag, snapshot: ManifestRow, html_path: Path):
    """Yield one JSON-serializable record per TEK17 provision."""
    current_chapter: str | None = None
    canonical_url = canonicalize_url(snapshot.final_url or snapshot.url)

    for chapter_block in doc.select("div.print-chapter-content"):
        block_header = chapter_block.select_one("div.block-header")
        if block_header:
            chapter_text = block_header.get_text(" ", strip=True)
            if chapter_text:
                current_chapter = chapter_text

        for heading in chapter_block.select("h2.big-header"):
            heading_text = heading.get_text(" ", strip=True)
            if not heading_text.startswith("§"):
                continue

            try:
                section_id, title = parse_section_heading(heading_text)
            except ValueError:
                continue

            section_node = find_section_body(heading)
            if section_node is None:
                continue

            reg_text, guidance_text = split_reg_and_guidance(section_node)

            yield {
                "source": "dibk",
                "doc_type": "provision",
                "root_print_url": snapshot.url,
                "final_url": snapshot.final_url,
                "canonical_root_print_url": canonical_url,
                "downloaded_at": snapshot.downloaded_at,
                "source_path": str(html_path),
                "sha256": snapshot.sha256,
                "chapter": current_chapter,
                "section_id": section_id,
                "title": title,
                "reg_text": reg_text,
                "guidance_text": guidance_text,
                "full_text": build_full_text(reg_text, guidance_text),
            }


def run_parse_root_print(manifest_path: Path, out_path: Path) -> None:
    """Parse the latest TEK17 root-print snapshot into per-provision JSONL."""
    repo_root = _repo_root()
    resolved_manifest_path = _resolve_repo_path(manifest_path, repo_root)
    resolved_out_path = _resolve_repo_path(out_path, repo_root)

    snapshot = select_latest_valid_snapshot(resolved_manifest_path)
    html_path = _resolve_snapshot_path(snapshot.path, repo_root)

    if not html_path.exists():
        raise FileNotFoundError(f"Snapshot missing: {html_path}")

    soup = BeautifulSoup(html_path.read_bytes(), "lxml")
    document_root = soup.select_one(".print-document-content") or soup.select_one("main") or soup

    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with resolved_out_path.open("w", encoding="utf-8") as output_file:
        for record in iter_provision_records(document_root, snapshot, html_path):
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} provision records to {resolved_out_path}")
    print(f"Source snapshot: {html_path}")