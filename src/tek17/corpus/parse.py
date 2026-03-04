from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup

# Captures provision identifiers in headings, e.g. "§ 12-6. Kommunikasjonsvei"
SECTION_ID_RE = re.compile(r"§\s*(\d{1,2}-\d{1,2}[a-zA-Z]?)")


def canonicalize_url(u: str) -> str:
    """
    Remove query params and fragments so URLs can be compared deterministically.
    """
    p = urlparse(u)
    return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))


@dataclass(frozen=True)
class ManifestRow:
    """
    One line from the root-print download manifest.
    We select the latest valid HTML snapshot (manifest is append-only).
    """
    url: str
    final_url: str
    status: int
    downloaded_at: str
    path: str
    sha256: str
    content_type: Optional[str]


def iter_manifest(manifest_path: Path) -> Iterator[ManifestRow]:
    """
    Stream manifest rows from JSONL.
    We ignore malformed lines instead of failing the whole pipeline.
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield ManifestRow(
                url=r.get("url", ""),
                final_url=r.get("final_url", r.get("url", "")),
                status=int(r.get("status", 0) or 0),
                downloaded_at=r.get("downloaded_at", ""),
                path=r.get("path", ""),
                sha256=r.get("sha256", ""),
                content_type=r.get("content_type"),
            )


def clean_text(s: str) -> str:
    """
    Normalize extracted text:
    - strip whitespace
    - remove empty lines
    - remove consecutive duplicate lines
    """
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln]

    out = []
    prev = None
    for ln in lines:
        if ln != prev:
            out.append(ln)
        prev = ln

    return "\n".join(out)


def parse_section_heading(h_text: str) -> tuple[str, str]:
    """
    Extract section id and title from a provision heading.

    Input:  "§ 1-2. Forskriftens anvendelse på særskilte tiltak"
    Output: ("§ 1-2", "Forskriftens anvendelse på særskilte tiltak")

    Assumption: In the root-print HTML, each provision starts with a heading
    containing the § identifier.
    """
    txt = h_text.strip()
    m = SECTION_ID_RE.search(txt)
    if not m:
        raise ValueError(f"Could not parse section id from heading: {h_text}")

    section_id = f"§ {m.group(1)}"

    # Remove the "§ x-y." prefix from the heading to get a clean title
    title = re.sub(r"^§\s*\d{1,2}-\d{1,2}[a-zA-Z]?\s*[\.\-–:]?\s*", "", txt).strip()
    return section_id, title


def split_reg_and_guidance(section_node: BeautifulSoup) -> tuple[str, str]:
    """
    Split one provision's HTML block into:
    - reg_text: regulation text (ledd + lists + tables)
    - guidance_text: guidance blocks (veiledning accordion content)

    DOM assumptions based on DiBK root-print layout:
    - Guidance appears inside <ul class="guidance-text"> ... <div class="content">...</div>
    - Regulation text includes everything else in the section.

    Limitation:
    - If DiBK changes the accordion structure or class names,
      this heuristic may need updating.
    """
    guidance_parts: list[str] = []

    # Each accordion item typically has a label (.accordion-title) and a body (div.content)
    for li in section_node.select("ul.guidance-text li.accordion-navigation"):
        label = li.select_one(".accordion-title")
        label_txt = label.get_text(" ", strip=True) if label else "Veiledning"

        body = li.select_one("div.content")
        body_txt = clean_text(body.get_text("\n", strip=True)) if body else ""

        if body_txt:
            guidance_parts.append(f"{label_txt}\n{body_txt}")

    guidance_text = "\n\n".join(guidance_parts).strip()

    # Extract regulation text by removing guidance blocks and then reading remaining text.
    # BeautifulSoup does not provide a cheap deep-clone, so we parse a string copy.
    tmp = BeautifulSoup(str(section_node), "lxml")
    for ul in tmp.select("ul.guidance-text"):
        ul.decompose()
    reg_text = clean_text(tmp.get_text("\n", strip=True))

    return reg_text, guidance_text


def run_parse_root_print(manifest_path: Path, out_path: Path) -> None:
    """
    Parse the latest downloaded root-print HTML snapshot into per-§ JSONL.

    Output: one JSONL record per provision, including:
    - section_id, title, optional chapter context
    - reg_text and guidance_text (authoritative fields)
    - full_text = reg_text + guidance_text (derived convenience field)

    Downstream use:
    - chunking and retrieval experiments should preferably chunk reg/guidance separately
      to support ablation studies (reg-only vs reg+guidance retrieval).
    """
    # Select the most recent valid HTML snapshot (manifest is append-only; last match wins).
    chosen: Optional[ManifestRow] = None
    for row in iter_manifest(manifest_path):
        if row.status == 200 and row.path and row.content_type and "text/html" in row.content_type.lower():
            chosen = row

    if not chosen:
        raise RuntimeError("No valid HTML snapshot found in manifest.")

    html_path = Path(chosen.path)
    if not html_path.exists():
        raise FileNotFoundError(f"Snapshot missing: {html_path}")

    soup = BeautifulSoup(html_path.read_bytes(), "lxml")

    # Root-print pages contain the full content within `.print-document-content`.
    # Fallback to <main> if the selector changes.
    doc = soup.select_one(".print-document-content") or soup.select_one("main") or soup

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Track chapter headings for context. In root-print HTML, chapter labels often
    # appear as `div.block-header` within `div.print-chapter-content`.
    current_chapter: Optional[str] = None

    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        # Iterate chapter blocks in order; each may contain multiple provisions.
        for chapter_block in doc.select("div.print-chapter-content"):
            block_header = chapter_block.select_one("div.block-header")
            if block_header:
                header_txt = block_header.get_text(" ", strip=True)
                if header_txt:
                    current_chapter = header_txt

            # Provisions are marked by `h2.big-header` headings in the print layout.
            for h in chapter_block.select("h2.big-header"):
                h_txt = h.get_text(" ", strip=True)
                if not h_txt.startswith("§"):
                    continue

                try:
                    section_id, title = parse_section_heading(h_txt)
                except ValueError:
                    continue

                # In the print layout, the provision body is usually the next sibling
                # `<section class="section-big ...">`. We stop scanning if we hit the next provision.
                section_node = None
                nxt = h.find_next_sibling()
                while nxt is not None:
                    if getattr(nxt, "name", None) == "section" and "section-big" in (nxt.get("class") or []):
                        section_node = nxt
                        break
                    if getattr(nxt, "name", None) == "h2" and "big-header" in (nxt.get("class") or []):
                        break
                    nxt = nxt.find_next_sibling()

                if section_node is None:
                    continue

                reg_text, guidance_text = split_reg_and_guidance(section_node)

                # Derived convenience field; do not chunk separately if you already chunk reg/guidance.
                full_parts = []
                if reg_text:
                    full_parts.append(reg_text)
                if guidance_text:
                    full_parts.append(guidance_text)
                full_text = "\n\n".join(full_parts).strip()

                record = {
                    "source": "dibk",
                    "doc_type": "provision",
                    "root_print_url": chosen.url,
                    "final_url": chosen.final_url,
                    "canonical_root_print_url": canonicalize_url(chosen.final_url or chosen.url),
                    "downloaded_at": chosen.downloaded_at,
                    "source_path": str(html_path),
                    "sha256": chosen.sha256,
                    "chapter": current_chapter,
                    "section_id": section_id,
                    "title": title,
                    "reg_text": reg_text,
                    "guidance_text": guidance_text,
                    "full_text": full_text,
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} provision records to {out_path}")
    print(f"Source snapshot: {html_path}")
