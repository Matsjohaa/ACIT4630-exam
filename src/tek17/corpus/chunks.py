from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_JSONL_PATH = Path("data/processed/tek17_dibk.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/tek17_chunks.jsonl")

from tek17.rag.config import CHUNK_SIZE, CHUNK_OVERLAP

HEADER_BODY_SEPARATOR = "\n\n"
FALLBACK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

LEDD_START_RE = re.compile(r"^\(\d+\)\s+")
LETTER_ITEM_RE = re.compile(r"^[a-zæøå]\)\s+", re.IGNORECASE)
NUMBER_ITEM_RE = re.compile(r"^\d+\)\s+")
BULLET_RE = re.compile(r"^[\-–•]\s+")


def load_jsonl_records(path: Path) -> list[dict]:
    """Load JSONL records from disk."""
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def build_documents(records: list[dict]) -> list[tuple[str, dict]]:
    """
    Build chunkable documents from parsed TEK17 provision records.

    Each provision may produce:
    - one regulation document
    - one guidance document
    """
    documents: list[tuple[str, dict]] = []

    for record in records:
        section_id = record.get("section_id") or "unknown"
        title = record.get("title") or ""
        chapter = record.get("chapter") or ""

        base_metadata = {
            "source": "dibk",
            "section_id": section_id,
            "title": title,
            "chapter": chapter,
        }

        regulation_text = (record.get("reg_text") or "").strip()
        if regulation_text:
            metadata = {**base_metadata, "text_type": "regulation"}
            header = f"{section_id} – {title}\n(Forskriftstekst)\n\n"
            documents.append((header + regulation_text, metadata))

        guidance_text = (record.get("guidance_text") or "").strip()
        if guidance_text:
            metadata = {**base_metadata, "text_type": "guidance"}
            header = f"{section_id} – {title}\n(Veiledning)\n\n"
            documents.append((header + guidance_text, metadata))

    return documents


def is_structural_boundary(line: str) -> bool:
    """Return true if a line looks like the start of a new regulatory unit."""
    return bool(
        LEDD_START_RE.match(line)
        or LETTER_ITEM_RE.match(line)
        or NUMBER_ITEM_RE.match(line)
        or BULLET_RE.match(line)
    )


def split_into_structural_units(text: str) -> list[str]:
    """
    Split TEK17 text into structure-aware units using simple formatting cues.

    The goal is to preserve regulation-like boundaries such as:
    - ledd: (1), (2), ...
    - lettered lists: a), b), ...
    - numbered lists: 1), 2), ...
    - bullets: -, –, •
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return []

    units: list[str] = []
    current_unit: list[str] = []

    for line in lines:
        if current_unit and is_structural_boundary(line):
            units.append("\n".join(current_unit).strip())
            current_unit = [line]
        else:
            current_unit.append(line)

    if current_unit:
        units.append("\n".join(current_unit).strip())

    # If nearly every line became its own unit, the heuristic is too aggressive.
    if len(lines) >= 8 and len(units) >= len(lines) - 1:
        return ["\n".join(lines).strip()]

    return units


def build_fallback_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create a generic character-based fallback splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=FALLBACK_SEPARATORS,
    )


def split_large_unit(
        *,
        header: str,
        unit: str,
        unit_index: int,
        chunk_size: int,
        chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """Split one oversized structural unit using character-based fallback chunking."""
    max_body_size = max(1, chunk_size - len(header))
    adjusted_overlap = min(chunk_overlap, max(0, max_body_size - 1))

    splitter = build_fallback_splitter(
        chunk_size=max_body_size,
        chunk_overlap=adjusted_overlap,
    )

    return [
        (header + part, unit_index, unit_index)
        for part in splitter.split_text(unit)
    ]


def pack_units_into_chunks(
        *,
        header: str,
        units: list[str],
        chunk_size: int,
        chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """
    Pack structural units into chunks.

    Returns:
        list of (chunk_text, unit_start_index, unit_end_index)
    where indices are 1-based and inclusive.
    """
    if not units:
        return []

    chunks: list[tuple[str, int, int]] = []
    separator = HEADER_BODY_SEPARATOR
    header_length = len(header)
    start_index = 0

    while start_index < len(units):
        max_body_size = max(1, chunk_size - header_length)

        if len(units[start_index]) > max_body_size:
            chunks.extend(
                split_large_unit(
                    header=header,
                    unit=units[start_index],
                    unit_index=start_index + 1,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
            start_index += 1
            continue

        end_index = start_index
        current_units: list[str] = []
        current_length = 0

        while end_index < len(units):
            unit = units[end_index]
            added_length = len(unit) + (len(separator) if current_units else 0)

            if header_length + current_length + added_length <= chunk_size or not current_units:
                current_units.append(unit)
                current_length += added_length
                end_index += 1
            else:
                break

        chunk_text = header + separator.join(current_units)
        chunks.append((chunk_text, start_index + 1, end_index))

        if end_index >= len(units):
            break

        if chunk_overlap <= 0:
            start_index = end_index
            continue

        overlap_chars = 0
        next_start_index = end_index

        while next_start_index > start_index:
            next_start_index -= 1
            overlap_chars += len(units[next_start_index]) + len(separator)
            if overlap_chars >= chunk_overlap:
                break

        if next_start_index <= start_index:
            next_start_index = max(start_index + 1, end_index - 1)

        start_index = next_start_index

    return chunks


def split_header_and_body(text: str) -> tuple[str, str]:
    """
    Split a document into header and body.

    Expected document format:
        § x-y – Title
        (Forskriftstekst|Veiledning)

        <body>
    """
    if HEADER_BODY_SEPARATOR not in text:
        return "", text

    header, body = text.split(HEADER_BODY_SEPARATOR, 1)
    return header + HEADER_BODY_SEPARATOR, body


def write_chunk_record(file, text: str, metadata: dict) -> None:
    """Write one chunk record as JSONL."""
    record = {
        "text": text,
        "metadata": metadata,
    }
    file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_and_save_chunks(
        jsonl_path: Path = DEFAULT_JSONL_PATH,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    """Read parsed TEK17 records and write chunked JSONL output."""
    resolved_jsonl_path = jsonl_path.resolve()
    resolved_chunks_path = chunks_path.resolve()

    if not resolved_jsonl_path.exists():
        raise FileNotFoundError(f"JSONL corpus not found: {resolved_jsonl_path}")

    records = load_jsonl_records(resolved_jsonl_path)
    documents = build_documents(records)

    resolved_chunks_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with resolved_chunks_path.open("w", encoding="utf-8") as file:
        for document_text, metadata in documents:
            header, body = split_header_and_body(document_text)
            units = split_into_structural_units(body)

            packed_chunks = pack_units_into_chunks(
                header=header,
                units=units,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if not packed_chunks and body.strip():
                fallback_splitter = build_fallback_splitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for chunk_text in fallback_splitter.split_text(document_text):
                    write_chunk_record(file, chunk_text, metadata)
                    written += 1
                continue

            for chunk_text, unit_start, unit_end in packed_chunks:
                chunk_metadata = dict(metadata)
                chunk_metadata["para_start"] = unit_start
                chunk_metadata["para_end"] = unit_end

                write_chunk_record(file, chunk_text, chunk_metadata)
                written += 1

    print(f"Wrote {written} chunks to {resolved_chunks_path}")