from __future__ import annotations

from pathlib import Path
import json
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_JSONL_PATH = Path("data/processed/tek17_dibk.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/tek17_chunks.jsonl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


_LEDD_START_RE = re.compile(r"^\(\d+\)\s+")
_LETTER_ITEM_RE = re.compile(r"^[a-zæøå]\)\s+", re.IGNORECASE)
_NUM_ITEM_RE = re.compile(r"^\d+\)\s+")
_BULLET_RE = re.compile(r"^[\-–•]\s+")


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_documents(records: list[dict]) -> list[tuple[str, dict]]:
    """Turn provision records into (text, metadata) pairs ready for chunking.

    Each provision produces up to TWO documents:
    1. Regulation text   (tag: regulation)
    2. Guidance text     (tag: guidance)

    This separation lets downstream retrieval experiments ablate reg-only
    vs reg+guidance.
    """
    docs: list[tuple[str, dict]] = []
    for rec in records:
        section_id = rec.get("section_id", "unknown") or "unknown"
        title = rec.get("title", "") or ""
        chapter = rec.get("chapter", "") or ""

        base_meta = {
            "source": "dibk",
            "section_id": section_id,
            "title": title,
            "chapter": chapter,
        }

        reg = (rec.get("reg_text", "") or "").strip()
        if reg:
            meta = {**base_meta, "text_type": "regulation"}
            header = f"{section_id} – {title}\n(Forskriftstekst)\n\n"
            docs.append((header + reg, meta))

        guidance = (rec.get("guidance_text", "") or "").strip()
        if guidance:
            meta = {**base_meta, "text_type": "guidance"}
            header = f"{section_id} – {title}\n(Veiledning)\n\n"
            docs.append((header + guidance, meta))

    return docs


def _split_into_units(text: str) -> list[str]:
    """Split TEK17 text into paragraph/ledd-ish units.

    We work on the *already extracted* plain text and use simple heuristics
    based on common markers in Norwegian regulations:
    - (1), (2), ... (ledd)
    - a), b), ... (lettered list items)
    - 1), 2), ... (numbered list items)
    - -, – , • (bullets)

    If no markers are found, we fall back to a single unit.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    def is_new_unit(line: str) -> bool:
        return bool(
            _LEDD_START_RE.match(line)
            or _LETTER_ITEM_RE.match(line)
            or _NUM_ITEM_RE.match(line)
            or _BULLET_RE.match(line)
        )

    units: list[str] = []
    buf: list[str] = []
    for line in lines:
        if buf and is_new_unit(line):
            units.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)

    if buf:
        units.append("\n".join(buf).strip())

    # Avoid producing lots of tiny one-liners if the heuristic triggers too often.
    # If *almost every* line became its own unit, treat the whole text as one unit.
    if len(lines) >= 8 and len(units) >= len(lines) - 1:
        return ["\n".join(lines).strip()]

    return units


def _pack_units_into_chunks(
    *,
    header: str,
    units: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """Pack units into size-limited chunks with overlap at unit boundaries.

    Returns a list of (chunk_text, unit_start_index, unit_end_index) where
    indices are 1-based and inclusive.
    """
    if not units:
        return []

    sep = "\n\n"
    header_len = len(header)

    def units_len(slice_units: list[str]) -> int:
        if not slice_units:
            return 0
        return len(sep.join(slice_units))

    chunks: list[tuple[str, int, int]] = []
    start = 0

    # Fallback splitter for a single very large unit.
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(1, chunk_size - header_len),
        chunk_overlap=min(chunk_overlap, max(0, chunk_size - header_len - 1)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    while start < len(units):
        max_body_len = max(1, chunk_size - header_len)

        # Single unit too large: split it with fallback splitter.
        if len(units[start]) > max_body_len:
            unit_index = start + 1
            for part in fallback_splitter.split_text(units[start]):
                chunks.append((header + part, unit_index, unit_index))
            start += 1
            continue

        end = start
        cur_units: list[str] = []
        cur_len = 0

        while end < len(units):
            unit = units[end]
            add_len = len(unit) + (len(sep) if cur_units else 0)
            if header_len + cur_len + add_len <= chunk_size or not cur_units:
                cur_units.append(unit)
                cur_len += add_len
                end += 1
            else:
                break

        chunk_text = header + sep.join(cur_units)
        chunks.append((chunk_text, start + 1, end))

        if end >= len(units):
            break

        if chunk_overlap <= 0:
            start = end
            continue

        # Determine new start based on overlap budget (in characters), but avoid stalling.
        overlap_chars = 0
        overlap_start = end
        while overlap_start > start:
            overlap_start -= 1
            overlap_chars += len(units[overlap_start]) + len(sep)
            if overlap_chars >= chunk_overlap:
                break

        if overlap_start <= start:
            overlap_start = max(start + 1, end - 1)

        start = overlap_start

    return chunks


def build_and_save_chunks(
    jsonl_path: Path = DEFAULT_JSONL_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    """Read the TEK17 JSONL corpus and write chunked records to JSONL.

    Output format (one JSON object per line):
    {
      "text": str,
      "metadata": { ... }
    }
    """
    jsonl_path = jsonl_path.resolve()
    chunks_path = chunks_path.resolve()

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"JSONL corpus not found: {jsonl_path}\n"
            "Run `python -m tek17 download-dibk` then `python -m tek17 parse-dibk` first."
        )

    records = _load_jsonl(jsonl_path)
    docs = _build_documents(records)

    # We chunk at paragraph/ledd-ish boundaries first, then pack into chunks.
    # This keeps chunks aligned with regulation structure better than pure
    # character-based splitting.

    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with chunks_path.open("w", encoding="utf-8") as f:
        for text, meta in docs:
            # Keep the first two lines as a header and chunk the remaining body.
            # Header format from _build_documents():
            #   "§ x-y – Title\n(<type>)\n\n"
            if "\n\n" in text:
                header, body = text.split("\n\n", 1)
                header = header + "\n\n"
            else:
                header, body = "", text

            units = _split_into_units(body)
            packed = _pack_units_into_chunks(
                header=header,
                units=units,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # If unit splitting failed (e.g. empty body), fall back to plain splitting.
            if not packed and body.strip():
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                for chunk_text in splitter.split_text(text):
                    rec = {"text": chunk_text, "metadata": meta}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
                continue

            for chunk_text, unit_start, unit_end in packed:
                meta2 = dict(meta)
                meta2["para_start"] = unit_start
                meta2["para_end"] = unit_end
                rec = {"text": chunk_text, "metadata": meta2}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} chunks to {chunks_path}")
