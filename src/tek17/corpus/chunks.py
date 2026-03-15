from __future__ import annotations

from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_JSONL_PATH = Path("data/processed/tek17_dibk.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/tek17_chunks.jsonl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


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
            header = f"{section_id} \\u2013 {title}\n(Forskriftstekst)\n\n"
            docs.append((header + reg, meta))

        guidance = (rec.get("guidance_text", "") or "").strip()
        if guidance:
            meta = {**base_meta, "text_type": "guidance"}
            header = f"{section_id} \\u2013 {title}\n(Veiledning)\n\n"
            docs.append((header + guidance, meta))

    return docs


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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with chunks_path.open("w", encoding="utf-8") as f:
        for text, meta in docs:
            for chunk_text in splitter.split_text(text):
                rec = {"text": chunk_text, "metadata": meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} chunks to {chunks_path}")
