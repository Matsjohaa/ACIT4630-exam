"""Sanity-check the TEK17 vector store (ChromaDB).

Prints collection count and a stable fingerprint of `chroma.sqlite3`.

Example:
  /path/to/python analysis/scripts/check_vectorstore.py

If you get an error or count=0:
  python -m tek17 download-dibk
  python -m tek17 parse-dibk
  python -m tek17 ingest
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tek17.rag.config import CHROMA_COLLECTION, CHROMA_DIR
from tek17.rag.retrieval.client import vectorstore_snapshot


def main() -> int:
    p = argparse.ArgumentParser(description="Check TEK17 Chroma vectorstore")
    p.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    p.add_argument("--collection", type=str, default=CHROMA_COLLECTION)
    args = p.parse_args()

    snap = vectorstore_snapshot(args.chroma_dir, args.collection)

    print("Vectorstore snapshot")
    print("- chroma_dir   :", snap.get("chroma_dir"))
    print("- collection   :", snap.get("collection"))
    print("- count        :", snap.get("count"))
    print("- sqlite_sha256:", snap.get("sqlite_sha256"))
    print("- sqlite_size  :", snap.get("sqlite_size"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
