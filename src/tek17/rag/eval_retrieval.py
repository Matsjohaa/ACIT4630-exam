"""Simple retrieval evaluation for the TEK17 RAG system.

This script reads an eval JSONL file (see `analysis/questions/README.md`), embeds
each question, queries the Chroma collection directly, and reports how
often at least one of the configured `target_sections` is retrieved.

Run, for example, from the project root:

    python -m tek17.rag.eval_retrieval \
        --eval-file analysis/questions/tek17_eval_questions.example.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tek17.rag.embedding.client import embed_query
from tek17.rag.retrieval.client import get_collection, query_collection


# Keep these consistent with the main server configuration
DEFAULT_CHROMA_DIR = Path("data/vectorstore/chroma")
COLLECTION_NAME = "tek17"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 6


def _load_eval_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _contains_target(retrieved_sections: Iterable[str], target_sections: Iterable[str]) -> bool:
    retrieved_set = {s.strip() for s in retrieved_sections if s}
    targets = {t.strip() for t in target_sections if t}
    if not targets:
        return False
    return not retrieved_set.isdisjoint(targets)


def evaluate_retrieval(
    eval_file: Path,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    if not eval_file.exists():
        raise SystemExit(f"Eval file not found: {eval_file}")

    items = _load_eval_items(eval_file)
    if not items:
        raise SystemExit("Eval file is empty")

    collection = get_collection(chroma_dir, collection_name)

    total_with_targets = 0
    hits = 0

    print(f"Using collection '{collection_name}' in {chroma_dir}")
    print(f"Top-k = {top_k}, eval items = {len(items)}")
    print("id\twith_targets\thit\ttarget_sections\tretrieved_section_ids")

    for item in items:
        qid = item.get("id", "")
        question = item.get("question", "")
        target_sections = item.get("target_sections") or []

        if not question:
            continue

        has_targets = bool(target_sections)
        if has_targets:
            total_with_targets += 1

        q_embedding = embed_query(
            question,
            provider="ollama",
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        documents, metadatas, distances = query_collection(
            collection=collection,
            query_embedding=q_embedding,
            top_k=top_k,
        )

        retrieved_sections = [
            (m or {}).get("section_id", "") for m in metadatas
        ]

        hit = _contains_target(retrieved_sections, target_sections) if has_targets else False
        if hit:
            hits += 1

        print(
            f"{qid}\t{int(has_targets)}\t{int(hit)}\t"
            f"{target_sections}\t{retrieved_sections}"
        )

    if total_with_targets:
        recall_at_k = hits / total_with_targets
        print()
        print(f"Questions with target_sections: {total_with_targets}")
        print(f"Hits (at least one target retrieved): {hits}")
        print(f"Recall@{top_k}: {recall_at_k:.3f}")
    else:
        print("No items with non-empty target_sections; nothing to score.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval for TEK17 RAG.")
    parser.add_argument(
        "--eval-file",
        type=Path,
        required=True,
        help="Path to eval JSONL file (see analysis/questions/README.md)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve per question",
    )
    args = parser.parse_args()

    evaluate_retrieval(
        eval_file=args.eval_file,
        top_k=args.top_k,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
