"""Simple retrieval evaluation for the TEK17 RAG system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tek17.rag.config import (
    CHUNKS_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
    HYBRID_ALPHA,
    RETRIEVAL_METHOD,
    TOP_K,
)
from tek17.rag.ingest import embed_query
from tek17.rag.retrieval.client import get_collection, retrieve


def _load_eval_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    return items


def _normalize_section_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = normalized.replace("§", "")
    normalized = " ".join(normalized.split())
    return normalized


def _contains_target(
    retrieved_sections: Iterable[str],
    target_sections: Iterable[str],
) -> bool:
    retrieved_set = {
        _normalize_section_id(section)
        for section in retrieved_sections
        if str(section).strip()
    }
    target_set = {
        _normalize_section_id(section)
        for section in target_sections
        if str(section).strip()
    }

    if not target_set:
        return False

    return not retrieved_set.isdisjoint(target_set)


def _normalize_retrieval_method(method: str) -> str:
    normalized = (method or RETRIEVAL_METHOD).strip().lower()
    if normalized == "sparce":
        return "sparse"
    return normalized


def evaluate_retrieval(
    eval_file: Path,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    top_k: int = TOP_K,
    retrieval_method: str = RETRIEVAL_METHOD,
    hybrid_alpha: float = HYBRID_ALPHA,
    out_path: Path | None = None,
) -> None:
    if not eval_file.exists():
        raise SystemExit(f"Eval file not found: {eval_file}")

    items = _load_eval_items(eval_file)
    if not items:
        raise SystemExit("Eval file is empty")

    collection = get_collection(chroma_dir, collection_name)

    total_with_targets = 0
    hits = 0
    normalized_method = _normalize_retrieval_method(retrieval_method)

    out_file = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8")

    try:
        print(f"Using collection '{collection_name}' in {chroma_dir}")
        print(f"Top-k = {top_k}, eval items = {len(items)}")
        print("id\twith_targets\thit\ttarget_sections\tretrieved_section_ids")

        for item in items:
            question_id = item.get("id", "")
            question = item.get("question", "")
            target_sections = item.get("target_sections") or []

            if not question:
                continue

            has_targets = bool(target_sections)
            if has_targets:
                total_with_targets += 1

            query_embedding: list[float] | None = None
            if normalized_method in {"dense", "hybrid"}:
                query_embedding = embed_query(
                    question,
                    provider=EMBED_PROVIDER,
                    model=EMBED_MODEL,
                    base_url=EMBED_BASE_URL,
                )

            _, metadatas, _ = retrieve(
                collection=collection,
                query_text=question,
                query_embedding=query_embedding,
                top_k=top_k,
                method=normalized_method,
                chunks_path=CHUNKS_PATH,
                hybrid_alpha=hybrid_alpha,
            )

            retrieved_sections = [
                (metadata or {}).get("section_id", "")
                for metadata in metadatas
            ]

            hit = _contains_target(retrieved_sections, target_sections) if has_targets else False
            if hit:
                hits += 1

            print(
                f"{question_id}\t{int(has_targets)}\t{int(hit)}\t"
                f"{target_sections}\t{retrieved_sections}"
            )

            if out_file is not None:
                row = {
                    "id": question_id,
                    "question": question,
                    "target_sections": target_sections,
                    "retrieved_sections": retrieved_sections,
                    "has_targets": has_targets,
                    "hit": bool(hit),
                    "top_k": top_k,
                    "retrieval_method": normalized_method,
                    "hybrid_alpha": hybrid_alpha,
                    "embed_provider": EMBED_PROVIDER,
                    "embed_model": EMBED_MODEL,
                    "collection_name": collection_name,
                    "chroma_dir": str(chroma_dir),
                }
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        if not total_with_targets:
            print("No items with non-empty target_sections; nothing to score.")
            return

        recall_at_k = hits / total_with_targets

        print()
        print(f"Questions with target_sections: {total_with_targets}")
        print(f"Hits (at least one target retrieved): {hits}")
        print(f"Recall@{top_k}: {recall_at_k:.3f}")

    finally:
        if out_file is not None:
            out_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval for TEK17 RAG.")
    parser.add_argument(
        "--eval-file",
        type=Path,
        required=True,
        help="Path to eval JSONL file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--retrieval-method",
        choices=["dense", "sparse", "hybrid", "sparce"],
        default=RETRIEVAL_METHOD,
        help="Retrieval method to evaluate.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=HYBRID_ALPHA,
        help="Hybrid weighting: alpha*dense + (1-alpha)*sparse.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSONL file for results",
    )
    args = parser.parse_args()

    evaluate_retrieval(
        eval_file=args.eval_file,
        top_k=args.top_k,
        retrieval_method=args.retrieval_method,
        hybrid_alpha=args.hybrid_alpha,
        out_path=args.out,
    )


if __name__ == "__main__":  # pragma: no cover
    main()