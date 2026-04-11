"""RAG refusal behaviour evaluation for the TEK17 system.

This script reads an eval JSONL file, sends questions to a running TEK17
RAG server, and compares the model's behaviour with the `should_refuse`
labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from tek17.rag.config import SERVER_URL, TOP_K


def _load_eval_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    return items


def _classify_refusal(answer: str) -> bool:
    """Return True if the answer appears to be a refusal."""
    text = answer.lower()

    if "kan_ikke_svare" in text:
        return True

    refusal_patterns = [
        "kan ikke svare",
        "kan ikke gi et sikkert svar",
        "har ikke nok informasjon",
        "finner ikke nok informasjon",
        "kan ikke gi råd",
        "kan ikke gi et konkret svar",
        "utenfor det som dekkes av tek17",
    ]

    return any(pattern in text for pattern in refusal_patterns)


def evaluate_refusal(
    eval_file: Path,
    server_url: str = SERVER_URL,
    top_k: int = TOP_K,
    out_path: Path | None = None,
) -> None:
    if not eval_file.exists():
        raise SystemExit(f"Eval file not found: {eval_file}")

    items = _load_eval_items(eval_file)
    if not items:
        raise SystemExit("Eval file is empty")

    total = 0
    correct = 0

    tp_refuse = 0
    fp_refuse = 0
    tn_refuse = 0
    fn_refuse = 0

    out_file = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8")

    try:
        print(f"Using RAG server at {server_url}")
        print("id\tshould_refuse\tmodel_refused\tretrieval_hit\tstatus")

        for item in items:
            question_id = item.get("id", "")
            question = item.get("question", "")
            target_sections = item.get("target_sections") or []
            should_refuse = bool(item.get("should_refuse", False))

            if not question:
                continue

            total += 1

            try:
                response = requests.post(
                    f"{server_url}/query",
                    json={"question": question, "top_k": top_k},
                    timeout=60,
                )
                response.raise_for_status()
            except Exception as exc:
                print(f"{question_id}\t{should_refuse}\tERROR\t0\trequest_failed: {exc}")

                if out_file is not None:
                    row = {
                        "id": question_id,
                        "question": question,
                        "target_sections": target_sections,
                        "should_refuse": should_refuse,
                        "model_refused": None,
                        "retrieval_hit": False,
                        "status": "request_failed",
                        "error": str(exc),
                        "answer": None,
                        "retrieved_section_ids": [],
                        "server_url": server_url,
                        "top_k": top_k,
                    }
                    out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            data = response.json()
            answer = data.get("answer", "")
            sources = data.get("sources", []) or []

            retrieved_section_ids = [
                (source or {}).get("section_id", "")
                for source in sources
            ]

            retrieval_hit = False
            if target_sections:
                retrieved_set = {section.strip() for section in retrieved_section_ids if section}
                target_set = {section.strip() for section in target_sections if section}
                retrieval_hit = not retrieved_set.isdisjoint(target_set)

            model_refused = _classify_refusal(answer)

            if should_refuse and model_refused:
                tp_refuse += 1
                status = "correct_refusal"
            elif not should_refuse and not model_refused:
                tn_refuse += 1
                status = "correct_answer"
            elif not should_refuse and model_refused:
                fp_refuse += 1
                status = "over_refusal"
            else:
                fn_refuse += 1
                status = "under_refusal"

            if (should_refuse and model_refused) or (not should_refuse and not model_refused):
                correct += 1

            print(
                f"{question_id}\t{should_refuse}\t{model_refused}\t{int(retrieval_hit)}\t{status}"
            )

            if out_file is not None:
                row = {
                    "id": question_id,
                    "question": question,
                    "target_sections": target_sections,
                    "should_refuse": should_refuse,
                    "model_refused": model_refused,
                    "retrieval_hit": retrieval_hit,
                    "status": status,
                    "answer": answer,
                    "retrieved_section_ids": retrieved_section_ids,
                    "server_url": server_url,
                    "top_k": top_k,
                }
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        if not total:
            print("No valid eval items.")
            return

        accuracy = correct / total

        print()
        print(f"Total eval items: {total}")
        print(f"Refusal behaviour accuracy: {accuracy:.3f}")
        print("Confusion matrix (refusal vs. label):")
        print(f"  TP (correct refusals)      : {tp_refuse}")
        print(f"  FP (over-refusals)         : {fp_refuse}")
        print(f"  TN (correct answers)       : {tn_refuse}")
        print(f"  FN (under-refusals)        : {fn_refuse}")

    finally:
        if out_file is not None:
            out_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate refusal behaviour of the TEK17 RAG server.",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        required=True,
        help="Path to eval JSONL file",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=SERVER_URL,
        help="Base URL of the running RAG server",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of chunks to request per question",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSONL file for results",
    )
    args = parser.parse_args()

    evaluate_refusal(
        eval_file=args.eval_file,
        server_url=args.server_url,
        top_k=args.top_k,
        out_path=args.out,
    )


if __name__ == "__main__":  # pragma: no cover
    main()