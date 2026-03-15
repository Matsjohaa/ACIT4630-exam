"""RAG refusal behaviour evaluation for the TEK17 system.

This script reads an eval JSONL file (see `analysis/questions/README.md`), sends
questions to a running TEK17 RAG server (`tek17.rag.server`), and
compares the model's behaviour with the `should_refuse` labels.

The script uses a simple heuristic to detect whether the model
"refused" or not based on the answer text; this is intentionally
transparent so you can adjust it to your own criteria.

Example usage (server must already be running):

    # In one terminal
    python -m tek17 serve

    # In another terminal
    python -m tek17.rag.eval_refusal \
        --eval-file analysis/questions/tek17_eval_questions.example.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_SERVER_URL = "http://localhost:8000"
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


def _classify_refusal(answer: str) -> bool:
    """Very simple heuristic to decide if the model refused.

    Returns True if we believe the model refused to answer the
    question (e.g. says it cannot answer or that information is not
    available), False otherwise.
    """

    text = answer.lower()

    # Norwegian-ish refusal patterns – adjust as needed for your eval set.
    patterns = [
        "kan ikke svare",
        "kan ikke gi et sikkert svar",
        "har ikke nok informasjon",
        "finner ikke nok informasjon",
        "kan ikke gi råd",
        "kan ikke gi et konkret svar",
        "utenfor det som dekkes av tek17",
    ]

    return any(p in text for p in patterns)


def evaluate_refusal(
    eval_file: Path,
    server_url: str = DEFAULT_SERVER_URL,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    if not eval_file.exists():
        raise SystemExit(f"Eval file not found: {eval_file}")

    items = _load_eval_items(eval_file)
    if not items:
        raise SystemExit("Eval file is empty")

    total = 0
    correct = 0

    # Confusion matrix counters
    tp_refuse = 0  # should_refuse=True and model_refused=True
    fp_refuse = 0  # should_refuse=False but model_refused=True
    tn_refuse = 0  # should_refuse=False and model_refused=False
    fn_refuse = 0  # should_refuse=True but model_refused=False

    print(f"Using RAG server at {server_url}")
    print("id\tshould_refuse\tmodel_refused\tretrieval_hit\tstatus")

    for item in items:
        qid = item.get("id", "")
        question = item.get("question", "")
        target_sections = item.get("target_sections") or []
        should_refuse = bool(item.get("should_refuse", False))

        if not question:
            continue

        total += 1

        try:
            resp = requests.post(
                f"{server_url}/query",
                json={"question": question, "top_k": top_k},
                timeout=60,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"{qid}\t{should_refuse}\tERROR\t0\trequest_failed: {e}")
            continue

        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("sources", []) or []

        # Determine whether retrieval found at least one expected section
        retrieved_section_ids = [
            (s or {}).get("section_id", "") for s in sources
        ]
        retrieval_hit = False
        if target_sections:
            retrieved_set = {s.strip() for s in retrieved_section_ids if s}
            targets = {t.strip() for t in target_sections if t}
            retrieval_hit = not retrieved_set.isdisjoint(targets)

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
        else:  # should_refuse and not model_refused
            fn_refuse += 1
            status = "under_refusal"

        if (should_refuse and model_refused) or (not should_refuse and not model_refused):
            correct += 1

        print(
            f"{qid}\t{should_refuse}\t{model_refused}\t{int(retrieval_hit)}\t{status}"
        )

    if total:
        accuracy = correct / total
        print()
        print(f"Total eval items: {total}")
        print(f"Refusal behaviour accuracy: {accuracy:.3f}")
        print("Confusion matrix (refusal vs. label):")
        print(f"  TP (correct refusals)      : {tp_refuse}")
        print(f"  FP (over-refusals)         : {fp_refuse}")
        print(f"  TN (correct answers)       : {tn_refuse}")
        print(f"  FN (under-refusals)        : {fn_refuse}")
    else:
        print("No valid eval items.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate refusal behaviour of the TEK17 RAG server.",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        required=True,
        help="Path to eval JSONL file (see analysis/questions/README.md)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help="Base URL of the running RAG server",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to request per question",
    )
    args = parser.parse_args()

    evaluate_refusal(
        eval_file=args.eval_file,
        server_url=args.server_url,
        top_k=args.top_k,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
