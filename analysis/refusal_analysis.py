from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tek17.rag.config import ANALYSIS_DIR


DEFAULT_LOG_DIR = ANALYSIS_DIR / "logging"
DEFAULT_OUT_CSV = DEFAULT_LOG_DIR / "conditional_refusal_summary.csv"

CATEGORY_NAMES = [
    "retrieval_miss_correct_refusal",
    "over_refusal_with_evidence",
    "answer_without_evidence",
    "correct_answer",
    "correct_refusal_with_context",
    "under_refusal_with_context",
    "unsafe_answer_no_evidence",
    "other",
]


def classify_case(hit: bool, should_refuse: bool, model_refused: bool) -> str:
    if not hit and should_refuse and model_refused:
        return "retrieval_miss_correct_refusal"
    if hit and not should_refuse and model_refused:
        return "over_refusal_with_evidence"
    if not hit and not should_refuse and not model_refused:
        return "answer_without_evidence"
    if hit and not should_refuse and not model_refused:
        return "correct_answer"
    if hit and should_refuse and model_refused:
        return "correct_refusal_with_context"
    if hit and should_refuse and not model_refused:
        return "under_refusal_with_context"
    if not hit and should_refuse and not model_refused:
        return "unsafe_answer_no_evidence"
    return "other"


def empty_category_counts() -> dict[str, int]:
    return {name: 0 for name in CATEGORY_NAMES}


def summarize_run(path: Path) -> dict[str, Any] | None:
    hit_refuse = 0
    hit_total = 0
    nohit_refuse = 0
    nohit_total = 0

    model = None
    retrieval = None
    top_k = None
    temperature = None
    mode = None
    hybrid_alpha = None

    category_counts = empty_category_counts()
    n_rows = 0

    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            row = json.loads(line)

            if i == 0:
                model = row.get("model")
                retrieval = row.get("retrieval_method")
                top_k = row.get("top_k")
                temperature = row.get("temperature")
                mode = row.get("mode")
                hybrid_alpha = row.get("hybrid_alpha")

            hit = row.get("retrieval_hit")
            model_refused = row.get("model_refused")
            should_refuse = row.get("should_refuse")

            if hit is None or model_refused is None or should_refuse is None:
                continue

            hit = bool(hit)
            model_refused = bool(model_refused)
            should_refuse = bool(should_refuse)

            n_rows += 1

            if hit:
                hit_total += 1
                if model_refused:
                    hit_refuse += 1
            else:
                nohit_total += 1
                if model_refused:
                    nohit_refuse += 1

            category = classify_case(hit, should_refuse, model_refused)
            category_counts[category] += 1

    if n_rows == 0:
        return None

    p_hit = hit_refuse / hit_total if hit_total else None
    p_nohit = nohit_refuse / nohit_total if nohit_total else None
    gap = (p_nohit - p_hit) if (p_hit is not None and p_nohit is not None) else None

    summary = {
        "file": path.name,
        "mode": mode,
        "model": model,
        "retrieval_method": retrieval,
        "top_k": top_k,
        "temperature": temperature,
        "hybrid_alpha": hybrid_alpha,
        "n_rows": n_rows,
        "hit_n": hit_total,
        "hit_refuse_n": hit_refuse,
        "no_hit_n": nohit_total,
        "no_hit_refuse_n": nohit_refuse,
        "P(refusal|hit)": round(p_hit, 3) if p_hit is not None else None,
        "P(refusal|no_hit)": round(p_nohit, 3) if p_nohit is not None else None,
        "gap_nohit_minus_hit": round(gap, 3) if gap is not None else None,
    }

    for name in CATEGORY_NAMES:
        summary[name] = category_counts[name]

    return summary


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== Conditional refusal + taxonomy (all runs) ===\n")

    for row in rows:
        extra = ""
        if row["retrieval_method"] == "hybrid" and row["hybrid_alpha"] is not None:
            extra = f" (alpha={row['hybrid_alpha']})"

        p_hit_str = f"{row['P(refusal|hit)']:.3f}" if row["P(refusal|hit)"] is not None else "NA"
        p_nohit_str = f"{row['P(refusal|no_hit)']:.3f}" if row["P(refusal|no_hit)"] is not None else "NA"
        gap_str = f"{row['gap_nohit_minus_hit']:.3f}" if row["gap_nohit_minus_hit"] is not None else "NA"

        print(
            f"{str(row['model']):15} | "
            f"{str(row['retrieval_method']):7}{extra:12} | "
            f"P(hit)={p_hit_str} | "
            f"P(no_hit)={p_nohit_str} | "
            f"gap={gap_str}"
        )
        print(
            "  taxonomy: "
            f"miss->correct_refusal={row['retrieval_miss_correct_refusal']}, "
            f"over_refusal_with_evidence={row['over_refusal_with_evidence']}, "
            f"answer_without_evidence={row['answer_without_evidence']}, "
            f"correct_answer={row['correct_answer']}, "
            f"correct_refusal_with_context={row['correct_refusal_with_context']}, "
            f"under_refusal_with_context={row['under_refusal_with_context']}, "
            f"unsafe_answer_no_evidence={row['unsafe_answer_no_evidence']}, "
            f"other={row['other']}"
        )


def write_summary_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize conditional refusal metrics across evaluation runs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing per-run JSONL files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Output CSV path for aggregated summary.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []

    for path in sorted(args.log_dir.glob("*.jsonl")):
        summary = summarize_run(path)
        if summary is not None:
            rows.append(summary)

    rows.sort(
        key=lambda row: (
            str(row["model"]),
            str(row["retrieval_method"]),
            row["top_k"] if row["top_k"] is not None else -1,
        )
    )

    if not rows:
        print("No valid JSONL runs found.")
        return

    print_summary(rows)
    write_summary_csv(rows, args.out_csv)
    print(f"\nSaved summary to: {args.out_csv}")


if __name__ == "__main__":
    main()