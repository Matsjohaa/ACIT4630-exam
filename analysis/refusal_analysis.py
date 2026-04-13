from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tek17.rag.config import (
    ANALYSIS_DIR,
    CONDITIONAL_REFUSAL_CATEGORY_NAMES,
)


DEFAULT_LOG_DIR = ANALYSIS_DIR / "logging"
DEFAULT_OUT_CSV = DEFAULT_LOG_DIR / "conditional_refusal_summary.csv"


def _coalesce_bool(row: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            return value
    return None


def classify_case(
    any_hit: bool,
    full_hit: bool,
    partial_hit: bool,
    should_refuse: bool,
    model_refused: bool,
    requires_qualification: bool = False,
    qualification_warning_present: bool = False,
) -> str:
    if requires_qualification and (not should_refuse) and (not model_refused):
        if not any_hit:
            return "answer_without_evidence"
        if not qualification_warning_present:
            return "missing_qualification_warning"
        return "correct_qualified_answer"
    if (not any_hit) and should_refuse and model_refused:
        return "retrieval_miss_correct_refusal"
    if partial_hit and (not should_refuse) and model_refused:
        return "over_refusal_with_partial_evidence"
    if full_hit and (not should_refuse) and model_refused:
        return "over_refusal_with_full_evidence"
    if (not any_hit) and (not should_refuse) and (not model_refused):
        return "answer_without_evidence"
    if partial_hit and (not should_refuse) and (not model_refused):
        return "partial_support_answer"
    if full_hit and (not should_refuse) and (not model_refused):
        return "correct_answer"
    if partial_hit and should_refuse and model_refused:
        return "correct_refusal_with_partial_context"
    if full_hit and should_refuse and model_refused:
        return "correct_refusal_with_full_context"
    if partial_hit and should_refuse and (not model_refused):
        return "under_refusal_with_partial_context"
    if full_hit and should_refuse and (not model_refused):
        return "under_refusal_with_full_context"
    if (not any_hit) and should_refuse and (not model_refused):
        return "unsafe_answer_no_evidence"
    return "other"


def empty_category_counts() -> dict[str, int]:
    return {name: 0 for name in CONDITIONAL_REFUSAL_CATEGORY_NAMES}


def summarize_run(path: Path) -> dict[str, Any] | None:
    any_hit_refuse = 0
    any_hit_total = 0
    no_hit_refuse = 0
    no_hit_total = 0
    partial_hit_refuse = 0
    partial_hit_total = 0
    full_hit_refuse = 0
    full_hit_total = 0

    model = None
    retrieval = None
    top_k = None
    temperature = None
    mode = None
    hybrid_alpha = None
    prompt_version = None

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
                prompt_version = row.get("prompt_version")

            model_refused = row.get("model_refused")
            should_refuse = row.get("should_refuse")

            any_hit = _coalesce_bool(row, "any_hit", "retrieval_hit")
            full_hit = _coalesce_bool(row, "full_hit")
            partial_hit = _coalesce_bool(row, "partial_hit")

            if model_refused is None or should_refuse is None or any_hit is None:
                continue

            model_refused = bool(model_refused)
            should_refuse = bool(should_refuse)
            any_hit = bool(any_hit)
            full_hit = bool(full_hit) if full_hit is not None else False
            partial_hit = bool(partial_hit) if partial_hit is not None else False

            n_rows += 1

            if any_hit:
                any_hit_total += 1
                if model_refused:
                    any_hit_refuse += 1
            else:
                no_hit_total += 1
                if model_refused:
                    no_hit_refuse += 1

            if partial_hit:
                partial_hit_total += 1
                if model_refused:
                    partial_hit_refuse += 1

            if full_hit:
                full_hit_total += 1
                if model_refused:
                    full_hit_refuse += 1

            category = classify_case(
                any_hit=any_hit,
                full_hit=full_hit,
                partial_hit=partial_hit,
                should_refuse=should_refuse,
                model_refused=model_refused,
            )
            category_counts[category] += 1

    if n_rows == 0:
        return None

    p_any_hit = any_hit_refuse / any_hit_total if any_hit_total else None
    p_no_hit = no_hit_refuse / no_hit_total if no_hit_total else None
    p_partial_hit = partial_hit_refuse / partial_hit_total if partial_hit_total else None
    p_full_hit = full_hit_refuse / full_hit_total if full_hit_total else None

    gap_nohit_minus_any = (
        p_no_hit - p_any_hit
        if (p_any_hit is not None and p_no_hit is not None)
        else None
    )
    gap_partial_minus_full = (
        p_partial_hit - p_full_hit
        if (p_partial_hit is not None and p_full_hit is not None)
        else None
    )

    summary = {
        "file": path.name,
        "mode": mode,
        "model": model,
        "retrieval_method": retrieval,
        "top_k": top_k,
        "temperature": temperature,
        "hybrid_alpha": hybrid_alpha,
        "prompt_version": prompt_version,
        "n_rows": n_rows,
        "any_hit_n": any_hit_total,
        "any_hit_refuse_n": any_hit_refuse,
        "no_hit_n": no_hit_total,
        "no_hit_refuse_n": no_hit_refuse,
        "partial_hit_n": partial_hit_total,
        "partial_hit_refuse_n": partial_hit_refuse,
        "full_hit_n": full_hit_total,
        "full_hit_refuse_n": full_hit_refuse,
        "P(refusal|any_hit)": round(p_any_hit, 3) if p_any_hit is not None else None,
        "P(refusal|no_hit)": round(p_no_hit, 3) if p_no_hit is not None else None,
        "P(refusal|partial_hit)": round(p_partial_hit, 3) if p_partial_hit is not None else None,
        "P(refusal|full_hit)": round(p_full_hit, 3) if p_full_hit is not None else None,
        "gap_nohit_minus_anyhit": round(gap_nohit_minus_any, 3) if gap_nohit_minus_any is not None else None,
        "gap_partial_minus_full": round(gap_partial_minus_full, 3) if gap_partial_minus_full is not None else None,
    }

    for name in CONDITIONAL_REFUSAL_CATEGORY_NAMES:
        summary[name] = category_counts[name]

    return summary


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== Conditional refusal + taxonomy (all runs) ===\n")

    for row in rows:
        extra = ""
        if row["retrieval_method"] == "hybrid" and row["hybrid_alpha"] is not None:
            extra = f" (alpha={row['hybrid_alpha']})"

        p_any_str = (
            f"{row['P(refusal|any_hit)']:.3f}"
            if row["P(refusal|any_hit)"] is not None
            else "NA"
        )
        p_nohit_str = (
            f"{row['P(refusal|no_hit)']:.3f}"
            if row["P(refusal|no_hit)"] is not None
            else "NA"
        )
        p_partial_str = (
            f"{row['P(refusal|partial_hit)']:.3f}"
            if row["P(refusal|partial_hit)"] is not None
            else "NA"
        )
        p_full_str = (
            f"{row['P(refusal|full_hit)']:.3f}"
            if row["P(refusal|full_hit)"] is not None
            else "NA"
        )
        gap_nohit_any_str = (
            f"{row['gap_nohit_minus_anyhit']:.3f}"
            if row["gap_nohit_minus_anyhit"] is not None
            else "NA"
        )
        gap_partial_full_str = (
            f"{row['gap_partial_minus_full']:.3f}"
            if row["gap_partial_minus_full"] is not None
            else "NA"
        )

        print(
            f"{str(row['model']):15} | "
            f"{str(row['retrieval_method']):7}{extra:12} | "
            f"P(any_hit)={p_any_str} | "
            f"P(no_hit)={p_nohit_str} | "
            f"P(partial)={p_partial_str} | "
            f"P(full)={p_full_str}"
        )
        print(
            f"  gaps: no_hit-any_hit={gap_nohit_any_str}, "
            f"partial-full={gap_partial_full_str}"
        )
        print(
            "  taxonomy: "
            f"miss->correct_refusal={row['retrieval_miss_correct_refusal']}, "
            f"over_refusal_with_partial_evidence={row['over_refusal_with_partial_evidence']}, "
            f"over_refusal_with_full_evidence={row['over_refusal_with_full_evidence']}, "
            f"answer_without_evidence={row['answer_without_evidence']}, "
            f"partial_support_answer={row['partial_support_answer']}, "
            f"correct_answer={row['correct_answer']}, "
            f"correct_refusal_with_partial_context={row['correct_refusal_with_partial_context']}, "
            f"correct_refusal_with_full_context={row['correct_refusal_with_full_context']}, "
            f"under_refusal_with_partial_context={row['under_refusal_with_partial_context']}, "
            f"under_refusal_with_full_context={row['under_refusal_with_full_context']}, "
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