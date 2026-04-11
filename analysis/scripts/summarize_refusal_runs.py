"""Summarize one or more refusal-eval JSONL runs into CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_REFUSAL_TYPES = [
    "out_of_scope",
    "in_domain_missing_context",
    "(unspecified)",
]

DEFAULT_IN_SCOPE_SLICES = [
    "in_scope_single",
    "in_scope_multi",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def _infer_question_type(row: dict[str, Any]) -> str:
    question_type = row.get("question_type")
    if isinstance(question_type, str) and question_type.strip():
        return question_type.strip()

    if bool(row.get("should_refuse")):
        return "refusal"

    target_sections = row.get("target_sections") or []
    if isinstance(target_sections, list) and len(target_sections) >= 2:
        return "in_scope_multi"

    question_id = row.get("id")
    if isinstance(question_id, str) and "_multi_" in question_id:
        return "in_scope_multi"

    return "in_scope_single"


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / (denom ** 0.5) if denom else 0.0

    return {
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "refusal_rate_pred": _safe_div(tp + fp, total),
        "refusal_rate_true": _safe_div(tp + fn, total),
    }


def _coalesce_bool(row: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            return value
    return False


def _build_refusal_type_columns(
    refusal_type_counts: dict[str, int],
    refusal_type_refused: dict[str, int],
) -> dict[str, object]:
    columns: dict[str, object] = {}

    for refusal_type in DEFAULT_REFUSAL_TYPES:
        total = int(refusal_type_counts.get(refusal_type, 0))
        refused = int(refusal_type_refused.get(refusal_type, 0))

        columns[f"refuse_n__{refusal_type}"] = total
        columns[f"refuse_refused__{refusal_type}"] = refused
        columns[f"refuse_refused_rate__{refusal_type}"] = (
            _safe_div(refused, total) if total else None
        )

    return columns


def _build_slice_columns(
    slice_counts: dict[str, int],
    slice_over_refusal: dict[str, int],
    slice_unsupported: dict[str, int],
    slice_partial_support: dict[str, int],
    slice_ungrounded: dict[str, int],
    slice_strict_correct: dict[str, int],
    slice_strict_den: dict[str, int],
) -> dict[str, object]:
    columns: dict[str, object] = {}

    for slice_name in DEFAULT_IN_SCOPE_SLICES:
        total = slice_counts.get(slice_name, 0)

        columns[f"slice_n__{slice_name}"] = total
        columns[f"slice_over_refusal_rate__{slice_name}"] = (
            _safe_div(slice_over_refusal[slice_name], total) if total else None
        )
        columns[f"slice_unsupported_rate__{slice_name}"] = (
            _safe_div(slice_unsupported[slice_name], total) if total else None
        )
        columns[f"slice_partial_support_rate__{slice_name}"] = (
            _safe_div(slice_partial_support[slice_name], total) if total else None
        )
        columns[f"slice_ungrounded_rate__{slice_name}"] = (
            _safe_div(slice_ungrounded[slice_name], total) if total else None
        )
        columns[f"slice_strict_correct_rate__{slice_name}"] = (
            _safe_div(
                slice_strict_correct[slice_name],
                slice_strict_den[slice_name],
            )
            if slice_strict_den[slice_name]
            else None
        )

    return columns


def _summarize_file(path: Path) -> dict[str, object]:
    rows = _load_jsonl(path)
    if not rows:
        return {"file": str(path), "n": 0}

    first = rows[0]

    retrieval_method = str(first.get("retrieval_method", "")).strip().lower()
    if retrieval_method == "sparce":
        retrieval_method = "sparse"

    top_k = first.get("top_k")
    hybrid_alpha = first.get("hybrid_alpha")
    model = first.get("model")
    temperature = first.get("temperature")
    mode = first.get("mode")
    prompt_version = first.get("prompt_version")
    system_prompt_sha256 = first.get("system_prompt_sha256")

    tp = fp = tn = fn = 0

    any_hits = 0
    full_hits = 0
    partial_hits = 0

    n_total = len(rows)
    n_errors = 0

    n_answer_correct = 0
    n_answer_correct_den = 0
    n_answer_correct_strict = 0
    n_answer_correct_strict_den = 0
    n_unsupported_non_refusal = 0
    n_partial_support_non_refusal = 0
    n_ungrounded_non_refusal = 0

    slice_counts = {name: 0 for name in DEFAULT_IN_SCOPE_SLICES}
    slice_over_refusal = {name: 0 for name in DEFAULT_IN_SCOPE_SLICES}
    slice_unsupported = {name: 0 for name in DEFAULT_IN_SCOPE_SLICES}
    slice_partial_support = {name: 0 for name in DEFAULT_IN_SCOPE_SLICES}
    slice_ungrounded = {name: 0 for name in DEFAULT_IN_SCOPE_SLICES}
    slice_strict_correct = {name: 0 for name in DEFAULT_IN_SCOPE_SLICES}
    slice_strict_den = {name: 0 for name in DEFAULT_IN_SCOPE_SLICES}

    refusal_type_counts: dict[str, int] = {}
    refusal_type_refused: dict[str, int] = {}

    for row in rows:
        if row.get("status") == "query_failed":
            n_errors += 1
            if row.get("unsupported_non_refusal"):
                n_unsupported_non_refusal += 1
            if row.get("partial_support_non_refusal"):
                n_partial_support_non_refusal += 1
            if row.get("ungrounded_non_refusal"):
                n_ungrounded_non_refusal += 1
            continue

        should_refuse = bool(row.get("should_refuse"))
        model_refused = bool(row.get("model_refused"))
        question_type = _infer_question_type(row)

        if should_refuse and model_refused:
            tp += 1
        elif (not should_refuse) and model_refused:
            fp += 1
        elif (not should_refuse) and (not model_refused):
            tn += 1
        else:
            fn += 1

        any_hit = _coalesce_bool(row, "any_hit", "retrieval_hit")
        full_hit = _coalesce_bool(row, "full_hit")
        partial_hit = _coalesce_bool(row, "partial_hit")

        if any_hit:
            any_hits += 1
        if full_hit:
            full_hits += 1
        if partial_hit:
            partial_hits += 1

        if row.get("unsupported_non_refusal"):
            n_unsupported_non_refusal += 1

        if row.get("partial_support_non_refusal"):
            n_partial_support_non_refusal += 1

        if row.get("ungrounded_non_refusal"):
            n_ungrounded_non_refusal += 1

        if isinstance(row.get("answer_correct"), bool):
            n_answer_correct_den += 1
            if bool(row.get("answer_correct")):
                n_answer_correct += 1

        if isinstance(row.get("answer_correct_strict"), bool):
            n_answer_correct_strict_den += 1
            if bool(row.get("answer_correct_strict")):
                n_answer_correct_strict += 1

        if not should_refuse and question_type in slice_counts:
            slice_counts[question_type] += 1
            if model_refused:
                slice_over_refusal[question_type] += 1
            if row.get("unsupported_non_refusal"):
                slice_unsupported[question_type] += 1
            if row.get("partial_support_non_refusal"):
                slice_partial_support[question_type] += 1
            if row.get("ungrounded_non_refusal"):
                slice_ungrounded[question_type] += 1
            if isinstance(row.get("answer_correct_strict"), bool):
                slice_strict_den[question_type] += 1
                if bool(row.get("answer_correct_strict")):
                    slice_strict_correct[question_type] += 1

        if should_refuse:
            refusal_type = row.get("refusal_type")
            key = (
                str(refusal_type).strip()
                if isinstance(refusal_type, str) and str(refusal_type).strip()
                else "(unspecified)"
            )
            refusal_type_counts[key] = refusal_type_counts.get(key, 0) + 1
            if model_refused:
                refusal_type_refused[key] = refusal_type_refused.get(key, 0) + 1

    metrics = _compute_metrics(tp, fp, tn, fn)

    n_ok = n_total - n_errors
    answer_correct_rate = (
        _safe_div(n_answer_correct, n_answer_correct_den)
        if n_answer_correct_den
        else None
    )
    answer_correct_strict_rate = (
        _safe_div(n_answer_correct_strict, n_answer_correct_strict_den)
        if n_answer_correct_strict_den
        else None
    )

    refusal_type_columns = _build_refusal_type_columns(
        refusal_type_counts,
        refusal_type_refused,
    )
    slice_columns = _build_slice_columns(
        slice_counts,
        slice_over_refusal,
        slice_unsupported,
        slice_partial_support,
        slice_ungrounded,
        slice_strict_correct,
        slice_strict_den,
    )

    refusal_type_breakdown = None
    if refusal_type_counts:
        refusal_type_breakdown = json.dumps(
            {
                key: {
                    "n": refusal_type_counts.get(key, 0),
                    "refused": refusal_type_refused.get(key, 0),
                    "refused_rate": (
                        _safe_div(
                            refusal_type_refused.get(key, 0),
                            refusal_type_counts.get(key, 0),
                        )
                        if refusal_type_counts.get(key)
                        else 0.0
                    ),
                }
                for key in sorted(refusal_type_counts)
            },
            ensure_ascii=False,
        )

    return {
        "file": str(path),
        "n": n_total,
        "n_ok": n_ok,
        "n_errors": n_errors,
        "error_rate": _safe_div(n_errors, n_total),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "any_hit_rate": _safe_div(any_hits, n_ok),
        "full_hit_rate": _safe_div(full_hits, n_ok),
        "partial_hit_rate": _safe_div(partial_hits, n_ok),
        "retrieval_hit_rate": _safe_div(any_hits, n_ok),
        "unsupported_non_refusal": n_unsupported_non_refusal,
        "unsupported_non_refusal_rate": _safe_div(n_unsupported_non_refusal, n_total),
        "partial_support_non_refusal": n_partial_support_non_refusal,
        "partial_support_non_refusal_rate": _safe_div(n_partial_support_non_refusal, n_total),
        "ungrounded_non_refusal": n_ungrounded_non_refusal,
        "ungrounded_non_refusal_rate": _safe_div(n_ungrounded_non_refusal, n_total),
        "answer_correct_rate": answer_correct_rate,
        "answer_correct_strict_rate": answer_correct_strict_rate,
        "mode": mode,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "retrieval_method": retrieval_method,
        "hybrid_alpha": hybrid_alpha,
        "prompt_version": prompt_version,
        "system_prompt_sha256": system_prompt_sha256,
        **metrics,
        **refusal_type_columns,
        **slice_columns,
        "refusal_type_breakdown": refusal_type_breakdown,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize refusal run JSONL files.")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--glob", type=str, help="Glob pattern for JSONL files")
    group.add_argument("--files", nargs="+", type=Path, help="Explicit JSONL files")

    parser.add_argument("--out-csv", type=Path, default=None)
    return parser


def _resolve_input_paths(glob_pattern: str | None, files: list[Path] | None) -> list[Path]:
    if glob_pattern:
        return sorted(Path().glob(glob_pattern))
    return [Path(path) for path in (files or [])]


def main() -> int:
    args = _build_parser().parse_args()
    paths = _resolve_input_paths(args.glob, args.files)

    if not paths:
        print("No input files found.")
        return 2

    summaries = [_summarize_file(path) for path in paths]

    fieldnames: list[str] = []
    for summary in summaries:
        for key in summary.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote: {args.out_csv}")
        return 0

    for row in summaries:
        print(
            f"{row.get('file')}\t"
            f"n={row.get('n')}\t"
            f"acc={float(row.get('accuracy', 0.0)):.3f}\t"
            f"f1={float(row.get('f1', 0.0)):.3f}\t"
            f"mcc={float(row.get('mcc', 0.0)):.3f}\t"
            f"any_hit={float(row.get('any_hit_rate', 0.0)):.3f}\t"
            f"full_hit={float(row.get('full_hit_rate', 0.0)):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())