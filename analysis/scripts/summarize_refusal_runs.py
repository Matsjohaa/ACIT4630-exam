"""Summarize one or more refusal-eval JSONL runs into CSV.

Input files are the JSONL logs written by `analysis/scripts/test_refusal.py`.

Example:
  python analysis/scripts/summarize_refusal_runs.py \
    --glob 'analysis/logging/refusal_openai_gpt-5.1_*.jsonl' \
    --out-csv analysis/logging/refusal_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _infer_question_type(r: dict) -> str:
    qt = r.get("question_type")
    if isinstance(qt, str) and qt.strip():
        return qt.strip()

    # Fallback inference for older logs
    if bool(r.get("should_refuse")):
        return "refusal"

    target_sections = r.get("target_sections") or []
    if isinstance(target_sections, list) and len(target_sections) >= 2:
        return "in_scope_multi"

    rid = r.get("id")
    if isinstance(rid, str) and "_multi_" in rid:
        return "in_scope_multi"
    return "in_scope_single"


def _compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    # Matthews correlation coefficient
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / (denom ** 0.5) if denom else 0.0

    refusal_rate_pred = (tp + fp) / total if total else 0.0
    refusal_rate_true = (tp + fn) / total if total else 0.0

    bal_acc = (recall + specificity) / 2.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "refusal_rate_pred": refusal_rate_pred,
        "refusal_rate_true": refusal_rate_true,
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _summarize_file(path: Path) -> dict[str, object]:
    rows = _load_jsonl(path)
    if not rows:
        return {
            "file": str(path),
            "n": 0,
        }

    # Pull run-level fields from the first row (they should be constant per run)
    first = rows[0]
    retrieval_method = str(first.get("retrieval_method", ""))
    if retrieval_method == "sparce":
        retrieval_method = "sparse"

    top_k = first.get("top_k")
    hybrid_alpha = first.get("hybrid_alpha")
    model = first.get("model")
    temperature = first.get("temperature")
    mode = first.get("mode")

    tp = fp = tn = fn = 0
    retrieval_hits = 0

    n_total = len(rows)
    n_errors = 0

    # Lightweight correctness/support (optional)
    n_answer_correct = 0
    n_answer_correct_den = 0
    n_answer_correct_strict = 0
    n_answer_correct_strict_den = 0
    n_unsupported_non_refusal = 0
    n_ungrounded_non_refusal = 0

    # Slice: in-scope single vs multi
    slice_counts: dict[str, int] = {"in_scope_single": 0, "in_scope_multi": 0}
    slice_over_refusal: dict[str, int] = {"in_scope_single": 0, "in_scope_multi": 0}
    slice_unsupported: dict[str, int] = {"in_scope_single": 0, "in_scope_multi": 0}
    slice_ungrounded: dict[str, int] = {"in_scope_single": 0, "in_scope_multi": 0}
    slice_strict_correct: dict[str, int] = {"in_scope_single": 0, "in_scope_multi": 0}
    slice_strict_den: dict[str, int] = {"in_scope_single": 0, "in_scope_multi": 0}

    # Refusal type breakdown (optional)
    refuse_type_counts: dict[str, int] = {}
    refuse_type_refused: dict[str, int] = {}

    for r in rows:
        status = r.get("status")
        if status == "query_failed":
            n_errors += 1
            # Still count unsupported_non_refusal if present (should be False),
            # but skip refusal confusion.
            if r.get("unsupported_non_refusal"):
                n_unsupported_non_refusal += 1
            continue

        should_refuse = bool(r.get("should_refuse"))
        model_refused = bool(r.get("model_refused"))
        qtype = _infer_question_type(r)

        if should_refuse and model_refused:
            tp += 1
        elif (not should_refuse) and model_refused:
            fp += 1
        elif (not should_refuse) and (not model_refused):
            tn += 1
        else:
            fn += 1

        if r.get("retrieval_hit"):
            retrieval_hits += 1

        if r.get("unsupported_non_refusal"):
            n_unsupported_non_refusal += 1

        if r.get("ungrounded_non_refusal"):
            n_ungrounded_non_refusal += 1

        if isinstance(r.get("answer_correct"), bool):
            n_answer_correct_den += 1
            if bool(r.get("answer_correct")):
                n_answer_correct += 1

        if isinstance(r.get("answer_correct_strict"), bool):
            n_answer_correct_strict_den += 1
            if bool(r.get("answer_correct_strict")):
                n_answer_correct_strict += 1

        # Slice stats (only meaningful for in-scope questions)
        if not should_refuse and qtype in slice_counts:
            slice_counts[qtype] += 1
            if model_refused:
                slice_over_refusal[qtype] += 1
            if r.get("unsupported_non_refusal"):
                slice_unsupported[qtype] += 1
            if r.get("ungrounded_non_refusal"):
                slice_ungrounded[qtype] += 1
            if isinstance(r.get("answer_correct_strict"), bool):
                slice_strict_den[qtype] += 1
                if bool(r.get("answer_correct_strict")):
                    slice_strict_correct[qtype] += 1

        if should_refuse:
            refusal_type = r.get("refusal_type")
            key = str(refusal_type).strip() if isinstance(refusal_type, str) and str(refusal_type).strip() else "(unspecified)"
            refuse_type_counts[key] = refuse_type_counts.get(key, 0) + 1
            if model_refused:
                refuse_type_refused[key] = refuse_type_refused.get(key, 0) + 1

    metrics = _compute_metrics(tp, fp, tn, fn)

    n_ok = n_total - n_errors
    answer_correct_rate = (n_answer_correct / n_answer_correct_den) if n_answer_correct_den else None
    answer_correct_strict_rate = (
        (n_answer_correct_strict / n_answer_correct_strict_den) if n_answer_correct_strict_den else None
    )

    # Explicit refusal_type columns (commonly used types)
    refuse_types = ["out_of_scope", "in_domain_missing_context", "(unspecified)"]
    refuse_cols: dict[str, object] = {}
    for t in refuse_types:
        n_t = int(refuse_type_counts.get(t, 0))
        refused_t = int(refuse_type_refused.get(t, 0))
        refuse_cols[f"refuse_n__{t}"] = n_t
        refuse_cols[f"refuse_refused__{t}"] = refused_t
        refuse_cols[f"refuse_refused_rate__{t}"] = (refused_t / n_t) if n_t else None

    # Explicit single vs multi slice columns
    slice_cols: dict[str, object] = {}
    for s in ["in_scope_single", "in_scope_multi"]:
        n_s = slice_counts.get(s, 0)
        slice_cols[f"slice_n__{s}"] = n_s
        slice_cols[f"slice_over_refusal_rate__{s}"] = (slice_over_refusal[s] / n_s) if n_s else None
        slice_cols[f"slice_unsupported_rate__{s}"] = (slice_unsupported[s] / n_s) if n_s else None
        slice_cols[f"slice_ungrounded_rate__{s}"] = (slice_ungrounded[s] / n_s) if n_s else None
        slice_cols[f"slice_strict_correct_rate__{s}"] = (
            (slice_strict_correct[s] / slice_strict_den[s]) if slice_strict_den[s] else None
        )

    return {
        "file": str(path),
        "n": n_total,
        "n_ok": n_ok,
        "n_errors": n_errors,
        "error_rate": (n_errors / n_total) if n_total else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "retrieval_hit_rate": (retrieval_hits / n_ok) if n_ok else 0.0,
        "unsupported_non_refusal": n_unsupported_non_refusal,
        "unsupported_non_refusal_rate": (n_unsupported_non_refusal / n_total) if n_total else 0.0,
        "ungrounded_non_refusal": n_ungrounded_non_refusal,
        "ungrounded_non_refusal_rate": (n_ungrounded_non_refusal / n_total) if n_total else 0.0,
        "answer_correct_rate": answer_correct_rate,
        "answer_correct_strict_rate": answer_correct_strict_rate,
        "mode": mode,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "retrieval_method": retrieval_method,
        "hybrid_alpha": hybrid_alpha,
        **metrics,
        **refuse_cols,
        **slice_cols,
        "refusal_type_breakdown": json.dumps(
            {
                k: {
                    "n": refuse_type_counts.get(k, 0),
                    "refused": refuse_type_refused.get(k, 0),
                    "refused_rate": (refuse_type_refused.get(k, 0) / refuse_type_counts[k]) if refuse_type_counts.get(k) else 0.0,
                }
                for k in sorted(refuse_type_counts.keys())
            },
            ensure_ascii=False,
        )
        if refuse_type_counts
        else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize refusal run JSONL files.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--glob", type=str, help="Glob for JSONL files")
    g.add_argument("--files", nargs="+", type=Path, help="Explicit JSONL files")

    p.add_argument("--out-csv", type=Path, default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.glob:
        paths = sorted(Path().glob(args.glob))
    else:
        paths = [Path(p) for p in args.files]

    if not paths:
        print("No input files found.")
        return 2

    summaries = [_summarize_file(p) for p in paths]

    fieldnames: list[str] = []
    for s in summaries:
        for k in s.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in summaries:
                w.writerow(row)
        print(f"Wrote: {args.out_csv}")
    else:
        # Print a tiny table
        for row in summaries:
            print(
                f"{row.get('file')}\t"
                f"n={row.get('n')}\tacc={float(row.get('accuracy', 0.0)):.3f}\t"
                f"f1={float(row.get('f1', 0.0)):.3f}\t"
                f"mcc={float(row.get('mcc', 0.0)):.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
