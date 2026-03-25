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

    for r in rows:
        should_refuse = bool(r.get("should_refuse"))
        model_refused = bool(r.get("model_refused"))

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

    metrics = _compute_metrics(tp, fp, tn, fn)

    return {
        "file": str(path),
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "retrieval_hit_rate": (retrieval_hits / len(rows)) if rows else 0.0,
        "mode": mode,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "retrieval_method": retrieval_method,
        "hybrid_alpha": hybrid_alpha,
        **metrics,
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
