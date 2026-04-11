"""Compare two refusal-eval JSONL runs produced by analysis/scripts/test_refusal.py."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metrics:
    total: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    refusal_rate_pred: float
    retrieval_hit_rate: float
    query_failed_rate: float


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                continue

            rows[row_id] = row

    return rows


def _confusion(rows: dict[str, dict[str, Any]]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0

    for row in rows.values():
        if row.get("status") == "query_failed":
            continue

        should_refuse = bool(row.get("should_refuse", False))
        model_refused = bool(row.get("model_refused", False))

        if should_refuse and model_refused:
            tp += 1
        elif (not should_refuse) and model_refused:
            fp += 1
        elif (not should_refuse) and (not model_refused):
            tn += 1
        else:
            fn += 1

    return tp, fp, tn, fn


def _metrics(rows: dict[str, dict[str, Any]]) -> Metrics:
    tp, fp, tn, fn = _confusion(rows)

    valid_rows = [row for row in rows.values() if row.get("status") != "query_failed"]
    total = tp + fp + tn + fn
    retrieval_hits = sum(1 for row in valid_rows if bool(row.get("retrieval_hit", False)))
    query_failed = sum(1 for row in rows.values() if row.get("status") == "query_failed")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    return Metrics(
        total=total,
        accuracy=_safe_div(tp + tn, total),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        refusal_rate_pred=_safe_div(tp + fp, total),
        retrieval_hit_rate=_safe_div(retrieval_hits, total),
        query_failed_rate=_safe_div(query_failed, len(rows)),
    )


def _format_metrics(metrics: Metrics) -> str:
    return (
        f"n={metrics.total} "
        f"acc={metrics.accuracy:.3f} "
        f"prec={metrics.precision:.3f} "
        f"rec={metrics.recall:.3f} "
        f"spec={metrics.specificity:.3f} "
        f"f1={metrics.f1:.3f} "
        f"refuse_pred={metrics.refusal_rate_pred:.3f} "
        f"hit_rate={metrics.retrieval_hit_rate:.3f} "
        f"query_failed={metrics.query_failed_rate:.3f}"
    )


def _format_delta(a: float, b: float) -> str:
    return f"{(b - a):+.3f}"


def _build_flip_row(
    row_id: str,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": row_id,
        "question": str(row_a.get("question", "")).strip() or str(row_b.get("question", "")).strip(),
        "should_refuse": bool(row_a.get("should_refuse", False)),
        "a_model_refused": bool(row_a.get("model_refused", False)),
        "b_model_refused": bool(row_b.get("model_refused", False)),
        "a_retrieval_hit": row_a.get("retrieval_hit"),
        "b_retrieval_hit": row_b.get("retrieval_hit"),
        "a_status": row_a.get("status"),
        "b_status": row_b.get("status"),
        "a_model": row_a.get("model"),
        "b_model": row_b.get("model"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two refusal-eval JSONL runs.")
    parser.add_argument("--a", type=Path, required=True, help="Run A JSONL")
    parser.add_argument("--b", type=Path, required=True, help="Run B JSONL")
    parser.add_argument("--show", type=int, default=50, help="Max flips to print")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSONL output file for refusal flips",
    )
    args = parser.parse_args()

    rows_a = _load_jsonl_by_id(args.a)
    rows_b = _load_jsonl_by_id(args.b)

    common_ids = sorted(set(rows_a).intersection(rows_b))
    if not common_ids:
        raise SystemExit("No common ids between the two runs.")

    aligned_a = {row_id: rows_a[row_id] for row_id in common_ids}
    aligned_b = {row_id: rows_b[row_id] for row_id in common_ids}

    metrics_a = _metrics(aligned_a)
    metrics_b = _metrics(aligned_b)

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print()
    print("A:", _format_metrics(metrics_a))
    print("B:", _format_metrics(metrics_b))
    print()
    print("Delta (B - A):")
    print(f"  accuracy         {_format_delta(metrics_a.accuracy, metrics_b.accuracy)}")
    print(f"  precision        {_format_delta(metrics_a.precision, metrics_b.precision)}")
    print(f"  recall           {_format_delta(metrics_a.recall, metrics_b.recall)}")
    print(f"  specificity      {_format_delta(metrics_a.specificity, metrics_b.specificity)}")
    print(f"  f1               {_format_delta(metrics_a.f1, metrics_b.f1)}")
    print(f"  refusal_rate     {_format_delta(metrics_a.refusal_rate_pred, metrics_b.refusal_rate_pred)}")
    print(f"  retrieval_hit    {_format_delta(metrics_a.retrieval_hit_rate, metrics_b.retrieval_hit_rate)}")
    print(f"  query_failed     {_format_delta(metrics_a.query_failed_rate, metrics_b.query_failed_rate)}")

    flips = [
        _build_flip_row(row_id, aligned_a[row_id], aligned_b[row_id])
        for row_id in common_ids
        if bool(aligned_a[row_id].get("model_refused", False))
        != bool(aligned_b[row_id].get("model_refused", False))
    ]

    retrieval_flips = [
        row_id
        for row_id in common_ids
        if aligned_a[row_id].get("retrieval_hit") != aligned_b[row_id].get("retrieval_hit")
    ]

    print()
    print(f"Refusal flips (A != B): {len(flips)}")
    print(f"Retrieval-hit flips    : {len(retrieval_flips)}")

    shown = flips[: max(0, args.show)]
    for row in shown:
        print(
            f"- {row['id']} "
            f"should_refuse={int(row['should_refuse'])} "
            f"A_refused={int(row['a_model_refused'])} "
            f"B_refused={int(row['b_model_refused'])} "
            f"A_hit={row['a_retrieval_hit']} "
            f"B_hit={row['b_retrieval_hit']} :: "
            f"{row['question']}"
        )

    if len(flips) > len(shown):
        print(f"... ({len(flips) - len(shown)} more)")

    if args.out is not None:
        _write_jsonl(args.out, flips)
        print()
        print(f"Wrote refusal flips: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())