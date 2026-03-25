"""Compare two refusal-eval JSONL runs produced by `analysis/scripts/test_refusal.py`.

Example:

    /path/to/python analysis/scripts/compare_refusal_runs.py \
        --a analysis/logging/refusal_ollama_llama3.2_t0_top6.jsonl \
        --b analysis/logging/refusal_openai_gpt-4.1-mini_t0_top6.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metrics:
    total: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    refusal_rate_pred: float


def _load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("id", "")).strip()
            if not rid:
                continue
            rows[rid] = row
    return rows


def _confusion(rows: dict[str, dict]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for r in rows.values():
        y = bool(r.get("should_refuse", False))
        yhat = bool(r.get("model_refused", False))
        if y and yhat:
            tp += 1
        elif (not y) and yhat:
            fp += 1
        elif (not y) and (not yhat):
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _metrics(tp: int, fp: int, tn: int, fn: int) -> Metrics:
    total = tp + fp + tn + fn

    def safe(num: float, den: float) -> float:
        return num / den if den else 0.0

    accuracy = safe(tp + tn, total)
    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    f1 = safe(2.0 * precision * recall, precision + recall)
    refusal_rate_pred = safe(tp + fp, total)

    return Metrics(
        total=total,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        refusal_rate_pred=refusal_rate_pred,
    )


def _fmt(m: Metrics) -> str:
    return (
        f"n={m.total} acc={m.accuracy:.3f} prec={m.precision:.3f} "
        f"rec={m.recall:.3f} spec={m.specificity:.3f} f1={m.f1:.3f} "
        f"refuse_pred={m.refusal_rate_pred:.3f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two refusal-eval JSONL runs")
    p.add_argument("--a", type=Path, required=True, help="Run A JSONL")
    p.add_argument("--b", type=Path, required=True, help="Run B JSONL")
    p.add_argument("--show", type=int, default=50, help="Max flips to print")
    args = p.parse_args()

    a = _load(args.a)
    b = _load(args.b)

    common = sorted(set(a).intersection(b))
    if not common:
        raise SystemExit("No common ids between the two runs.")

    a_c = {k: a[k] for k in common}
    b_c = {k: b[k] for k in common}

    ma = _metrics(*_confusion(a_c))
    mb = _metrics(*_confusion(b_c))

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print()
    print("A:", _fmt(ma))
    print("B:", _fmt(mb))

    flips = [
        k
        for k in common
        if bool(a_c[k].get("model_refused", False)) != bool(b_c[k].get("model_refused", False))
    ]

    print()
    print(f"Flips (A != B): {len(flips)}")

    shown = flips[: max(0, args.show)]
    for k in shown:
        y = bool(a_c[k].get("should_refuse", False))
        ar = bool(a_c[k].get("model_refused", False))
        br = bool(b_c[k].get("model_refused", False))
        q = str(a_c[k].get("question", "")).strip() or str(b_c[k].get("question", "")).strip()
        print(f"- {k} should_refuse={int(y)} A_refused={int(ar)} B_refused={int(br)} :: {q}")

    if len(flips) > len(shown):
        print(f"... ({len(flips) - len(shown)} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
