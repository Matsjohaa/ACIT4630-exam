"""Summarize all runs in analysis/logging/ with key metrics."""
import json
from pathlib import Path

log_dir = Path(__file__).resolve().parents[1] / "logging"

print(f"{'Config':<58} {'Acc':>5} {'Prec':>5} {'Rec':>5} {'F1':>5} | {'P(any)':>7} {'P(no)':>7} {'Gap':>7}")
print("-" * 120)

for p in sorted(log_dir.glob("refusal_*.jsonl")):
    rows = [json.loads(l) for l in p.open()]
    tp = sum(1 for r in rows if r.get("model_refused") is True and r.get("should_refuse") is True)
    fp = sum(1 for r in rows if r.get("model_refused") is True and r.get("should_refuse") is False)
    tn = sum(1 for r in rows if r.get("model_refused") is False and r.get("should_refuse") is False)
    fn = sum(1 for r in rows if r.get("model_refused") is False and r.get("should_refuse") is True)
    n = tp + fp + tn + fn
    if n == 0:
        continue
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    any_hit_n = sum(1 for r in rows if r.get("any_hit") is True)
    any_hit_ref = sum(1 for r in rows if r.get("any_hit") is True and r.get("model_refused") is True)
    no_hit_n = sum(1 for r in rows if r.get("any_hit") is False)
    no_hit_ref = sum(1 for r in rows if r.get("any_hit") is False and r.get("model_refused") is True)

    p_any = any_hit_ref / any_hit_n if any_hit_n else None
    p_no = no_hit_ref / no_hit_n if no_hit_n else None
    gap = (p_no - p_any) if (p_any is not None and p_no is not None) else None

    name = p.stem.replace("refusal_", "").rsplit("_2026", 1)[0]

    cond = ""
    if p_any is not None and p_no is not None:
        cond = f" | {p_any:>7.3f} {p_no:>7.3f} {gap:>7.3f}"

    print(f"{name:<58} {acc:>5.3f} {prec:>5.3f} {rec:>5.3f} {f1:>5.3f}{cond}")
