import json
from pathlib import Path
import csv

log_dir = Path("analysis/logging")
rows_out = []

for path in sorted(log_dir.glob("*.jsonl")):
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

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)

            if i == 0:
                model = row.get("model")
                retrieval = row.get("retrieval_method")
                top_k = row.get("top_k")
                temperature = row.get("temperature")
                mode = row.get("mode")
                hybrid_alpha = row.get("hybrid_alpha")

            hit = row.get("retrieval_hit")
            refused = row.get("model_refused")

            if hit is None or refused is None:
                continue

            if hit:
                hit_total += 1
                if refused:
                    hit_refuse += 1
            else:
                nohit_total += 1
                if refused:
                    nohit_refuse += 1

    if hit_total == 0 and nohit_total == 0:
        continue

    p_hit = hit_refuse / hit_total if hit_total else None
    p_nohit = nohit_refuse / nohit_total if nohit_total else None
    gap = (p_nohit - p_hit) if (p_hit is not None and p_nohit is not None) else None

    rows_out.append({
        "file": path.name,
        "mode": mode,
        "model": model,
        "retrieval_method": retrieval,
        "top_k": top_k,
        "temperature": temperature,
        "hybrid_alpha": hybrid_alpha,
        "hit_n": hit_total,
        "hit_refuse_n": hit_refuse,
        "no_hit_n": nohit_total,
        "no_hit_refuse_n": nohit_refuse,
        "P(refusal|hit)": round(p_hit, 3) if p_hit is not None else None,
        "P(refusal|no_hit)": round(p_nohit, 3) if p_nohit is not None else None,
        "gap_nohit_minus_hit": round(gap, 3) if gap is not None else None,
    })

rows_out = sorted(
    rows_out,
    key=lambda x: (
        str(x["model"]),
        str(x["retrieval_method"]),
        x["top_k"] if x["top_k"] is not None else -1,
    ),
)

print("\n=== Conditional refusal (all runs) ===\n")
for r in rows_out:
    extra = ""
    if r["retrieval_method"] == "hybrid" and r["hybrid_alpha"] is not None:
        extra = f" (alpha={r['hybrid_alpha']})"

    print(
        f"{str(r['model']):15} | "
        f"{str(r['retrieval_method']):7}{extra:12} | "
        f"P(hit)={r['P(refusal|hit)']:.3f} | "
        f"P(no_hit)={r['P(refusal|no_hit)']:.3f} | "
        f"gap={r['gap_nohit_minus_hit']:.3f}"
    )

out_csv = log_dir / "conditional_refusal_summary.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    writer.writeheader()
    writer.writerows(rows_out)

print(f"\nSaved summary to: {out_csv}")