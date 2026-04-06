import json
from pathlib import Path
import csv

log_dir = Path("analysis/logging")
rows_out = []


def classify_case(hit, should_refuse, model_refused):
    if not hit and should_refuse and model_refused:
        return "retrieval_miss_correct_refusal"
    elif hit and not should_refuse and model_refused:
        return "over_refusal_with_evidence"
    elif not hit and not should_refuse and not model_refused:
        return "answer_without_evidence"
    elif hit and not should_refuse and not model_refused:
        return "correct_answer"
    elif hit and should_refuse and model_refused:
        return "correct_refusal_with_context"
    elif hit and should_refuse and not model_refused:
        return "under_refusal_with_context"
    elif not hit and should_refuse and not model_refused:
        return "unsafe_answer_no_evidence"
    else:
        return "other"


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

    category_counts = {
        "retrieval_miss_correct_refusal": 0,
        "over_refusal_with_evidence": 0,
        "answer_without_evidence": 0,
        "correct_answer": 0,
        "correct_refusal_with_context": 0,
        "under_refusal_with_context": 0,
        "unsafe_answer_no_evidence": 0,
        "other": 0,
    }

    n_rows = 0

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
            model_refused = row.get("model_refused")
            should_refuse = row.get("should_refuse")

            if hit is None or model_refused is None or should_refuse is None:
                continue

            hit = bool(hit)
            model_refused = bool(model_refused)
            should_refuse = bool(should_refuse)

            n_rows += 1

            # Existing conditional refusal logic
            if hit:
                hit_total += 1
                if model_refused:
                    hit_refuse += 1
            else:
                nohit_total += 1
                if model_refused:
                    nohit_refuse += 1

            # New error taxonomy
            category = classify_case(hit, should_refuse, model_refused)
            category_counts[category] += 1

    if n_rows == 0:
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
        "n_rows": n_rows,
        "hit_n": hit_total,
        "hit_refuse_n": hit_refuse,
        "no_hit_n": nohit_total,
        "no_hit_refuse_n": nohit_refuse,
        "P(refusal|hit)": round(p_hit, 3) if p_hit is not None else None,
        "P(refusal|no_hit)": round(p_nohit, 3) if p_nohit is not None else None,
        "gap_nohit_minus_hit": round(gap, 3) if gap is not None else None,
        "retrieval_miss_correct_refusal": category_counts["retrieval_miss_correct_refusal"],
        "over_refusal_with_evidence": category_counts["over_refusal_with_evidence"],
        "answer_without_evidence": category_counts["answer_without_evidence"],
        "correct_answer": category_counts["correct_answer"],
        "correct_refusal_with_context": category_counts["correct_refusal_with_context"],
        "under_refusal_with_context": category_counts["under_refusal_with_context"],
        "unsafe_answer_no_evidence": category_counts["unsafe_answer_no_evidence"],
        "other": category_counts["other"],
    })

rows_out = sorted(
    rows_out,
    key=lambda x: (
        str(x["model"]),
        str(x["retrieval_method"]),
        x["top_k"] if x["top_k"] is not None else -1,
    ),
)

print("\n=== Conditional refusal + taxonomy (all runs) ===\n")
for r in rows_out:
    extra = ""
    if r["retrieval_method"] == "hybrid" and r["hybrid_alpha"] is not None:
        extra = f" (alpha={r['hybrid_alpha']})"

    p_hit_str = f"{r['P(refusal|hit)']:.3f}" if r["P(refusal|hit)"] is not None else "NA"
    p_nohit_str = f"{r['P(refusal|no_hit)']:.3f}" if r["P(refusal|no_hit)"] is not None else "NA"
    gap_str = f"{r['gap_nohit_minus_hit']:.3f}" if r["gap_nohit_minus_hit"] is not None else "NA"

    print(
        f"{str(r['model']):15} | "
        f"{str(r['retrieval_method']):7}{extra:12} | "
        f"P(hit)={p_hit_str} | "
        f"P(no_hit)={p_nohit_str} | "
        f"gap={gap_str}"
    )
    print(
        f"  taxonomy: "
        f"miss->correct_refusal={r['retrieval_miss_correct_refusal']}, "
        f"over_refusal_with_evidence={r['over_refusal_with_evidence']}, "
        f"answer_without_evidence={r['answer_without_evidence']}, "
        f"correct_answer={r['correct_answer']}, "
        f"correct_refusal_with_context={r['correct_refusal_with_context']}, "
        f"under_refusal_with_context={r['under_refusal_with_context']}, "
        f"unsafe_answer_no_evidence={r['unsafe_answer_no_evidence']}, "
        f"other={r['other']}"
    )

if rows_out:
    out_csv = log_dir / "conditional_refusal_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSaved summary to: {out_csv}")
else:
    print("No valid JSONL runs found.")