"""Headless refusal testing for the TEK17 RAG system (no Streamlit UI).

Supports two modes:

- local: runs retrieval + embed + LLM directly (no FastAPI server needed)
- server: sends requests to a running FastAPI server (/query)

Examples:

    # Single question (local mode)
    python analysis/scripts/test_refusal.py \
        --question "Kan jeg bygge rekkverk med 80 cm høyde?" \
        --mode local

    # Batch eval (local mode)
    python analysis/scripts/test_refusal.py \
        --eval-file analysis/questions/tek17_eval_questions.example.jsonl \
        --mode local \
        --out analysis/logging/refusal_runs.example.jsonl

    # Batch eval (server mode; requires: python -m tek17 serve)
    python analysis/scripts/test_refusal.py \
        --eval-file analysis/questions/tek17_eval_questions.dibk_example.jsonl \
        --mode server \
        --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from tek17.rag.prompts import SYSTEM_PROMPT


DEFAULT_SERVER_URL = "http://localhost:8000"


Mode = Literal["local", "server"]


@dataclass(frozen=True)
class RunConfig:
    mode: Mode

    # Common
    top_k: int
    model: str | None
    temperature: float | None

    retrieval_method: str
    hybrid_alpha: float

    # Server mode
    server_url: str

    # Local mode
    chroma_dir: Path
    collection: str
    ollama_url: str
    llm_provider: Literal["ollama", "openai"]
    embed_provider: Literal["ollama", "openai"]
    llm_model: str
    embed_model: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _has_refusal_tag(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False
    # Allow the model to wrap it in quotes.
    return "kan_ikke_svare:" in text


def _classify_refusal(answer: str) -> bool:
    """Simple heuristic refusal detector.

    """

    text = (answer or "").strip().lower()
    if not text:
        return True

    if _has_refusal_tag(text):
        return True

    patterns = [
        # Norwegian-ish
        "kan ikke svare",
        "kan ikke gi et sikkert svar",
        "kan ikke gi et konkret svar",
        "har ikke nok informasjon",
        "finner ikke nok informasjon",
        "ikke nok informasjon",
        "mangler informasjon",
        "har ikke grunnlag",
        "finner ikke grunnlag",
        "utenfor det som dekkes av tek17",
        "utenfor tek17",
        # English-ish
        "i can't answer",
        "i cannot answer",
        "i don't have enough information",
        "not enough information",
        "outside the scope",
    ]
    return any(p in text for p in patterns)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (low, high). If n==0 returns (0, 0).
    """
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def _compute_refusal_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Compute standard metrics treating 'refuse' as the positive class."""
    total = tp + fp + tn + fn
    accuracy = _safe_div(tp + tn, total)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # TPR / sensitivity
    specificity = _safe_div(tn, tn + fp)  # TNR
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)

    # Matthews correlation coefficient
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    refusal_rate_pred = _safe_div(tp + fp, total)
    refusal_rate_true = _safe_div(tp + fn, total)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "refusal_rate_pred": refusal_rate_pred,
        "refusal_rate_true": refusal_rate_true,
    }


def _retrieval_hit(sources: list[dict[str, Any]], target_sections: list[str]) -> bool:
    if not target_sections:
        return False

    retrieved = {
        str((s or {}).get("section_id", "")).strip()
        for s in sources
        if str((s or {}).get("section_id", "")).strip()
    }
    targets = {str(t).strip() for t in target_sections if str(t).strip()}
    return not retrieved.isdisjoint(targets)


def _query_server(question: str, cfg: RunConfig) -> dict[str, Any]:
    import requests

    payload: dict[str, Any] = {
        "question": question,
        "top_k": cfg.top_k,
        "retrieval_method": cfg.retrieval_method,
        "hybrid_alpha": cfg.hybrid_alpha,
    }
    if cfg.model is not None:
        payload["model"] = cfg.model
    if cfg.temperature is not None:
        payload["temperature"] = cfg.temperature

    resp = requests.post(f"{cfg.server_url}/query", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _query_local(question: str, cfg: RunConfig) -> dict[str, Any]:
    from tek17.rag.embedding.client import embed_query
    from tek17.rag.llm.client import chat
    from tek17.rag.retrieval.client import get_collection, retrieve

    collection = get_collection(cfg.chroma_dir, cfg.collection)

    q_embedding: list[float] | None = None
    if cfg.retrieval_method in {"dense", "hybrid"}:
        q_embedding = embed_query(
            question,
            provider=cfg.embed_provider,
            model=cfg.embed_model,
            base_url=cfg.ollama_url,
        )

    from tek17.rag.config import CHUNKS_PATH

    documents, metadatas, distances = retrieve(
        collection=collection,
        query_text=question,
        query_embedding=q_embedding,
        top_k=cfg.top_k,
        method=cfg.retrieval_method,
        chunks_path=CHUNKS_PATH,
        hybrid_alpha=cfg.hybrid_alpha,
    )

    sources: list[dict[str, Any]] = []
    context_parts: list[str] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        context_parts.append(doc)
        sources.append(
            {
                "section_id": meta.get("section_id", ""),
                "title": meta.get("title", ""),
                "chapter": meta.get("chapter", ""),
                "text_type": meta.get("text_type", ""),
                "text": doc,
                "distance": dist,
            }
        )

    context_block = "\n\n---\n\n".join(context_parts)

    user_msg = (
        f"Kontekst fra TEK17:\n\n{context_block}\n\n"
        f"---\n\nSpørsmål: {question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    answer = chat(
        messages,
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        base_url=cfg.ollama_url,
        temperature=cfg.temperature if cfg.temperature is not None else 0.3,
    )

    return {
        "answer": answer,
        "sources": sources,
        "model": cfg.llm_model,
        "question": question,
        "retrieval_method": cfg.retrieval_method,
    }


def _run_query(question: str, cfg: RunConfig) -> dict[str, Any]:
    if cfg.mode == "server":
        return _query_server(question, cfg)
    return _query_local(question, cfg)


def _print_sources(sources: list[dict[str, Any]], max_sources: int = 6) -> None:
    shown = sources[:max_sources]
    for i, s in enumerate(shown, start=1):
        section_id = (s or {}).get("section_id", "")
        title = (s or {}).get("title", "")
        chapter = (s or {}).get("chapter", "")
        text_type = (s or {}).get("text_type", "")
        distance = (s or {}).get("distance", None)
        label = f"{section_id} – {title}".strip(" –")
        meta = " | ".join([p for p in [text_type, chapter] if p])
        dist = f" (dist={distance:.4f})" if isinstance(distance, (float, int)) else ""
        if meta:
            print(f"  [{i}] {label} [{meta}]{dist}")
        else:
            print(f"  [{i}] {label}{dist}")


def run_single(question: str, cfg: RunConfig) -> int:
    try:
        data = _run_query(question, cfg)
    except Exception as exc:
        print(f"ERROR: query failed: {exc}")
        return 2

    answer = data.get("answer", "")
    sources = data.get("sources", []) or []
    refused = _classify_refusal(answer)

    print(f"mode: {cfg.mode}")
    print(f"refused: {refused}")
    print()
    print(answer)

    if sources:
        print()
        print("sources:")
        _print_sources(sources)

    return 0


def run_eval(eval_file: Path, cfg: RunConfig, out: Path | None) -> int:
    if not eval_file.exists():
        print(f"ERROR: eval file not found: {eval_file}")
        return 2

    items = _load_jsonl(eval_file)
    if not items:
        print("ERROR: eval file is empty")
        return 2

    total = 0
    correct = 0

    tp_refuse = fp_refuse = tn_refuse = fn_refuse = 0
    # Stratified by retrieval hit
    tp_hit = fp_hit = tn_hit = fn_hit = 0
    tp_miss = fp_miss = tn_miss = fn_miss = 0

    rows_out: list[dict[str, Any]] = []

    print(f"mode: {cfg.mode}")
    if cfg.mode == "server":
        print(f"server_url: {cfg.server_url}")
    else:
        print(f"chroma_dir: {cfg.chroma_dir}")
        print(f"collection: {cfg.collection}")
    print("id\tshould_refuse\tmodel_refused\tretrieval_hit\tstatus")

    for item in items:
        qid = str(item.get("id", ""))
        question = str(item.get("question", "")).strip()
        target_sections = item.get("target_sections") or []
        should_refuse = bool(item.get("should_refuse", False))

        if not question:
            continue

        total += 1

        try:
            data = _run_query(question, cfg)
        except Exception as exc:
            print(f"{qid}\t{should_refuse}\tERROR\t0\tquery_failed: {exc}")
            continue

        answer = data.get("answer", "")
        sources = data.get("sources", []) or []

        retrieval_hit = _retrieval_hit(sources, list(target_sections) if isinstance(target_sections, list) else [])
        refused_tagged = _has_refusal_tag(answer)
        refused_heuristic = _classify_refusal(answer)
        model_refused = refused_tagged or refused_heuristic

        if should_refuse and model_refused:
            tp_refuse += 1
            status = "correct_refusal"
        elif (not should_refuse) and (not model_refused):
            tn_refuse += 1
            status = "correct_answer"
        elif (not should_refuse) and model_refused:
            fp_refuse += 1
            status = "over_refusal"
        else:
            fn_refuse += 1
            status = "under_refusal"

        if retrieval_hit:
            if should_refuse and model_refused:
                tp_hit += 1
            elif (not should_refuse) and (not model_refused):
                tn_hit += 1
            elif (not should_refuse) and model_refused:
                fp_hit += 1
            else:
                fn_hit += 1
        else:
            if should_refuse and model_refused:
                tp_miss += 1
            elif (not should_refuse) and (not model_refused):
                tn_miss += 1
            elif (not should_refuse) and model_refused:
                fp_miss += 1
            else:
                fn_miss += 1

        if (should_refuse and model_refused) or ((not should_refuse) and (not model_refused)):
            correct += 1

        print(f"{qid}\t{should_refuse}\t{model_refused}\t{int(retrieval_hit)}\t{status}")

        if out is not None:
            rows_out.append(
                {
                    "id": qid,
                    "question": question,
                    "should_refuse": should_refuse,
                    "model_refused": model_refused,
                    "refused_tagged": refused_tagged,
                    "refused_heuristic": refused_heuristic,
                    "status": status,
                    "retrieval_hit": retrieval_hit,
                    "target_sections": target_sections,
                    "answer": answer,
                    "sources": sources,
                    "mode": cfg.mode,
                    "top_k": cfg.top_k,
                    "retrieval_method": cfg.retrieval_method,
                    "hybrid_alpha": cfg.hybrid_alpha,
                    "model": data.get("model", cfg.model or cfg.llm_model),
                    "temperature": cfg.temperature,
                }
            )

    if total:
        accuracy = correct / total
        acc_lo, acc_hi = _wilson_ci(correct, total)
        metrics = _compute_refusal_metrics(tp_refuse, fp_refuse, tn_refuse, fn_refuse)
        print()
        print(f"Total eval items: {total}")
        print(f"Refusal behaviour accuracy: {accuracy:.3f} (95% CI {acc_lo:.3f}–{acc_hi:.3f})")
        print("Confusion matrix (refusal vs. label):")
        print(f"  TP (correct refusals)      : {tp_refuse}")
        print(f"  FP (over-refusals)         : {fp_refuse}")
        print(f"  TN (correct answers)       : {tn_refuse}")
        print(f"  FN (under-refusals)        : {fn_refuse}")

        print()
        print("Metrics (refusal = positive class):")
        print(f"  Precision (PPV)            : {metrics['precision']:.3f}")
        print(f"  Recall (TPR/sensitivity)   : {metrics['recall']:.3f}")
        print(f"  Specificity (TNR)          : {metrics['specificity']:.3f}")
        print(f"  F1                         : {metrics['f1']:.3f}")
        print(f"  Balanced accuracy          : {metrics['balanced_accuracy']:.3f}")
        print(f"  MCC                        : {metrics['mcc']:.3f}")
        print(f"  Refusal rate (pred / true) : {metrics['refusal_rate_pred']:.3f} / {metrics['refusal_rate_true']:.3f}")

        # Stratified metrics can help distinguish retrieval issues from refusal logic.
        n_hit = tp_hit + fp_hit + tn_hit + fn_hit
        n_miss = tp_miss + fp_miss + tn_miss + fn_miss
        if n_hit and n_miss:
            m_hit = _compute_refusal_metrics(tp_hit, fp_hit, tn_hit, fn_hit)
            m_miss = _compute_refusal_metrics(tp_miss, fp_miss, tn_miss, fn_miss)
            print()
            print("Breakdown by retrieval_hit:")
            print(f"  retrieval_hit=1 (n={n_hit}) acc={m_hit['accuracy']:.3f} f1={m_hit['f1']:.3f} mcc={m_hit['mcc']:.3f}")
            print(f"  retrieval_hit=0 (n={n_miss}) acc={m_miss['accuracy']:.3f} f1={m_miss['f1']:.3f} mcc={m_miss['mcc']:.3f}")

    if out is not None:
        _write_jsonl(out, rows_out)
        print()
        print(f"Wrote detailed results: {out}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless refusal testing for TEK17 RAG (no UI).",
    )

    parser.add_argument("--mode", choices=["local", "server"], default="local")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", type=str, help="Single question to probe")
    group.add_argument("--eval-file", type=Path, help="Eval JSONL (see analysis/questions/)")

    parser.add_argument("--top-k", type=int, default=6)

    parser.add_argument(
        "--retrieval-method",
        choices=["dense", "sparse", "sparce", "hybrid"],
        default="dense",
        help="Retrieval method for local mode.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Hybrid weighting: alpha*dense + (1-alpha)*sparse.",
    )

    # Common overrides
    parser.add_argument("--model", type=str, default=None, help="Override model (server mode only)")
    parser.add_argument("--temperature", type=float, default=None)

    # Server mode
    parser.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL)

    # Local mode
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/vectorstore/chroma"))
    parser.add_argument("--collection", type=str, default="tek17")

    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--embed-provider", choices=["ollama", "openai"], default="ollama")

    parser.add_argument("--llm-model", type=str, default="llama3.2")
    parser.add_argument("--embed-model", type=str, default="nomic-embed-text")

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write detailed per-item results as JSONL (batch mode only)",
    )

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    retrieval_method = str(args.retrieval_method).strip().lower()
    if retrieval_method == "sparce":
        retrieval_method = "sparse"

    cfg = RunConfig(
        mode=args.mode,
        top_k=args.top_k,
        model=args.model,
        temperature=args.temperature,
        retrieval_method=retrieval_method,
        hybrid_alpha=float(args.hybrid_alpha),
        server_url=args.server_url,
        chroma_dir=args.chroma_dir,
        collection=args.collection,
        ollama_url=args.ollama_url,
        llm_provider=args.llm_provider,
        embed_provider=args.embed_provider,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
    )

    if args.question is not None:
        question = str(args.question).strip()
        if not question:
            print("ERROR: --question was provided but is empty")
            return 2
        return run_single(question, cfg)

    # eval-file
    if cfg.mode == "server" and args.out is not None:
        # Still fine; just clarify in output that this is from the server.
        pass

    if args.eval_file is None:
        print("ERROR: --eval-file is required in eval mode")
        return 2

    return run_eval(args.eval_file, cfg, out=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
