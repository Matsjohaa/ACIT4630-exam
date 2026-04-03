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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from tek17.rag.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_SHA256
from tek17.rag.config import PROMPT_VERSION


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
    # Allow the model to wrap it in quotes and/or omit the colon.
    return "kan_ikke_svare" in text


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


_STOPWORDS = {
    "og",
    "eller",
    "som",
    "med",
    "for",
    "til",
    "av",
    "på",
    "i",
    "jf",
    "kapittel",
    "paragraf",
    "ledd",
    "bokstav",
    "gjelder",
    "skal",
    "kan",
    "må",
    "ikke",
    "det",
    "den",
    "de",
    "et",
    "en",
    "er",
    "å",
    "når",
    "hva",
    "hvordan",
    "hvilke",
    "hvilken",
    "hvor",
    "jeg",
    "du",
    "vi",
    "man",
    "teK17".lower(),
}


def _content_words(text: str) -> set[str]:
    t = (text or "").lower()
    # Keep letters (incl. Norwegian) and digits.
    t = re.sub(r"[^0-9a-zæøå]+", " ", t)
    words = {w for w in t.split() if len(w) >= 4 and w not in _STOPWORDS}
    return words


def _groundedness_score(answer: str, sources: list[dict[str, Any]]) -> float:
    """Heuristic grounding score in [0,1].

    Measures overlap between answer content-words and retrieved context text.
    This is intentionally lightweight: it won't prove correctness, but it *will*
    catch many cases where the model answers confidently without using the context.
    """

    a_words = _content_words(answer)
    if not a_words:
        return 0.0

    ctx_parts: list[str] = []
    for s in sources or []:
        txt = (s or {}).get("text")
        if isinstance(txt, str) and txt.strip():
            ctx_parts.append(txt)

    if not ctx_parts:
        return 0.0

    c_words = _content_words("\n".join(ctx_parts))
    if not c_words:
        return 0.0

    overlap = len(a_words.intersection(c_words))
    return overlap / max(1, len(a_words))


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
    from tek17.rag.llm.client import chat_result
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

    result = chat_result(
        messages,
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        base_url=cfg.ollama_url,
        temperature=cfg.temperature if cfg.temperature is not None else 0.3,
    )
    answer = result.content

    return {
        "answer": answer,
        "sources": sources,
        "model": cfg.llm_model,
        "question": question,
        "retrieval_method": cfg.retrieval_method,
        "llm_finish_reason": result.finish_reason,
        "llm_usage": result.usage,
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

    # Lightweight answer correctness rubric (see below):
    # - For should_refuse items: correct iff model_refused
    # - For in-scope items: correct iff model did NOT refuse AND retrieved at least one target section
    #   (this intentionally flags "non-refusal hallucinations" where the model answers without support)
    correct_answer = 0
    incorrect_answer = 0
    unsupported_non_refusal = 0
    ungrounded_non_refusal = 0
    correct_answer_strict = 0
    incorrect_answer_strict = 0
    query_failed = 0

    tp_refuse = fp_refuse = tn_refuse = fn_refuse = 0
    # Stratified by retrieval hit
    tp_hit = fp_hit = tn_hit = fn_hit = 0
    tp_miss = fp_miss = tn_miss = fn_miss = 0

    # Split refusal questions by refusal_type (optional field on dataset items)
    refuse_type_counts: dict[str, int] = {}
    refuse_type_refused: dict[str, int] = {}

    rows_out: list[dict[str, Any]] = []

    # Vectorstore snapshot (local mode only; cached in the client)
    vs_snapshot: dict[str, Any] | None = None
    if cfg.mode == "local":
        try:
            from tek17.rag.retrieval.client import vectorstore_snapshot

            vs_snapshot = vectorstore_snapshot(cfg.chroma_dir, cfg.collection)
        except Exception:
            vs_snapshot = None
    else:
        # Best-effort: fetch count from the server
        try:
            import requests

            stats = requests.get(f"{cfg.server_url}/collection/stats", timeout=10).json()
            if isinstance(stats, dict) and "count" in stats:
                vs_snapshot = {
                    "chroma_dir": None,
                    "collection": stats.get("collection"),
                    "count": stats.get("count"),
                    "sqlite_sha256": None,
                    "sqlite_size": None,
                }
        except Exception:
            vs_snapshot = None

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
        question_type = item.get("question_type")
        if isinstance(question_type, str):
            question_type = question_type.strip() or None
        else:
            question_type = None
        refusal_type = item.get("refusal_type")
        if isinstance(refusal_type, str):
            refusal_type = refusal_type.strip() or None
        else:
            refusal_type = None

        if not question:
            continue

        total += 1

        try:
            data = _run_query(question, cfg)
        except Exception as exc:
            query_failed += 1
            incorrect_answer += 1
            print(f"{qid}\t{should_refuse}\tERROR\t0\tquery_failed: {exc}")
            if out is not None:
                rows_out.append(
                    {
                        "prompt_version": PROMPT_VERSION,
                        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
                        "id": qid,
                        "question": question,
                        "question_type": question_type,
                        "should_refuse": should_refuse,
                        "refusal_type": refusal_type,
                        "model_refused": None,
                        "refused_tagged": None,
                        "refused_heuristic": None,
                        "status": "query_failed",
                        "retrieval_hit": None,
                        "answer_correct": False,
                        "answer_correct_strict": False,
                        "answer_supported": None,
                        "unsupported_non_refusal": False,
                        "ungrounded_non_refusal": False,
                        "groundedness_score": None,
                        "answer_grounded": None,
                        "target_sections": target_sections,
                        "answer": None,
                        "sources": None,
                        "llm_finish_reason": None,
                        "llm_usage": None,
                        "error": str(exc),
                        "mode": cfg.mode,
                        "top_k": cfg.top_k,
                        "retrieval_method": cfg.retrieval_method,
                        "hybrid_alpha": cfg.hybrid_alpha,
                        "model": (cfg.model if cfg.mode == "server" else cfg.llm_model),
                        "temperature": cfg.temperature,
                        "embed_provider": (cfg.embed_provider if cfg.mode == "local" else None),
                        "embed_model": (cfg.embed_model if cfg.mode == "local" else None),
                        "vectorstore": vs_snapshot,
                    }
                )
            continue

        answer = data.get("answer", "")
        sources = data.get("sources", []) or []

        llm_finish_reason = data.get("llm_finish_reason")
        llm_usage = data.get("llm_usage")

        retrieval_hit = _retrieval_hit(sources, list(target_sections) if isinstance(target_sections, list) else [])
        refused_tagged = _has_refusal_tag(answer)
        refused_heuristic = _classify_refusal(answer)
        model_refused = refused_tagged or refused_heuristic

        status = ""

        groundedness = _groundedness_score(answer, sources)
        grounded = groundedness >= 0.25

        # Lightweight correctness/support labels.
        if should_refuse:
            answer_correct = bool(model_refused)
            answer_supported: bool | None = None
            answer_correct_strict = bool(model_refused)
        else:
            if model_refused:
                answer_correct = False
                answer_supported = None
                answer_correct_strict = False
            else:
                answer_supported = bool(retrieval_hit)
                answer_correct = bool(retrieval_hit)
                # Strict: require both retrieval_hit AND that the answer appears grounded in retrieved text.
                answer_correct_strict = bool(retrieval_hit) and bool(grounded)

        if (not should_refuse) and (not model_refused) and retrieval_hit and (not grounded):
            ungrounded_non_refusal += 1
            status = "ungrounded_answer"

        if (not should_refuse) and (not model_refused) and (not retrieval_hit):
            unsupported_non_refusal += 1
            status = "unsupported_answer"

        if should_refuse and model_refused:
            tp_refuse += 1
            status = "correct_refusal"
        elif (not should_refuse) and (not model_refused):
            tn_refuse += 1
            # If we already flagged this as unsupported, keep that.
            if status != "unsupported_answer":
                status = "correct_answer"
        elif (not should_refuse) and model_refused:
            fp_refuse += 1
            status = "over_refusal"
        else:
            fn_refuse += 1
            status = "under_refusal"

        # Track refusal behaviour by refusal_type.
        if should_refuse:
            key = refusal_type or "(unspecified)"
            refuse_type_counts[key] = refuse_type_counts.get(key, 0) + 1
            if model_refused:
                refuse_type_refused[key] = refuse_type_refused.get(key, 0) + 1

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

        if answer_correct:
            correct_answer += 1
        else:
            incorrect_answer += 1

        if answer_correct_strict:
            correct_answer_strict += 1
        else:
            incorrect_answer_strict += 1

        print(f"{qid}\t{should_refuse}\t{model_refused}\t{int(retrieval_hit)}\t{status}")

        if out is not None:
            rows_out.append(
                {
                    "prompt_version": PROMPT_VERSION,
                    "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
                    "id": qid,
                    "question": question,
                    "question_type": question_type,
                    "should_refuse": should_refuse,
                    "refusal_type": refusal_type,
                    "model_refused": model_refused,
                    "refused_tagged": refused_tagged,
                    "refused_heuristic": refused_heuristic,
                    "status": status,
                    "retrieval_hit": retrieval_hit,
                    "answer_correct": answer_correct,
                    "answer_correct_strict": answer_correct_strict,
                    "answer_supported": answer_supported,
                    "unsupported_non_refusal": (not should_refuse) and (not model_refused) and (not retrieval_hit),
                    "ungrounded_non_refusal": (not should_refuse) and (not model_refused) and bool(retrieval_hit) and (not grounded),
                    "groundedness_score": groundedness,
                    "answer_grounded": grounded if (not should_refuse) and (not model_refused) else None,
                    "target_sections": target_sections,
                    "answer": answer,
                    "sources": sources,
                    "llm_finish_reason": llm_finish_reason,
                    "llm_usage": llm_usage,
                    "mode": cfg.mode,
                    "top_k": cfg.top_k,
                    "retrieval_method": cfg.retrieval_method,
                    "hybrid_alpha": cfg.hybrid_alpha,
                    "model": data.get("model", cfg.model or cfg.llm_model),
                    "temperature": cfg.temperature,
                    "embed_provider": (cfg.embed_provider if cfg.mode == "local" else None),
                    "embed_model": (cfg.embed_model if cfg.mode == "local" else None),
                    "vectorstore": vs_snapshot,
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

        print()
        correctness_acc = (correct_answer / total) if total else 0.0
        corr_lo, corr_hi = _wilson_ci(correct_answer, total)
        print("Lightweight answer correctness (support-based):")
        print(f"  accuracy                  : {correctness_acc:.3f} (95% CI {corr_lo:.3f}–{corr_hi:.3f})")
        print(f"  unsupported_non_refusals   : {unsupported_non_refusal} ({(unsupported_non_refusal/total):.3f} of all items)")
        print(f"  ungrounded_non_refusals    : {ungrounded_non_refusal} ({(ungrounded_non_refusal/total):.3f} of all items)")
        print(f"  query_failed               : {query_failed} ({(query_failed/total):.3f} of all items)")

        strict_acc = (correct_answer_strict / total) if total else 0.0
        strict_lo, strict_hi = _wilson_ci(correct_answer_strict, total)
        print("Strict answer correctness (retrieval_hit + groundedness):")
        print(f"  accuracy                  : {strict_acc:.3f} (95% CI {strict_lo:.3f}–{strict_hi:.3f})")

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

        if refuse_type_counts:
            print()
            print("Refusal questions by refusal_type:")
            for k in sorted(refuse_type_counts.keys()):
                n_k = refuse_type_counts[k]
                refused_k = refuse_type_refused.get(k, 0)
                rate_k = (refused_k / n_k) if n_k else 0.0
                print(f"  {k}: n={n_k} refused={refused_k} rate={rate_k:.3f}")

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
