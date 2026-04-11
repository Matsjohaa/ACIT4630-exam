"""Headless refusal testing for the TEK17 RAG system."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from tek17.rag.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_STATS_TIMEOUT_S,
    CONTENT_STOPWORDS,
    CONTENT_WORD_MIN_LEN,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
    EVAL_MODE,
    GROUNDEDNESS_THRESHOLD,
    HYBRID_ALPHA,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    REFUSAL_PATTERNS,
    REFUSAL_TAG,
    REQUEST_TIMEOUT_S,
    RETRIEVAL_METHOD,
    SERVER_URL,
    SOURCE_PREVIEW_LIMIT,
    TOP_K,
    WILSON_Z,
)
from tek17.rag.ingest import embed_query
from tek17.rag.llm.dispatcher import chat_result
from tek17.rag.prompts import get_system_prompt, get_system_prompt_sha256
from tek17.rag.retrieval.client import get_collection, retrieve, vectorstore_snapshot


Mode = Literal["local", "server"]
Provider = Literal["ollama", "openai"]


@dataclass(frozen=True)
class RunConfig:
    mode: Mode
    top_k: int
    model: str | None
    temperature: float | None
    retrieval_method: str
    hybrid_alpha: float
    server_url: str
    chroma_dir: Path
    collection: str
    embed_base_url: str | None
    llm_provider: Provider
    embed_provider: Provider
    llm_model: str
    embed_model: str
    llm_base_url: str | None
    prompt_version: str


@dataclass(frozen=True)
class RetrievalCoverage:
    any_hit: bool
    full_hit: bool
    partial_hit: bool
    retrieved_sections: list[str]
    normalized_retrieved_sections: list[str]
    normalized_target_sections: list[str]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_retrieval_method(method: str) -> str:
    normalized = (method or RETRIEVAL_METHOD).strip().lower()
    if normalized == "sparce":
        return "sparse"
    return normalized


def _normalize_prompt_version(version: str) -> str:
    return (version or PROMPT_VERSION).strip().lower()


def _has_refusal_tag(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False
    return REFUSAL_TAG.lower() in text


def _classify_refusal(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return True

    if _has_refusal_tag(text):
        return True

    return any(pattern in text for pattern in REFUSAL_PATTERNS)


def _content_words(text: str) -> set[str]:
    normalized = re.sub(r"[^0-9a-zæøå]+", " ", (text or "").lower())
    return {
        word
        for word in normalized.split()
        if len(word) >= CONTENT_WORD_MIN_LEN and word not in CONTENT_STOPWORDS
    }


def _groundedness_score(answer: str, sources: list[dict[str, Any]]) -> float:
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0

    context_parts: list[str] = []
    for source in sources or []:
        text = (source or {}).get("text")
        if isinstance(text, str) and text.strip():
            context_parts.append(text)

    if not context_parts:
        return 0.0

    context_words = _content_words("\n".join(context_parts))
    if not context_words:
        return 0.0

    overlap = len(answer_words.intersection(context_words))
    return overlap / max(1, len(answer_words))


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _wilson_ci(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0

    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(
        (phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n)
    )
    return max(0.0, center - half), min(1.0, center + half)


def _compute_refusal_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

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


def _normalize_section_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = normalized.replace("§", "")
    normalized = " ".join(normalized.split())
    return normalized


def _normalize_section_set(sections: Iterable[str]) -> set[str]:
    return {
        _normalize_section_id(section)
        for section in sections
        if str(section).strip()
    }


def _retrieval_coverage(
    sources: list[dict[str, Any]],
    target_sections: list[str],
) -> RetrievalCoverage:
    retrieved_sections = [
        str((source or {}).get("section_id", "")).strip()
        for source in sources
        if str((source or {}).get("section_id", "")).strip()
    ]
    normalized_retrieved = _normalize_section_set(retrieved_sections)
    normalized_targets = _normalize_section_set(target_sections)

    if not normalized_targets:
        return RetrievalCoverage(
            any_hit=False,
            full_hit=False,
            partial_hit=False,
            retrieved_sections=retrieved_sections,
            normalized_retrieved_sections=sorted(normalized_retrieved),
            normalized_target_sections=sorted(normalized_targets),
        )

    any_hit = not normalized_retrieved.isdisjoint(normalized_targets)
    full_hit = normalized_targets.issubset(normalized_retrieved)
    partial_hit = any_hit and not full_hit

    return RetrievalCoverage(
        any_hit=any_hit,
        full_hit=full_hit,
        partial_hit=partial_hit,
        retrieved_sections=retrieved_sections,
        normalized_retrieved_sections=sorted(normalized_retrieved),
        normalized_target_sections=sorted(normalized_targets),
    )


def _query_server(question: str, cfg: RunConfig) -> dict[str, Any]:
    import requests

    payload: dict[str, Any] = {
        "question": question,
        "top_k": cfg.top_k,
        "retrieval_method": cfg.retrieval_method,
        "hybrid_alpha": cfg.hybrid_alpha,
        "prompt_version": cfg.prompt_version,
    }
    if cfg.model is not None:
        payload["model"] = cfg.model
    if cfg.temperature is not None:
        payload["temperature"] = cfg.temperature

    response = requests.post(
        f"{cfg.server_url}/query",
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def _query_local(question: str, cfg: RunConfig) -> dict[str, Any]:
    collection = get_collection(cfg.chroma_dir, cfg.collection)

    query_embedding: list[float] | None = None
    if cfg.retrieval_method in {"dense", "hybrid"}:
        query_embedding = embed_query(
            question,
            provider=cfg.embed_provider,
            model=cfg.embed_model,
            base_url=cfg.embed_base_url,
        )

    documents, metadatas, distances = retrieve(
        collection=collection,
        query_text=question,
        query_embedding=query_embedding,
        top_k=cfg.top_k,
        method=cfg.retrieval_method,
        chunks_path=CHUNKS_PATH,
        hybrid_alpha=cfg.hybrid_alpha,
    )

    sources: list[dict[str, Any]] = []
    context_parts: list[str] = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        context_parts.append(document)
        sources.append(
            {
                "section_id": metadata.get("section_id", ""),
                "title": metadata.get("title", ""),
                "chapter": metadata.get("chapter", ""),
                "text_type": metadata.get("text_type", ""),
                "text": document,
                "distance": distance,
            }
        )

    context_block = "\n\n---\n\n".join(context_parts)
    user_message = (
        f"Kontekst fra TEK17:\n\n{context_block}\n\n"
        f"---\n\nSpørsmål: {question}"
    )
    system_prompt = get_system_prompt(cfg.prompt_version)

    result = chat_result(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        temperature=cfg.temperature if cfg.temperature is not None else LLM_TEMPERATURE,
    )

    return {
        "answer": result.content,
        "sources": sources,
        "model": cfg.llm_model,
        "question": question,
        "retrieval_method": cfg.retrieval_method,
        "llm_finish_reason": result.finish_reason,
        "llm_usage": result.usage,
        "prompt_version": cfg.prompt_version,
    }


def _run_query(question: str, cfg: RunConfig) -> dict[str, Any]:
    if cfg.mode == "server":
        return _query_server(question, cfg)
    return _query_local(question, cfg)


def _print_sources(
    sources: list[dict[str, Any]],
    max_sources: int = SOURCE_PREVIEW_LIMIT,
) -> None:
    for index, source in enumerate(sources[:max_sources], start=1):
        section_id = (source or {}).get("section_id", "")
        title = (source or {}).get("title", "")
        chapter = (source or {}).get("chapter", "")
        text_type = (source or {}).get("text_type", "")
        distance = (source or {}).get("distance")

        label = f"{section_id} – {title}".strip(" –")
        meta = " | ".join(part for part in [text_type, chapter] if part)
        distance_text = f" (dist={distance:.4f})" if isinstance(distance, (float, int)) else ""

        if meta:
            print(f"  [{index}] {label} [{meta}]{distance_text}")
        else:
            print(f"  [{index}] {label}{distance_text}")


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
    print(f"prompt_version: {cfg.prompt_version}")
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

    prompt_sha256 = get_system_prompt_sha256(cfg.prompt_version)
    items = _load_jsonl(eval_file)
    if not items:
        print("ERROR: eval file is empty")
        return 2

    total = 0
    correct = 0
    correct_answer = 0
    correct_answer_strict = 0
    unsupported_non_refusal = 0
    partial_support_non_refusal = 0
    ungrounded_non_refusal = 0
    query_failed = 0

    tp_refuse = fp_refuse = tn_refuse = fn_refuse = 0

    tp_any = fp_any = tn_any = fn_any = 0
    tp_full = fp_full = tn_full = fn_full = 0
    tp_none = fp_none = tn_none = fn_none = 0
    tp_partial = fp_partial = tn_partial = fn_partial = 0

    refusal_type_counts: dict[str, int] = {}
    refusal_type_refused: dict[str, int] = {}
    rows_out: list[dict[str, Any]] = []

    vs_snapshot: dict[str, Any] | None = None
    if cfg.mode == "local":
        try:
            vs_snapshot = vectorstore_snapshot(cfg.chroma_dir, cfg.collection)
        except Exception:
            vs_snapshot = None
    else:
        try:
            import requests

            stats = requests.get(
                f"{cfg.server_url}/collection/stats",
                timeout=COLLECTION_STATS_TIMEOUT_S,
            ).json()
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
    print(f"prompt_version: {cfg.prompt_version}")
    if cfg.mode == "server":
        print(f"server_url: {cfg.server_url}")
    else:
        print(f"chroma_dir: {cfg.chroma_dir}")
        print(f"collection: {cfg.collection}")
    print("id\tshould_refuse\tmodel_refused\tany_hit\tfull_hit\tpartial_hit\tstatus")

    for item in items:
        question_id = str(item.get("id", ""))
        question = str(item.get("question", "")).strip()
        target_sections = item.get("target_sections") or []
        should_refuse = bool(item.get("should_refuse", False))

        question_type = item.get("question_type")
        question_type = question_type.strip() if isinstance(question_type, str) and question_type.strip() else None

        refusal_type = item.get("refusal_type")
        refusal_type = refusal_type.strip() if isinstance(refusal_type, str) and refusal_type.strip() else None

        if not question:
            continue

        total += 1

        try:
            data = _run_query(question, cfg)
        except Exception as exc:
            query_failed += 1
            print(f"{question_id}\t{should_refuse}\tERROR\t0\t0\t0\tquery_failed: {exc}")

            if out is not None:
                rows_out.append(
                    {
                        "prompt_version": cfg.prompt_version,
                        "system_prompt_sha256": prompt_sha256,
                        "id": question_id,
                        "question": question,
                        "question_type": question_type,
                        "should_refuse": should_refuse,
                        "refusal_type": refusal_type,
                        "model_refused": None,
                        "refused_tagged": None,
                        "refused_heuristic": None,
                        "status": "query_failed",
                        "retrieval_hit": None,
                        "any_hit": None,
                        "full_hit": None,
                        "partial_hit": None,
                        "answer_correct": False,
                        "answer_correct_strict": False,
                        "answer_supported": None,
                        "unsupported_non_refusal": False,
                        "partial_support_non_refusal": False,
                        "ungrounded_non_refusal": False,
                        "groundedness_score": None,
                        "answer_grounded": None,
                        "target_sections": target_sections,
                        "normalized_target_sections": None,
                        "normalized_retrieved_sections": None,
                        "answer": None,
                        "sources": None,
                        "llm_finish_reason": None,
                        "llm_usage": None,
                        "error": str(exc),
                        "mode": cfg.mode,
                        "top_k": cfg.top_k,
                        "retrieval_method": cfg.retrieval_method,
                        "hybrid_alpha": cfg.hybrid_alpha,
                        "model": cfg.model if cfg.mode == "server" else cfg.llm_model,
                        "temperature": cfg.temperature,
                        "embed_provider": cfg.embed_provider if cfg.mode == "local" else None,
                        "embed_model": cfg.embed_model if cfg.mode == "local" else None,
                        "vectorstore": vs_snapshot,
                    }
                )
            continue

        answer = data.get("answer", "")
        sources = data.get("sources", []) or []

        coverage = _retrieval_coverage(
            sources,
            list(target_sections) if isinstance(target_sections, list) else [],
        )
        any_hit = coverage.any_hit
        full_hit = coverage.full_hit
        partial_hit = coverage.partial_hit
        retrieval_hit = any_hit

        refused_tagged = _has_refusal_tag(answer)
        refused_heuristic = _classify_refusal(answer)
        model_refused = refused_tagged or refused_heuristic

        groundedness = _groundedness_score(answer, sources)
        grounded = groundedness >= GROUNDEDNESS_THRESHOLD

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
                answer_supported = bool(full_hit)
                answer_correct = bool(full_hit)
                answer_correct_strict = bool(full_hit) and bool(grounded)

        status = ""
        if (not should_refuse) and (not model_refused) and (not any_hit):
            unsupported_non_refusal += 1
            status = "unsupported_answer"
        elif (not should_refuse) and (not model_refused) and partial_hit:
            partial_support_non_refusal += 1
            status = "partial_support_answer"
        elif (not should_refuse) and (not model_refused) and full_hit and (not grounded):
            ungrounded_non_refusal += 1
            status = "ungrounded_answer"

        if should_refuse and model_refused:
            tp_refuse += 1
            status = "correct_refusal"
        elif (not should_refuse) and (not model_refused):
            tn_refuse += 1
            if not status:
                status = "correct_answer"
        elif (not should_refuse) and model_refused:
            fp_refuse += 1
            status = "over_refusal"
        else:
            fn_refuse += 1
            status = "under_refusal"

        if should_refuse:
            key = refusal_type or "(unspecified)"
            refusal_type_counts[key] = refusal_type_counts.get(key, 0) + 1
            if model_refused:
                refusal_type_refused[key] = refusal_type_refused.get(key, 0) + 1

        if any_hit:
            if should_refuse and model_refused:
                tp_any += 1
            elif (not should_refuse) and (not model_refused):
                tn_any += 1
            elif (not should_refuse) and model_refused:
                fp_any += 1
            else:
                fn_any += 1
        else:
            if should_refuse and model_refused:
                tp_none += 1
            elif (not should_refuse) and (not model_refused):
                tn_none += 1
            elif (not should_refuse) and model_refused:
                fp_none += 1
            else:
                fn_none += 1

        if full_hit:
            if should_refuse and model_refused:
                tp_full += 1
            elif (not should_refuse) and (not model_refused):
                tn_full += 1
            elif (not should_refuse) and model_refused:
                fp_full += 1
            else:
                fn_full += 1

        if partial_hit:
            if should_refuse and model_refused:
                tp_partial += 1
            elif (not should_refuse) and (not model_refused):
                tn_partial += 1
            elif (not should_refuse) and model_refused:
                fp_partial += 1
            else:
                fn_partial += 1

        if (should_refuse and model_refused) or ((not should_refuse) and (not model_refused)):
            correct += 1
        if answer_correct:
            correct_answer += 1
        if answer_correct_strict:
            correct_answer_strict += 1

        print(
            f"{question_id}\t{should_refuse}\t{model_refused}\t"
            f"{int(any_hit)}\t{int(full_hit)}\t{int(partial_hit)}\t{status}"
        )

        if out is not None:
            rows_out.append(
                {
                    "prompt_version": cfg.prompt_version,
                    "system_prompt_sha256": prompt_sha256,
                    "id": question_id,
                    "question": question,
                    "question_type": question_type,
                    "should_refuse": should_refuse,
                    "refusal_type": refusal_type,
                    "model_refused": model_refused,
                    "refused_tagged": refused_tagged,
                    "refused_heuristic": refused_heuristic,
                    "status": status,
                    "retrieval_hit": retrieval_hit,
                    "any_hit": any_hit,
                    "full_hit": full_hit,
                    "partial_hit": partial_hit,
                    "answer_correct": answer_correct,
                    "answer_correct_strict": answer_correct_strict,
                    "answer_supported": answer_supported,
                    "unsupported_non_refusal": (not should_refuse) and (not model_refused) and (not any_hit),
                    "partial_support_non_refusal": (not should_refuse) and (not model_refused) and partial_hit,
                    "ungrounded_non_refusal": (not should_refuse) and (not model_refused) and full_hit and (not grounded),
                    "groundedness_score": groundedness,
                    "answer_grounded": grounded if (not should_refuse) and (not model_refused) else None,
                    "target_sections": target_sections,
                    "normalized_target_sections": coverage.normalized_target_sections,
                    "retrieved_section_ids": coverage.retrieved_sections,
                    "normalized_retrieved_sections": coverage.normalized_retrieved_sections,
                    "answer": answer,
                    "sources": sources,
                    "llm_finish_reason": data.get("llm_finish_reason"),
                    "llm_usage": data.get("llm_usage"),
                    "mode": cfg.mode,
                    "top_k": cfg.top_k,
                    "retrieval_method": cfg.retrieval_method,
                    "hybrid_alpha": cfg.hybrid_alpha,
                    "model": data.get("model", cfg.model or cfg.llm_model),
                    "temperature": cfg.temperature,
                    "embed_provider": cfg.embed_provider if cfg.mode == "local" else None,
                    "embed_model": cfg.embed_model if cfg.mode == "local" else None,
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

        correctness_acc = correct_answer / total
        corr_lo, corr_hi = _wilson_ci(correct_answer, total)

        print()
        print("Lightweight answer correctness (full-hit support-based):")
        print(f"  accuracy                   : {correctness_acc:.3f} (95% CI {corr_lo:.3f}–{corr_hi:.3f})")
        print(f"  unsupported_non_refusals   : {unsupported_non_refusal} ({(unsupported_non_refusal / total):.3f} of all items)")
        print(f"  partial_support_answers    : {partial_support_non_refusal} ({(partial_support_non_refusal / total):.3f} of all items)")
        print(f"  ungrounded_non_refusals    : {ungrounded_non_refusal} ({(ungrounded_non_refusal / total):.3f} of all items)")
        print(f"  query_failed               : {query_failed} ({(query_failed / total):.3f} of all items)")

        strict_acc = correct_answer_strict / total
        strict_lo, strict_hi = _wilson_ci(correct_answer_strict, total)

        print("Strict answer correctness (full_hit + groundedness):")
        print(f"  accuracy                   : {strict_acc:.3f} (95% CI {strict_lo:.3f}–{strict_hi:.3f})")

        n_any = tp_any + fp_any + tn_any + fn_any
        n_none = tp_none + fp_none + tn_none + fn_none
        n_full = tp_full + fp_full + tn_full + fn_full
        n_partial = tp_partial + fp_partial + tn_partial + fn_partial

        if n_any:
            metrics_any = _compute_refusal_metrics(tp_any, fp_any, tn_any, fn_any)
            print()
            print("Breakdown by any_hit:")
            print(
                f"  any_hit=1 (n={n_any}) "
                f"acc={metrics_any['accuracy']:.3f} "
                f"f1={metrics_any['f1']:.3f} "
                f"mcc={metrics_any['mcc']:.3f}"
            )

        if n_none:
            metrics_none = _compute_refusal_metrics(tp_none, fp_none, tn_none, fn_none)
            print(
                f"  any_hit=0 (n={n_none}) "
                f"acc={metrics_none['accuracy']:.3f} "
                f"f1={metrics_none['f1']:.3f} "
                f"mcc={metrics_none['mcc']:.3f}"
            )

        if n_full:
            metrics_full = _compute_refusal_metrics(tp_full, fp_full, tn_full, fn_full)
            print()
            print("Breakdown by full_hit:")
            print(
                f"  full_hit=1 (n={n_full}) "
                f"acc={metrics_full['accuracy']:.3f} "
                f"f1={metrics_full['f1']:.3f} "
                f"mcc={metrics_full['mcc']:.3f}"
            )

        if n_partial:
            metrics_partial = _compute_refusal_metrics(tp_partial, fp_partial, tn_partial, fn_partial)
            print()
            print("Breakdown by partial_hit:")
            print(
                f"  partial_hit=1 (n={n_partial}) "
                f"acc={metrics_partial['accuracy']:.3f} "
                f"f1={metrics_partial['f1']:.3f} "
                f"mcc={metrics_partial['mcc']:.3f}"
            )

        if refusal_type_counts:
            print()
            print("Refusal questions by refusal_type:")
            for key in sorted(refusal_type_counts):
                n_key = refusal_type_counts[key]
                refused_key = refusal_type_refused.get(key, 0)
                rate_key = refused_key / n_key if n_key else 0.0
                print(f"  {key}: n={n_key} refused={refused_key} rate={rate_key:.3f}")

    if out is not None:
        _write_jsonl(out, rows_out)
        print()
        print(f"Wrote detailed results: {out}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless refusal testing for TEK17 RAG (no UI).",
    )

    parser.add_argument("--mode", choices=["local", "server"], default=EVAL_MODE)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", type=str, help="Single question to probe")
    group.add_argument("--eval-file", type=Path, help="Eval JSONL file maintained manually")

    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--retrieval-method",
        choices=["dense", "sparse", "hybrid", "sparce"],
        default=RETRIEVAL_METHOD,
        help="Retrieval method for the run.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=HYBRID_ALPHA,
        help="Hybrid weighting: alpha*dense + (1-alpha)*sparse.",
    )

    parser.add_argument("--model", type=str, default=None, help="Model override in server mode")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--server-url", type=str, default=SERVER_URL)
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection", type=str, default=CHROMA_COLLECTION)
    parser.add_argument("--embed-base-url", type=str, default=EMBED_BASE_URL)
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default=LLM_PROVIDER)
    parser.add_argument("--embed-provider", choices=["ollama", "openai"], default=EMBED_PROVIDER)
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL)
    parser.add_argument("--embed-model", type=str, default=EMBED_MODEL)

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write detailed per-item results as JSONL in batch mode",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=PROMPT_VERSION,
        help="Prompt variant to use",
    )

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    prompt_version = _normalize_prompt_version(args.prompt_version)
    if prompt_version not in PROMPT_VERSIONS:
        print(
            "ERROR: invalid --prompt-version. "
            f"Expected one of: {', '.join(sorted(PROMPT_VERSIONS))}"
        )
        return 2

    cfg = RunConfig(
        mode=args.mode,
        top_k=args.top_k,
        model=args.model,
        temperature=args.temperature,
        retrieval_method=_normalize_retrieval_method(args.retrieval_method),
        hybrid_alpha=float(args.hybrid_alpha),
        server_url=args.server_url,
        chroma_dir=args.chroma_dir,
        collection=args.collection,
        embed_base_url=args.embed_base_url,
        llm_provider=args.llm_provider,
        embed_provider=args.embed_provider,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        llm_base_url=LLM_BASE_URL,
        prompt_version=prompt_version,
    )

    if args.question is not None:
        question = str(args.question).strip()
        if not question:
            print("ERROR: --question was provided but is empty")
            return 2
        return run_single(question, cfg)

    if args.eval_file is None:
        print("ERROR: --eval-file is required in eval mode")
        return 2

    return run_eval(args.eval_file, cfg, out=args.out)


if __name__ == "__main__":
    raise SystemExit(main())