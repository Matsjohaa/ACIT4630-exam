"""Run a small parameter sweep for refusal evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tek17.rag.config import (
    BENCHMARK_OUT_DIR,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
    HYBRID_ALPHA,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    PROMPT_VERSION,
    RETRIEVAL_METHOD,
    SERVER_URL,
    SWEEP_REPEAT,
    TEST_MODE,
    TOP_K,
)


def _csv_list(value: str) -> list[str]:
    parts = [part.strip() for part in (value or "").split(",")]
    return [part for part in parts if part]


def _csv_ints(value: str) -> list[int]:
    return [int(part) for part in _csv_list(value)]


def _csv_floats(value: str) -> list[float]:
    return [float(part) for part in _csv_list(value)]


def _normalize_retrieval_method(method: str) -> str:
    normalized = (method or RETRIEVAL_METHOD).strip().lower()
    if normalized == "sparce":
        return "sparse"
    return normalized


def _sanitize_filename_part(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace(":", "-")
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep refusal evaluation runs.")

    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=BENCHMARK_OUT_DIR)

    parser.add_argument("--mode", choices=["local", "server"], default=TEST_MODE)
    parser.add_argument("--server-url", type=str, default=SERVER_URL)

    parser.add_argument("--top-k", type=str, default=str(TOP_K))
    parser.add_argument("--retrieval-method", type=str, default=RETRIEVAL_METHOD)
    parser.add_argument("--hybrid-alpha", type=str, default=str(HYBRID_ALPHA))
    parser.add_argument("--temperature", type=str, default=str(LLM_TEMPERATURE))
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=PROMPT_VERSION,
        help="Comma-separated prompt variants to sweep, e.g. baseline,relaxed,strict",
    )

    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection", type=str, default=CHROMA_COLLECTION)

    parser.add_argument("--embed-base-url", type=str, default=EMBED_BASE_URL or "")
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default=LLM_PROVIDER)
    parser.add_argument("--embed-provider", choices=["ollama", "openai"], default=EMBED_PROVIDER)

    parser.add_argument("--llm-model", type=str, default=LLM_MODEL)
    parser.add_argument("--embed-model", type=str, default=EMBED_MODEL)

    parser.add_argument("--repeat", type=int, default=SWEEP_REPEAT)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    test_refusal = (Path(__file__).resolve().parent / "test_refusal.py").resolve()
    if not test_refusal.exists():
        raise SystemExit(f"Could not find test runner: {test_refusal}")

    top_ks = _csv_ints(args.top_k)
    methods = [_normalize_retrieval_method(method) for method in _csv_list(args.retrieval_method)]
    alphas = _csv_floats(args.hybrid_alpha)
    temperatures = _csv_floats(args.temperature)
    prompt_versions = [version.strip().lower() for version in _csv_list(args.prompt_version)]

    if not prompt_versions:
        prompt_versions = [PROMPT_VERSION]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[int, str, float, float, str]] = []
    for _ in range(int(args.repeat)):
        for prompt_version in prompt_versions:
            for top_k in top_ks:
                for method in methods:
                    if method == "hybrid":
                        for alpha in alphas:
                            for temperature in temperatures:
                                runs.append(
                                    (top_k, method, float(alpha), float(temperature), prompt_version)
                                )
                    else:
                        fallback_alpha = float(alphas[0]) if alphas else HYBRID_ALPHA
                        for temperature in temperatures:
                            runs.append(
                                (top_k, method, fallback_alpha, float(temperature), prompt_version)
                            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for index, (top_k, method, alpha, temperature, prompt_version) in enumerate(runs, start=1):
        alpha_part = f"_a{alpha:g}" if method == "hybrid" else ""
        out_file = args.out_dir / (
            f"refusal_{_sanitize_filename_part(args.llm_provider)}"
            f"_{_sanitize_filename_part(args.llm_model)}"
            f"_{_sanitize_filename_part(prompt_version)}"
            f"_t{temperature:g}_top{top_k}_{method}"
            f"{alpha_part}_{timestamp}_{index:03d}.jsonl"
        )

        cmd = [
            sys.executable,
            str(test_refusal),
            "--mode",
            args.mode,
            "--eval-file",
            str(args.eval_file),
            "--top-k",
            str(top_k),
            "--retrieval-method",
            method,
            "--hybrid-alpha",
            str(alpha),
            "--temperature",
            str(temperature),
            "--prompt-version",
            prompt_version,
            "--server-url",
            args.server_url,
            "--chroma-dir",
            str(args.chroma_dir),
            "--collection",
            args.collection,
            "--embed-base-url",
            args.embed_base_url,
            "--llm-provider",
            args.llm_provider,
            "--embed-provider",
            args.embed_provider,
            "--llm-model",
            args.llm_model,
            "--embed-model",
            args.embed_model,
            "--out",
            str(out_file),
        ]

        print(
            f"[{index}/{len(runs)}] "
            f"prompt={prompt_version} top_k={top_k} method={method} "
            f"alpha={alpha:g} temp={temperature:g}"
        )
        print(f"  -> {out_file}")

        if args.dry_run:
            print("  (dry-run)")
            continue

        process = subprocess.run(cmd)
        if process.returncode != 0:
            print(f"ERROR: run failed with exit code {process.returncode}")
            if not args.continue_on_error:
                return int(process.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())