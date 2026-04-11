"""Run a fixed refusal benchmark across multiple LLM backends.

This script orchestrates:
- analysis/scripts/test_refusal.py
- analysis/scripts/summarize_refusal_runs.py
- analysis/scripts/compare_refusal_runs.py

Default run order:
- Ollama model from config / CLI
- one or more OpenAI models

It writes:
- one JSONL file per run
- one summary CSV across all runs
- optional pairwise comparison JSONL files for refusal flips
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tek17.rag.config import (
    BENCHMARK_COMPARE_SHOW,
    BENCHMARK_OPENAI_MODELS,
    BENCHMARK_OUT_DIR,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
    HYBRID_ALPHA,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVAL_METHOD,
    SERVER_URL,
    TOP_K,
)


def _csv_list(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark refusal behaviour across multiple models."
    )

    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=BENCHMARK_OUT_DIR)

    parser.add_argument("--mode", choices=["local", "server"], default="local")
    parser.add_argument("--server-url", type=str, default=SERVER_URL)

    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--retrieval-method",
        choices=["dense", "sparse", "hybrid"],
        default=RETRIEVAL_METHOD,
    )
    parser.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA)
    parser.add_argument("--temperature", type=float, default=LLM_TEMPERATURE)

    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection", type=str, default=CHROMA_COLLECTION)

    parser.add_argument("--embed-base-url", type=str, default=EMBED_BASE_URL or "")
    parser.add_argument("--embed-provider", choices=["ollama", "openai"], default=EMBED_PROVIDER)
    parser.add_argument("--embed-model", type=str, default=EMBED_MODEL)

    parser.add_argument("--ollama-model", type=str, default=LLM_MODEL)
    parser.add_argument(
        "--openai-models",
        type=str,
        default=BENCHMARK_OPENAI_MODELS,
        help="Comma-separated OpenAI model ids to run in order.",
    )

    parser.add_argument(
        "--compare-pairs",
        action="store_true",
        help="Also run pairwise comparisons between consecutive runs.",
    )
    parser.add_argument(
        "--compare-show",
        type=int,
        default=BENCHMARK_COMPARE_SHOW,
        help="Maximum number of refusal flips to print in pairwise comparisons.",
    )

    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser


def _run_command(command: list[str], *, continue_on_error: bool) -> int:
    process = subprocess.run(command)
    if process.returncode != 0 and not continue_on_error:
        return int(process.returncode)
    return 0


def _build_run_list(
    ollama_model: str,
    openai_models: str,
) -> list[tuple[str, str, str]]:
    runs: list[tuple[str, str, str]] = [("ollama", ollama_model, "ollama")]
    for model_name in _csv_list(openai_models):
        runs.append(("openai", model_name, "openai"))
    return runs


def _safe_filename(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def main() -> int:
    args = _build_parser().parse_args()

    script_dir = Path(__file__).resolve().parent
    test_refusal_script = (script_dir / "test_refusal.py").resolve()
    summarize_script = (script_dir / "summarize_refusal_runs.py").resolve()
    compare_script = (script_dir / "compare_refusal_runs.py").resolve()

    if not test_refusal_script.exists():
        raise SystemExit(f"Could not find test runner: {test_refusal_script}")
    if not summarize_script.exists():
        raise SystemExit(f"Could not find summarizer: {summarize_script}")
    if args.compare_pairs and not compare_script.exists():
        raise SystemExit(f"Could not find comparison script: {compare_script}")
    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    runs = _build_run_list(args.ollama_model, args.openai_models)
    out_files: list[Path] = []

    for index, (llm_provider, llm_model, tag) in enumerate(runs, start=1):
        safe_model_name = _safe_filename(llm_model)
        out_file = args.out_dir / (
            f"refusal_benchmark_{tag}_{safe_model_name}_"
            f"{args.retrieval_method}_top{args.top_k}_t{args.temperature:g}_"
            f"{timestamp}_{index:02d}.jsonl"
        )

        command = [
            sys.executable,
            str(test_refusal_script),
            "--mode",
            args.mode,
            "--eval-file",
            str(args.eval_file),
            "--top-k",
            str(args.top_k),
            "--retrieval-method",
            str(args.retrieval_method),
            "--hybrid-alpha",
            str(args.hybrid_alpha),
            "--temperature",
            str(args.temperature),
            "--server-url",
            str(args.server_url),
            "--chroma-dir",
            str(args.chroma_dir),
            "--collection",
            str(args.collection),
            "--embed-base-url",
            str(args.embed_base_url),
            "--llm-provider",
            str(llm_provider),
            "--embed-provider",
            str(args.embed_provider),
            "--llm-model",
            str(llm_model),
            "--embed-model",
            str(args.embed_model),
            "--out",
            str(out_file),
        ]

        print(
            f"[{index}/{len(runs)}] "
            f"{llm_provider}:{llm_model} "
            f"({args.retrieval_method}, top_k={args.top_k}, t={args.temperature:g})"
        )
        print(f"  -> {out_file}")

        if args.dry_run:
            continue

        return_code = _run_command(
            command,
            continue_on_error=bool(args.continue_on_error),
        )
        if return_code != 0:
            return return_code

        out_files.append(out_file)

    if args.dry_run:
        return 0

    if out_files:
        summary_csv = args.out_dir / f"refusal_benchmark_summary_{timestamp}.csv"
        summary_command = [
            sys.executable,
            str(summarize_script),
            "--files",
            *[str(path) for path in out_files],
            "--out-csv",
            str(summary_csv),
        ]

        print(f"\nSummarizing -> {summary_csv}")
        return_code = _run_command(summary_command, continue_on_error=False)
        if return_code != 0:
            return return_code

    if args.compare_pairs and len(out_files) >= 2:
        print("\nRunning pairwise comparisons")
        for index in range(len(out_files) - 1):
            run_a = out_files[index]
            run_b = out_files[index + 1]

            compare_out = args.out_dir / (
                f"refusal_compare_{run_a.stem}__vs__{run_b.stem}.jsonl"
            )

            compare_command = [
                sys.executable,
                str(compare_script),
                "--a",
                str(run_a),
                "--b",
                str(run_b),
                "--show",
                str(args.compare_show),
                "--out",
                str(compare_out),
            ]

            print(f"  {run_a.name} vs {run_b.name}")
            return_code = _run_command(compare_command, continue_on_error=False)
            if return_code != 0:
                return return_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())