"""Run a small, fixed refusal benchmark across multiple LLM backends.

This script orchestrates `analysis/scripts/test_refusal.py` for a few model
configurations and then summarizes the outputs into a single CSV.

Default run order:
- Ollama (local)
- OpenAI mini
- OpenAI GPT-5.2

Example:
  python analysis/scripts/benchmark_refusal_models.py \
    --eval-file analysis/questions/tek17_eval_questions.auto_v3_multistep.jsonl \
    --retrieval-method dense --top-k 6 --temperature 0

Notes
- For apples-to-apples comparisons, this defaults to keeping embeddings fixed
  (embed_provider=ollama, embed_model=nomic-embed-text) while varying only the
  LLM provider/model.
- Requires OPENAI_API_KEY for the OpenAI runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _csv_list(value: str) -> list[str]:
    parts = [p.strip() for p in (value or "").split(",")]
    return [p for p in parts if p]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark refusal behaviour across multiple models.")

    p.add_argument("--eval-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("analysis/logging"))

    p.add_argument("--mode", choices=["local", "server"], default="local")
    p.add_argument("--server-url", type=str, default="http://localhost:8000")

    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--retrieval-method", choices=["dense", "sparse", "hybrid"], default="dense")
    p.add_argument("--hybrid-alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)

    # Local-mode knobs (passed through; harmless in server mode)
    p.add_argument("--chroma-dir", type=Path, default=Path("data/vectorstore/chroma"))
    p.add_argument("--collection", type=str, default="tek17")

    p.add_argument("--ollama-url", type=str, default="http://localhost:11434")

    p.add_argument("--embed-provider", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--embed-model", type=str, default="nomic-embed-text")

    p.add_argument("--ollama-model", type=str, default="llama3.2")
    p.add_argument(
        "--openai-models",
        type=str,
        default="gpt-4.1-mini,gpt-5.2",
        help="Comma-separated OpenAI model ids to run (in order).",
    )

    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p


def _run(cmd: list[str], *, continue_on_error: bool) -> int:
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and not continue_on_error:
        return int(proc.returncode)
    return 0


def main() -> int:
    args = _build_parser().parse_args()

    test_refusal = (Path(__file__).resolve().parent / "test_refusal.py").resolve()
    summarize = (Path(__file__).resolve().parent / "summarize_refusal_runs.py").resolve()
    if not test_refusal.exists():
        raise SystemExit(f"Could not find test runner: {test_refusal}")
    if not summarize.exists():
        raise SystemExit(f"Could not find summarizer: {summarize}")

    if not args.eval_file.exists():
        raise SystemExit(f"Eval file not found: {args.eval_file}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build run list.
    runs: list[tuple[str, str, str]] = []
    runs.append(("ollama", args.ollama_model, "ollama"))
    for m in _csv_list(args.openai_models):
        runs.append(("openai", m, "openai"))

    out_files: list[Path] = []

    for idx, (llm_provider, llm_model, tag) in enumerate(runs, start=1):
        out_file = args.out_dir / (
            f"refusal_benchmark_{tag}_{llm_model}_"
            f"{args.retrieval_method}_top{args.top_k}_t{args.temperature:g}_"
            f"{timestamp}_{idx:02d}.jsonl"
        )

        cmd = [
            sys.executable,
            str(test_refusal),
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
            "--ollama-url",
            str(args.ollama_url),
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

        print(f"[{idx}/{len(runs)}] {llm_provider}:{llm_model} ({args.retrieval_method}, top_k={args.top_k}, t={args.temperature:g})")
        print(f"  -> {out_file}")

        if args.dry_run:
            continue

        rc = _run(cmd, continue_on_error=bool(args.continue_on_error))
        if rc != 0:
            return rc

        out_files.append(out_file)

    # Summarize
    if not args.dry_run and out_files:
        summary_csv = args.out_dir / f"refusal_benchmark_summary_{timestamp}.csv"
        sum_cmd = [
            sys.executable,
            str(summarize),
            "--files",
            *[str(p) for p in out_files],
            "--out-csv",
            str(summary_csv),
        ]

        print(f"\nSummarizing -> {summary_csv}")
        rc = _run(sum_cmd, continue_on_error=False)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
