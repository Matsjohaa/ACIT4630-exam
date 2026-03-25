"""Run a small parameter sweep for refusal evaluation.

This is a thin orchestrator around `analysis/scripts/test_refusal.py` that:
- runs a grid of (top_k, retrieval_method, hybrid_alpha, temperature)
- writes one JSONL log file per run

Example:
  python analysis/scripts/sweep_refusal.py \
    --eval-file analysis/questions/tek17_eval_questions.dibk_example.jsonl \
    --llm-provider openai --llm-model gpt-5.1 --temperature 0 \
    --top-k 10 --retrieval-method dense,sparse,hybrid --hybrid-alpha 0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _csv_list(value: str) -> list[str]:
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def _csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for p in _csv_list(value):
        out.append(int(p))
    return out


def _csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for p in _csv_list(value):
        out.append(float(p))
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep refusal evaluation runs.")

    p.add_argument("--eval-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("analysis/logging"))

    p.add_argument("--mode", choices=["local", "server"], default="local")
    p.add_argument("--server-url", type=str, default="http://localhost:8000")

    p.add_argument("--top-k", type=str, default="6")
    p.add_argument("--retrieval-method", type=str, default="dense")
    p.add_argument("--hybrid-alpha", type=str, default="0.5")
    p.add_argument("--temperature", type=str, default="0")

    # Local-mode knobs (passed through; harmless in server mode)
    p.add_argument("--chroma-dir", type=Path, default=Path("data/vectorstore/chroma"))
    p.add_argument("--collection", type=str, default="tek17")

    p.add_argument("--ollama-url", type=str, default="http://localhost:11434")

    p.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--embed-provider", choices=["ollama", "openai"], default="ollama")

    p.add_argument("--llm-model", type=str, default="llama3.2")
    p.add_argument("--embed-model", type=str, default="nomic-embed-text")

    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p


def main() -> int:
    args = _build_parser().parse_args()

    test_refusal = (Path(__file__).resolve().parent / "test_refusal.py").resolve()
    if not test_refusal.exists():
        raise SystemExit(f"Could not find test runner: {test_refusal}")

    top_ks = _csv_ints(args.top_k)
    methods = [m.lower() for m in _csv_list(args.retrieval_method)]
    alphas = _csv_floats(args.hybrid_alpha)
    temps = _csv_floats(args.temperature)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[int, str, float, float]] = []
    for _ in range(int(args.repeat)):
        for top_k in top_ks:
            for method in methods:
                if method == "sparce":
                    method = "sparse"
                if method == "hybrid":
                    for alpha in alphas:
                        for temp in temps:
                            runs.append((top_k, method, float(alpha), float(temp)))
                else:
                    for temp in temps:
                        runs.append((top_k, method, float(alphas[0] if alphas else 0.5), float(temp)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, (top_k, method, alpha, temp) in enumerate(runs, start=1):
        alpha_part = f"a{alpha:g}" if method == "hybrid" else ""
        out_file = args.out_dir / (
            f"refusal_{args.llm_provider}_{args.llm_model}_"
            f"t{temp:g}_top{top_k}_{method}{('_' + alpha_part) if alpha_part else ''}_{timestamp}_{idx:03d}.jsonl"
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
            str(temp),
            "--server-url",
            args.server_url,
            "--chroma-dir",
            str(args.chroma_dir),
            "--collection",
            args.collection,
            "--ollama-url",
            args.ollama_url,
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

        print(f"[{idx}/{len(runs)}] top_k={top_k} method={method} alpha={alpha:g} temp={temp:g}")
        print("  ->", out_file)

        if args.dry_run:
            print("  (dry-run)")
            continue

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"ERROR: run failed with exit code {proc.returncode}")
            if not args.continue_on_error:
                return int(proc.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
