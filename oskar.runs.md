
# Reproducing our refusal + hybrid-b1 results (TEK17 RAG)

This document explains how to reproduce the exact artifacts we used in the analysis and paper tables.

## Quick answer: will reruns match the analysis files?

**Yes, you can get the same or very similar results** if you keep the *entire pipeline identical*:

- Same evaluation set (same JSONL questions and labels)
- Same retrieval configuration (hybrid + same b1, same `top_k`)
- Same prompt version (`baseline`) and code version
- Same embeddings model/provider and the *same* Chroma vectorstore snapshot
- Same LLM settings (notably `temperature=0`)

**What can still change even with `temperature=0`:**

- Provider-side nondeterminism / model updates (OpenAI models can change over time)
- Occasional timeouts / transient API failures (shows up as `query_failed` / `n_errors`)
- If the vectorstore differs (re-ingested corpus, different sqlite, different chunking), retrieval changes and metrics can shift

So: rerunning should be close, but **the most reliable way to reproduce the *exact* reported numbers is to re-summarize from the saved JSONL logs** included in this repo.

---

## Where the “source of truth” artifacts are

This repo contains the final evaluation logs and derived summaries under:

### Model comparison (3 models, fixed hybrid b1=0.5)

Folder:

- `analysis/gpt_alpha_runs/model_comparison/`

Contents:

- `refusal_openai_gpt-4.1-mini_baseline_t0_top6_hybrid_20260416135008_002.jsonl`
- `refusal_openai_gpt-5.2_baseline_t0_top6_hybrid_20260416135008_001.jsonl`
- `refusal_ollama_llama3.2_baseline_t0_top6_hybrid_20260416135008_003.jsonl`
- `summary.csv` (paper table source)
- `flips_*.jsonl` (pairwise differences for qualitative analysis)

### Hybrid b1 sweep (gpt-4.1-mini, b1 in {0.25, 0.5, 0.75, 0.9})

Folder:

- `analysis/gpt_alpha_runs/hybrid_alpha_sweep/gpt-4.1-mini/`

Contents:

- `refusal_openai_gpt-4.1-mini_baseline_t0_top6_hybrid_a0.25_20260416_151039_001.jsonl`
- `refusal_openai_gpt-4.1-mini_baseline_t0_top6_hybrid_20260416135008_002.jsonl` (this is the b1=0.5 baseline reused from model comparison)
- `refusal_openai_gpt-4.1-mini_baseline_t0_top6_hybrid_a0.75_20260416_151039_002.jsonl`
- `refusal_openai_gpt-4.1-mini_baseline_t0_top6_hybrid_a0.9_20260416_151039_003.jsonl`
- `conditional_refusal_summary.csv` (paper b1-sweep table source)

---

## Reproduce paper tables from the saved logs (recommended)

These steps do **not** call any LLM. They only read the JSONL logs and regenerate the derived CSVs.

Run everything from repo root.

### 1) Regenerate model-comparison summary.csv

```bash
./.venv/bin/python analysis/scripts/summarize_refusal_runs.py \
	--glob "analysis/gpt_alpha_runs/model_comparison/*.jsonl" \
	--out-csv "analysis/gpt_alpha_runs/model_comparison/summary.csv"
```

This recreates the metrics table (accuracy/F1/MCC, refusal rates, hit rates, answer correctness, `n_errors`, etc.).

### 2) (Optional) Regenerate flip files

OpenAI 4.1-mini vs OpenAI 5.2:

```bash
./.venv/bin/python analysis/scripts/compare_refusal_runs.py \
	--a "analysis/gpt_alpha_runs/model_comparison/refusal_openai_gpt-4.1-mini_baseline_t0_top6_hybrid_20260416135008_002.jsonl" \
	--b "analysis/gpt_alpha_runs/model_comparison/refusal_openai_gpt-5.2_baseline_t0_top6_hybrid_20260416135008_001.jsonl" \
	--show 50 \
	--out "analysis/gpt_alpha_runs/model_comparison/flips_openai_4.1-mini_vs_5.2.jsonl"
```

Llama3.2 vs OpenAI 5.2:

```bash
./.venv/bin/python analysis/scripts/compare_refusal_runs.py \
	--a "analysis/gpt_alpha_runs/model_comparison/refusal_ollama_llama3.2_baseline_t0_top6_hybrid_20260416135008_003.jsonl" \
	--b "analysis/gpt_alpha_runs/model_comparison/refusal_openai_gpt-5.2_baseline_t0_top6_hybrid_20260416135008_001.jsonl" \
	--show 50 \
	--out "analysis/gpt_alpha_runs/model_comparison/flips_llama3.2_vs_openai_5.2.jsonl"
```

Llama3.2 vs OpenAI 4.1-mini:

```bash
./.venv/bin/python analysis/scripts/compare_refusal_runs.py \
	--a "analysis/gpt_alpha_runs/model_comparison/refusal_ollama_llama3.2_baseline_t0_top6_hybrid_20260416135008_003.jsonl" \
	--b "analysis/gpt_alpha_runs/model_comparison/refusal_openai_gpt-4.1-mini_baseline_t0_top6_hybrid_20260416135008_002.jsonl" \
	--show 50 \
	--out "analysis/gpt_alpha_runs/model_comparison/flips_llama3.2_vs_openai_4.1-mini.jsonl"
```

### 3) Regenerate the b1-sweep conditional_refusal_summary.csv

This reproduces the *conditional refusal* table and taxonomy counts across b1 values.

```bash
./.venv/bin/python analysis/refusal_analysis.py \
	--log-dir "analysis/gpt_alpha_runs/hybrid_alpha_sweep/gpt-4.1-mini" \
	--out-csv "analysis/gpt_alpha_runs/hybrid_alpha_sweep/gpt-4.1-mini/conditional_refusal_summary.csv"
```

---

## Re-run the evaluations from scratch (to regenerate JSONLs)

This calls LLMs/embeddings again. Results should be similar, but may not match exactly (see nondeterminism caveats above).

### Prerequisites

1) Python environment

- Create a venv and install dependencies (one-time). Use whatever workflow you prefer (`pip`, `uv`, etc.). The project is defined in `pyproject.toml`.

2) Vectorstore must already exist

- The evaluation runs use a persisted Chroma store under `data/vectorstore/chroma/`.
- If you rebuild/re-ingest the corpus, retrieval will change.

3) Ollama must be running (for embeddings)

- We kept embeddings constant using Ollama:
	- `--embed-provider ollama`
	- `--embed-model nomic-embed-text`

4) OpenAI API key (only needed for OpenAI LLM runs)

- Put `OPENAI_API_KEY` in `.env` and load it:

```bash
set -a; source .env; set +a
```

### Common settings (kept constant across runs)

- Eval set: `analysis/questions/tek17_eval_questions.dibk_manual_v2_38.jsonl`
- Retrieval: `--retrieval-method hybrid --top-k 6`
- Determinism: `--temperature 0`
- Prompt: `--prompt-version baseline`
- Embeddings: `--embed-provider ollama --embed-model nomic-embed-text`

### A) Re-run the 3-model comparison (hybrid b1=0.5)

From repo root:

```bash
set -a; source .env; set +a
eval_file="analysis/questions/tek17_eval_questions.dibk_manual_v2_38.jsonl"
run_ts=$(date +%Y%m%d%H%M%S)

# OpenAI gpt-5.2
./.venv/bin/python analysis/scripts/test_refusal.py \
	--mode local \
	--eval-file "$eval_file" \
	--retrieval-method hybrid --hybrid-alpha 0.5 \
	--top-k 6 --temperature 0 \
	--embed-provider ollama --embed-model nomic-embed-text \
	--llm-provider openai --llm-model gpt-5.2 \
	--prompt-version baseline \
	--out "analysis/gpt_alpha_runs/model_comparison/refusal_openai_gpt-5.2_baseline_t0_top6_hybrid_${run_ts}_001.jsonl"

# OpenAI gpt-4.1-mini
./.venv/bin/python analysis/scripts/test_refusal.py \
	--mode local \
	--eval-file "$eval_file" \
	--retrieval-method hybrid --hybrid-alpha 0.5 \
	--top-k 6 --temperature 0 \
	--embed-provider ollama --embed-model nomic-embed-text \
	--llm-provider openai --llm-model gpt-4.1-mini \
	--prompt-version baseline \
	--out "analysis/gpt_alpha_runs/model_comparison/refusal_openai_gpt-4.1-mini_baseline_t0_top6_hybrid_${run_ts}_002.jsonl"

# Ollama llama3.2 (local)
./.venv/bin/python analysis/scripts/test_refusal.py \
	--mode local \
	--eval-file "$eval_file" \
	--retrieval-method hybrid --hybrid-alpha 0.5 \
	--top-k 6 --temperature 0 \
	--embed-provider ollama --embed-model nomic-embed-text \
	--llm-provider ollama --llm-model llama3.2 \
	--prompt-version baseline \
	--out "analysis/gpt_alpha_runs/model_comparison/refusal_ollama_llama3.2_baseline_t0_top6_hybrid_${run_ts}_003.jsonl"
```

Then regenerate summary + flips (same commands as earlier, but your filenames will have the new timestamp).

### B) Re-run the hybrid b1 sweep (gpt-4.1-mini)

We ran 3 new b1 values and **reused** the b1=0.5 JSONL from the model-comparison run.

If you want to reproduce the same structure:

```bash
set -a; source .env; set +a
eval_file="analysis/questions/tek17_eval_questions.dibk_manual_v2_38.jsonl"

./.venv/bin/python analysis/scripts/sweep_refusal.py \
	--eval-file "$eval_file" \
	--out-dir analysis/gpt_alpha_runs/hybrid_alpha_sweep/gpt-4.1-mini \
	--mode local \
	--retrieval-method hybrid \
	--top-k 6 \
	--hybrid-alpha "0.25,0.75,0.9" \
	--temperature "0" \
	--prompt-version baseline \
	--llm-provider openai \
	--llm-model gpt-4.1-mini \
	--embed-provider ollama \
	--embed-model nomic-embed-text
```

Finally regenerate the conditional refusal summary:

```bash
./.venv/bin/python analysis/refusal_analysis.py \
	--log-dir "analysis/gpt_alpha_runs/hybrid_alpha_sweep/gpt-4.1-mini" \
	--out-csv "analysis/gpt_alpha_runs/hybrid_alpha_sweep/gpt-4.1-mini/conditional_refusal_summary.csv"
```

---

## Notes / gotchas

- The eval file name contains `v2_38`, but the run logs for these experiments use `n=33` questions.
- For the model comparison, llama3.2 had `n_errors=1` (one timeout). If you re-run, you may get 0 errors (or a different error count), which will change `n_ok`.
- OpenAI runs should not inherit an Ollama base URL. The codebase was adjusted so OpenAI gets no `base_url` unless explicitly configured.

