# TEK17 RAG

RAG-based question-answering system for TEK17 (Byggteknisk forskrift).  
Supports dense, sparse, and hybrid retrieval with Ollama or OpenAI backends.  
Includes evaluation scripts for retrieval quality and refusal behaviour.

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com) with `llama3.2` and `nomic-embed-text` pulled
- (Optional) OpenAI API key in `.env` for GPT experiments

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Configuration

All settings live in `tek17.conf` (INI format, project root).  
CLI flags override the config file (e.g. `python -m tek17 chunk --chunk-size 400`).  
Secrets (API keys) go in `.env`.

## Pipeline

The quickest way to build the corpus and run an evaluation:

```bash
./run_pipeline.sh
```

This parses the TEK17 snapshot, chunks it, ingests into ChromaDB, and runs an
evaluation on the manual question set.

Or run each step individually:

```bash
python -m tek17 download-dibk   # 1. Download TEK17 snapshot from DiBK (only needed once)
python -m tek17 parse-dibk      # 2. Parse HTML into per-provision JSONL
python -m tek17 chunk           # 3. Chunk provisions
python -m tek17 ingest          # 4. Embed and ingest into ChromaDB
```

## Running experiments

```bash
# Single evaluation run (uses settings from tek17.conf)
python analysis/scripts/test_refusal.py \
  --eval-file analysis/questions/tek17_eval_questions.dibk_manual_v2_38.jsonl

# Override specific settings via flags
python analysis/scripts/test_refusal.py \
  --eval-file analysis/questions/tek17_eval_questions.dibk_manual_v2_38.jsonl \
  --retrieval-method hybrid --top-k 14 --temperature 0

# Parameter sweep (retrieval method × top-k)
python analysis/scripts/sweep_refusal.py

# Multi-model benchmark (Ollama + OpenAI)
python analysis/scripts/benchmark_refusal_models.py

# Refusal taxonomy analysis on logged runs
python analysis/refusal_analysis.py
```

Results are written to `analysis/logging/`.

## Project structure

```
tek17.conf              # central config (INI)
src/tek17/              # installable package
  cli.py                # Typer CLI
  corpus/               # download, parse, chunk
  rag/                  # config, ingest, prompts, LLM dispatch, retrieval
analysis/
  scripts/              # sweep, benchmark, evaluation helpers
  refusal_analysis.py   # conditional-refusal taxonomy
  questions/            # evaluation question sets
data/
  raw/                  # downloaded HTML snapshots
  processed/            # parsed JSONL + chunks
  vectorstore/          # ChromaDB
```
