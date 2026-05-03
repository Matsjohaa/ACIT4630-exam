#!/usr/bin/env bash
set -euo pipefail


# This script runs the entire TEK17 RAG pipeline, 
# from parsing the DIBK document to evaluating 
# the model's performance on refusal questions, using variables in tek17.conf.


# Build corpus and vector store
python -m tek17 parse-dibk
python -m tek17 chunk
python -m tek17 ingest

# Run evaluation
python analysis/scripts/test_refusal.py \
  --eval-file analysis/questions/tek17_eval_questions.dibk_manual_v2_38.jsonl
