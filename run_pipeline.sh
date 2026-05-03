#!/usr/bin/env bash
set -euo pipefail

# This script runs the entire TEK17 RAG pipeline, 
# from parsing the DIBK document to evaluating 
# the model's performance on refusal questions, using variables in tek17.conf.


# Activate venv if not already active
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Ensure package is installed
pip install -e "$SCRIPT_DIR" -q

# Read config values for display and output naming
LLM_PROVIDER=$(python -c "import configparser; c=configparser.ConfigParser(); c.read('tek17.conf'); print(c.get('llm','provider',fallback='ollama'))")
LLM_MODEL=$(python -c "import configparser; c=configparser.ConfigParser(); c.read('tek17.conf'); print(c.get('llm','model',fallback='llama3.2'))")
LLM_TEMP=$(python -c "import configparser; c=configparser.ConfigParser(); c.read('tek17.conf'); print(c.get('llm','temperature',fallback='0.0'))")
RET_METHOD=$(python -c "import configparser; c=configparser.ConfigParser(); c.read('tek17.conf'); print(c.get('retrieval','method',fallback='dense'))")
TOP_K=$(python -c "import configparser; c=configparser.ConfigParser(); c.read('tek17.conf'); print(c.get('retrieval','top_k',fallback='6'))")
HYBRID_ALPHA=$(python -c "import configparser; c=configparser.ConfigParser(); c.read('tek17.conf'); print(c.get('retrieval','hybrid_alpha',fallback='0.5'))")
CHUNK_SIZE=$(python -c "import configparser; c=configparser.ConfigParser(); c.read('tek17.conf'); print(c.get('chunking','chunk_size',fallback='800'))")
PROMPT_VER=$(python -c "import configparser; c=configparser.ConfigParser(); c.read('tek17.conf'); print(c.get('prompt','version',fallback='baseline'))")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="${LLM_PROVIDER}_${LLM_MODEL}_${PROMPT_VER}_t${LLM_TEMP}_top${TOP_K}_${RET_METHOD}"
OUT_FILE="analysis/logging/refusal_${RUN_TAG}_${TIMESTAMP}.jsonl"

echo "=========================================="
echo " TEK17 Pipeline"
echo "=========================================="
echo " LLM:        ${LLM_PROVIDER} / ${LLM_MODEL}"
echo " Temperature: ${LLM_TEMP}"
echo " Retrieval:  ${RET_METHOD} (top_k=${TOP_K}, alpha=${HYBRID_ALPHA})"
echo " Chunk size: ${CHUNK_SIZE}"
echo " Prompt:     ${PROMPT_VER}"
echo " Output:     ${OUT_FILE}"
echo "=========================================="

# Build corpus and vector store
python -m tek17 parse-dibk
python -m tek17 chunk
python -m tek17 ingest

# Run evaluation (save results to logging)
PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}" python analysis/scripts/test_refusal.py \
  --eval-file analysis/questions/tek17_eval_questions.dibk_manual_v2_38.jsonl \
  --out "$OUT_FILE"

echo ""
echo "Results saved to: ${OUT_FILE}"
