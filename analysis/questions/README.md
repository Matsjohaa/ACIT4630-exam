# TEK17 Evaluation Dataset

This folder contains the evaluation question set used for testing refusal behaviour and retrieval performance in the TEK17 RAG system.

## JSONL schema

Each line in the evaluation JSONL should be a single JSON object with at least the following fields:

- `id` (string): Unique identifier for the eval item.
- `question` (string): The natural-language question posed to the system.
- `target_sections` (array of strings): One or more TEK17 section identifiers that contain the information needed to answer the question (e.g. `["§ 11-9"]`). Leave empty if the question is intentionally unanswerable from TEK17.
- `difficulty` (string): Rough difficulty label, e.g. `"easy"`, `"medium"`, or `"hard"`.
- `should_refuse` (boolean):
  - `false` if the model **should** be able to answer based on TEK17.
  - `true` if the model **should** refuse (e.g. question is out-of-scope, purely hypothetical, or clearly not covered by TEK17).

Optional fields you may choose to include:

- `notes` (string): Any human notes or guidance about what a good answer would look like, or why the question is labelled as `should_refuse`.
- `chapter_hint` (string): If you want to constrain retrieval analysis to specific chapters.

- `refusal_type` (string): Only relevant when `should_refuse=true`. Use this to distinguish refusal sub-cases that often behave differently:
  - `out_of_scope`: Not answerable from TEK17 at all (municipal processes, PBL/SAK10 details, fees, etc.).
  - `in_domain_missing_context`: In the building domain, but requires project-specific calculations/inputs or professional judgement beyond what TEK17 context alone can support.

- `question_type` (string): Optional explicit slice label used in summaries. Suggested values:
  - `in_scope_single`
  - `in_scope_multi`
  - `refusal`

Note: our summarization script can also infer `question_type` if it is missing.

## Files

- Question set used in our experiments (213 items):
  - [analysis/questions/tek17_eval_questions.auto_v3_multistep.jsonl](analysis/questions/tek17_eval_questions.auto_v3_multistep.jsonl)

This dataset includes:

- in-scope single-section questions
- some multi-section questions (2–3 target sections)
- refusal questions split by `refusal_type` (`out_of_scope` vs `in_domain_missing_context`)

## Notes on correctness

The refusal eval script reports two different "correctness" signals:

- Support-based correctness: treats an in-scope answer as correct if the system *did not refuse* and retrieval returned at least one of the `target_sections`.
  This is good for flagging "non-refusal hallucinations" when `retrieval_hit=false`.

- Strict correctness (heuristic): additionally requires that the answer appears *grounded* in the retrieved context text (simple word-overlap score).
  This is still not a gold semantic judge, but it catches some cases where `retrieval_hit=true` but the answer does not actually use the context.

## Regenerating the dataset

To regenerate the dataset (from the project root):

```bash
python analysis/scripts/generate_eval_questions.py \
  --out analysis/questions/tek17_eval_questions.auto_v3_multistep.jsonl \
  --n-in-scope 160 \
  --n-refuse 80 \
  --seed 17 \
  --style paraphrase \
  --multi-frac 0.25 \
  --multi-max-sections 3
```

Script: [analysis/scripts/generate_eval_questions.py](analysis/scripts/generate_eval_questions.py)

Before running evals, ensure the vector store is built and non-empty:

```bash
python analysis/scripts/check_vectorstore.py
```

If the check fails or `count=0`, rebuild:

```bash
python -m tek17 download-dibk
python -m tek17 parse-dibk
python -m tek17 ingest
```

When running evaluation scripts, you should point them to the concrete JSONL file you created (e.g. `tek17_eval_questions.jsonl`).

## Running the same experiments

Model benchmark (3 LLMs, retrieval fixed to dense):

- Script: [analysis/scripts/benchmark_refusal_models.py](analysis/scripts/benchmark_refusal_models.py)
- Runner: [analysis/scripts/test_refusal.py](analysis/scripts/test_refusal.py)

```bash
python analysis/scripts/benchmark_refusal_models.py \
  --eval-file analysis/questions/tek17_eval_questions.auto_v3_multistep.jsonl \
  --retrieval-method dense --top-k 6 --temperature 0
```

Retrieval benchmark (1 LLM fixed; compare dense vs sparse vs hybrid):

- Runner: [analysis/scripts/test_refusal.py](analysis/scripts/test_refusal.py)
- Summarizer: [analysis/scripts/summarize_refusal_runs.py](analysis/scripts/summarize_refusal_runs.py)

```bash
# Example: OpenAI gpt-4.1-mini, top_k=6, temperature=0

python analysis/scripts/test_refusal.py \
  --mode local \
  --eval-file analysis/questions/tek17_eval_questions.auto_v3_multistep.jsonl \
  --llm-provider openai --llm-model gpt-4.1-mini \
  --retrieval-method dense --top-k 6 --temperature 0 \
  --out analysis/logging/refusal_retrieval_benchmark_openai_gpt-4.1-mini_dense_top6_t0.jsonl

python analysis/scripts/test_refusal.py \
  --mode local \
  --eval-file analysis/questions/tek17_eval_questions.auto_v3_multistep.jsonl \
  --llm-provider openai --llm-model gpt-4.1-mini \
  --retrieval-method sparse --top-k 6 --temperature 0 \
  --out analysis/logging/refusal_retrieval_benchmark_openai_gpt-4.1-mini_sparse_top6_t0.jsonl

python analysis/scripts/test_refusal.py \
  --mode local \
  --eval-file analysis/questions/tek17_eval_questions.auto_v3_multistep.jsonl \
  --llm-provider openai --llm-model gpt-4.1-mini \
  --retrieval-method hybrid --hybrid-alpha 0.5 --top-k 6 --temperature 0 \
  --out analysis/logging/refusal_retrieval_benchmark_openai_gpt-4.1-mini_hybrid_top6_t0_a0.5.jsonl

python analysis/scripts/summarize_refusal_runs.py \
  --files \
    analysis/logging/refusal_retrieval_benchmark_openai_gpt-4.1-mini_dense_top6_t0.jsonl \
    analysis/logging/refusal_retrieval_benchmark_openai_gpt-4.1-mini_sparse_top6_t0.jsonl \
    analysis/logging/refusal_retrieval_benchmark_openai_gpt-4.1-mini_hybrid_top6_t0_a0.5.jsonl \
  --out-csv analysis/logging/refusal_retrieval_benchmark_openai_gpt-4.1-mini_top6_t0_summary.csv
```
