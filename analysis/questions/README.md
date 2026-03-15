# TEK17 Evaluation Dataset

This folder contains evaluation questions for testing the TEK17 RAG system.

## JSONL schema

Each line in `tek17_eval_questions.jsonl` (or a copy based on the example file) should be a single JSON object with at least the following fields:

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

## Files

- `tek17_eval_questions.example.jsonl`: Small example file illustrating the schema. Duplicate and modify this file to create your actual evaluation set.

When running evaluation scripts, you should point them to the concrete JSONL file you created (e.g. `tek17_eval_questions.jsonl`).
