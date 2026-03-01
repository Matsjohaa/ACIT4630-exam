# TEK17 RAG – Corpus Preparation Pipeline

## Overview

This repository provides a pipeline for extracting and structuring  TEK17 with guidance from DiBK.

The pipeline:
1. Downloads a single authoritative **root-print snapshot** of TEK17.
2. Parses it into one structured record per provision (§ x-y).
3. Separates regulation text and guidance text.
4. Produces a clean JSONL corpus for retrieval experiments.
---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

---

## CLI Usage

### Download TEK17 snapshot

```bash
python -m tek17 download-dibk
```

Optional flags:
- `--force` – re-download snapshot
- `--url` – override default root-print URL
- `--out-dir` – raw storage directory
- `--manifest` – manifest file path

Raw HTML is stored in:

```
data/raw/dibk_root_print/YYYY-MM-DD/
```

---

### Parse into structured corpus

```bash
python -m tek17 parse-dibk
```

Output:

```
data/processed/tek17_dibk.jsonl
```

Each JSONL record represents one provision (§ x-y).

---

## Output Schema

Each record contains (example):

```json
{
  "source": "dibk",
  "doc_type": "provision",
  "root_print_url": "https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17?subtype=root&print=true",
  "final_url": "https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17?subtype=root&print=true",
  "canonical_root_print_url": "https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17",
  "downloaded_at": "2026-03-01T19:31:16",
  "source_path": "data/raw/dibk_root_print/YYYY-MM-DD/tek17_full_root_print.html",
  "sha256": "346402fea344cc9cc77dd1af149680dab0e426966ffdd781b4c4ffd763a7941b",
  "chapter": "Kapittel X ...",
  "section_id": "§ 1-1",
  "title": "Formål",
  "reg_text": "...",
  "guidance_text": "...",
  "full_text": "..."
}
```

For retrieval experiments, `reg_text` and `guidance_text` should be chunked separately.

---

## Reproducibility

- Root-print snapshot logged in append-only manifest  
- SHA256 hash stored for integrity verification  
- Parsing is deterministic  

To regenerate the corpus:

```bash
python -m tek17 download-dibk --force
python -m tek17 parse-dibk
```

---

## Repository Structure

```
src/tek17/
data/
```

This corpus is ready for chunking, retrieval, and refusal-behaviour experiments.
