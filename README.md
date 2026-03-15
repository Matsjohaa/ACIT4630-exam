# TEK17 RAG

## Overview

End-to-end pipeline for extracting, indexing and querying TEK17 (Byggteknisk forskrift) with guidance from DiBK.

The system:
1. Downloads an authoritative **root-print snapshot** of TEK17.
2. Parses it into one structured record per provision (§ x-y).
3. Separates regulation text and guidance text into a JSONL corpus.
4. Chunks and embeds the corpus into a **ChromaDB** vector store via **Ollama**.
5. Serves a **FastAPI** RAG server that retrieves relevant chunks and generates answers.
6. Provides a **Streamlit** chat UI for interactive Q&A.

---

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

> **Note:** Python 3.11 is recommended. ChromaDB has compatibility issues with Python 3.14.

### Ollama

```bash
brew install ollama          # macOS
brew services start ollama
ollama pull nomic-embed-text # embedding model
ollama pull llama3.2         # LLM
```

---

## Quick Start

```bash
# 1. Download TEK17 snapshot
python -m tek17 download-dibk

# 2. Parse into structured corpus
python -m tek17 parse-dibk

# 3. Ingest into vector store
python -m tek17 ingest

# 4. Start RAG server (terminal 1)
python -m tek17 serve

# 5. Launch Streamlit UI (terminal 2)
python -m tek17 ui
```

---

## CLI Reference

| Command | Description |
|---|---|
| `python -m tek17 hello` | Sanity check |
| `python -m tek17 download-dibk` | Download TEK17 root-print snapshot |
| `python -m tek17 parse-dibk` | Parse snapshot into per-§ JSONL |
| `python -m tek17 ingest` | Chunk, embed and store in ChromaDB |
| `python -m tek17 serve` | Start FastAPI RAG server (port 8000) |
| `python -m tek17 ui` | Launch Streamlit chat UI (port 8501) |

All commands support `--help` for full option docs.

---

## Repository Structure

```
src/tek17/
├── __init__.py
├── __main__.py
├── cli.py                  # Typer CLI – all commands
├── corpus/                 # Data extraction pipeline
│   ├── __init__.py
│   ├── download.py         # Download TEK17 root-print from DiBK
│   └── parse.py            # Parse HTML into per-§ JSONL records
├── rag/                    # Retrieval-augmented generation
│   ├── __init__.py
│   ├── ingest.py           # Chunk → embed → ChromaDB
│   └── server.py           # FastAPI server with /query endpoint
└── app/                    # Frontend
    ├── __init__.py
    └── ui.py               # Streamlit chat client
```

### Data directories (gitignored)

```
data/
├── raw/                    # Downloaded HTML snapshots + manifest
├── processed/              # Parsed JSONL corpus
└── vectorstore/            # ChromaDB persistent store
```

---

## Output Schema

Each JSONL record represents one provision (§ x-y):

```json
{
  "source": "dibk",
  "doc_type": "provision",
  "section_id": "§ 1-1",
  "title": "Formål",
  "chapter": "Kapittel 1 Felles bestemmelser",
  "reg_text": "...",
  "guidance_text": "...",
  "full_text": "..."
}
```

For retrieval experiments, `reg_text` and `guidance_text` are chunked separately.

---

## Reproducibility

- Root-print snapshot logged in append-only manifest
- SHA256 hash stored for integrity verification
- Parsing is deterministic

To regenerate the corpus:

```bash
python -m tek17 download-dibk --force
python -m tek17 parse-dibk
python -m tek17 ingest
```

### Canonical TEK17 snapshot for this project

For all experiments, analyses and evaluation in this repo we use a **fixed TEK17 snapshot** to ensure everyone (group + examiner) works on the exact same document version.

- **Root-print URL:** https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17?subtype=root&print=true
- **Canonical URL:** https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17
- **Downloaded at:** `2026-03-13T11:56:22`
- **Raw HTML file:** `data/raw/dibk_root_print/2026-03-13/tek17_full_root_print.html`
- **Processed corpus file:** `data/processed/tek17_dibk.jsonl`
- **SHA256 (raw HTML):** `56e9dd740ea27b3b699c45285f2c4a21fcab455a85e4d00f69379ad44eb618f5`

All RAG indexing, retrieval and refusal analysis in this project is expected to be based on this snapshot. If you re-download TEK17 at a later date, treat it as a **new version** and do not overwrite this canonical snapshot.

````
