# TEK17 RAG

RAG system for TEK17 (Byggteknisk forskrift) with:

- A reproducible TEK17 corpus pipeline (download → parse → chunks)
- Local ChromaDB vector store and Ollama-based LLM
- Evaluation of retrieval quality and refusal behaviour

The goal is that anyone in the group can:

1. Rebuild the corpus and vector store
2. Run the RAG API + UI
3. Run the eval scripts and understand what they measure

---

## 1. Setup

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

> Python 3.11 is recommended. ChromaDB has issues on 3.14.

Install and prepare Ollama (macOS example):

```bash
brew install ollama
brew services start ollama
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2           # LLM
```

---

## 2. End-to-end quick start

From a clean checkout with the virtualenv activated:

```bash
# 1) Download TEK17 root-print snapshot from DiBK
python -m tek17 download-dibk

# 2) Parse HTML into one JSONL record per §
python -m tek17 parse-dibk

# 3) Chunk, embed and ingest into ChromaDB
python -m tek17 ingest

# 4) Start the RAG API server (terminal 1)
python -m tek17 serve

# 5) Start the Streamlit chat UI (terminal 2)
python -m tek17 ui
```

Then open the Streamlit URL (default `http://localhost:8501`) and ask questions about TEK17.

---

## 3. CLI commands (what they do)

All main commands are under the `tek17` package (Typer-based CLI):

| Command | What it does |
| --- | --- |
| `python -m tek17 hello` | Quick sanity check that CLI wiring works |
| `python -m tek17 download-dibk` | Download TEK17 root-print snapshot from DiBK |
| `python -m tek17 parse-dibk` | Parse snapshot into per-§ JSONL (`data/processed/tek17_dibk.jsonl`) |
| `python -m tek17 ingest` | Chunk TEK17, embed with Ollama, store in ChromaDB |
| `python -m tek17 serve` | Start FastAPI RAG server on port 8000 |
| `python -m tek17 ui` | Start Streamlit chat UI on port 8501 |

Each command supports `--help` for more options.

---

## 4. How the system works (high level)

When a user asks a question (via UI or API):

1. **Embedding** – the server calls `embed_query(...)` (Ollama `nomic-embed-text`) on the question.
2. **Retrieval** – ChromaDB is queried for the most similar TEK17 chunks.
3. **Context building** – the retrieved chunks are concatenated into a context block.
4. **LLM call** – the context + question are sent to the LLM (Ollama `llama3.2` by default).
5. **Response** – the API returns the answer and the list of source chunks (with §, title, chapter, etc.).
6. **Logging** – each `/query` call is written to `analysis/logging/rag_queries.jsonl` for later analysis.

This makes it possible to inspect exactly what the model saw when it answered or refused.

---

## 5. Repository structure

Core Python package (under `src/tek17`):

```text
src/tek17/
├── __init__.py
├── __main__.py           # Entrypoint: python -m tek17
├── cli.py                # Typer CLI – wires all commands
├── app/
│   ├── __init__.py
│   └── ui.py             # Streamlit chat frontend
├── corpus/               # TEK17 corpus pipeline (no ML)
│   ├── __init__.py
│   ├── download.py       # Download TEK17 root-print from DiBK
│   ├── parse.py          # Parse HTML into per-§ JSONL records
│   └── chunks.py         # Build and save text chunks for RAG
└── rag/                  # RAG stack (embeddings, retrieval, LLM, eval)
    ├── __init__.py
    ├── config.py         # Central configuration (paths, models, etc.)
    ├── ingest.py         # Orchestrate chunking → embeddings → ChromaDB
    ├── server.py         # FastAPI RAG server with /query, /models, /stats
    ├── embedding/
    │   ├── __init__.py
    │   ├── client.py     # Embedding API (Ollama, pluggable later)
    │   └── chroma_ingest.py # Read chunks, embed and upsert into Chroma
    ├── retrieval/
    │   ├── __init__.py
    │   └── client.py     # Chroma client and query wrapper
    ├── llm/
    │   ├── __init__.py
    │   └── client.py     # LLM client (Ollama chat; multi-LLM ready)
    ├── eval_retrieval.py # Script: retrieval quality evaluation
    └── eval_refusal.py   # Script: refusal behaviour evaluation
```

Data and analysis directories (gitignored):

```text
data/
├── raw/                      # Downloaded HTML snapshots + manifest
│   └── dibk_root_print/
├── processed/                # Parsed corpus and chunked text
│   ├── tek17_dibk.jsonl      # One record per provision (§ x-y)
│   └── tek17_chunks.jsonl    # Chunked texts with metadata
└── vectorstore/
    └── chroma/               # ChromaDB persistent store

analysis/
├── logging/
│   └── rag_queries.jsonl     # One JSON object per /query request
└── questions/
    ├── README.md             # Evaluation schema and guidance
    ├── tek17_eval_questions.example.jsonl
    └── tek17_eval_questions.dibk_example.jsonl
```

---

## 6. Corpus and chunk schema

After parsing, the main TEK17 corpus lives in `data/processed/tek17_dibk.jsonl`.
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

For RAG, `corpus/chunks.py` turns each provision into smaller chunks and
writes them to `data/processed/tek17_chunks.jsonl`:

```json
{
  "text": "...",
  "metadata": {
    "source": "dibk",
    "section_id": "§ 12-14",
    "title": "Trapper",
    "chapter": "Kapittel 12 Planløsning og bygningsdeler",
    "text_type": "reg_text"  
  }
}
```

These metadata are what later show up as `sources` in the API responses.

---

## 7. Evaluation and refusal analysis

Evaluation question sets live under `analysis/questions/`:

- `README.md` – describes the JSONL schema:
  - `id`, `question`, `target_sections`, `difficulty`, `should_refuse`, `notes`.
- `tek17_eval_questions.example.jsonl` – small generic example.
- `tek17_eval_questions.dibk_example.jsonl` – questions based on DiBK’s “gråsonespørsmål”.

Two scripts (run as modules) help you measure performance:

1. **Retrieval-only evaluation** (`rag/eval_retrieval.py`)

   Checks how often we retrieve at least one of the `target_sections`.

   ```bash
   python -m tek17.rag.eval_retrieval \
     --eval-file analysis/questions/tek17_eval_questions.dibk_example.jsonl
   ```

2. **Refusal behaviour evaluation** (`rag/eval_refusal.py`)

   Requires the RAG server to be running.
   It sends each eval `question` to `/query`, detects refusals using a simple
   text heuristic, and compares with the `should_refuse` label.

   ```bash
   python -m tek17 serve
   python -m tek17.rag.eval_refusal \
     --eval-file analysis/questions/tek17_eval_questions.dibk_example.jsonl
   ```

Both scripts print per-question results and simple summary metrics.

---

## 8. Reproducibility and snapshot

- Root-print snapshot is logged in an append-only manifest.
- SHA256 hash is stored so we can verify the HTML has not changed.
- Parsing and chunking are deterministic given the same input.

To **rebuild** the corpus and vector store from the canonical snapshot:

```bash
python -m tek17 download-dibk --force   # optional, keeps manifest history
python -m tek17 parse-dibk
python -m tek17 ingest
```

Canonical TEK17 snapshot used in this project:

- **Root-print URL:** https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17?subtype=root&print=true
- **Canonical URL:** https://www.dibk.no/regelverk/byggteknisk-forskrift-tek17
- **Downloaded at:** `2026-03-13T11:56:22`
- **Raw HTML file:** `data/raw/dibk_root_print/2026-03-13/tek17_full_root_print.html`
- **Processed corpus file:** `data/processed/tek17_dibk.jsonl`
- **SHA256 (raw HTML):** `56e9dd740ea27b3b699c45285f2c4a21fcab455a85e4d00f69379ad44eb618f5`

All RAG indexing, retrieval and refusal analysis in this project is
based on this snapshot. If you re-download TEK17 later, treat it as a
**new version** and do not overwrite this canonical snapshot.

