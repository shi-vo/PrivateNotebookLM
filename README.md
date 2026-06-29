# Contract Review & Q&A — Local RAG App

A fully functional, local-first Retrieval-Augmented Generation (RAG) application for contracts. Upload PDF, DOCX, or TXT contracts, ask questions about them in plain English, and get answers grounded only in what you uploaded — with citations pointing back to the exact page or section they came from. No data and no API calls ever leave your machine.

## Features

- Upload PDF, DOCX, or TXT contracts through a simple browser UI — no command line needed after setup.
- Ask natural-language questions; answers are grounded only in your uploaded documents, never the model's general knowledge.
- Every answer comes with clickable citation tags showing the exact source document, page/section, and excerpt.
- If nothing relevant is found, the app says so explicitly instead of guessing — answers are never shown without at least one supporting citation.
- Scope questions to a single document or search across all of them, via a simple dropdown.
- Documents and their vector index persist to disk — restart the server and everything is still there, no re-uploading.
- One-click document removal, with the underlying vectors deleted immediately.
- A `/health` endpoint and UI banner that tell you exactly which `ollama pull` commands to run if a model isn't ready.

## How It Works

```
Upload (PDF/DOCX/TXT)
   → Extract text (page-by-page for PDFs, section-by-section for DOCX/TXT)
   → Split into overlapping, sentence-bounded chunks (~700 tokens, ~100 overlap)
   → Embed each chunk (Ollama: nomic-embed-text)
   → Store vectors + metadata in a FAISS index, saved to disk

Question
   → Embed the question the same way
   → FAISS similarity search (top 5 chunks, optionally restricted to one document)
   → Build a prompt that labels each chunk as a numbered source
   → Local LLM (Ollama: qwen2.5) answers using only those sources
   → Parse which sources the model actually cited; if parsing fails or
     the model didn't follow the format, fall back to showing all
     retrieved chunks as citations rather than showing none
   → Return the answer + citations to the browser
```

The backend is FastAPI; the frontend is plain HTML/CSS/JS with no framework and no build step, served directly by FastAPI.

## Why Fully Local

This is the core design constraint, not an afterthought. Both the embedding model (`nomic-embed-text`) and the chat model (`qwen2.5:7b-instruct`, or a smaller `3b` variant on lighter hardware) run through [Ollama](https://ollama.com) on your own machine. The FAISS index, uploaded files, and all chunk metadata are stored as plain files on your disk. Once the models are pulled, the app needs no internet connection to function — your contracts and the questions you ask about them never leave your computer.

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Embeddings + LLM | Ollama (`nomic-embed-text`, `qwen2.5:7b-instruct`) |
| Vector store | FAISS (`faiss-cpu`), persisted to disk |
| Document parsing | `pypdf` / `pdfplumber` (PDF), `python-docx` (DOCX) |
| Chunking | `tiktoken`-based token counting with sentence-aware splitting |
| Config | `.env` via `python-dotenv` |

## Project Structure

```
main.py                  FastAPI app, route definitions
app/
  config.py              All settings, loaded from .env
  document_loader.py     PDF/DOCX/TXT text extraction
  chunker.py              Token-aware chunking with overlap
  embeddings.py           Ollama embedding calls (with retry)
  vector_store.py         FAISS index + metadata, load/save/search
  llm_client.py           Ollama chat calls + /health check
  rag_pipeline.py         Retrieval, prompting, citation parsing/fallback
  document_manager.py     Upload/delete orchestration + doc registry
static/                  index.html, style.css, script.js (vanilla JS)
data/                    uploads/, faiss_index/, processed/ — created on first run
```

## Prerequisites

- Python 3.10+ (3.13 supported)
- [Ollama](https://ollama.com) installed and runnable as `ollama serve`
- ~8GB+ free RAM/VRAM (16GB+ recommended for the default 7B model)

## Setup

```bash
cd RAG_MVP
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # defaults work out of the box
```

Pull the models (one-time, requires internet just for this step):

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct
```

If you have less than ~16GB of RAM/VRAM, use the smaller chat model instead — pull it and set `LLM_MODEL=qwen2.5:3b-instruct` in `.env`:

```bash
ollama pull qwen2.5:3b-instruct
```

## Running

In one terminal:

```bash
ollama serve
```

In another, from the project root:

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000`. If Ollama isn't reachable or a model isn't pulled yet, the page shows a banner with the exact command to fix it.

## Using It

1. Upload one or more contracts. Each shows up in the document list as soon as indexing finishes — no page reload needed.
2. Pick a scope from the dropdown: "All Documents" searches everything; selecting a specific document restricts answers to just that one.
3. Ask a question. Each answer comes with citation tags — click one to expand the exact excerpt (with page or section) it was drawn from.
4. If nothing relevant was found in your documents, the assistant says so instead of guessing. Every answer that isn't a flat "no information found" is backed by at least one citation; this is enforced server-side even if the local model's output format is unreliable (see the comments in `app/rag_pipeline.py`).
5. Remove a document with the "Remove" button — its vectors are deleted from the index immediately and the change is persisted to disk.

## API Reference

| Endpoint | Description |
|---|---|
| `GET /health` | Ollama reachability + model readiness, with exact `ollama pull` commands if something's missing |
| `POST /upload` | Multipart file upload → `{doc_id, filename, chunk_count, status}` |
| `GET /documents` | List of indexed documents → `[{doc_id, filename, upload_date, chunk_count}]` |
| `POST /ask` | `{question, doc_id}` (`doc_id`: `"all"` or a specific id) → `{answer, citations}` |
| `DELETE /documents/{doc_id}` | Removes a document and all of its vectors |

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Change if Ollama runs elsewhere |
| `EMBED_MODEL` | `nomic-embed-text` | Swappable without code changes |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Use `qwen2.5:3b-instruct` on constrained hardware |
| `CHUNK_SIZE` | `700` | Tokens per chunk (approximate) |
| `CHUNK_OVERLAP` | `100` | Token overlap between consecutive chunks |
| `TOP_K` | `5` | Chunks retrieved per question |

## Persistence

The FAISS index and document registry live under `data/faiss_index/` and are written to disk after every upload and delete. Restarting the server (`Ctrl-C`, then `uvicorn main:app --reload` again) reloads them — your documents stay queryable without re-uploading. This was verified directly: uploads and deletes made before a restart are still correct, and still correctly searchable, after a simulated restart with a fresh process.

## Scope & Limitations (MVP)

This is intentionally a minimal, working MVP rather than a scalable product:

- Single-user, no accounts or authentication — meant to run locally for one person.
- No folders, tags, or document organization — a flat list of uploaded documents.
- No multi-document checkbox selection — scope is either "All Documents" or exactly one document.
- Document-filtered search is brute-force (search the full index, then filter) — fine at MVP scale (a few dozen contracts), not optimized for large corpora.
- Answer quality depends entirely on the local model you run; smaller models (e.g. the `3b` fallback) are less reliable at following the citation-output format, which is why the server-side fallback logic in `app/rag_pipeline.py` exists.

## Extending: Swapping in OpenAI's API

The app isn't locked into Ollama by accident — `app/embeddings.py` and `app/llm_client.py` are the only two places that actually talk to a model, and both are thin wrappers around a model name and base URL already read from `.env`. If you ever wanted higher-quality answers, or to run this on a machine without a GPU, swapping in OpenAI's API would mean rewriting those two files to call OpenAI instead of Ollama and adding an API key to `.env` — nothing else in the pipeline (chunking, FAISS storage, citation logic, the frontend) would need to change. It isn't wired up today because the point of this version is "100% local, no API calls," but the architecture doesn't fight you if that changes later.

## Notes on Testing

The pipeline (chunking, persistence, citation parsing/fallback, doc-scoped filtering, the API layer end to end) was verified offline using a deterministic fake stand-in for Ollama. That covers all the logic that doesn't depend on actual model quality. What it can't verify is real answer quality — do a quick smoke test on your machine with `ollama serve` running: upload a contract, ask a question you know the answer to, and confirm the citation actually supports the answer.
