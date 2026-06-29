"""
FastAPI entrypoint for the local Contract Review & Q&A RAG app.

Run with:  uvicorn main:app --reload
(with `ollama serve` running separately in the background)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import BASE_DIR, ensure_dirs
from app.document_loader import EmptyDocumentError, UnsupportedFileType
from app.document_manager import DocumentNotFoundError, manager
from app.embeddings import EmbeddingError
from app.llm_client import check_ollama_status
from app.rag_pipeline import answer_question
from app.vector_store import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load any existing FAISS index + documents registry from
    # disk so previously uploaded contracts are immediately queryable —
    # this is what makes uploads survive an app restart.
    ensure_dirs()
    store.load()
    manager.load()
    logger.info(
        "Startup complete: %d document(s), %d vector(s) loaded from disk.",
        len(manager.documents), store.total_vectors,
    )
    yield
    # Nothing special needed on shutdown — every mutation is persisted
    # immediately after the request that caused it.


app = FastAPI(title="Contract Review & Q&A RAG", lifespan=lifespan)


# --------------------------------------------------------------------------
# Request/response models
# --------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    doc_id: str = "all"


# --------------------------------------------------------------------------
# API routes (registered before the static-file catch-all mount below)
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    status = check_ollama_status()
    status["ok"] = bool(
        status["ollama_reachable"] and status["embed_model_ready"] and status["llm_model_ready"]
    )
    return status


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail=f"'{file.filename}' is empty.")

    try:
        result = manager.upload_document(file.filename, file_bytes)
        return result
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmptyDocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmbeddingError as e:
        # Most likely Ollama isn't running or the embedding model isn't
        # pulled — surface that clearly rather than a generic 500.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error processing upload '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Failed to process '{file.filename}': {e}")


@app.get("/documents")
def list_documents():
    return manager.list_documents()


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    doc_id: Optional[str] = req.doc_id if req.doc_id and req.doc_id != "all" else "all"
    if doc_id != "all" and not manager.exists(doc_id):
        raise HTTPException(status_code=404, detail=f"No document with doc_id '{doc_id}'.")

    result = answer_question(req.question, doc_id=doc_id)
    return result


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        removed = manager.delete_document(doc_id)
        return {"doc_id": doc_id, "chunks_removed": removed, "status": "deleted"}
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --------------------------------------------------------------------------
# Static frontend — mounted last so it acts as a catch-all under "/"
# without shadowing the API routes registered above. html=True makes
# "/" resolve to static/index.html automatically.
# --------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
