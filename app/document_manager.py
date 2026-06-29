"""
Orchestrates the upload pipeline (extract -> chunk -> embed -> index ->
persist) and owns the lightweight documents registry used to populate
the frontend's dropdown and to validate doc_id on /ask and DELETE.

The registry (data/faiss_index/documents.json) is deliberately separate
from the FAISS chunk-level metadata in vector_store.py: this file tracks
one record per *document* (filename, upload date, chunk count, where
the original upload is stored on disk), while vector_store's metadata
tracks one record per *chunk*. Keeping them separate means deleting a
document is just "drop its chunks from the vector store" + "drop its
one registry entry" + "delete its file" — no need to scan chunk
metadata to answer "what documents do I have".
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.chunker import chunk_units
from app.config import (
    DOCUMENTS_REGISTRY_PATH,
    SUPPORTED_EXTENSIONS,
    TOP_K,
    UPLOADS_DIR,
    ensure_dirs,
)
from app.document_loader import EmptyDocumentError, UnsupportedFileType, load_document
from app.embeddings import EmbeddingError, embed_texts
from app.vector_store import store

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    pass


class DocumentManager:
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the documents registry from disk. Called once on startup,
        right alongside vector_store.store.load()."""
        ensure_dirs()
        if DOCUMENTS_REGISTRY_PATH.exists():
            try:
                with open(DOCUMENTS_REGISTRY_PATH, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                logger.info("Loaded %d document(s) from registry.", len(self.documents))
            except Exception as e:
                logger.error("Failed to load documents registry, starting empty: %s", e)
                self.documents = {}
        else:
            self.documents = {}

    def _save(self) -> None:
        ensure_dirs()
        with open(DOCUMENTS_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def list_documents(self) -> List[Dict[str, Any]]:
        return [
            {
                "doc_id": doc_id,
                "filename": rec["filename"],
                "upload_date": rec["upload_date"],
                "chunk_count": rec["chunk_count"],
            }
            for doc_id, rec in sorted(self.documents.items(), key=lambda kv: kv[1]["upload_date"])
        ]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(doc_id)

    def exists(self, doc_id: str) -> bool:
        return doc_id in self.documents

    # ------------------------------------------------------------------
    # Upload pipeline
    # ------------------------------------------------------------------
    def upload_document(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """Extract -> chunk -> embed -> add to FAISS -> persist.

        Raises UnsupportedFileType / EmptyDocumentError / EmbeddingError on
        failure — main.py maps these to clean HTTP error responses. On any
        failure, nothing partial is left in the index (we only add to
        FAISS once embeddings for the whole document succeed).
        """
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileType(
                f"Unsupported file type '{ext}'. Supported types: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        ensure_dirs()
        doc_id = uuid.uuid4().hex

        # Store the original upload under a doc_id-prefixed name so two
        # uploads with the same filename never collide on disk, while the
        # human-readable original filename is preserved for citations.
        stored_name = f"{doc_id}{ext}"
        stored_path = UPLOADS_DIR / stored_name
        stored_path.write_bytes(file_bytes)

        try:
            units = load_document(stored_path, doc_id)
            chunks = chunk_units(units)
            if not chunks:
                raise EmptyDocumentError(f"No usable text chunks extracted from '{filename}'.")

            texts = [c["text"] for c in chunks]
            vectors = embed_texts(texts)

            metadatas = [
                {
                    "doc_id": doc_id,
                    "doc_name": filename,
                    "page_or_section": c["page_or_section"],
                    "chunk_text": c["text"],
                }
                for c in chunks
            ]

            with self._lock:
                store.add(vectors, metadatas)
                store.save()

                self.documents[doc_id] = {
                    "filename": filename,
                    "upload_date": datetime.now(timezone.utc).isoformat(),
                    "chunk_count": len(chunks),
                    "stored_filename": stored_name,
                }
                self._save()

            logger.info("Indexed '%s' (doc_id=%s) -> %d chunks.", filename, doc_id, len(chunks))
            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_count": len(chunks),
                "status": "indexed",
            }
        except (UnsupportedFileType, EmptyDocumentError, EmbeddingError):
            # Clean up the file we just wrote since indexing didn't succeed.
            stored_path.unlink(missing_ok=True)
            raise
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str) -> int:
        """Remove a document's vectors, metadata, registry entry, and
        original uploaded file. Returns the number of chunks removed.
        Raises DocumentNotFoundError if doc_id is unknown."""
        with self._lock:
            rec = self.documents.get(doc_id)
            if rec is None:
                raise DocumentNotFoundError(f"No document with doc_id '{doc_id}'.")

            removed = store.delete_doc(doc_id)
            store.save()

            stored_path = UPLOADS_DIR / rec["stored_filename"]
            stored_path.unlink(missing_ok=True)

            del self.documents[doc_id]
            self._save()

            logger.info("Deleted document '%s' (doc_id=%s), removed %d chunks.", rec["filename"], doc_id, removed)
            return removed


# Module-level singleton, mirroring vector_store.store — loaded once at
# startup in main.py's lifespan handler.
manager = DocumentManager()
