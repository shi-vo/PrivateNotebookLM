"""
FAISS-backed vector store with JSON-backed metadata, persisted to disk so
uploaded documents survive an app restart.

Design notes:
- We use a flat (IndexFlatIP) index wrapped in IndexIDMap so each vector
  keeps a stable integer id across add/remove/save/load cycles. Vectors
  are L2-normalized before insertion and before querying, so inner
  product = cosine similarity.
- Metadata (doc_id, doc_name, page_or_section, chunk_text) is stored in
  a plain dict keyed by the same integer id (as a string, since JSON
  object keys must be strings), in metadata.json next to index.faiss.
- "Filter by doc_id" is implemented as brute-force post-filtering: at
  MVP scale (a few dozen contracts, low thousands of chunks) this is
  fast and far simpler than maintaining per-document sub-indices. When
  a doc filter is active we search the *entire* index (k = ntotal) so
  we don't miss matches, then keep only chunks from that doc and take
  the first top_k by score.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K, ensure_dirs

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.next_id: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load an existing index + metadata from disk if present,
        otherwise start from an empty store. Called once on app startup."""
        ensure_dirs()
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            try:
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                self.dim = self.index.d
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.metadata = raw.get("metadata", {})
                self.next_id = raw.get("next_id", 0)
                logger.info(
                    "Loaded existing FAISS index: %d vectors, dim=%d, %d metadata entries",
                    self.index.ntotal, self.dim, len(self.metadata),
                )
            except Exception as e:
                logger.error("Failed to load existing index/metadata, starting fresh: %s", e)
                self.index = None
                self.dim = None
                self.metadata = {}
                self.next_id = 0
        else:
            logger.info("No existing FAISS index found at %s — starting with an empty store.", FAISS_INDEX_PATH)
            self.index = None
            self.dim = None
            self.metadata = {}
            self.next_id = 0

    def save(self) -> None:
        """Persist the current index + metadata to disk. Called after
        every successful upload and delete so an unexpected crash never
        loses more than the in-flight request."""
        ensure_dirs()
        with self._lock:
            if self.index is not None:
                faiss.write_index(self.index, str(FAISS_INDEX_PATH))
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump({"metadata": self.metadata, "next_id": self.next_id}, f)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.dim = dim
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        elif self.dim != dim:
            raise ValueError(
                f"Embedding dimension mismatch: existing index has dim {self.dim}, "
                f"new vectors have dim {dim}. Did the embedding model change? "
                f"Delete data/faiss_index/ to reset if you switched EMBED_MODEL."
            )

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[int]:
        """Add vectors + parallel metadata dicts. Returns the assigned row ids."""
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas must be the same length")
        if not vectors:
            return []

        with self._lock:
            arr = np.array(vectors, dtype="float32")
            self._ensure_index(arr.shape[1])
            faiss.normalize_L2(arr)

            ids = np.arange(self.next_id, self.next_id + len(vectors), dtype="int64")
            self.index.add_with_ids(arr, ids)

            for rid, meta in zip(ids.tolist(), metadatas):
                self.metadata[str(rid)] = meta

            self.next_id += len(vectors)
            return ids.tolist()

    def delete_doc(self, doc_id: str) -> int:
        """Remove every chunk belonging to doc_id from the index and
        metadata. Returns the number of chunks removed."""
        with self._lock:
            ids_to_remove = [int(k) for k, meta in self.metadata.items() if meta.get("doc_id") == doc_id]
            if ids_to_remove and self.index is not None:
                self.index.remove_ids(np.array(ids_to_remove, dtype="int64"))
            for k in (str(i) for i in ids_to_remove):
                self.metadata.pop(k, None)
            return len(ids_to_remove)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: List[float],
        top_k: int = TOP_K,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to top_k chunks most similar to query_vector, each as
        {score, row_id, doc_id, doc_name, page_or_section, chunk_text}.

        doc_id=None or "all" searches every indexed document. Any other
        value restricts results to that document's chunks only.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        filtering = doc_id is not None and doc_id != "all"
        # When filtering to one document we don't know in advance how
        # many of the globally-top vectors belong to it, so we search
        # the whole index and filter client-side. Fine at MVP scale.
        k = self.index.ntotal if filtering else min(top_k, self.index.ntotal)

        q = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)

        results: List[Dict[str, Any]] = []
        for score, rid in zip(scores[0].tolist(), ids[0].tolist()):
            if rid == -1:
                continue
            meta = self.metadata.get(str(rid))
            if meta is None:
                continue
            if filtering and meta.get("doc_id") != doc_id:
                continue
            results.append({"score": float(score), "row_id": int(rid), **meta})
            if len(results) >= top_k:
                break
        return results

    def chunk_count_for_doc(self, doc_id: str) -> int:
        return sum(1 for meta in self.metadata.values() if meta.get("doc_id") == doc_id)

    @property
    def total_vectors(self) -> int:
        return 0 if self.index is None else self.index.ntotal


# Module-level singleton — one store per process, loaded once at startup
# in main.py's lifespan handler and shared by every request.
store = VectorStore()
