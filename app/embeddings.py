"""
Thin wrapper around Ollama's embedding endpoint.

Two things make this more than a one-line call:
1. The first request after `ollama serve` starts (or after a model has
   been evicted from VRAM) can be slow while the model loads, and can
   occasionally fail with a transient connection error. We retry with
   backoff so a single slow load doesn't fail an upload.
2. We batch all chunk texts for a document into one /embed call (the
   Ollama `embed` endpoint accepts a list), which is much faster than
   one HTTP round-trip per chunk.
"""
from __future__ import annotations

import logging
import time
from typing import List

import httpx
from ollama import Client, ResponseError

from app.config import EMBED_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0


class EmbeddingError(Exception):
    """Raised when embedding ultimately fails after retries."""


def _get_client() -> Client:
    return Client(host=OLLAMA_BASE_URL)


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Embed a batch of texts. Returns one vector per input text, in order.

    Retries on transient connection errors / model-loading delays. Raises
    EmbeddingError with a user-actionable message if the model isn't
    pulled or Ollama isn't reachable at all.
    """
    if not texts:
        return []

    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.embed(model=model, input=texts)
            embeddings = response.embeddings
            if embeddings is None or len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Ollama returned {len(embeddings) if embeddings else 0} "
                    f"embeddings for {len(texts)} inputs."
                )
            return [list(e) for e in embeddings]
        except ResponseError as e:
            # e.g. model not pulled -> 404. Not worth retrying.
            if e.status_code == 404 or "not found" in str(e).lower():
                raise EmbeddingError(
                    f"Embedding model '{model}' is not available in Ollama. "
                    f"Run: ollama pull {model}"
                ) from e
            last_error = e
            logger.warning("Ollama embed ResponseError (attempt %d/%d): %s", attempt, _MAX_RETRIES, e)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_error = e
            logger.warning(
                "Ollama unreachable while embedding (attempt %d/%d): %s", attempt, _MAX_RETRIES, e
            )
        except Exception as e:  # pragma: no cover - defensive catch-all
            last_error = e
            logger.warning("Unexpected error embedding (attempt %d/%d): %s", attempt, _MAX_RETRIES, e)

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    raise EmbeddingError(
        f"Could not reach Ollama at {OLLAMA_BASE_URL} to embed text after "
        f"{_MAX_RETRIES} attempts. Is `ollama serve` running? Last error: {last_error}"
    )


def embed_text(text: str, model: str = EMBED_MODEL) -> List[float]:
    """Convenience wrapper for embedding a single string (e.g. a user question)."""
    return embed_texts([text], model=model)[0]
