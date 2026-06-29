"""
Thin wrapper around Ollama's chat endpoint, plus a health-check helper
used by GET /health to tell the frontend exactly what setup step is
missing (Ollama not running vs. a specific model not pulled).
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, TypedDict

import httpx
from ollama import Client, ResponseError

from app.config import EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 2.0


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMError(Exception):
    """Raised when chat generation ultimately fails after retries."""


def _get_client() -> Client:
    return Client(host=OLLAMA_BASE_URL)


def chat_completion(
    messages: List[ChatMessage],
    model: str = LLM_MODEL,
    temperature: float = 0.1,
) -> str:
    """Send a chat request to Ollama and return the assistant's text.

    temperature defaults low (0.1) since this is a grounded Q&A task —
    we want the model sticking close to the provided sources, not being
    creative.
    """
    client = _get_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
            )
            return response.message.content or ""
        except ResponseError as e:
            if e.status_code == 404 or "not found" in str(e).lower():
                raise LLMError(
                    f"LLM model '{model}' is not available in Ollama. Run: ollama pull {model}"
                ) from e
            last_error = e
            logger.warning("Ollama chat ResponseError (attempt %d/%d): %s", attempt, _MAX_RETRIES, e)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_error = e
            logger.warning("Ollama unreachable while chatting (attempt %d/%d): %s", attempt, _MAX_RETRIES, e)
        except Exception as e:  # pragma: no cover - defensive catch-all
            last_error = e
            logger.warning("Unexpected error during chat (attempt %d/%d): %s", attempt, _MAX_RETRIES, e)

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    raise LLMError(
        f"Could not reach Ollama at {OLLAMA_BASE_URL} for chat generation after "
        f"{_MAX_RETRIES} attempts. Is `ollama serve` running? Last error: {last_error}"
    )


def check_ollama_status() -> dict:
    """Used by GET /health. Returns a dict the frontend can render directly:

        {
            "ollama_reachable": bool,
            "embed_model_ready": bool,
            "llm_model_ready": bool,
            "available_models": [...],
            "missing_commands": ["ollama pull nomic-embed-text", ...],
            "error": str | None,
        }
    """
    result = {
        "ollama_reachable": False,
        "embed_model_ready": False,
        "llm_model_ready": False,
        "available_models": [],
        "missing_commands": [],
        "error": None,
    }
    try:
        client = _get_client()
        response = client.list()
        available = [m.model for m in response.models]
        result["ollama_reachable"] = True
        result["available_models"] = available

        # Ollama model names in `list` often include an implicit ":latest"
        # tag (e.g. "nomic-embed-text:latest") even if the user pulled it
        # without specifying a tag, and configured model names may or may
        # not include a tag themselves — compare on the base name too.
        def _matches(configured: str, available_names: List[str]) -> bool:
            if configured in available_names:
                return True
            base = configured.split(":")[0]
            return any(name.split(":")[0] == base for name in available_names)

        result["embed_model_ready"] = _matches(EMBED_MODEL, available)
        result["llm_model_ready"] = _matches(LLM_MODEL, available)

        if not result["embed_model_ready"]:
            result["missing_commands"].append(f"ollama pull {EMBED_MODEL}")
        if not result["llm_model_ready"]:
            result["missing_commands"].append(f"ollama pull {LLM_MODEL}")
    except Exception as e:
        result["error"] = (
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Start it with `ollama serve`. ({e})"
        )
        result["missing_commands"] = [
            "ollama serve",
            f"ollama pull {EMBED_MODEL}",
            f"ollama pull {LLM_MODEL}",
        ]
    return result
