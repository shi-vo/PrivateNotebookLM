"""
Central configuration, loaded from environment variables / .env.

Every other module imports its settings from here rather than reading
os.environ directly, so there is exactly one place that knows how to
parse and default each setting.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of the app/ package
BASE_DIR = Path(__file__).resolve().parent.parent

# tiktoken (used by chunker.py for approximate token counting) downloads
# its BPE encoding file from the internet the first time it's used,
# unless it finds it in a local cache directory. Pin that cache inside
# data/ so the download happens at most once per machine and every run
# after that is fully offline, per the "100% local" constraint. Must be
# set before tiktoken is imported anywhere.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(BASE_DIR / "data" / "tiktoken_cache"))

# Load .env from project root if present. Safe to call even if the file
# doesn't exist (e.g. first run before the user copies .env.example).
load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")

# --- Chunking ---
CHUNK_SIZE = _get_int("CHUNK_SIZE", 700)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 100)

# --- Retrieval ---
TOP_K = _get_int("TOP_K", 5)

# --- Paths ---
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
UPLOADS_DIR = BASE_DIR / os.getenv("UPLOADS_DIR", "data/uploads")
FAISS_INDEX_DIR = BASE_DIR / os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
PROCESSED_DIR = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")

FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
METADATA_PATH = FAISS_INDEX_DIR / "metadata.json"
DOCUMENTS_REGISTRY_PATH = FAISS_INDEX_DIR / "documents.json"

# Embedding dimension for nomic-embed-text. Determined lazily at runtime
# from the first real embedding call (see embeddings.py) but kept here
# as the documented default for reference.
DEFAULT_EMBED_DIM = 768

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

TIKTOKEN_CACHE_DIR = DATA_DIR / "tiktoken_cache"


def ensure_dirs() -> None:
    """Create all data directories if they don't already exist."""
    for d in (DATA_DIR, UPLOADS_DIR, FAISS_INDEX_DIR, PROCESSED_DIR, TIKTOKEN_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
