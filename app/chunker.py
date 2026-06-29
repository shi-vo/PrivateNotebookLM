"""
Turn extracted text units (one per page or section, see document_loader.py)
into overlapping, token-bounded chunks suitable for embedding.

Design choice: chunking happens *within* each unit independently — a
chunk never spans two pages or two sections. This keeps citations exact
(every chunk has exactly one page/section label) at the cost of
occasionally producing a small trailing chunk for short pages/sections.
For contract review, precise "Page 4" style citations matter more than
perfectly even chunk sizes.

Splitting always happens on sentence boundaries (never mid-sentence).
If a single sentence is itself longer than CHUNK_SIZE tokens (rare —
e.g. a huge run-on clause with no punctuation), it is hard-split on
token boundaries as a last resort so the pipeline never crashes on
pathological input.
"""
from __future__ import annotations

import logging
import re
from typing import List, TypedDict

import tiktoken

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, ensure_dirs
from app.document_loader import TextUnit

logger = logging.getLogger(__name__)

# tiktoken's cl100k_base is used purely as an approximate, fast,
# dependency-light token counter. It doesn't match the real Ollama
# model tokenizer (qwen/nomic use their own), but it's a good-enough
# proxy for keeping chunks in a sane size range, per the spec.
#
# IMPORTANT: tiktoken downloads this encoding's BPE file from the
# internet the *first* time it's used on a machine (then caches it
# locally under TIKTOKEN_CACHE_DIR, set in config.py, so every run
# after that is offline). If that one-time download can't happen
# (no internet during setup, or a restrictive network), we fall back
# to a character-count approximation so chunking — and the whole
# app — keeps working fully offline. This only affects how evenly
# chunks are sized, never correctness.
ensure_dirs()
try:
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning(
        "tiktoken's cl100k_base encoding could not be loaded (%s). "
        "Falling back to a character-based token approximation — chunking "
        "will still work, just with less precise token counts.", e,
    )
    _ENCODING = None

_CHARS_PER_TOKEN_APPROX = 4  # standard rule-of-thumb for English text

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(‘“])")


class Chunk(TypedDict):
    doc_id: str
    page_or_section: str
    text: str
    token_count: int


def _count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN_APPROX)


def _split_into_sentences(text: str) -> List[str]:
    """Split on paragraph breaks first, then sentences within each
    paragraph, so paragraph order is preserved and we never merge
    sentences across an intentional blank-line break in a weird order."""
    sentences: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts = _SENTENCE_SPLIT_RE.split(para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def _hard_split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    """Last-resort split for a single sentence that exceeds max_tokens,
    cutting on token boundaries (or character boundaries, if tiktoken's
    encoding isn't available — see module docstring)."""
    if _ENCODING is not None:
        tokens = _ENCODING.encode(sentence)
        return [
            _ENCODING.decode(tokens[i : i + max_tokens])
            for i in range(0, len(tokens), max_tokens)
        ]
    max_chars = max_tokens * _CHARS_PER_TOKEN_APPROX
    return [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]


def chunk_unit(unit: TextUnit, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Chunk a single text unit (one page or section) into one or more
    overlapping Chunks, never splitting a sentence in half."""
    sentences = _split_into_sentences(unit["text"])

    # Break any pathologically long single sentence into token-bounded
    # pieces up front so the packing loop below never has to special-case it.
    normalized_sentences: List[str] = []
    for s in sentences:
        if _count_tokens(s) > chunk_size:
            normalized_sentences.extend(_hard_split_long_sentence(s, chunk_size))
        else:
            normalized_sentences.append(s)
    sentences = normalized_sentences

    if not sentences:
        return []

    sentence_tokens = [_count_tokens(s) for s in sentences]

    chunks: List[Chunk] = []
    start = 0
    n = len(sentences)

    while start < n:
        cur_tokens = 0
        end = start
        while end < n and cur_tokens + sentence_tokens[end] <= chunk_size:
            cur_tokens += sentence_tokens[end]
            end += 1

        # Guarantee progress even if a single sentence alone exceeds
        # chunk_size (shouldn't happen after normalization, but be safe).
        if end == start:
            end = start + 1
            cur_tokens = sentence_tokens[start]

        chunk_text = " ".join(sentences[start:end]).strip()
        chunks.append(
            {
                "doc_id": unit["doc_id"],
                "page_or_section": unit["page_or_section"],
                "text": chunk_text,
                "token_count": cur_tokens,
            }
        )

        if end >= n:
            break

        # Step back from `end` to build ~overlap tokens of context for
        # the next chunk's start, without exceeding chunk_size.
        overlap_tokens = 0
        new_start = end
        while new_start > start and overlap_tokens < overlap:
            new_start -= 1
            overlap_tokens += sentence_tokens[new_start]

        # Avoid an infinite loop if overlap computation didn't advance.
        start = new_start if new_start > start else end

    return chunks


def chunk_units(units: List[TextUnit], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Chunk a full document's worth of text units in order."""
    all_chunks: List[Chunk] = []
    for unit in units:
        all_chunks.extend(chunk_unit(unit, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
