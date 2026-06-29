"""
The actual RAG flow: embed question -> retrieve chunks -> build a
source-labeled prompt -> ask the LLM -> map its claimed sources back to
real citations.

This module is the trickiest part of the whole app, per the project
spec, so the citation-mapping logic below is commented in detail.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from app.config import TOP_K
from app.embeddings import EmbeddingError, embed_text
from app.llm_client import LLMError, chat_completion
from app.vector_store import store

logger = logging.getLogger(__name__)

NO_INFO_MESSAGE = (
    "I don't have enough information in the uploaded documents to answer that."
)

SYSTEM_PROMPT = """You are a contract review assistant. You answer questions \
strictly using the numbered sources provided in the user message — never from \
outside/general knowledge, even if you happen to know the answer.

Rules:
1. If the sources do not contain enough information to answer, respond exactly: \
"I don't have enough information in the uploaded documents to answer that." Do not guess.
2. Be concise and direct. Quote or closely paraphrase the relevant contract language \
where helpful.
3. After your answer, on its own final line, write exactly:
Sources used: [n, n, ...]
listing the number(s) of the sources you actually relied on (e.g. "Sources used: [1, 3]"). \
If you used none, write "Sources used: []".
"""


class Citation(TypedDict):
    doc_name: str
    page_or_section: str
    excerpt: str


class AskResult(TypedDict):
    answer: str
    citations: List[Citation]


def _build_source_block(retrieved: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks as numbered, source-labeled blocks the LLM
    can refer back to by number, e.g.:

        [Source 1: ABC_Vendor_Agreement.pdf, Page 4]
        <chunk text>
    """
    blocks = []
    for i, chunk in enumerate(retrieved, start=1):
        header = f"[Source {i}: {chunk['doc_name']}, {chunk['page_or_section']}]"
        blocks.append(f"{header}\n{chunk['chunk_text']}")
    return "\n\n".join(blocks)


# Matches "Sources used: [1, 3]" (any case, any whitespace, optionally no
# brackets' worth of trailing punctuation). We search the *whole* response
# rather than assuming it's the last line, since smaller local models
# sometimes add trailing whitespace/newlines or extra commentary after it.
_SOURCES_USED_RE = re.compile(r"sources\s+used\s*:\s*\[([^\]]*)\]", re.IGNORECASE)


def _parse_sources_used(raw_response: str) -> tuple[str, Optional[List[int]]]:
    """Split the model's raw response into (clean_answer_text, source_numbers).

    Returns source_numbers=None if the "Sources used: [...]" marker could
    not be found/parsed at all (a true parse failure). Returns an empty
    list [] if the marker was found but explicitly listed no sources, or
    listed values that weren't parseable integers.
    """
    match = _SOURCES_USED_RE.search(raw_response)
    if not match:
        return raw_response.strip(), None

    # Strip the marker (and everything from it onward) out of the
    # user-visible answer text.
    clean_answer = raw_response[: match.start()].strip()

    numbers_str = match.group(1).strip()
    if not numbers_str:
        return clean_answer, []

    numbers: List[int] = []
    for piece in numbers_str.split(","):
        piece = piece.strip()
        if piece.isdigit():
            numbers.append(int(piece))
        # silently skip anything non-numeric (e.g. model wrote "1, three") —
        # we still return whatever valid numbers we did find.
    return clean_answer, numbers


def _citation_from_chunk(chunk: Dict[str, Any]) -> Citation:
    return {
        "doc_name": chunk["doc_name"],
        "page_or_section": chunk["page_or_section"],
        "excerpt": chunk["chunk_text"],
    }


def _resolve_citations(
    retrieved: List[Dict[str, Any]], source_numbers: Optional[List[int]]
) -> List[Citation]:
    """Map the LLM's claimed source numbers (1-indexed, matching the order
    chunks were listed in the prompt) back to real citations.

    Falls back to returning *all* retrieved chunks as citations whenever
    we'd otherwise end up with zero — covering three distinct failure
    modes of small local models:
      (a) it never wrote a parseable "Sources used: [...]" line at all,
      (b) it wrote the line but every number in it was out of range/garbage,
      (c) it wrote "Sources used: []" (claims no sources) even though we
          *did* retrieve plausibly-relevant chunks for the question.
    In all three cases, showing the retrieved chunks lets the user verify
    the grounding themselves rather than seeing an answer with no
    citations at all, which the spec explicitly disallows whenever chunks
    were retrieved in the first place.
    """
    if not retrieved:
        # Nothing was retrieved at all (empty index / empty doc filter) —
        # zero citations is the only honest option here.
        return []

    valid_indices = []
    if source_numbers:
        valid_indices = [n for n in source_numbers if 1 <= n <= len(retrieved)]

    if not valid_indices:
        logger.info(
            "Citation parsing fallback triggered (parsed=%s) — returning all %d retrieved chunks.",
            source_numbers, len(retrieved),
        )
        return [_citation_from_chunk(c) for c in retrieved]

    return [_citation_from_chunk(retrieved[i - 1]) for i in valid_indices]


def answer_question(question: str, doc_id: Optional[str] = None, top_k: int = TOP_K) -> AskResult:
    """Run the full RAG flow for one question and return {answer, citations}."""
    question = (question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "citations": []}

    # 1. Embed the question.
    try:
        query_vector = embed_text(question)
    except EmbeddingError as e:
        return {"answer": f"Could not reach the embedding model: {e}", "citations": []}

    # 2. Retrieve top_k chunks, optionally filtered to a single document.
    retrieved = store.search(query_vector, top_k=top_k, doc_id=doc_id)

    if not retrieved:
        # Nothing in the index at all (or nothing for the selected doc) —
        # short-circuit without calling the LLM. There is nothing for it
        # to ground an answer in, and no citations are possible.
        return {"answer": NO_INFO_MESSAGE, "citations": []}

    # 3. Build the source-labeled prompt and ask the LLM.
    source_block = _build_source_block(retrieved)
    user_prompt = (
        f"{source_block}\n\n"
        f"Question: {question}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = chat_completion(messages)
    except LLMError as e:
        return {"answer": f"Could not reach the language model: {e}", "citations": []}

    # 4. Parse "Sources used: [...]" out of the response.
    clean_answer, source_numbers = _parse_sources_used(raw_response)
    if not clean_answer:
        # Model put everything into the sources line somehow; fall back to
        # showing the raw response rather than an empty bubble.
        clean_answer = raw_response.strip()

    # 5. Map source numbers -> real citations, with the never-zero fallback.
    citations = _resolve_citations(retrieved, source_numbers)

    return {"answer": clean_answer, "citations": citations}
