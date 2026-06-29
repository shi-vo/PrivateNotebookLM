# Contract Review & Q&A — What This Is

This is a Retrieval-Augmented Generation (RAG) application for contracts: upload one or more contracts, ask questions about them in plain English, and get answers that are grounded only in what you uploaded, with citations pointing back to the exact page or section they came from. It is a major, fully functional MVP — every core piece of the pipeline (upload, extraction, chunking, embedding, vector search, grounded answering with citations, persistence across restarts) is built, wired together, and working end to end, not a partial proof of concept. It runs entirely on your own computer.

## How it works

When you upload a PDF, DOCX, or TXT file, the backend extracts its text page by page (for PDFs) or section by section (for DOCX/TXT), then splits that text into overlapping chunks of a few hundred words each, always breaking on sentence boundaries so no chunk ends mid-sentence. Each chunk is turned into a vector embedding and stored in a FAISS index on disk, alongside a small metadata record (which document it came from, which page or section, the chunk's text). That index is saved after every upload or delete, so closing and reopening the app doesn't lose anything — your documents stay there and stay searchable.

When you ask a question, the app embeds the question the same way, searches the FAISS index for the most relevant chunks (optionally restricted to one document if you've picked one from the dropdown instead of "All Documents"), and hands those chunks to a local language model with strict instructions to answer using only that material — and to say so plainly if the answer isn't in there, rather than guessing. The model is also asked to report which chunks it actually used, and the app parses that out to build the citation tags you see under each answer. If the model's response doesn't follow that format cleanly (small local models aren't always reliable about this), the app falls back to showing all the retrieved chunks as citations rather than showing an answer with no source at all — an answer is never presented as ungrounded.

The frontend is plain HTML, CSS, and JavaScript — no framework, no build step — talking to a FastAPI backend over a handful of endpoints (upload, ask, list documents, delete, health check).

## Everything runs locally

This is the core design constraint, not an afterthought: there are no calls to any external API, anywhere, at any point. Both the embedding model (`nomic-embed-text`) and the chat model (`qwen2.5:7b-instruct`, or a smaller `3b` variant on lighter hardware) run through [Ollama](https://ollama.com) on your own machine. The FAISS index, the uploaded files, and all chunk metadata are stored as plain files on your disk. Once the models are pulled, the app needs no internet connection to function — your contracts and the questions you ask about them never leave your computer.

## If you ever wanted to use OpenAI's API instead

The app isn't locked into Ollama by accident — `app/embeddings.py` and `app/llm_client.py` are the only two places that actually talk to a model, and both are thin wrappers around a model name and a base URL, both already read from `.env`. If you later wanted higher-quality answers, or to run this on a machine without a GPU, swapping in OpenAI's API would mean rewriting those two files to call OpenAI instead of Ollama and adding an API key to `.env` — nothing else in the pipeline (chunking, FAISS storage, citation logic, the frontend) would need to change. That's a deliberate trade-off to keep in mind: it's not wired up today because the whole point of this version was "100% local, no API calls," but the architecture doesn't fight you if your needs change later.
