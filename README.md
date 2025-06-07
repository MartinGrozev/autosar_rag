# AUTOSAR Official Documentation RAG Assistant

This project implements a local Retrieval-Augmented Generation (RAG) assistant for answering AUTOSAR documentation questions using your local official PDF specs. It is designed for laptops with no GPU, running on CPU (tested on Core i5 10th Gen with 32GB RAM).

## Features

- **PDF Chunking & Indexing:** Parses AUTOSAR PDFs, cleans and chunks text, and builds a hybrid vector (FAISS) and BM25 index.
- **Embedding:** Uses Sentence Transformers to create dense embeddings for efficient semantic search.
- **Hybrid Search:** Combines FAISS dense search and BM25 keyword search with Reciprocal Rank Fusion.
- **API Server:** FastAPI server provides `/ask` endpoint for answering questions (filterable by document, module, or keyword).
- **Ollama Integration:** Optionally uses a local Ollama server for LLM-based answer generation from retrieved context.
- **Modular Design:** Easily extend or replace the embedding model, LLM, or search strategy.

## Setup

1. **Clone the repository** (or copy project files).
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Prepare AUTOSAR PDFs:**
    - Place all desired AUTOSAR PDF documents in a directory (default: `C:/Users/HP/Desktop/AUTOSAR`).
4. **Index the documents:**
    - Edit `main.py` and set `AUTOSAR_PATH` to your PDF folder.
    - Run:
        ```bash
        python main.py
        ```
    - This will parse PDFs, extract text chunks, and create FAISS/BM25 indices in the `faiss_data` directory.

5. **Start the API server:**
    ```bash
    uvicorn rag_server:app --reload
    ```
    - The FastAPI server will be available at `http://localhost:8000`.

6. **Query the Assistant:**
    - Send POST requests to `http://localhost:8000/ask` with your question (optionally filter by source/module/keyword).

## File Overview

- `main.py` — Simple entry point to run the document indexer.
- `rag_indexer.py` — Handles PDF parsing, chunking, keyword extraction, embedding, and index creation.
- `rag_query.py` — Provides hybrid search, context assembly, and prompt generation.
- `utils_embeddings.py` — SentenceTransformer model loader and embedding functions.
- `utils_text_processing.py` — PDF chunker, keyword extractor, BM25 tokenizer, and text utilities.
- `rag_server.py` — FastAPI REST API exposing the `/ask` endpoint.

## FAQ / Troubleshooting

- **Q: How do I add new documents?**
  - Add your PDF to the AUTOSAR path and re-run `main.py` to re-index.
- **Q: I get errors about missing NLTK stopwords.**
  - Run `python -m nltk.downloader stopwords` if you want full NLTK support, or ignore (the script will use a basic list).
- **Q: How do I change the embedding model?**
  - Edit `MODEL_NAME` in `utils_embeddings.py`.

## Hardware Requirements

- No GPU required. Works on CPU with 16GB+ RAM recommended for large docsets.

## Credits

- Inspired by and using open-source libraries: [FAISS](https://github.com/facebookresearch/faiss), [Sentence Transformers](https://www.sbert.net/), [rank-bm25](https://github.com/dorianbrown/rank_bm25), [FastAPI](https://fastapi.tiangolo.com/), [PyMuPDF](https://github.com/pymupdf/PyMuPDF).
- For AUTOSAR specs, see [AUTOSAR](https://www.autosar.org/).

---

*This project is for local developer use with official AUTOSAR documentation. It does not include or redistribute any copyrighted AUTOSAR material.*
