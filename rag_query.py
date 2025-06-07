# rag_query.py

import faiss
import pickle
import numpy as np
from utils_embeddings import embed_texts
from rank_bm25 import BM25Okapi
import re
import os # Added for os.path.join
from utils_text_processing import bm25_tokenizer # <--- ADD THIS IMPORT

# Settings
CHUNK_COUNT = 6
#OLLAMA_MODEL = "qwen3:0.6b"
#OLLAMA_MODEL = "gemma3:4b-it-qat"
OLLAMA_MODEL = "qwen3:1.7b"
#OLLAMA_MODEL = "qwen3:4b"
#OLLAMA_MODEL = "qwen3:30b-a3b"
#OLLAMA_MODEL = "qwen3:0.6b-fp16"
#OLLAMA_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-128K-GGUF:latest"

FAISS_CANDIDATES_MULTIPLIER = 5
BM25_CANDIDATES_MULTIPLIER = 5
RRF_K = 60
OUTPUT_DIR = "faiss_data" # Define output_dir for loading files
BM25_TOKENIZED_CORPUS_FILE = os.path.join(OUTPUT_DIR, "bm25_tokenized_corpus.pkl") # New


# Load FAISS index and chunks
print("💾 Loading FAISS index, chunks, and metadata...")
with open(os.path.join(OUTPUT_DIR, "chunks.pkl"), "rb") as f:
    chunks_text_list = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "rb") as f:
    metadata_list = pickle.load(f)
index_faiss = faiss.read_index(os.path.join(OUTPUT_DIR, "index.faiss"))
print("✅ Data loaded.")

print("⚙️ Loading pre-tokenized corpus and initializing BM25...")
try:
    with open(BM25_TOKENIZED_CORPUS_FILE, "rb") as f_bm25_corpus:
        loaded_tokenized_corpus_bm25 = pickle.load(f_bm25_corpus)
    bm25 = BM25Okapi(loaded_tokenized_corpus_bm25)
    print(f"✅ BM25 model initialized with pre-tokenized corpus from {BM25_TOKENIZED_CORPUS_FILE}.")
except FileNotFoundError:
    print(f"⚠️ BM25 pre-tokenized corpus not found at {BM25_TOKENIZED_CORPUS_FILE}.")
    print("    Please run the indexer first to create it.")
    print("    Falling back to on-the-fly tokenization for BM25 (will be slow for many chunks).")
    # Fallback: tokenize on the fly if file not found (not ideal for many chunks)
    tokenized_corpus_bm25_onthefly = [bm25_tokenizer(chunk) for chunk in chunks_text_list]
    bm25 = BM25Okapi(tokenized_corpus_bm25_onthefly)
    print("✅ BM25 model initialized (on-the-fly tokenization).")
except Exception as e:
    print(f"❌ Error loading/initializing BM25: {e}")
    bm25 = None # Ensure bm25 is None if it fails
# --- End BM25 Setup ---

def search_chunks_hybrid(query, top_k=CHUNK_COUNT, filter_by=None):
    # ... (FAISS search part remains the same)
    query_vector_faiss = embed_texts([query])
    faiss_candidate_count = top_k * FAISS_CANDIDATES_MULTIPLIER
    if query_vector_faiss.ndim == 1:
        query_vector_faiss = np.expand_dims(query_vector_faiss, axis=0)
    D_faiss, I_faiss = index_faiss.search(query_vector_faiss.astype('float32'), faiss_candidate_count)
    faiss_results_with_scores = {}
    if I_faiss.size > 0:
        for i, idx in enumerate(I_faiss[0]):
            if idx != -1:
                faiss_results_with_scores[idx] = D_faiss[0][i]
    print(f"👍 FAISS: Found {len(faiss_results_with_scores)} initial candidates.")


    # --- BM25 Search (Sparse) ---
    bm25_results_with_scores = {}
    if bm25: # Check if bm25 model was initialized successfully
        tokenized_query_bm25 = bm25_tokenizer(query)
        bm25_candidate_count = top_k * BM25_CANDIDATES_MULTIPLIER
        print(f"📝 BM25: Searching for top {bm25_candidate_count} candidates...")
        bm25_scores = bm25.get_scores(tokenized_query_bm25)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_candidate_count]
        for idx in bm25_top_indices:
            bm25_results_with_scores[idx] = bm25_scores[idx]
        print(f"👍 BM25: Found {len(bm25_results_with_scores)} initial candidates.")
    else:
        print("⚠️ BM25 model not available, skipping BM25 search.")

    # --- Reciprocal Rank Fusion (RRF) ---
    # ... (RRF logic remains the same, but now it might only have FAISS results if BM25 failed)
    print(f"🔄 Combining results with RRF (k={RRF_K})...")
    rrf_scores = {}
    
    sorted_faiss_by_dist = sorted(faiss_results_with_scores.items(), key=lambda item: item[1])
    for rank, (idx, dist) in enumerate(sorted_faiss_by_dist):
        if idx not in rrf_scores: rrf_scores[idx] = 0.0
        rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

    if bm25_results_with_scores: # Only add BM25 scores if available
        # bm25_top_indices was already created and sorted by relevance if bm25 search ran
        # Need to re-create bm25_top_indices if it wasn't if we want to use it here,
        # or directly use sorted bm25_results_with_scores
        sorted_bm25_by_score = sorted(bm25_results_with_scores.items(), key=lambda item: item[1], reverse=True)
        for rank, (idx, score) in enumerate(sorted_bm25_by_score):
            if idx not in rrf_scores: rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)
    
    # ... (Rest of filtering and final selection remains the same)
    sorted_combined_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
    final_results = []
    retrieved_indices_set = set()
    for idx in sorted_combined_indices:
        if idx in retrieved_indices_set: continue
        meta = metadata_list[idx]
        passes_filter = True
        if filter_by:
            if 'source' in filter_by and meta.get('source') != filter_by['source']: passes_filter = False
            if passes_filter and 'module' in filter_by and meta.get('module') != filter_by['module']: passes_filter = False
            if passes_filter and 'keyword' in filter_by:
                keywords_in_meta = meta.get('keywords', [])
                if not isinstance(keywords_in_meta, list) or \
                   not any(isinstance(kw, str) and filter_by['keyword'].lower() in kw.lower() for kw in keywords_in_meta):
                    passes_filter = False
        if passes_filter:
            final_results.append((chunks_text_list[idx], meta))
            retrieved_indices_set.add(idx)
            if len(final_results) >= top_k: break
    print(f"✅ Hybrid search: Selected {len(final_results)} final chunks after RRF and filtering.")
    return final_results

def build_prompt(query, retrieved_chunks):
    context_parts = []
    for i, (chunk_text, meta) in enumerate(retrieved_chunks):
        context_parts.append(
            f"[CONTEXT {i+1}]\n"
            f"Source: {meta.get('source', 'N/A')}\n"
            f"Page: {meta.get('page_number', 'N/A')}\n"
            f"Section: {meta.get('section_id_str', 'N/A')} - {meta.get('section_title', 'N/A')}\n"
            f"Module: {meta.get('module', 'N/A')}\n"
            f"Keywords: {', '.join(meta.get('keywords', [])) if meta.get('keywords') else 'N/A'}\n\n"
            f"{chunk_text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert AUTOSAR documentation assistant.

For the given question, follow these steps:
1) First think about the question itself - why is it asked and what could it be connected to.For example if the question is about how to generate a signature, consider what a signature is and what is needed to generate it.
2) then carefully review the provided context.
3) Think step by step, summarizing the relevant information, inferring details as needed, and showing your reasoning in detail.
4) After the reasoning, provide your final answer based on your reasoning (not referencing the process or that you are using RAG/context).

**Format your output as follows:**

Final answer here.

If you are unsure or the answer cannot be determined, write "I don't know." as the answer.

---
AUTOSAR documentation :
{context}
---

Question:
{query}

Answer:
"""

    return prompt

def query_ollama(prompt, model=OLLAMA_MODEL):
    from ollama import Client
    # Ensure Ollama server is running at this host and port
    client = Client(host='http://localhost:11434')
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False # Set to True if you want streaming
        )
        return response['message']['content']
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return f"Ollama error: {e}"

def ask_rag(query, filter_by=None):
    print(f"💬 Asking (hybrid): {query}")
    if filter_by:
        print(f"📎 Filter: {filter_by}")
    
    retrieved_chunks = search_chunks_hybrid(query, top_k=CHUNK_COUNT, filter_by=filter_by)

    if not retrieved_chunks:
        print("\n⚠️ No relevant chunks found after hybrid search and filtering.")
        return "I don't know. No relevant information was found in the documents based on your query and filters."

    print("\n🔍 Retrieved Chunks (Hybrid):")
    for i, (chunk, meta) in enumerate(retrieved_chunks):
        print(f"\n--- Chunk {i+1} (Hybrid) ---")
        print(f"[Source: {meta.get('source', '?')}, Page: {meta.get('page_number', '?')}, Section: {meta.get('section_id_str','?')} - {meta.get('section_title','?')}, Module: {meta.get('module','?')}]")
        # Print snippet of chunk
        #print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        print(chunk) 

    prompt = build_prompt(query, retrieved_chunks)
    # print(f"\n📝 Generated Prompt (Hybrid):\n{prompt[:1000]}...\n") # For debugging prompt
    return query_ollama(prompt)