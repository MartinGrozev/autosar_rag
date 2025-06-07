# rag_indexer.py

import os
import pickle
import faiss
# Import TF-IDF function from utils_text_processing
from utils_text_processing import extract_autosar_chunks, bm25_tokenizer, extract_tfidf_keywords_for_corpus
from utils_embeddings import embed_texts
# import shutil

AUTOSAR_PATH = "C:/Users/HP/Desktop/AUTOSAR"
OUTPUT_DIR = "faiss_data"
BM25_TOKENIZED_CORPUS_FILE = os.path.join(OUTPUT_DIR, "bm25_tokenized_corpus.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def index_documents(root_folder_path):
    all_provisional_chunks_data = [] # Store chunk data before adding global keywords

    print(f"🚀 Starting document indexing from root: {root_folder_path}")

    for dirpath, dirnames, filenames in os.walk(root_folder_path):
        print(f"📂 Exploring directory: {dirpath}")
        for filename in filenames:
            if not filename.endswith(".pdf"):
                continue
            pdf_path = os.path.join(dirpath, filename)
            print(f"📄 Processing: {pdf_path}")
            
            # extract_autosar_chunks now returns a list of dicts without 'keywords'
            provisional_chunks_from_pdf = extract_autosar_chunks(pdf_path)

            for chunk_data in provisional_chunks_from_pdf:
                # Add source and source_path here, as it's per PDF
                chunk_data['source'] = filename
                chunk_data['source_path'] = pdf_path
                all_provisional_chunks_data.append(chunk_data)

    if not all_provisional_chunks_data:
        print(f"⚠️ No provisional chunks were created from any PDF in '{root_folder_path}'. Exiting.")
        return

    # --- Global TF-IDF Keyword Extraction ---
    print(f"✨ Extracting global TF-IDF keywords for {len(all_provisional_chunks_data)} total chunks...")
    all_chunk_texts_for_tfidf = [chunk['text'] for chunk in all_provisional_chunks_data]
    global_keywords_for_all_chunks = extract_tfidf_keywords_for_corpus(all_chunk_texts_for_tfidf)
    
    # Add keywords back to each chunk's metadata
    all_metadata_store = []
    all_chunks_text_for_embedding = [] # This will be the final list of texts for embedding

    if len(global_keywords_for_all_chunks) != len(all_provisional_chunks_data):
        print(f"❌ ERROR: Keyword list length ({len(global_keywords_for_all_chunks)}) doesn't match chunk list length ({len(all_provisional_chunks_data)}).")
        # Fallback or raise error
        for i, chunk_data in enumerate(all_provisional_chunks_data):
            chunk_data['keywords'] = [] # Assign empty keywords
            all_metadata_store.append(chunk_data) # Keep all original metadata except keywords
            all_chunks_text_for_embedding.append(chunk_data['text'])

    else:
        for i, chunk_data in enumerate(all_provisional_chunks_data):
            chunk_data['keywords'] = global_keywords_for_all_chunks[i]
            # Prepare final metadata and text list for saving and embedding
            all_metadata_store.append({
                "source": chunk_data['source'],
                "source_path": chunk_data['source_path'],
                "section_id_str": chunk_data["section_id_str"],
                "section_title": chunk_data["section_title"],
                "module": chunk_data.get("module"),
                "keywords": chunk_data.get("keywords", []), # Should now have TF-IDF keywords
                "page_number": chunk_data.get("page_number")
            })
            all_chunks_text_for_embedding.append(chunk_data['text'])
    print("✅ Global TF-IDF keywords assigned to chunks.")
    # --- End Global TF-IDF ---


    if not all_chunks_text_for_embedding: # Should be redundant if provisional_chunks_data check passed
        print(f"⚠️ No chunks available for embedding after keyword assignment. Exiting.")
        return

    # --- BM25: Tokenize corpus for saving ---
    print(f"⚙️ Tokenizing {len(all_chunks_text_for_embedding)} chunks for BM25 persistence...")
    tokenized_corpus_bm25_to_save = [bm25_tokenizer(chunk_text) for chunk_text in all_chunks_text_for_embedding]
    with open(BM25_TOKENIZED_CORPUS_FILE, "wb") as f_bm25_corpus:
        pickle.dump(tokenized_corpus_bm25_to_save, f_bm25_corpus)
    print(f"✅ BM25 tokenized corpus saved to {BM25_TOKENIZED_CORPUS_FILE}")
    # --- End BM25 Save ---

    print(f"🧠 Embedding {len(all_chunks_text_for_embedding)} chunks with metadata...")
    # ... (rest of embedding and FAISS indexing as before) ...
    vectors = embed_texts(all_chunks_text_for_embedding).astype("float32")
    if vectors.shape[0] == 0:
        print("⚠️ No vectors to add to FAISS index. Exiting.")
        return
    dimension = vectors.shape[1]
    if dimension == 0:
        print(f"⚠️ Embedding dimension is 0. Cannot create FAISS index. Check embed_texts function.")
        return

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    with open(os.path.join(OUTPUT_DIR, "chunks.pkl"), "wb") as f_chunks:
        pickle.dump(all_chunks_text_for_embedding, f_chunks) # Final texts
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f_metadata:
        pickle.dump(all_metadata_store, f_metadata) # Final metadata with global keywords
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "index.faiss"))
    print(f"✅ All {len(all_chunks_text_for_embedding)} chunks indexed and saved from all subdirectories!")


if __name__ == "__main__":
    # Optional: Delete existing faiss_data for a completely fresh index
    # import shutil
    # if os.path.exists(OUTPUT_DIR):
    #     print(f"🧹 Deleting existing output directory: {OUTPUT_DIR}")
    #     shutil.rmtree(OUTPUT_DIR)
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    index_documents(AUTOSAR_PATH)