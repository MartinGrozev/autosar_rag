# utils_embeddings.py

# Use SentenceTransformers library
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the desired Sentence Transformer model once
# all-mpnet-base-v2 is a good general-purpose choice.
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
print(f"Loading embedding model: {MODEL_NAME}")
try:
    # You can specify device='cuda' if you have a GPU and PyTorch with CUDA installed
    model = SentenceTransformer(MODEL_NAME, device='cpu') # Use 'cuda' if GPU available
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load Sentence Transformer model: {e}")
    print("Make sure you have installed it: pip install sentence-transformers")
    model = None

def embed_texts(texts):
    """Embed a list of texts into dense vectors using Sentence Transformers."""
    if model is None:
        raise RuntimeError("Sentence Transformer model failed to load. Cannot embed texts.")
    if not texts:
        print("Warning: embed_texts called with empty list.")
        # Return an empty array with the correct dimension if possible, otherwise handle upstream
        # For FAISS, we need the dimension. Assuming all-mpnet-base-v2 dimension (768)
        return np.empty((0, 768), dtype=np.float32) 

    print(f"Embedding {len(texts)} texts using {MODEL_NAME}...")
    
    # The encode method handles batching efficiently.
    # Set show_progress_bar=True for long lists.
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True # Directly output NumPy array
    )
    
    print(f"Embedding complete. Shape: {embeddings.shape}")
    return embeddings.astype('float32') # Ensure float32 for FAISS

# Example usage (optional, for testing)
if __name__ == '__main__':
    sample_texts = ["This is the first sentence.", "This is the second sentence."]
    if model:
        embeddings_result = embed_texts(sample_texts)
        print("Sample embeddings:")
        print(embeddings_result)
        print(f"Dimension: {embeddings_result.shape[1]}")
    else:
        print("Model not loaded, skipping embedding example.")