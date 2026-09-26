import faiss
import numpy as np

class FaissBlocker:
    def __init__(self, dimension):
        """
        Initialize the FAISS index. 
        'dimension' must match the size of your embeddings (384 for our model).
        """
        # Avoid FAISS/OpenMP mutex failures on macOS when used with PyTorch.
        faiss.omp_set_num_threads(1)
        # Using Inner Product (Cosine Similarity)
        self.index = faiss.IndexFlatIP(dimension)
        
    def add_vectors(self, vectors):
        """
        Add the 'Source 2' (target) vectors to the index.
        """
        # FAISS requires float32 data type
        vectors = np.float32(vectors)
        faiss.normalize_L2(vectors) # Normalize for cosine similarity
        self.index.add(vectors)
        print(f"Added {len(vectors)} vectors to the index.")
        
    def search(self, query_vectors, top_k=5):
        """
        Search the index for the closest matches to 'Source 1' (query) vectors.
        """
        query_vectors = np.float32(query_vectors)
        faiss.normalize_L2(query_vectors)
        
        # Returns distances (D) and indices (I) of the top_k matches
        distances, indices = self.index.search(query_vectors, top_k)
        return distances, indices

# --- Test the function ---
if __name__ == "__main__":
    # 1. Create dummy data (1000 random vectors of size 384)
    dimension = 384
    source2_vectors = np.random.random((1000, dimension)).astype('float32')
    source1_vectors = np.random.random((5, dimension)).astype('float32')
    
    # 2. Initialize blocker
    blocker = FaissBlocker(dimension)
    
    # 3. Add Source 2 to the index
    blocker.add_vectors(source2_vectors)
    
    # 4. Search for Source 1 matches
    distances, indices = blocker.search(source1_vectors, top_k=3)
    
    print("\nSearch complete! Here are the top 3 matches for the first query:")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"Rank {rank+1}: Match ID {idx} (Similarity: {dist:.4f})")