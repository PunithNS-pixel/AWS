from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Loads a multilingual model that works well for English, Hindi, Tamil, etc.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def get_embeddings(self, text_list):
        """
        Converts a list of strings into a numpy array of vectors.
        """
        # show_progress_bar=True helps you see how fast it's running
        embeddings = self.model.encode(text_list, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

# --- Test the function ---
if __name__ == "__main__":
    embedder = Embedder()
    
    sample_texts = [
        "Example Business Inc",
        "Example Business Incorporated",
        "Completely Different Corp"
    ]
    
    vectors = embedder.get_embeddings(sample_texts)
    print(f"Successfully generated embeddings with shape: {vectors.shape}")
    print("(This means 3 texts were converted into vectors of 384 numbers each!)")