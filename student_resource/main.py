import os
from collections import defaultdict
import pandas as pd
from src.data_loader import load_data_in_batches
from src.embedder import Embedder
from src.blocker import FaissBlocker

# SET THIS TO FALSE WHEN YOU ARE READY FOR THE FULL OVERNIGHT RUN
DEBUG = False # Set to True for debugging with a smaller dataset
DEBUG_ROWS = 10000

def get_id_column(chunk):
    """Return the identifier column used by the input dataset."""
    for column in ("entity_id", "id"):
        if column in chunk.columns:
            return column
    raise KeyError(f"Expected an identifier column ('entity_id' or 'id'); found {list(chunk.columns)}")


def run_full_pipeline():
    print(f"Initializing Pipeline... (DEBUG MODE: {DEBUG})")
    embedder = Embedder()
    blocker = FaissBlocker(dimension=384)
    
    s2_ids = []
    
    # 1. Index Source 2
    print("\n--- Indexing Source 2 ---")
    s2_path = "dataset/train/train_source2.tsv"
    
    for chunk in load_data_in_batches(s2_path, chunk_size=5000):
        combined_text = chunk['business_name'] + " " + chunk['business_address']
        vectors = embedder.get_embeddings(combined_text.tolist())
        
        blocker.add_vectors(vectors)
        s2_ids.extend(chunk[get_id_column(chunk)].tolist())
        
        if DEBUG and len(s2_ids) >= DEBUG_ROWS:
            print(f"Debug limit reached for Source 2 ({DEBUG_ROWS} rows).")
            break
        
    # 2. Search with Source 1
    print("\n--- Searching & Saving Results ---")
    s1_path = "dataset/train/train_source1.tsv"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    
    candidate_ids_by_s1 = defaultdict(list)
    s1_count = 0
    
    for chunk in load_data_in_batches(s1_path, chunk_size=1000):
        combined_text = chunk['business_name'] + " " + chunk['business_address']
        vectors = embedder.get_embeddings(combined_text.tolist())
        
        distances, indices = blocker.search(vectors, top_k=5)
        
        for i, s1_id in enumerate(chunk[get_id_column(chunk)].tolist()):
            for match_idx in indices[i]:
                if match_idx < len(s2_ids): # Safety check
                    s2_id = s2_ids[match_idx]
                    if s2_id not in candidate_ids_by_s1[s1_id]:
                        candidate_ids_by_s1[s1_id].append(s2_id)
                    
        s1_count += len(chunk)
        if DEBUG and s1_count >= DEBUG_ROWS:
            print(f"Debug limit reached for Source 1 ({DEBUG_ROWS} rows).")
            break
                
    # 3. Write to TSV
    output_rows = [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(candidate_ids),
            "matched_entity_ids": candidate_ids[0] if candidate_ids else "",
        }
        for s1_id, candidate_ids in candidate_ids_by_s1.items()
    ]
    output_df = pd.DataFrame(output_rows)
    candidate_df = output_df[["source1_entity_id", "candidate_entity_ids"]]
    matching_df = output_df[["source1_entity_id", "matched_entity_ids"]]

    print(f"Saving {len(candidate_df)} candidate rows to {candidate_path}...")
    candidate_df.to_csv(candidate_path, sep='\t', index=False)
    matching_df.to_csv(matching_path, sep='\t', index=False)
    print(f"Saving {len(matching_df)} matching rows to {matching_path}...")
    print("Done!")

if __name__ == "__main__":
    run_full_pipeline()