import os
from collections import defaultdict
import pandas as pd
from src.data_loader import load_data_in_batches
from src.embedder import Embedder
from src.blocker import FaissBlocker

# Set this to True for a small smoke run.
DEBUG = False
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

    target_ids = []

    # Index both target sources from the test split used for submission.
    print("\n--- Indexing Source 2 and Source 3 ---")
    target_paths = [
        "dataset/test/test_source2.tsv",
        "dataset/test/test_source3.tsv",
    ]
    for target_path in target_paths:
        target_count = 0
        for chunk in load_data_in_batches(target_path, chunk_size=5000):
            combined_text = chunk['business_name'] + " " + chunk['business_address']
            vectors = embedder.get_embeddings(combined_text.tolist())

            blocker.add_vectors(vectors)
            target_ids.extend(chunk[get_id_column(chunk)].tolist())
            target_count += len(chunk)

            if DEBUG and target_count >= DEBUG_ROWS:
                print(f"Debug limit reached for {target_path} ({DEBUG_ROWS} rows).")
                break

    # Search with Source 1 from the same test split.
    s1_path = "dataset/test/test_source1.tsv"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(output_dir, "matching_results.tsv")

    candidate_ids_by_s1 = defaultdict(list)
    s1_count = 0

    print("\n--- Searching & Saving Results ---")
    for chunk in load_data_in_batches(s1_path, chunk_size=1000):
        combined_text = chunk['business_name'] + " " + chunk['business_address']
        vectors = embedder.get_embeddings(combined_text.tolist())

        distances, indices = blocker.search(vectors, top_k=5)

        for i, s1_id in enumerate(chunk[get_id_column(chunk)].tolist()):
            candidate_ids_by_s1.setdefault(s1_id, [])
            for match_idx in indices[i]:
                if match_idx < len(target_ids):
                    target_id = target_ids[match_idx]
                    if target_id not in candidate_ids_by_s1[s1_id]:
                        candidate_ids_by_s1[s1_id].append(target_id)

        s1_count += len(chunk)
        if DEBUG and s1_count >= DEBUG_ROWS:
            print(f"Debug limit reached for Source 1 ({DEBUG_ROWS} rows).")
            break

    # Include S1 rows with no candidates as required by the submission format.
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