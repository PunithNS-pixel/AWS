import csv

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from src.normalize import normalize_text
except ModuleNotFoundError:
    from normalize import normalize_text


def process_chunk(chunk):
    """
    Applies normalization to the business name and address in a chunk of data.
    Supports either a pandas DataFrame or a list of row dictionaries.
    """
    if pd is not None and hasattr(chunk, 'columns'):
        chunk = chunk.copy()
        chunk['business_name'] = chunk['business_name'].fillna('').apply(normalize_text)
        chunk['business_address'] = chunk['business_address'].fillna('').apply(normalize_text)
        return chunk

    normalized_rows = []
    for row in chunk:
        normalized_row = dict(row)
        normalized_row['business_name'] = normalize_text(normalized_row.get('business_name', ''))
        normalized_row['business_address'] = normalize_text(normalized_row.get('business_address', ''))
        normalized_rows.append(normalized_row)
    return normalized_rows


def load_data_in_batches(file_path, chunk_size=100000):
    """
    Generator function to yield processed chunks of data.
    Falls back to the stdlib csv reader if pandas is unavailable.
    """
    print(f"Loading data from {file_path}...")

    if pd is not None:
        for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size):
            yield process_chunk(chunk)
        return

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= chunk_size:
                yield process_chunk(batch)
                batch = []
        if batch:
            yield process_chunk(batch)


# --- Test the function ---
if __name__ == "__main__":
    import os

    test_file = "dataset/train/train_source1.tsv"

    if os.path.exists(test_file):
        first_chunk = next(load_data_in_batches(test_file, chunk_size=1000))
        print(f"Successfully loaded a chunk with {len(first_chunk)} rows.")

        if pd is not None and hasattr(first_chunk, 'columns'):
            print(first_chunk[['business_name', 'business_address']].head(2))
        else:
            for row in first_chunk[:2]:
                print({k: row[k] for k in ('business_name', 'business_address')})
    else:
        print(f"Could not find {test_file}. Please check your file path.")