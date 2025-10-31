import os
import numpy as np
import pandas as pd
import faiss

# Paths to your pre-generated files
INDEX_PATH = "../databases/vector_index.faiss"
METADATA_PATH = "../data/vector_metadata.csv"

def test_existing_database():
    #  Load metadata
    assert os.path.exists(INDEX_PATH), f"FAISS index not found: {INDEX_PATH}"
    assert os.path.exists(METADATA_PATH), f"Metadata CSV not found: {METADATA_PATH}"
    metadata_df = pd.read_csv(METADATA_PATH)
    assert not metadata_df.empty, "Metadata CSV is empty."
    assert "id" in metadata_df.columns
    assert "embedding" in metadata_df.columns

    #  Load FAISS index
    index = faiss.read_index(INDEX_PATH)
    if not isinstance(index, faiss.IndexIDMap):
        index = faiss.IndexIDMap(index)
    assert index.ntotal == len(metadata_df), "FAISS index size mismatch with metadata."

    #  test a query
    first_vec = np.array(eval(metadata_df["embedding"].iloc[0]), dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(first_vec, k=1)
    assert indices[0][0] == metadata_df["id"].iloc[0], "Query did not return the expected ID."
