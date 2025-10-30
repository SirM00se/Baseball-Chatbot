import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss


# -----------------------------
# Load Data
# -----------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Load text data from CSV."""
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


# -----------------------------
# Create Sentence Embeddings
# -----------------------------
def create_embeddings(texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Generate normalized embeddings for text using a SentenceTransformer model."""
    print(f"🔄 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")
    print(f"Generated embeddings: {embeddings.shape}")
    return embeddings


# -----------------------------
# Build and Save FAISS Index
# -----------------------------
def build_faiss_index(embeddings: np.ndarray, index_path: str) -> np.ndarray:
    """Builds a FAISS L2 index and saves it to disk."""
    ids = np.arange(len(embeddings)).astype(np.int64)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to: {index_path}")
    print(f"Total vectors in index: {index.ntotal}")
    return ids


# -----------------------------
# Save Metadata (CSV)
# -----------------------------
def save_metadata(df: pd.DataFrame, ids: np.ndarray, embeddings: np.ndarray, output_path: str):
    """Attach IDs and embeddings to the DataFrame and save as CSV."""
    df["id"] = ids
    df["embedding"] = embeddings.tolist()
    df.to_csv(output_path, index=False)
    print(f"Saved metadata to: {output_path}")


# -----------------------------
# Main Execution Function
# -----------------------------
def main():
    data_path = "../data/baseballrules.csv"
    index_path = "../databases/vector_index.faiss"
    metadata_path = "../data/vector_metadata.csv"

    # Step 1: Load
    df = load_data(data_path)

    # Step 2: Generate embeddings
    embeddings = create_embeddings(df["text"].tolist())

    # Step 3: Build FAISS index
    ids = build_faiss_index(embeddings, index_path)

    # Step 4: Save metadata
    save_metadata(df, ids, embeddings, metadata_path)

    print("\nAll tasks complete!")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
