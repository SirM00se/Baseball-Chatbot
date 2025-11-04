import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os


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
    print(f"Loading model: {model_name}")
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
    # Validate embeddings
    if (embeddings is not None) and (len(embeddings) != 0) and (isinstance(embeddings, np.ndarray)) and (embeddings.dtype == np.float32) and (len(embeddings.shape) == 2):

        # Build IDs
        ids = np.arange(len(embeddings)).astype(np.int64)
        dimension = embeddings.shape[1]

        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index)

        # Add embeddings
        try:
            index.add_with_ids(embeddings, ids)
        except Exception as e:
            raise RuntimeError(f"Failed to add embeddings to FAISS index: {e}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Write index to disk
        try:
            faiss.write_index(index, index_path)
        except Exception as e:
            raise IOError(f"Failed to write FAISS index to '{index_path}': {e}")

        print(f"Saved FAISS index to: {index_path}")
        print(f"Total vectors in index: {index.ntotal}")
        return ids
    else:
        print("error in embeddings creation")
        return None

# -----------------------------
# Save Metadata (CSV)
# -----------------------------
def save_metadata(df: pd.DataFrame, ids: np.ndarray, embeddings: np.ndarray, output_path: str):
    """Attach IDs and embeddings to the DataFrame and save as CSV."""
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not isinstance(ids, (np.ndarray, list)):
            raise TypeError("ids must be a NumPy array or list.")
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a NumPy array.")
        if len(df) != len(ids) or len(df) != len(embeddings):
            raise ValueError("Length of df, ids, and embeddings must match.")

        # Attach IDs and embeddings
        df["id"] = ids
        df["embedding"] = embeddings.tolist()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        try:
            df.to_csv(output_path, index=False)
        except Exception as e:
            raise IOError(f"Failed to save CSV to '{output_path}': {e}")

        print(f"Saved metadata to: {output_path}")

    except (TypeError, ValueError, IOError) as e:
        print(f"Error saving metadata: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while saving metadata: {e}")
        raise


# -----------------------------
# Main Execution Function
# -----------------------------
def main():
    data_path = "../data/baseballrules.csv"
    index_path = "../databases/vector_index.faiss"
    metadata_path = "../data/vector_metadata.csv"
    try:
        # Step 1: Load
        df = load_data(data_path)

        # Step 2: Generate embeddings
        embeddings = create_embeddings(df["text"].tolist())

        # Step 3: Build FAISS index
        ids = build_faiss_index(embeddings, index_path)

        # Step 4: Save metadata
        save_metadata(df, ids, embeddings, metadata_path)
    except TypeError as e:
        print(f"Type error in input: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except RuntimeError as e:
        print(f"Runtime error (likely GPU issue): {e}")
    except OSError as e:
        print(f"Model file error: {e}")
    except MemoryError as e:
        print(f"Out of memory error: {e}")
    except Exception as e:
        print(f"Unexpected error during encoding: {e}")
    else:
        print("\nAll tasks complete!")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
