import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
df = pd.read_csv("baseballrules.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, normalize_embeddings=True).astype("float32")#generates embeddings
print("Embeddings shape:", embeddings.shape)
dimension = embeddings.shape[1]#generates dimensions
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("Vectors in index:", index.ntotal)
faiss.write_index(index, "vector_index.faiss")#creates vector database
df["embedding"] = embeddings.tolist()#adds embeddings to csv
df.to_csv("vector_metadata.csv", index=False)