import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
df = pd.read_csv("../data/baseballrules.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, normalize_embeddings=True).astype("float32")#generates embeddings
print("Embeddings shape:", embeddings.shape)
ids = np.arange(len(embeddings)).astype(np.int64)
dimension = embeddings.shape[1]#generates dimensions
index = faiss.IndexFlatL2(dimension)
index = faiss.IndexIDMap(index)
index.add_with_ids(embeddings, ids)
print("Vectors in index:", index.ntotal)
faiss.write_index(index, "../databases/vector_index.faiss")#creates vector database
df["id"] = ids
df["embedding"] = embeddings.tolist()#adds embeddings to csv
df.to_csv("../data/vector_metadata.csv", index=False)