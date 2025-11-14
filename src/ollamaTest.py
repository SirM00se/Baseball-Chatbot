import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3

def askQuestion() -> str:
    try:
        question = input("send a message ")
    except KeyboardInterrupt:
        print("Keyboard interrupt, try again")
        return askQuestion()
    return question

def createEmbeddingQuestion(question, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    model = SentenceTransformer(model_name)
    embedding = model.encode(question, convert_to_numpy=True, normalize_embeddings=True).astype('float32').reshape(1, -1)
    print(f"Generated embedding: {embedding.shape}")
    return embedding

def generateIDs(embedding):
    ids = [[],[]]
    index = faiss.read_index("../databases/vector_index.faiss")
    k = 5
    scores, retrieved_ids = index.search(embedding, k)
    for score in scores:
        ids[1].extend(score.tolist())
    for id in retrieved_ids:
        ids[0].extend(id.tolist())
    return ids

def querySQLite(ids):

    conn = sqlite3.connect("../databases/baseball_vectors.db")
    cursor = conn.cursor()
    searchids = ids[0]
    placeholders = ','.join(['?'] * len(searchids))
    # Fetch all matching rows
    query = f"SELECT text FROM rules WHERE id in ({placeholders})"
    cursor.execute(query, searchids)
    results = cursor.fetchall()
    texts = [row[0] for row in results]
    conn.close()
    return texts

def main():
    question = askQuestion()
    embedding = createEmbeddingQuestion(question)
    ids = generateIDs(embedding)
    texts = querySQLite(ids)
    for y in texts:
        print(y)

if __name__ == "__main__":
    main()