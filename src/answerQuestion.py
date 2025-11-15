import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import ollama

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
    return embedding

def generateIDs(embedding):
    index = faiss.read_index("../databases/vector_index.faiss")
    k = 5

    # Ensure embedding is [1, dim]
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)

    scores, ids = index.search(embedding, k)

    scores = scores[0].tolist()
    ids = ids[0].tolist()

    # Pair them together and sort by score
    pairs = sorted(zip(ids, scores), key=lambda x: x[1])  # x[1] = score

    # Unzip back into separate lists
    sorted_ids = [p[0] for p in pairs]
    sorted_scores = [p[1] for p in pairs]

    return sorted_ids, sorted_scores

def querySQLite(ids, scores):
    """
    ids    : list of sorted FAISS ids
    scores : list of sorted FAISS scores (same order)
    """
    conn = sqlite3.connect("../databases/baseball_vectors.db")
    cursor = conn.cursor()

    placeholders = ",".join(["?"] * len(ids))

    query = f"SELECT url, text, tag, id FROM rules WHERE id IN ({placeholders})"
    cursor.execute(query, ids)
    rows = cursor.fetchall()
    conn.close()

    # Build a mapping from id → row
    row_map = {row[3]: row for row in rows}

    # Rebuild chunks in the *same order* as the sorted ids
    retrieved_chunks = []

    for cid, score in zip(ids, scores):
        url, text, tag, id_from_db = row_map[cid]
        retrieved_chunks.append({
            "id": id_from_db,
            "url": url,
            "text": text,
            "tag": tag,
            "score": score
        })

    return retrieved_chunks

def make_ollama_json_prompt(question, chunks):
    """
    Creates a JSON-style RAG prompt for Ollama.
    """
    # Convert chunks to JSON-like structure
    chunk_list = [
        {
            "id": c["id"],
            "score": c["score"],
            "url": c["url"],
            "tag": c["tag"],
            "text": c["text"]
        }
        for c in chunks
    ]

    # Now create a JSON-based instruction
    prompt = f"""
You are a retrieval-augmented assistant.

Here is the context, provided as a JSON array of document chunks:

CHUNKS:
{chunk_list}

QUESTION:
{question}

INSTRUCTIONS:
- Answer using ONLY the information from the chunks.
- For every piece of information you include, include the corresponding URL from the chunk.
- If the answer is not in the chunks, say "I don't know."
- Do NOT invent missing information.
- Format your answer as plain text, but include URLs inline or in parentheses for citations.
    """

    return prompt

def callOllama(prompt):
    response = ollama.chat(
        model="llama3.1:8b",  # or your local Ollama model
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

def main():
    question = askQuestion()
    embedding = createEmbeddingQuestion(question)
    ids, scores = generateIDs(embedding)
    chunks = querySQLite(ids, scores)
    prompt = make_ollama_json_prompt(question, chunks)
    answer = callOllama(prompt)
    print(answer)

if __name__ == "__main__":
    main()