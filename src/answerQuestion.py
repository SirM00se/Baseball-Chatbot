import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import ollama
import sys

def askQuestion() -> str:
    """
    Allows user to input a question
    keeps asking the user if there is a keyboard interrupt

    :return:
        The question as a string
    """
    try:
        question = input("send a message ")
        return question
    except KeyboardInterrupt:
        print("Keyboard interrupt, try again")
        return askQuestion()


def createEmbeddingQuestion(question, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    creates an embedding for the user question

    :param question: user question
    :param model_name: Sentence Transformers model
    :return:
        The embedding of the user question
    """
    model = SentenceTransformer(model_name)

    #generates the embedding for the question
    embedding = model.encode(question, convert_to_numpy=True, normalize_embeddings=True).astype('float32').reshape(1, -1)
    return embedding

def generateIDs(embedding):
    """
    generates the closest ids in the vector database to the user question

    :param embedding: embedding of the user question
    :return:
        An array of sorted ids and scores
    """
    try:
        index = faiss.read_index("../databases/vector_index.faiss")
    except Exception as e:
        print(f"FAISS index loading error: {e}")
        sys.exit(1)

    k = 5 #checks for the top 5 results

    # Ensure embedding is [1, dim]
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)

    scores, ids = index.search(embedding, k)

    #checks if ids is empty
    if ids.size == 0:
        print("No vectors found for query.")
        return None

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
    Searches the metadata for chunks that include metadata and the score

    :param ids: faiss chunk ids
    :param scores: faiss scores
    :return:
        The metadata chunks
    """
    #intialize access to the database
    try:
        conn = sqlite3.connect("../databases/baseball_vectors.db")
    except sqlite3.Error as e:
        print(f"SQLite connection error: {e}")
        sys.exit(1)

    cursor = conn.cursor()

    #Allows the ability to safely pass ids
    placeholders = ",".join(["?"] * len(ids))

    #Queries the sql database
    query = f"SELECT url, text, tag, id FROM rules WHERE id IN ({placeholders})"
    cursor.execute(query, ids)
    rows = cursor.fetchall()
    conn.close()

    # Build a mapping from id → row
    row_map = {row[3]: row for row in rows}
    #checks if row is missing
    for row in rows:
        if row is None:
            print(f"Metadata for id {row} missing")

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

    :param question: user question
    :param chunks: metadata chunks
    :return:
        A JSON RAG prompt that can be used by Ollama
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

def callOllama(prompt, model):
    """
    Generates a response to the question with Ollama

    :param prompt: JSON RAG prompt
    :return:
        An answer to the user question
    """
    response = ollama.chat(
        model= model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

def main():
    try:
        # 1. Ask for a question
        question = askQuestion()

        # Validate question
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Query must be a non-empty string")

        # 2. Generate embedding
        embedding = createEmbeddingQuestion(question)

        # Validate embedding
        if not isinstance(embedding, (list, np.ndarray)):
            raise TypeError(f"Embedding must be a list or ndarray, got {type(embedding)}")
        embedding = np.array(embedding)
        if not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding contains NaN or infinite values")
        if np.linalg.norm(embedding) == 0:
            raise ValueError("Embedding is a zero vector")

        # 3. Generate IDs from FAISS
        ids, scores = generateIDs(embedding)
        if not ids:
            print("No IDs retrieved from FAISS; skipping query")
            return

        # 4. Query SQLite for metadata
        chunks = querySQLite(ids, scores)
        if not chunks:
            print("No chunks retrieved from SQLite; skipping prompt")
            return

        # 5. Create prompt
        prompt = make_ollama_json_prompt(question, chunks)
        if not prompt:
            print("Generated prompt is empty; cannot call Ollama")
            return

        # 6. Call Ollama
        model = "llama3.1:8b"
        try:
            answer = callOllama(prompt, model)
            if not answer:
                print("Ollama returned empty response")
            else:
                print("Answer:", answer)
        except ollama.ResponseError as e:
            print("Ollama ResponseError:", e.error)
            if getattr(e, 'status_code', None) == 404:
                print(f"Model {model} not found locally. Attempting to pull...")
                try:
                    ollama.pull(model)
                    print(f"Model {model} pulled successfully. Retry manually.")
                except Exception as pull_err:
                    print(f"Failed to pull model: {pull_err}")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except TypeError as te:
        print(f"TypeError: {te}")
    except Exception as ex:
        print(f"Unexpected error: {ex}", file=sys.stderr)

if __name__ == "__main__":
    main()