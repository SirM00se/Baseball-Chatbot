import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import sys
import json
import textwrap
import os
from difflib import SequenceMatcher
from groq import Groq

#Creates client
client = Groq()

#load models
try:
    #index = faiss.read_index("../databases/vector_index.faiss")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_PATH = os.path.join(BASE_DIR, "..", "databases", "vector_index.faiss")

    index = faiss.read_index(FAISS_PATH)
except Exception as e:
    print(f"FAISS index loading error: {e}")
    sys.exit(1)
stmodel = SentenceTransformer('all-MiniLM-L6-v2')
#lists that store context
question_list = []
qa_history = []
conversation_summary = ""

def createEmbeddingQuestion(question) -> np.ndarray:
    """
    creates an embedding for the user question

    :param question: user question
    :return:
        The embedding of the user question
    """

    # generates the embedding for the question
    embedding = stmodel.encode(question, convert_to_numpy=True, normalize_embeddings=True).astype('float32').reshape(1,
                                                                                                                   -1)
    return embedding


def generateIDs(embedding):
    """
    generates the closest ids in the vector database to the user question

    :param embedding: embedding of the user question
    :return:
        An array of sorted ids and scores
    """

    k = 5  # checks for the top 5 results

    # Ensure embedding is [1, dim]
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)

    scores, ids = index.search(embedding, k)

    # checks if ids is empty
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
    # intialize access to the database
    try:
        BASE_DIR2 = os.path.dirname(os.path.abspath(__file__))
        db_PATH = os.path.join(BASE_DIR2, "..", "databases", "baseball_vectors.db")
        conn = sqlite3.connect(db_PATH)
    except sqlite3.Error as e:
        print(f"SQLite connection error: {e}")
        sys.exit(1)

    cursor = conn.cursor()

    # Allows the ability to safely pass ids
    placeholders = ",".join(["?"] * len(ids))

    # Queries the sql database
    query = f"SELECT url, text, tag, id FROM rules WHERE id IN ({placeholders})"
    cursor.execute(query, ids)
    rows = cursor.fetchall()
    conn.close()

    # Build a mapping from id → row
    row_map = {row[3]: row for row in rows}
    # checks if row is missing
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


def make_json_prompt(question, chunks):
    """
    Creates a clean JSON-style RAG prompt for Ollama.

    :param question: str — current user question
    :param chunks: list[dict] — [{'text': ..., 'url': ...}, ...]
    :return: str — a JSON RAG prompt string
    """


    # Pair text + URL together to preserve mapping
    doc_chunks = [
        {"text": c.get("text", ""), "url": c.get("url", "")}
        for c in chunks
    ]

    # Create the data payload
    payload = {
        "chunks": doc_chunks,
        "current_question": question,
        "all_questions": question_list,
        "instructions": [
            "Answer using ONLY the information in the chunks.",
            "Answer only the CURRENT_QUESTION.",
            "Use ALL_QUESTIONS only as context but do not invent anything.",
            "Cite the source URL for every fact you use.",
            "If the answer is not in the chunks, say: \"I don't know.\"",
            "Do not hallucinate missing information.",
            "Respond in plain text with inline URLs.",
        ]
    }

    # Build the final prompt
    prompt = textwrap.dedent(f"""
    You are a retrieval-augmented assistant.
    Below is the context and instructions in JSON format.

    {json.dumps(payload, indent=2)}
    """)

    return prompt

def rewrite_question(question, model):
    """
    Rewrite the question to be self-contained using conversation summary,
    while preventing drift.
    :param question: original question
    :param model: Ollama model
    :return: the rewritten question
    """
    global conversation_summary

    system_prompt = """
You rewrite follow-up questions so they are self-contained.
Rules:
- Replace ambiguous references (e.g., “it”) with the specific entity.
- DO NOT add new facts.
- DO NOT expand or change the meaning.
- DO NOT include details not explicitly in the question.
- Keep the rewritten question concise and similar to the original.
Return ONLY the rewritten question.
"""

    user_prompt = f"""
Conversation summary (for reference only; do not add facts from it):
{conversation_summary}

Original question:
"{question}"

Rewrite following the rules:
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # stream parameter omitted; Groq streaming is different
    )

    rewritten_question = completion.choices[0].message.content.strip()

    # If rewrite drifted too far, fallback to original
    if too_different(question, rewritten_question):
        return question

    return rewritten_question


def update_summary(model, max_turns=5):
    """
    Updates the summary of they conversation
    :param model: Ollama model
    :param max_turns: number of turns in the conversation
    :return: the summary
    """
    global qa_history, conversation_summary

    # Take the last few turns for summarization
    recent_qa = qa_history[-max_turns:]

    history_text = "\n".join(f"Q: {qa['question']}\nA: {qa['answer']}" for qa in recent_qa)

    prompt = f"""
Summarize the following conversation into concise key facts.
Do not include irrelevant details.

{history_text}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        # stream parameter omitted; streaming handled differently in Groq
    )

    conversation_summary = completion.choices[0].message.content.strip()
    return conversation_summary

def too_different(original, rewritten, threshold=0.35):
    """
    Checks if new question is too different
    :param original: original question
    :param rewritten: rewritten question
    :param threshold: threshold that determines if question is too different
    :return: whether the question is too different
    """
    ratio = SequenceMatcher(None, original, rewritten).ratio()
    return ratio < threshold

def truncateContext(question_list):
    """
    Removes old questions to manage memory
    :param question_list: list of questions
    :return: the updated list
    """
    del question_list[0]
    return question_list

def callLLM(prompt, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def answer_question(question: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Runs one full RAG query using the existing pipeline.
    Returns the answer as a string.
    """
    global question_list, qa_history

    # add to context
    question_list.append(question)

    # conversation summary every 5 turns
    if (len(qa_history) == 1) or (len(qa_history) % 5 == 0):
        update_summary(model)

    # rewrite question
    rewritten_question = rewrite_question(question, model)

    # embedding
    embedding = createEmbeddingQuestion(rewritten_question)

    # retrieve IDs from FAISS
    ids, scores = generateIDs(embedding)
    if not ids:
        return "I don't know."

    # retrieve metadata
    chunks = querySQLite(ids, scores)
    if not chunks:
        return "I don't know."

    # build RAG prompt
    prompt = make_json_prompt(question, chunks)

    # call Ollama
    answer = callLLM(prompt, model)

    # save to history
    qa_history.append({"question": question, "answer": answer})

    return answer

