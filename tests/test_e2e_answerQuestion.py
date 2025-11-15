import sqlite3
import numpy as np
import faiss
import pytest
from unittest.mock import patch
import answerQuestion

@pytest.fixture
def fake_embedding():
    """Deterministic 384-dim vector to replace SentenceTransformers."""
    return np.ones((1, 384), dtype="float32")


@pytest.fixture
def fake_faiss_index():
    """Create an in-memory FAISS index with 5 vectors."""

    dim = 384

    # ID-mapped FAISS index ensures explicit control of IDs
    base = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap(base)

    vectors = np.random.rand(5, dim).astype("float32")
    ids = np.array([0,1,2,3,4], dtype="int64")

    index.add_with_ids(vectors, ids)

    return index


@pytest.fixture
def fake_sqlite_db(tmp_path):
    """Create an SQLite DB matching your schema."""

    db_path = tmp_path / "test.db"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Create schema
    cur.execute("""
                CREATE TABLE rules (
                                       url TEXT,
                                       text TEXT,
                                       tag TEXT,
                                       id INTEGER PRIMARY KEY
                )
                """)

    # Insert rows with ids 0–4
    for i in range(5):
        cur.execute(
            "INSERT INTO rules (url, text, tag, id) VALUES (?, ?, ?, ?)",
            (f"http://doc{i}.com", f"Sample text {i}", f"tag{i}", i)
        )

    conn.commit()
    conn.close()
    return str(db_path)

def test_e2e_pipeline(fake_embedding, fake_faiss_index, fake_sqlite_db):
    """
    tests if the fake chunks and question are in the prompt
    Also checks if the final answer is correct

    :param fake_embedding:
    :param fake_faiss_index:
    :param fake_sqlite_db:
    :return:
    """
    # -----------------------
    # 1) Mock external systems
    # -----------------------

    # Mock SentenceTransformer.encode → returns fake_embedding
    with patch("answerQuestion.SentenceTransformer") as mock_model:

        mock_model.return_value.encode.return_value = fake_embedding

        # Mock FAISS index loading to return our fake index
        with patch("answerQuestion.faiss.read_index") as mock_faiss_load:
            mock_faiss_load.return_value = fake_faiss_index

            # Mock SQLite path
            with patch("answerQuestion.sqlite3.connect", return_value=sqlite3.connect(fake_sqlite_db)):

                # Mock Ollama
                with patch("answerQuestion.ollama.chat") as mock_ollama:
                    mock_ollama.return_value = {
                        "message": {
                            "content": "Final answer from Ollama"
                        }
                    }

                    # -----------------------
                    # 2) Run your pipeline
                    # -----------------------
                    question = "What is sample text?"
                    embedding = answerQuestion.createEmbeddingQuestion(question)
                    ids, scores = answerQuestion.generateIDs(embedding)
                    ids = [int(i) for i in ids]
                    chunks = answerQuestion.querySQLite(ids, scores)
                    prompt = answerQuestion.make_ollama_json_prompt(question, chunks)
                    answer = answerQuestion.callOllama(prompt, model="llama3.1")

                    # -----------------------
                    # 3) Assertions
                    # -----------------------

                    # Embedding shape
                    assert embedding.shape == (1, 384)

                    # IDs returned
                    assert len(ids) == 5

                    # SQLite returned 5 chunks
                    assert len(chunks) == 5
                    assert chunks[0]["text"].startswith("Sample text")

                    # Prompt contains JSON-style chunks
                    assert "CHUNKS:" in prompt
                    assert "QUESTION" in prompt

                    # Final Ollama answer
                    assert answer == "Final answer from Ollama"

                    print("\n\nFINAL ANSWER:", answer)
                    print("\nGENERATED PROMPT:\n", prompt)