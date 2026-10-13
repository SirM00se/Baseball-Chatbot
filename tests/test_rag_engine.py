import sqlite3
import numpy as np
import faiss
import pytest
from unittest.mock import MagicMock, patch
from src.rag_engine import RAGEngine, SessionState


@pytest.fixture
def fake_embedding():
    """Deterministic 384-dim vector to replace SentenceTransformer embeddings."""
    return np.ones((1, 384), dtype="float32")


@pytest.fixture
def fake_faiss_file(tmp_path):
    """Creates a temporary FAISS index file."""
    dim = 384
    base = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap(base)
    vectors = np.ones((5, dim), dtype="float32")
    ids = np.array([0, 1, 2, 3, 4], dtype="int64")
    index.add_with_ids(vectors, ids)

    index_path = str(tmp_path / "test_vector_index.faiss")
    faiss.write_index(index, index_path)
    return index_path


@pytest.fixture
def fake_sqlite_db(tmp_path):
    """Creates a temporary SQLite DB matching the rules schema."""
    db_path = str(tmp_path / "test_rules.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE rules (
            url TEXT,
            text TEXT,
            tag TEXT,
            id INTEGER PRIMARY KEY
        )
    """)
    for i in range(5):
        cur.execute(
            "INSERT INTO rules (url, text, tag, id) VALUES (?, ?, ?, ?)",
            (f"https://mlb.com/glossary/rules/{i}", f"Rule description {i}", f"tag_{i}", i),
        )
    conn.commit()
    conn.close()
    return db_path


def test_rag_engine_initialization_and_multi_session(fake_faiss_file, fake_sqlite_db, fake_embedding):
    """Verifies RAGEngine initializes properly, isolates session states, and handles query execution."""
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Answer for question."
    mock_groq.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("src.rag_engine.SentenceTransformer") as mock_st:
        mock_st.return_value.encode.return_value = fake_embedding

        engine = RAGEngine(
            faiss_path=fake_faiss_file,
            db_path=fake_sqlite_db,
            groq_client=mock_groq,
        )

        # Session isolation test
        session_a = engine.get_session("user_a")
        session_b = engine.get_session("user_b")
        assert session_a is not session_b

        # Answer question for user A
        ans_a = engine.answer_question("What is an infield fly?", session_id="user_a")
        assert ans_a == "Answer for question."
        assert len(session_a.question_list) == 1
        assert len(session_b.question_list) == 0

        # Answer question for user B
        ans_b = engine.answer_question("What is a balk?", session_id="user_b")
        assert ans_b == "Answer for question."
        assert len(session_a.question_list) == 1
        assert len(session_b.question_list) == 1


def test_drift_prevention_too_different():
    """Tests the drift detection threshold comparison."""
    assert not RAGEngine.too_different("What is a balk in baseball?", "What is a balk in baseball?")
    assert RAGEngine.too_different("What is a balk?", "Completely unrelated text about astronomy.")


def test_missing_files_raise_exceptions(tmp_path):
    """Verifies that missing files raise FileNotFoundError instead of terminating interpreter."""
    non_existent_faiss = str(tmp_path / "missing.faiss")
    non_existent_db = str(tmp_path / "missing.db")

    with pytest.raises(FileNotFoundError):
        RAGEngine(faiss_path=non_existent_faiss, db_path=non_existent_db)
