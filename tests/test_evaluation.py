import re
from unittest.mock import MagicMock
import pytest
from src.rag_engine import RAGEngine


def normalize_tag(text: str) -> str:
    """Normalize tags and expectations by lowercasing and replacing non-alphanumeric chars with spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@pytest.mark.parametrize("query,expected_tag", [
    ("What is an infield fly?", "Infield Fly"),
    ("When does a balk occur?", "Balk"),
    ("What are the rules for designated hitters?", "Designated Hitter"),
])
def test_top3_retrieval_accuracy(query, expected_tag):
    engine = RAGEngine(groq_client=MagicMock())
    emb = engine.create_embedding_question(query)
    ids, scores = engine.generate_ids(emb, k=3)
    chunks = engine.query_sqlite(ids, scores)

    norm_expected = normalize_tag(expected_tag)
    normalized_tags = [normalize_tag(c["tag"]) for c in chunks]
    assert any(norm_expected in t for t in normalized_tags), f"Expected '{expected_tag}' not found in top 3 retrieved tags: {normalized_tags}"