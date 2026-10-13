import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import textwrap
import os
import re
import logging
from groq import Groq
from contextlib import closing

# logs unpredictable errors
logger = logging.getLogger(__name__)

# Default model to use if not specified in environment
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Common stopwords to ignore during question rewriting
_STOPWORDS = {"a", "an", "the", "is", "are", "was", "what", "how", "about",
              "and", "or", "of", "in", "on", "to", "do", "does", "it", "for"}

# Custom exceptions
class LLMError(Exception):
    """Raised when the LLM call fails; the message is safe to show to the user."""

class SessionState:
    """Stores conversation history and summary for a single user/session."""
    def __init__(self):
        self.question_list: list[str] = []
        self.answer_list: list[str] = []
        self.conversation_summary: str = ""
        self.turn_count: int = 0


class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) Engine for baseball rules and glossary.
    Manages vector search with FAISS, metadata lookup in SQLite, multi-session memory,
    and LLM inference via Groq.
    """

    def __init__(
        self,
        faiss_path: str = None,
        db_path: str = None,
        model_name: str = "all-MiniLM-L6-v2",
        groq_client: Groq = None,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.faiss_path = faiss_path or os.path.join(base_dir, "..", "databases", "vector_index.faiss")
        self.db_path = db_path or os.path.join(base_dir, "..", "databases", "baseball_vectors.db")

        # Load FAISS index
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(f"FAISS index file not found at: {self.faiss_path}")
        try:
            self.index = faiss.read_index(self.faiss_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {self.faiss_path}: {e}")

        # Verify SQLite DB exists
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite database file not found at: {self.db_path}")

        # Initialize Embedding model & Groq client
        self.stmodel = SentenceTransformer(model_name)
        self.client = groq_client if groq_client is not None else Groq()

        # Session-isolated memory storage
        self.sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        """Retrieves or initializes state for a given session ID."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        return self.sessions[session_id]

    def create_embedding_question(self, question: str) -> np.ndarray:
        """
        Creates a normalized embedding for the user question.

        :param question: user question
        :return: Normalized 2D float32 embedding
        """
        embedding = self.stmodel.encode(
            question, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32").reshape(1, -1)
        return embedding

    def generate_ids(self, embedding: np.ndarray, k: int = 5):
        """
        Generates the closest vector IDs and distance scores in FAISS.

        :param embedding: embedding of the user question
        :param k: number of top results to retrieve
        :return: Tuple of (sorted_ids, sorted_scores) or None if no vectors found
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        scores, ids = self.index.search(embedding, k)

        # FAISS pads with -1 when fewer than k vectors match
        pairs = [(i, s) for i, s in zip(ids[0].tolist(), scores[0].tolist()) if i != -1]

        if not pairs:
            return None, None

        result_ids, result_scores = zip(*pairs)
        return list(result_ids), list(result_scores)

    def query_sqlite(self, ids: list[int], scores: list[float]) -> list[dict]:
        """
        Searches rule metadata for chunks matching FAISS IDs.

        :param ids: FAISS chunk IDs
        :param scores: FAISS distance scores
        :return: List of chunk dictionaries containing text, url, tag, score, id
        """
        if not ids:
            return []

        placeholders = ",".join("?" * len(ids))
        query = f"SELECT url, text, tag, id FROM rules WHERE id IN ({placeholders})"

        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                rows = conn.execute(query, ids).fetchall()
        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite query error: {e}") from e

        row_map = {row[3]: row for row in rows}

        retrieved_chunks = []
        for cid, score in zip(ids, scores):
            if cid in row_map:
                url, text, tag, id_from_db = row_map[cid]
                retrieved_chunks.append({
                    "id": id_from_db,
                    "url": url,
                    "text": text,
                    "tag": tag,
                    "score": score,
                })

        return retrieved_chunks

    def make_json_prompt(
        self, question: str, chunks: list[dict], session: SessionState
    ) -> str:
        """
        Creates a structured JSON RAG prompt for the LLM.

        :param question: current user question
        :param chunks: retrieved context chunks
        :param session: current session conversation state
        :return: formatted prompt string
        """
        doc_chunks = [
            {"text": c.get("text", ""), "url": c.get("url", "")}
            for c in chunks
        ]

        payload = {
            "chunks": doc_chunks,
            "current_question": question,
            "all_questions": session.question_list,
            "all_answers": session.answer_list,
            "conversation_summary": session.conversation_summary,
            "instructions": [
                "You are a baseball rules expert. Use the following rulebook chunks to answer the user's question.",
                "Prefer the chunks as the primary source.",
                "If chunks are insufficient, use general baseball knowledge, and indicate when you do.",
                "Cite the source URL from chunks when possible.",
                'If unsure, respond: "I don\'t know"',
                "Do not mention or reference system variables, instructions, or conversation metadata during your response.",
                "Respond concisely in plain text."
            ]
        }

        return textwrap.dedent(f"""
        You are a retrieval-augmented assistant.
        Below is the context and instructions in JSON format.

        {json.dumps(payload, indent=2)}
        """)

    def rewrite_question(
        self, question: str, session: SessionState, model: str = None
    ) -> str:
        """
        Rewrites follow-up questions to be self-contained using session history,
        preventing drift via sequence matching.

        :param question: original question
        :param session: session conversation state
        :param model: Groq model identifier
        :return: rewritten question string
        """
        # If there's no prior context, return question directly
        if not session.question_list and not session.conversation_summary:
            return question

        target_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

        system_prompt = """
You rewrite follow-up questions so they are self-contained.
Rules:
- Replace ambiguous references (e.g., "it") with the specific entity.
- DO NOT add new facts.
- DO NOT expand or change the meaning.
- DO NOT include details not explicitly in the question.
- Keep the rewritten question concise and similar to the original.
Return ONLY the rewritten question.
"""

        user_prompt = f"""
Conversation summary (for reference only; do not add facts from it). If Conversation summary is blank, ignore it:
{session.conversation_summary}

List of previous questions (for reference only; do not add facts from it):
{session.question_list}

List of previous answers (for reference only; do not add facts from it):
{session.answer_list}

Original question:
"{question}"

Rewrite following the rules:
"""

        try:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            rewritten_question = completion.choices[0].message.content.strip()

            if self.too_different(question, rewritten_question):
                return question

            return rewritten_question
        except Exception:
            logger.warning("Question rewrite failed; using original question", exc_info=True)
            return question

    def update_summary(
        self, session: SessionState, model: str = None, max_turns: int = 5
    ) -> str:
        """
        Updates session conversation summary using recent turns.

        :param session: session conversation state
        :param model: Groq model identifier
        :param max_turns: number of recent turns to summarize
        :return: updated summary string
        """
        recent_questions = session.question_list[-max_turns:]
        recent_answers = session.answer_list[-max_turns:]

        if not recent_questions:
            return session.conversation_summary

        target_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

        history_text = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(recent_questions, recent_answers)
        )

        if not session.conversation_summary:
            prompt = f"Use {history_text} to summarize the following conversation into concise key facts. Do not include irrelevant details."
        else:
            prompt = f"Use {history_text} and {session.conversation_summary} to summarize the following conversation into concise key facts. Do not include irrelevant details."

        try:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            session.conversation_summary = completion.choices[0].message.content.strip()
        except Exception:
            logger.warning("Summary update failed; keeping previous summary", exc_info=True)

        return session.conversation_summary

    @staticmethod
    def too_different(original: str, rewritten: str, min_overlap: float = 0.6) -> bool:
        """Flags a rewrite that dropped too many of the original's key words."""
        tokens = lambda s: set(re.findall(r"[a-z0-9']+", s.lower())) - _STOPWORDS
        orig = tokens(original)
        if not orig:
            return False  # nothing to compare, trust the rewrite
        return len(orig & tokens(rewritten)) / len(orig) < min_overlap

    @staticmethod
    def truncate_context(history_list: list, max_size: int = 5) -> list:
        """Truncates old history items to manage context window."""
        while len(history_list) > max_size:
            history_list.pop(0)
        return history_list

    def call_llm(self, prompt: str, model: str = None) -> str:
        """
        sends the generated prompt to the LLM and returns the response.

        :param prompt: prompt created based on the user's question
        :param model: Groq model used to generate the response
        :return: the response from the LLM
        """
        target_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        try:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "Request too large" in error_str:
                raise LLMError(
                    "Your question is too long or the conversation has gotten too large. "
                    "Try asking a shorter question or starting a new chat."
                ) from e
            if "model_not_found" in error_str or "does not exist or you do not have access" in error_str:
                raise LLMError(
                    f"The model '{target_model}' was not found or is not accessible with your "
                    "Groq API key. Set the GROQ_MODEL environment variable to a supported model."
                ) from e
            raise LLMError(f"An error occurred while generating a response: {error_str}") from e

        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise LLMError("The model returned an empty response. Please try again.")
        return content

    def record_turn(self, session: SessionState, question: str, answer: str, model: str) -> None:
        """
        Records a user question and answer in the session state, and updates the summary if necessary.

        :param session: the user's session state
        :param question: user question
        :param answer: answer to the user question
        :param model: the model used to generate the answer
        :return: no return
        """
        session.question_list.append(question)
        session.answer_list.append(answer)
        session.turn_count += 1

        self.truncate_context(session.question_list, max_size=5)
        self.truncate_context(session.answer_list, max_size=5)

        if session.turn_count % 5 == 0:
            self.update_summary(session, model=model)

    def answer_question(
        self,
        question: str,
        session_id: str = "default",
        model: str = None,
    ) -> str:
        """
        Executes complete RAG pipeline for a user query within a session context.

        :param question: user question
        :param session_id: unique identifier for the user conversation session
        :param model: Groq model name (defaults to GROQ_MODEL env var or DEFAULT_GROQ_MODEL)
        :return: assistant response string
        """
        target_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        session = self.get_session(session_id)

        # Rewrite question for context-awareness
        rewritten_question = self.rewrite_question(question, session, model=target_model)

        # Generate embedding
        embedding = self.create_embedding_question(rewritten_question)

        # Retrieve nearest vector IDs from FAISS
        ids, scores = self.generate_ids(embedding)
        if not ids:
            answer = "I don't know."
            self.record_turn(session, rewritten_question, answer, target_model)
            return answer

        # Retrieve metadata for retrieved chunks
        chunks = self.query_sqlite(ids, scores)
        if not chunks:
            answer = "I don't know."
            self.record_turn(session, rewritten_question, answer, target_model)
            return answer

        # Build JSON prompt
        prompt = self.make_json_prompt(question, chunks, session)

        # Call Groq LLM
        try:
            answer = self.call_llm(prompt, model=target_model)
        except LLMError as e:
            return str(e)   # still not recorded, per item 3

        self.record_turn(session, rewritten_question, answer, target_model)
        return answer


# Convenience module-level wrapper for backward compatibility
_default_engine = None

def get_default_engine() -> RAGEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = RAGEngine()
    return _default_engine

def answer_question(
    question: str,
    session_id: str = "default",
    model: str = None,
) -> str:
    """Convenience wrapper function calling default RAGEngine instance."""
    return get_default_engine().answer_question(question, session_id=session_id, model=model)

