# Baseball Rules RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) conversational assistant for Major League Baseball (MLB) official rules and glossary terminology.

Live Demo: [baseballchatbot.org](https://baseballchatbot.org)

---

## Overview

The **Baseball Rules RAG Chatbot** enables baseball fans, umpires, and players to query complex MLB regulations and terminology in natural language. The system pairs dense vector search over scraped official rules with multi-turn conversational memory, drift detection, and high-speed LLM inference via the Groq API.

### Key Architecture & Engineering Highlights
- **End-to-End Data Pipeline**: Automated web scraping of MLB glossary and rulebooks using Selenium and BeautifulSoup, parsed and chunked for semantic retrieval.
- **Hybrid Storage & Indexing**: Dense text embeddings generated with `all-MiniLM-L6-v2` (Sentence-Transformers) and indexed in FAISS (`IndexFlatL2` / `IndexIDMap`) for similarity search, combined with SQLite for structured rule metadata and URL referencing.
- **Context & Drift-Aware Conversation**: Contextual query rewriting to resolve pronouns and implicit references across multi-turn chats, with `difflib.SequenceMatcher` drift prevention heuristics and periodic context summarization.
- **Session-Isolated FastAPI Backend**: Clean session management isolating concurrent user histories, exposed via FastAPI REST endpoints with static HTML/JS frontend delivery.
- **Production-Grade Testing**: Automated test suite containing unit, integration, and end-to-end pipeline validation with `pytest` and `unittest.mock`.

---

## Project Structure

```text
Baseball-Chatbot/
├── data/                         # Scraped rulebooks, glossaries, and CSV metadata
├── databases/                    # FAISS vector index and SQLite database
│   ├── baseball_vectors.db
│   └── vector_index.faiss
├── src/                          # Application source code
│   ├── dataScraper.py            # Automated scraper for MLB glossary rules
│   ├── AddBaseballRules.py       # Rulebook text processing and chunking
│   ├── vectordatabase.py         # Embedding generation and FAISS indexing
│   ├── SQLConversion.py          # SQLite metadata database builder
│   ├── rebuildVectorDatabase.py  # Pipeline orchestrator to regenerate index & DB
│   ├── rag_engine.py             # Core RAG engine, vector search & Groq LLM client
│   └── main.py                   # FastAPI REST API & static file server
├── static/                       # Frontend assets
│   ├── index.html                # Chat UI
│   ├── chat.js                   # Client-side session and async chat controller
│   └── styles.css                # Interface styling
├── tests/                        # Comprehensive test suite
│   ├── test_units_dataScraper.py
│   ├── test_integration_dataScraper.py
│   ├── test_e2e_dataScraper.py
│   ├── test_e2e_rulesdata.py
│   ├── test_e2e_answerQuestion.py
│   └── test_rag_engine.py
└── requirements.txt              # Project dependencies
```

---

## Getting Started

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/blakelittlejohn/Baseball-Chatbot.git
cd Baseball-Chatbot

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all required dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Set your Groq API key (and optionally override the default model, e.g. `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, or `qwen/qwen3.8-27b`):

```bash
export GROQ_API_KEY="your_groq_api_key_here"
# Optional: defaults to openai/gpt-oss-120b
export GROQ_MODEL="openai/gpt-oss-120b"
```

*(On Windows PowerShell: `$env:GROQ_API_KEY="your_groq_api_key_here"`)*

### 3. Run the Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser and navigate to `http://localhost:8000`.

---

## Rebuilding the Vector Database & Ingestion Pipeline

To re-scrape the latest rules and regenerate the FAISS index and SQLite metadata store:

```bash
python src/rebuildVectorDatabase.py
```

---

## Testing

Execute the complete test suite using `pytest`:

```bash
pytest tests/
```
