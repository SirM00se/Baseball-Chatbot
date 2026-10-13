# Baseball Rules RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) conversational assistant for Major League Baseball (MLB) official rules and glossary terminology.

Live Demo: [baseballchatbot.org](https://baseballchatbot.org)

---

## Overview

The **Baseball Rules RAG Chatbot** enables baseball fans, umpires, coaches, and players to query complex MLB regulations and terminology in natural language. The system pairs dense vector search over scraped official rules and glossary definitions with multi-turn conversational memory, drift detection, and high-speed LLM inference via the Groq API (as well as support for local LLMs via Ollama).

### Key Features & Architecture Highlights
- **End-to-End Ingestion Pipeline**: Automated web scraping of MLB glossary pages using Selenium and BeautifulSoup, combined with structured parsing and chunking of official MLB rulebooks and definition texts.
- **Dense Vector Search & Storage**: Dense text embeddings generated with `all-MiniLM-L6-v2` (Sentence-Transformers) and indexed in FAISS (`IndexFlatL2` with `IndexIDMap`) for rapid similarity search.
- **Relational Metadata Store**: SQLite (`databases/baseball_vectors.db`) stores text chunks, tags, rule IDs, and source citation URLs for rich grounded answers.
- **Context & Drift-Aware Conversational Memory**: Multi-turn session management with contextual query rewriting to resolve pronouns and implicit references, guarded by `difflib.SequenceMatcher` similarity heuristics to prevent topic drift, along with periodic context summarization.
- **FastAPI Web Backend & Modern UI**: Session-isolated FastAPI REST API exposing `/chat` and serving static frontend assets (`index.html`, `style.css`, `chat.js`) with responsive chat bubbles and markdown link rendering.
- **Local CLI Prototype Support**: Includes `src/answerQuestion.py` for running an interactive terminal assistant powered by local Ollama models (`llama3.1:8b`).
- **Comprehensive Testing Suite**: Unit, integration, and end-to-end test coverage across scraping, database retrieval, prompt generation, session isolation, and RAG engine logic using `pytest` and `unittest.mock`.

---

## Project Structure

```text
Baseball-Chatbot/
├── data/                             # Raw texts, parsed datasets, and vector metadata
│   ├── BaseballRules.txt             # Official MLB rules text
│   ├── Definition of Terms.txt       # Official glossary definitions text
│   ├── baseballrules.csv             # Combined scraped glossary and parsed rules dataset
│   ├── eval_dataset.json             # Golden evaluation dataset with questions and ground truth
│   └── vector_metadata.csv           # Text chunks with IDs and embedding vectors
├── databases/                        # Vector index and relational database storage
│   ├── baseball_vectors.db           # SQLite database storing rule text, URLs, and IDs
│   └── vector_index.faiss            # FAISS vector index for dense similarity search
├── src/                              # Application source code
│   ├── AddBaseballRules.py           # Parses and chunks local rulebook texts into CSV
│   ├── answerQuestion.py             # CLI interactive chatbot prototype using local Ollama
│   ├── dataScraper.py                # Automated Selenium & BeautifulSoup scraper for MLB glossary
│   ├── evaluate_ragas.py             # Automated RAG evaluation pipeline using Ragas
│   ├── main.py                       # FastAPI REST API and static web server
│   ├── rag_engine.py                 # Core RAG engine: FAISS search, SQLite lookup, & Groq LLM client
│   ├── rebuildVectorDatabase.py      # Automated pipeline runner to rebuild index and database
│   ├── SQLConversion.py              # Loads metadata CSV and writes SQLite database table
│   └── vectordatabase.py             # Generates sentence embeddings and builds FAISS index
├── static/                           # Frontend web interface assets
│   ├── chat.js                       # Session persistence, API communication, and link formatting
│   ├── index.html                    # Chatbot web interface and layout
│   └── style.css                     # Responsive styling and chat bubble animations
├── tests/                            # Pytest automated test suite
│   ├── test_e2e_dataScraper.py       # E2E tests for scraping pipeline
│   ├── test_e2e_rulesdata.py         # Vector index and SQLite retrieval validation tests
│   ├── test_evaluation.py            # Retrieval accuracy unit tests
│   ├── test_integration_dataScraper.py # Integration tests for HTML parsing and web scraping
│   ├── test_rag_engine.py            # Unit and session-isolation tests for RAG engine
│   └── test_units_dataScraper.py     # Unit tests for individual scraping helpers
├── ragas_results.csv                 # Exported evaluation metrics results
└── requirements.txt                  # Python dependencies
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- Google Chrome & ChromeDriver (if running the live web scraper)
- Groq API Key (for FastAPI web app) or Ollama with `llama3.1:8b` (for CLI prototype)

### 2. Clone & Install Dependencies

```bash
git clone https://github.com/blakelittlejohn/Baseball-Chatbot.git
cd Baseball-Chatbot

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all required dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Set your Groq API key (and optionally override the default model, such as `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, or `qwen/qwen3.8-27b`):

```bash
export GROQ_API_KEY="your_groq_api_key_here"
# Optional: defaults to openai/gpt-oss-120b
export GROQ_MODEL="openai/gpt-oss-120b"
```

*(On Windows PowerShell: `$env:GROQ_API_KEY="your_groq_api_key_here"`)*

*(Note: Make sure that you use Groq model accessible to your plan.)*

---

## Running the Application

### Option A: Web Application (FastAPI + Groq)

Start the FastAPI application using Uvicorn:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser and navigate to `http://localhost:8000` to interact with the web interface.

#### API Endpoints
- `GET /`: Serves the chatbot web UI (`static/index.html`).
- `POST /chat`: Accepts JSON `{"message": "string", "session_id": "optional_string"}` and returns `{"response": "string", "session_id": "string"}`.
- `GET /static/*`: Serves static assets (`chat.js`, `style.css`).

### Option B: Interactive CLI Assistant (Ollama)

To run the local terminal version powered by Ollama:

1. Ensure Ollama is running and pull the model:
   ```bash
   ollama pull llama3.1:8b
   ```
2. Run the CLI script:
   ```bash
   cd src
   python src/answerQuestion.py
   ```

---

## Rebuilding the Vector Database & Ingestion Pipeline

To re-scrape official MLB glossary pages, parse raw rule texts, and regenerate both the FAISS vector index and SQLite database:

```bash
cd src
python rebuildVectorDatabase.py
```

This pipeline executes the following stages sequentially:
1. `dataScraper.py`: Scrapes rule glossary pages from `https://www.mlb.com/glossary/rules` and writes `data/baseballrules.csv`.
2. `AddBaseballRules.py`: Parses `data/BaseballRules.txt` and `data/Definition of Terms.txt`, appending chunks to `data/baseballrules.csv`.
3. `vectordatabase.py`: Generates `all-MiniLM-L6-v2` embeddings, builds `databases/vector_index.faiss`, and exports `data/vector_metadata.csv`.
4. `SQLConversion.py`: Reads `data/vector_metadata.csv` and populates the `rules` table in `databases/baseball_vectors.db`.

---

## Testing

Execute the automated test suite using `pytest`:

```bash
python -m pytest tests/
```

---

## RAG Pipeline Evaluation (Ragas)

The repository includes an automated evaluation pipeline using the [Ragas](https://docs.ragas.io/) framework to assess retrieval and generation quality against a golden dataset (`data/eval_dataset.json`).

### What It Does

The `src/evaluate_ragas.py` script:
1. Loads evaluation test cases containing questions and ground-truth references.
2. Runs the question through the `RAGEngine` to retrieve relevant context chunks from the FAISS/SQLite vector store.
3. Generates the model's answer.
4. Computes core RAG evaluation metrics using LLM/embedding evaluators:
   - **Context Precision**: Evaluates whether retrieved contexts are relevant and ranked appropriately.
   - **Context Recall**: Measures whether all necessary ground-truth information was successfully retrieved.
   - **Faithfulness**: Measures whether the generated answer is grounded in the retrieved context (preventing hallucinations).
   - **Answer Relevancy**: Quantifies how directly relevant the generated response is to the input query.
5. Exports full evaluation scores and sample breakdowns to `ragas_results.csv`.

### Running Evaluation

Ensure your `OPENAI_API_KEY` is exported (used by Ragas for evaluation LLM and embedding scoring):

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
python src/evaluate_ragas.py
```

---

## Credits

- Baseball Icon by Vincent Le Moign on [Icon-Icons.com](https://icon-icons.com/authors/514-vincent-le-moign)
