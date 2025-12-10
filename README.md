# Baseball-Chatbot

A project that answers all kinds of questions about baseball.

## What this project does

This project will use RAG to answer questions about baseball

- Answers questions about baseball rules

- Answers questions about baseball players

- Returns statistics about baseball players

### What each file does
- dataScraper.py returns chunks from the mlb rulebook

- vectordatabase.py creates a vector database from the chunks returned from dataScraper.py

- SQLConversion.py turns the metadata csv into an SQL database for ease of querying

- rebuildVectorDatabase.py rebuilds the vector database using all three files above

## Installation

### From Source

```bash
pip install requests
```

```bash
pip install pandas
```

```bash
pip install selenium
```

```bash
pip install beautifulsoup4
```

```bash
pip install webdriver_manager
```

```bash
pip install sentence-transformers
```

```bash
pip install faiss-cpu
```

```bash
pip install numpy
```

```bash
pip install pytest
```

```bash
pip install ollama
```

```bash
pip install fastapi
```

#### Make sure to include a driver for selenium such as chromedriver in the local repository

## Testing
Currently has unit, installation, and end to end tests for dataScraper.py
Additional end to end test for the vector database and metadata

```bash
python tests/test_units_dataScraper.py
python tests/test_integration_dataScraper.py
python tests/test_e2e_dataScraper.py
python tests/test_e2e_rulesdata.py
```

if pytest installed:

```bash
pytest tests/
```
