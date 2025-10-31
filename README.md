# Baseball-Chatbot

A project that answers all kinds of questions about baseball.

## What this project does

This project will use RAG to answer questions about baseball

- Answers questions about baseball rules

- Answers questions about baseball players

- Returns statistics about baseball players

### What each file does
- dataScraper.py returns chunks from the mlb rulebook

- vectordatabase.py creates a vectordatabase from the chunks returned from dataScraper.py

- SQLConversion.py turns the metadata csv into an SQL database for ease of querying

- rebuildVectorDatabase.py rebuilds the vector database using all three files above

## Installation

### From Source

```bash
pip install requests
pip install pandas
pip install selenium
pip install beautifulsoup
pip install webdriver_manager
pip install sentence-transformers
pip install faiss-cpu
pip install numpy
pip install pytest
```

#### Make sure to include a driver for selenium such as chromedriver in the local repository
