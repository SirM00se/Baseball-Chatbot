# Baseball-Chatbot

A project that answers all kinds of questions about baseball.  
website: [baseballchatbot.org](https://baseballchatbot.org)

## What this project does

This project will use RAG to answer questions about baseball

- Answers questions about baseball rules (in progress)

- Answers questions about baseball players (not started)

- Returns statistics about baseball players (not started)

### What each file does
- dataScraper.py returns chunks from the mlb rulebook glossary

- AddBaseballRules.py returns from an mlb rulebook txt file

- vectordatabase.py creates a vector database from the chunks returned from dataScraper.py

- SQLConversion.py turns the metadata csv into an SQL database for ease of querying

- rebuildVectorDatabase.py rebuilds the vector database using all three files above

- answerQuestion.py answers questions about baseball rules within the user's terminal. It utilizes Ollama, so make sure that you have Ollama running with the llama3.1:8b model.

- rag_engine.py is the engine of the website that utilizes Groq to generate answers.

- main.py is the api that connects rag_engine.py to the webpage

- chat.js and index.html are my frontend files
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
pip install fastapi-cpu
```

```bash
pip install groq
```

```bash
pip install sqlalchemy
```
```bash
pip install uvicorn
```
#### Make sure to include a driver for selenium such as chromedriver in the local repository

## Testing
Currently has unit, installation, and end to end tests for dataScraper.py.  
Additional end to end test for the vector database and metadata.

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
