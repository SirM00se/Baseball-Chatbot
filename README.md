# Baseball-Chatbot

A project that answers all kinds of questions about baseball.

## What this project does

This project will use RAG to answer questions about baseball

*Answers questions about baseball rules
*Answers questions about baseball players
*Returns statistics about baseball players

### What each file does
*dataScraper.py returns chunks from the mlb rulebook
*vectordatabase.py creates a vectordatabase from the chunks returned from dataScraper.py
*SQLConversion.py turns the metadata csv into an SQL database for ease of querying
*rebuildVectorDatabase.py rebuilds the vector database using all three files above

## Installation

1. Install requests
2. Install pandas
3. Install selenium
4. Install beautifulsoup
5. Install webdriver_manager
6. Install sentence-transformers
7. Install faiss-cpu
8. Install numpy
9. Install tqdm
