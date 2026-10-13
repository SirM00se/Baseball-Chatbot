# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Accept Groq API key at build time
ARG GROQ_API_KEY
ENV GROQ_API_KEY=$GROQ_API_KEY

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source files needed for the API
COPY src/__init__.py src/main.py src/rag_engine.py ./src/

# Copy static frontend assets and databases
COPY static/ ./static/
COPY databases/ ./databases/

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
