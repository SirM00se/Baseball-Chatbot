import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.rag_engine import RAGEngine

rag_engine: RAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    rag_engine = RAGEngine()
    yield


app = FastAPI(
    title="Baseball Rules RAG Chatbot",
    description="Conversational assistant querying official MLB rules and glossary data using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.post("/chat")
async def api_chat(request: Request):
    global rag_engine
    data = await request.json()
    question = data.get("message", "")
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not question.strip():
        return JSONResponse(
            {"response": "Please enter a valid question.", "session_id": session_id}
        )

    if rag_engine is None:
        rag_engine = RAGEngine()

    answer = rag_engine.answer_question(question, session_id=session_id)

    return JSONResponse({"response": answer, "session_id": session_id})