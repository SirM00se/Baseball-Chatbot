from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.rag_engine import answer_question  # <-- use wrapped logic

app = FastAPI()

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.post("/chat")
async def api_chat(request: Request):
    data = await request.json()
    question = data.get("message", "")

    if not question.strip():
        return JSONResponse({"response": "Please enter a valid question."})

    # call your RAG pipeline
    answer = answer_question(question)

    return JSONResponse({"response": answer})