
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.concurrency import run_in_threadpool
from functools import partial
import uvicorn

app = FastAPI()

# Do not create the pipeline at import time (can block / raise during startup)
qa_pipeline = None

# Load model on startup (non-blocking for the event loop is handled when calling the pipeline)
@app.on_event("startup")
async def load_model():
    global qa_pipeline
    try:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    except Exception as e:
        # Keep the app running; endpoint will lazy-load on first request if needed
        print("Model load failed at startup:", e)

# ...existing code...
class ChatRequest(BaseModel):
    question: str
    context: str

class ChatResponse(BaseModel):
    answer: str

# Creating the /chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global qa_pipeline
    try:
        if qa_pipeline is None:
            # lazy-load if startup failed or wasn't performed
            qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

        # run the synchronous pipeline off the event loop
        func = partial(qa_pipeline, question=request.question, context=request.context)
        result = await run_in_threadpool(func)
        return ChatResponse(answer=result['answer'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# running the app server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
