"""
main.py — FastAPI backend for FinanceGPT
Run: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os

from backend.rag import generate_answer
from backend.memory import init_db, get_or_create_session, save_message, get_history, clear_session

# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="FinanceGPT API",
    description="India-first multilingual finance assistant",
    version="1.0.0",
)

# Allow frontend (Streamlit on HuggingFace) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # in production, set to your HuggingFace Space URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def on_startup():
    init_db()
    print("✅ FinanceGPT API started")


# ─── Request / Response Models ───────────────────────────────
class ChatRequest(BaseModel):
    session_id: str | None = None   # None = new session
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict]             # retrieved documents used


class ClearRequest(BaseModel):
    session_id: str


# ─── Endpoints ───────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "FinanceGPT API is running 🚀"}


@app.get("/health")
def health():
    """UptimeRobot pings this to keep Render server alive."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint.
    - Creates session if session_id is None
    - Retrieves conversation history
    - Runs RAG pipeline
    - Saves messages to memory
    - Returns answer + sources
    """
    # Create or reuse session
    session_id = req.session_id or str(uuid.uuid4())
    get_or_create_session(session_id)

    # Validate message
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (max 2000 chars)")

    # Get conversation history for multi-turn context
    history = get_history(session_id, last_n=8)

    # Run RAG
    answer, sources = generate_answer(
        query=message,
        conversation_history=history,
    )

    # Save to memory
    save_message(session_id, "user", message)
    save_message(session_id, "assistant", answer)

    # Return minimal source info (title + domain only, not full content)
    source_info = [
        {"title": s.get("title", ""), "domain": s.get("domain", "")}
        for s in sources
    ]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=source_info,
    )


@app.post("/clear")
def clear(req: ClearRequest):
    """Clear conversation history for a session."""
    clear_session(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


@app.get("/history/{session_id}")
def history(session_id: str):
    """Get conversation history for a session."""
    msgs = get_history(session_id, last_n=50)
    return {"session_id": session_id, "messages": msgs}
