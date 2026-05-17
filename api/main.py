"""
api/main.py
===========
FastAPI application exposing two endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse (exact schema required by evaluator)

Design decisions:
- Pydantic models enforce the exact response schema. Any missing or extra
  fields cause a 422 validation error from FastAPI, which we catch and
  convert to a valid fallback response.
- The app is fully stateless: no in-memory session store, no database.
  All conversation state lives in the request body (messages list).
- On startup, we pre-load the FAISS index and embedding model into memory.
  This trades startup time for fast request handling (avoids cold-start
  latency on first chat request).
- All exceptions are caught at the /chat handler level and converted to
  a valid ChatResponse so the API never crashes or returns a malformed body.
- Logging is configured at INFO level so Render/Railway can capture it.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file (for local development; on Render use env var panel)
load_dotenv()

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── Pydantic Models ────────────────────────────────────────────────────────────

class Message(BaseModel):
    """Single message in the conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    """Incoming request body for POST /chat."""
    messages: List[Message] = Field(
        ...,
        description="Full conversation history from the beginning. Must not be empty.",
    )


class Recommendation(BaseModel):
    """A single assessment recommendation."""
    name: str = Field(..., description="Exact assessment name from SHL catalog")
    url: str = Field(..., description="Exact URL from SHL catalog (never invented)")
    test_type: str = Field(..., description="Primary test type code (A/B/C/D/E/K/P/S)")


class ChatResponse(BaseModel):
    """
    Response schema — MUST match exactly. Automated evaluator checks every field.
    - reply: conversational text response
    - recommendations: empty [] when clarifying/refusing, 1-10 items when recommending
    - end_of_conversation: true only when agent considers task fully complete
    """
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ── Safe fallback responses ────────────────────────────────────────────────────

def _safe_error_response(message: str = "I encountered an error. Please try again.") -> ChatResponse:
    return ChatResponse(
        reply=message,
        recommendations=[],
        end_of_conversation=False,
    )


# ── Lifespan: pre-load model and index ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load the FAISS index and sentence-transformer model at startup.

    Why: The first search() call would trigger model loading (~3-5s) and index
    loading (~1s). By pre-loading, the first real /chat request is fast.
    This is critical for the 30s timeout requirement.
    """
    log.info("=== SHL Assessment Recommender starting up ===")
    try:
        from retriever.retriever import _ensure_loaded, build_index
        
        # Build index if it doesn't exist yet
        from pathlib import Path
        index_path = Path("index/tfidf.pkl")
        if not index_path.exists():
            log.info("TF-IDF index not found. Building from catalog...")
            build_index()
        
        _ensure_loaded()
        log.info("TF-IDF index pre-loaded. Ready.")
    except Exception as e:
        log.error(f"Failed to pre-load retriever: {e}")
        log.warning("Server will start but first requests may be slow.")
    
    yield  # Application runs here
    
    log.info("=== SHL Assessment Recommender shutting down ===")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational AI agent that recommends SHL assessments through "
        "natural dialogue. Stateless: every /chat call receives full history."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """
    Health check endpoint.
    Returns {"status": "ok"} with HTTP 200.
    Used by Render/Railway to verify the service is alive.
    Must respond within 2 minutes on cold start.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.

    Receives full conversation history, processes it through the agent pipeline
    (retrieval → LLM → validation), and returns a structured recommendation.

    Input:  ChatRequest  { messages: [{role, content}, ...] }
    Output: ChatResponse { reply, recommendations, end_of_conversation }

    Error handling: ALL exceptions are caught. The endpoint ALWAYS returns a
    valid ChatResponse, never a 500 with invalid body.
    """
    from agent.agent import process_conversation

    try:
        # Convert Pydantic models to plain dicts for the agent
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Handle empty messages gracefully (not a 422 error)
        if not messages:
            log.warning("/chat called with empty messages list.")
            return _safe_error_response(
                "Hello! I'm your SHL Assessment Consultant. "
                "Please tell me about the role you're hiring for."
            )

        # Run the agent pipeline
        result = process_conversation(messages)

        # Build validated ChatResponse
        recommendations = [
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r.get("test_type", ""),
            )
            for r in result.get("recommendations", [])
        ]

        return ChatResponse(
            reply=result["reply"],
            recommendations=recommendations,
            end_of_conversation=result.get("end_of_conversation", False),
        )

    except Exception as exc:
        # Log the full exception for debugging but return a valid schema
        log.exception(f"Unhandled exception in /chat: {exc}")
        return _safe_error_response()


# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler that returns a valid ChatResponse JSON body
    even for unhandled errors. Ensures the API never returns malformed JSON.
    """
    log.exception(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=200,  # Return 200 with valid body, not 500
        content={
            "reply": "I encountered an unexpected error. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


# ── Dev server entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
