"""
NovaMind FastAPI Backend
========================
A self-hosted AI backend powered by the custom-trained NovaMind LLM.
This completely replaces the Groq API — zero internet, zero API keys.

Endpoints:
    GET  /           → Landing page with NovaMind info
    GET  /health     → Health check + model stats
    POST /chat       → Send a message, get a response
    GET  /stream?message=...  → Stream tokens via Server-Sent Events
    POST /reset      → Clear conversation history

Run:
    cd novamind
    pip install fastapi uvicorn
    python main.py
"""

import sys
import os
import json
import time
from pathlib import Path

# Ensure the novamind package is importable
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional

from inference.chat import NovaChatEngine

# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════

MODEL_PATH = str(Path(__file__).parent / "weights" / "final_model")
TOKENIZER_PATH = str(Path(__file__).parent / "weights" / "tokenizer")

# ═══════════════════════════════════════════════════════════
#  Initialize the NovaMind Engine
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  🧠 NovaMind — Starting Up...")
print("=" * 60)

engine = NovaChatEngine(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    device="auto",           # Will use GPU if available, else CPU
    max_response_tokens=300,
)

print("=" * 60)
print("  ✅ NovaMind is LIVE — No API keys. No internet. All YOU.")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="NovaMind API",
    description="Self-hosted AI backend powered by a custom-trained transformer LLM.",
    version="1.0.0",
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
#  Request / Response Models
# ═══════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    time_ms: float

# ═══════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Beautiful landing page showing NovaMind is running."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NovaMind — Live</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', sans-serif;
                background: #0a0a0f;
                color: #e0e0e0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                overflow: hidden;
            }
            .container {
                text-align: center;
                padding: 3rem;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 24px;
                backdrop-filter: blur(20px);
                max-width: 520px;
                animation: fadeIn 1s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .logo {
                font-size: 3.5rem;
                margin-bottom: 0.5rem;
            }
            h1 {
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #7c3aed, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                color: #888;
                font-size: 0.95rem;
                margin-bottom: 2rem;
            }
            .status {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(34,197,94,0.1);
                border: 1px solid rgba(34,197,94,0.3);
                padding: 8px 20px;
                border-radius: 100px;
                font-size: 0.85rem;
                color: #22c55e;
                margin-bottom: 2rem;
            }
            .dot {
                width: 8px; height: 8px;
                background: #22c55e;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
            .endpoints {
                text-align: left;
                background: rgba(0,0,0,0.3);
                border-radius: 12px;
                padding: 1.2rem;
                font-size: 0.82rem;
                font-family: 'Courier New', monospace;
                line-height: 1.8;
            }
            .method {
                color: #22c55e;
                font-weight: 700;
            }
            .path { color: #7c3aed; }
            .desc { color: #666; }
            .bg-orb {
                position: fixed;
                border-radius: 50%;
                filter: blur(120px);
                opacity: 0.15;
                z-index: -1;
            }
            .orb1 { width: 400px; height: 400px; background: #7c3aed; top: -100px; left: -100px; }
            .orb2 { width: 350px; height: 350px; background: #06b6d4; bottom: -80px; right: -80px; }
        </style>
    </head>
    <body>
        <div class="bg-orb orb1"></div>
        <div class="bg-orb orb2"></div>
        <div class="container">
            <div class="logo">🧠</div>
            <h1>NovaMind</h1>
            <p class="subtitle">Custom-Trained Transformer LLM — Running Locally</p>
            <div class="status"><div class="dot"></div> Online</div>
            <div class="endpoints">
                <span class="method">POST</span> <span class="path">/chat</span> <span class="desc">— Send a message</span><br>
                <span class="method">GET&nbsp;</span> <span class="path">/stream</span> <span class="desc">— Stream tokens (SSE)</span><br>
                <span class="method">GET&nbsp;</span> <span class="path">/health</span> <span class="desc">— Model stats</span><br>
                <span class="method">POST</span> <span class="path">/reset</span> <span class="desc">— Clear history</span><br>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Returns model info and system status."""
    config = engine.model.config
    params = engine.model.count_parameters()
    return {
        "status": "online",
        "model": "NovaMind",
        "device": config.device,
        "parameters": f"{params['total_million']}M",
        "vocab_size": config.vocab_size,
        "embed_dim": config.embed_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "context_length": config.context_length,
        "history_length": len(engine.history),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a complete response.
    
    Body:
        {
            "message": "What is a black hole?",
            "history": []  // optional
        }
    """
    try:
        start = time.perf_counter()
        response = engine.chat(request.message, request.history)
        elapsed = (time.perf_counter() - start) * 1000

        return ChatResponse(
            response=response,
            tokens_generated=len(response.split()),
            time_ms=round(elapsed, 1),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream")
async def stream_chat(message: str):
    """
    Stream tokens as Server-Sent Events.
    
    Usage: GET /stream?message=Hello
    """
    async def generate():
        for token in engine.stream_chat(message):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/reset")
async def reset_history():
    """Clear the conversation history."""
    engine.reset_history()
    return {"status": "history_cleared"}


# ═══════════════════════════════════════════════════════════
#  Run the server
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting NovaMind server at http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
