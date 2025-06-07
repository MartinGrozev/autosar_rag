# rag_server.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag_query import ask_rag

app = FastAPI(
    title="AUTOSAR RAG Assistant",
    description="Local Retrieval-Augmented Generation (RAG) Assistant for AUTOSAR Official Documentation Q&A",
    version="1.0.0"
)

# 🔥 Allow frontend access (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    source: Optional[str] = None
    module: Optional[str] = None
    keyword: Optional[str] = None

@app.post("/ask", response_model=dict)
async def ask_rag_endpoint(body: QueryRequest):
    """
    Ask the RAG engine a question, with optional filtering.
    """
    print(f"💬 Asking: {body.query}")
    filters = {}
    if body.source:
        filters["source"] = body.source
    if body.module:
        filters["module"] = body.module
    if body.keyword:
        filters["keyword"] = body.keyword

    answer = ask_rag(body.query, filter_by=filters if filters else None)
    print(f"🟰 RAG Answer: {answer}")
    return {"answer": answer}
