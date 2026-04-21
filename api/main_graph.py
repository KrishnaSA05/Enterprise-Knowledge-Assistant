"""
api/main.py
-----------
FastAPI backend — now powered by the Phase 2 LangGraph workflow.

Changes from v1:
  - Replaced retriever/reranker/generator trio with build_graph()
  - QueryRequest gains: department, top_n, max_retries (all optional)
  - QueryResponse gains: query_type, rewritten_query, retry_count, sources
  - /query now calls run_query() and returns the full agent decision trail
  - /health reports graph readiness

Original endpoints are preserved and backward compatible:
  POST /query  — still accepts {question} with no other fields required
  GET  /health — still returns {"status": "ok"}
"""

import os
import sys
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.vector_store import FaissVectorStore
from graph.workflow import build_graph, run_query

INDEX_PATH = os.path.join(ROOT_DIR, "artifacts", "faiss")


# ── App state — graph compiled once at startup ────────────────────────────────

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index and compile the graph once on startup."""
    global _graph

    if not os.path.exists(f"{INDEX_PATH}.index"):
        raise RuntimeError(
            f"FAISS index not found at {INDEX_PATH}.index — "
            "run `python main_graph.py` or `python main.py` first to build it."
        )

    print("🔹 Loading FAISS index...")
    vector_store = FaissVectorStore(384)
    vector_store.load(INDEX_PATH)

    print("🔗 Compiling LangGraph workflow...")
    _graph = build_graph(vector_store)
    print("✅ Graph ready — API is live")

    yield  # app runs here

    print("🔻 Shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="Agentic RAG-based QA over Enterprise Documents — powered by LangGraph",
    version="2.0.0",
    lifespan=lifespan
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    department: Optional[str] = None   # e.g. "HR", "Engineering"
    top_k: Optional[int] = 10          # FAISS fetch count
    top_n: Optional[int] = 4           # post-rerank count
    max_retries: Optional[int] = 2     # circuit breaker ceiling


class SourceChunk(BaseModel):
    source_file: str
    section: str
    rerank_score: float


class QueryResponse(BaseModel):
    question: str               # original user question
    answer: str                 # final generated answer

    # Agent decision trail — useful for debugging and UI display
    query_type: str             # "retrieve" | "direct"
    rewritten_query: str        # same as question if no rewrite happened
    retrieval_grade: str        # "relevant" | "irrelevant" | "" (direct queries)
    answer_grade: str           # "faithful" | "hallucinated" | "" (direct queries)
    retry_count: int            # number of rewrite+retrieve loops

    sources: List[SourceChunk]  # chunks used to generate the answer


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "graph_ready": _graph is not None,
        "version": "2.0.0"
    }


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    # Run the full agentic pipeline
    result = run_query(
        graph=_graph,
        query=request.question,
        department=request.department,
        top_k=request.top_k,
        top_n=request.top_n,
        max_retries=request.max_retries
    )

    # Build source list from reranked chunks
    sources = [
        SourceChunk(
            source_file=chunk["metadata"].get("source_file", "Unknown"),
            section=chunk["metadata"].get("section", "Unknown"),
            rerank_score=round(chunk.get("rerank_score", 0.0), 4)
        )
        for chunk in result["reranked_chunks"]
    ]

    return QueryResponse(
        question=result["original_query"],
        answer=result["answer"],
        query_type=result["query_type"],
        rewritten_query=result["query"],
        retrieval_grade=result["retrieval_grade"],
        answer_grade=result["answer_grade"],
        retry_count=result["retry_count"],
        sources=sources
    )
