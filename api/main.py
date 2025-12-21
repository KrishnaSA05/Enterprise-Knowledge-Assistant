import os
import sys
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Retrieval + Generation
from retrieval.vector_store import FaissVectorStore
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm import Generator

# ----------------------------
# Load FAISS index at startup
# ----------------------------

INDEX_PATH = "artifacts/faiss"

vector_store = FaissVectorStore(384)
vector_store.load(INDEX_PATH)

retriever = Retriever(vector_store)
reranker = Reranker()
generator = Generator()

# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG-based Question Answering over Enterprise Documents",
    version="1.0.0"
)

# ----------------------------
# Request / Response Models
# ----------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10


class QueryResponse(BaseModel):
    question: str
    answer: str


# ----------------------------
# Routes
# ----------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    retrieved = retriever.retrieve(
        request.question,
        top_k=request.top_k
    )

    reranked = reranker.rerank(
        request.question,
        retrieved,
        top_n=4
    )

    answer = generator.generate_answer(
        request.question,
        reranked
    )

    return {
        "question": request.question,
        "answer": answer
    }
