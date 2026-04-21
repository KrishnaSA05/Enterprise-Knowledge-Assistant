"""
main_graph.py
-------------
Entry point for the Phase 2 LangGraph agentic pipeline.

Runs alongside the original main.py:
  python main.py          → original linear pipeline (unchanged)
  python main_graph.py    → full agentic LangGraph pipeline

What Phase 2 adds over Phase 1:
  - Router: skips retrieval for simple/direct queries
  - Retrieval grader: detects poor retrieval before reranking
  - Query rewriter: self-corrects on retrieval failure
  - Answer grader: catches hallucinations before returning
  - Circuit breakers: guarantees the graph always terminates
"""

import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ingestion.loader import load_markdown_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import HuggingFaceEmbedder
from retrieval.vector_store import FaissVectorStore
from graph.workflow import build_graph, run_query

INDEX_PATH = "artifacts/faiss"


def build_and_save_index(data_dir: str) -> FaissVectorStore:
    print("🔹 Building FAISS index...")
    docs   = load_markdown_documents(data_dir)
    chunks = chunk_documents(docs)

    embedder       = HuggingFaceEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    dim  = len(embedded_chunks[0]["embedding"])
    store = FaissVectorStore(dim)

    embeddings = np.array([c["embedding"] for c in embedded_chunks], dtype="float32")
    metadata   = [{**c["metadata"], "text": c["text"]} for c in embedded_chunks]

    store.add(embeddings, metadata)
    store.save(INDEX_PATH)
    print(f"✅ FAISS index saved with {len(metadata)} chunks")
    return store


def load_index() -> FaissVectorStore:
    print("🔹 Loading FAISS index from disk...")
    store = FaissVectorStore(384)
    store.load(INDEX_PATH)
    return store


def _print_result(result: dict):
    """Pretty-prints the final graph state."""
    print("\n" + "═" * 60)
    print(f"❓ ORIGINAL QUERY : {result['original_query']}")
    if result["query"] != result["original_query"]:
        print(f"✏️  REWRITTEN QUERY: {result['query']}")
    print(f"🔀 QUERY TYPE     : {result['query_type']}")
    print(f"📊 RETRIEVAL GRADE: {result['retrieval_grade'] or 'N/A (direct)'}")
    print(f"✅ ANSWER GRADE   : {result['answer_grade'] or 'N/A (direct)'}")
    print(f"🔁 RETRIES        : {result['retry_count']}")
    print("═" * 60)
    print("\n🤖 ANSWER:")
    print(result["answer"])

    if result["reranked_chunks"]:
        print("\n📚 SOURCES USED:")
        for i, chunk in enumerate(result["reranked_chunks"]):
            section = chunk["metadata"].get("section", "?")
            source  = chunk["metadata"].get("source_file", "?")
            score   = chunk.get("rerank_score", 0.0)
            print(f"  [{i+1}] {source} → {section} (score: {score:.4f})")
    else:
        print("\n📚 SOURCES: None (direct answer)")

    print("═" * 60)


def main():
    os.makedirs("artifacts", exist_ok=True)

    # 1️⃣ Load or build index
    if os.path.exists(f"{INDEX_PATH}.index"):
        vector_store = load_index()
    else:
        vector_store = build_and_save_index("data")

    # 2️⃣ Compile graph
    print("\n🔗 Compiling Phase 2 LangGraph workflow...")
    graph = build_graph(vector_store)
    print("✅ Graph compiled\n")

    # 3️⃣ Test queries — exercises all paths in the graph
    test_queries = [
        "What benefits does the company offer to employees?",   # RAG path
        "Hello! What can you help me with?",                    # Direct path
        "xyzzy frobulate quantum synergy policy",               # Bad query → rewrite path
    ]

    for question in test_queries:
        result = run_query(graph, query=question)
        _print_result(result)
        print()


if __name__ == "__main__":
    main()
