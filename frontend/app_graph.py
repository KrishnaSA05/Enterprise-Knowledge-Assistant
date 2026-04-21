"""
frontend/app.py
---------------
Streamlit UI — now powered by the Phase 2 LangGraph workflow.

Changes from v1:
  - @st.cache_resource loads build_graph() instead of the component trio
  - UI shows agent decision trail: query type, rewrite, grades, retry count
  - Sources section now shows rerank scores
  - Sidebar expander added for advanced settings (top_k, top_n, max_retries)
  - Color-coded badges for query_type and grade results

Original UX is fully preserved — the text input and Get Answer button
work exactly as before with no extra steps required from the user.
"""

import os
import sys
import numpy as np
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ingestion.loader import load_markdown_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import HuggingFaceEmbedder
from retrieval.vector_store import FaissVectorStore
from graph.workflow import build_graph, run_query

INDEX_PATH = os.path.join(ROOT_DIR, "artifacts", "faiss")


# ── Pipeline loader ───────────────────────────────────────────────────────────

@st.cache_resource
def load_graph():
    """
    Loads FAISS index and compiles the LangGraph workflow.
    Cached — runs once per Streamlit session.
    Builds the index automatically if it doesn't exist yet.
    """
    store = FaissVectorStore(384)

    if not os.path.exists(f"{INDEX_PATH}.index"):
        st.info("⚙️ Building vector index for the first time. Please wait...")

        docs   = load_markdown_documents(os.path.join(ROOT_DIR, "data"))
        chunks = chunk_documents(docs)

        embedder        = HuggingFaceEmbedder()
        embedded_chunks = embedder.embed_chunks(chunks)

        embeddings = np.array([c["embedding"] for c in embedded_chunks], dtype="float32")
        metadata   = [{**c["metadata"], "text": c["text"]} for c in embedded_chunks]

        store.add(embeddings, metadata)
        store.save(INDEX_PATH)
    else:
        store.load(INDEX_PATH)

    return build_graph(store)


graph = load_graph()


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Enterprise Knowledge Assistant")
st.caption("Powered by LangGraph · Agentic RAG · Groq LLaMA-3.1-8B")


# ── Sidebar — advanced settings ───────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Retrieved chunks (top_k)",
        min_value=3, max_value=15, value=10,
        help="Number of chunks fetched from FAISS before reranking"
    )

    top_n = st.slider(
        "Reranked chunks (top_n)",
        min_value=1, max_value=top_k, value=4,
        help="Number of chunks kept after cross-encoder reranking"
    )

    max_retries = st.slider(
        "Max retries",
        min_value=0, max_value=3, value=2,
        help="Circuit breaker: max query rewrites + answer retries"
    )

    department = st.text_input(
        "Department filter (optional)",
        placeholder="e.g. HR, Engineering",
        help="Scope retrieval to a specific department"
    )

    st.divider()
    st.caption("v2.0 · LangGraph Phase 2")


# ── Main input ────────────────────────────────────────────────────────────────

question = st.text_input(
    "Ask a question about company policies:",
    placeholder="What benefits does the company offer to employees?"
)

run_button = st.button("🔍 Get Answer", type="primary")


# ── Query execution ───────────────────────────────────────────────────────────

if run_button:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🤖 Thinking..."):
            result = run_query(
                graph=graph,
                query=question,
                department=department.strip() or None,
                top_k=top_k,
                top_n=top_n,
                max_retries=max_retries
            )

        # ── Agent decision trail ──────────────────────────────────────────────

        st.subheader("🔍 Agent Decision Trail")

        col1, col2, col3, col4 = st.columns(4)

        # Query type badge
        qt = result["query_type"]
        col1.metric(
            "Query Type",
            "📄 RAG" if qt == "retrieve" else "💬 Direct"
        )

        # Retrieval grade badge
        rg = result["retrieval_grade"] or "N/A"
        col2.metric(
            "Retrieval Grade",
            "✅ Relevant" if rg == "relevant" else ("❌ Irrelevant" if rg == "irrelevant" else rg)
        )

        # Answer grade badge
        ag = result["answer_grade"] or "N/A"
        col3.metric(
            "Answer Grade",
            "✅ Faithful" if ag == "faithful" else ("⚠️ Hallucinated" if ag == "hallucinated" else ag)
        )

        # Retry count
        col4.metric("Retries", result["retry_count"])

        # Show rewritten query if it changed
        if result["query"] != result["original_query"]:
            st.info(f"✏️ **Query was rewritten to:** {result['query']}")

        st.divider()

        # ── Answer ────────────────────────────────────────────────────────────

        st.subheader("🤖 Answer")
        st.write(result["answer"])

        # ── Sources ───────────────────────────────────────────────────────────

        if result["reranked_chunks"]:
            st.subheader("📚 Sources Used")

            for i, chunk in enumerate(result["reranked_chunks"]):
                source = chunk["metadata"].get("source_file", "Unknown")
                section = chunk["metadata"].get("section", "Unknown")
                score  = chunk.get("rerank_score", 0.0)
                text   = chunk["metadata"].get("text", "")[:300]  # preview

                with st.expander(f"[{i+1}] {source} → {section}  (score: {score:.4f})"):
                    st.caption(text + ("..." if len(chunk["metadata"].get("text","")) > 300 else ""))
        else:
            st.info("📚 No documents retrieved — answered directly from model knowledge.")
