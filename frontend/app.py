import os
import sys
import streamlit as st

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# RAG components
from ingestion.loader import load_markdown_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import HuggingFaceEmbedder
from retrieval.vector_store import FaissVectorStore
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm import Generator

INDEX_PATH = os.path.join(ROOT_DIR, "artifacts", "faiss")


@st.cache_resource
def load_rag_pipeline():
    store = FaissVectorStore(384)

    # ✅ Build index if missing
    if not os.path.exists(f"{INDEX_PATH}.index"):
        st.info("Building vector index for the first time. Please wait...")

        docs = load_markdown_documents(os.path.join(ROOT_DIR, "data"))
        chunks = chunk_documents(docs)

        embedder = HuggingFaceEmbedder()
        embedded_chunks = embedder.embed_chunks(chunks)

        embeddings = [c["embedding"] for c in embedded_chunks]
        metadata = [{**c["metadata"], "text": c["text"]} for c in embedded_chunks]

        import numpy as np
        store.add(np.array(embeddings, dtype="float32"), metadata)
        store.save(INDEX_PATH)
    else:
        store.load(INDEX_PATH)

    retriever = Retriever(store)
    reranker = Reranker()
    generator = Generator()

    return retriever, reranker, generator


retriever, reranker, generator = load_rag_pipeline()

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📘")
st.title("📘 Enterprise Knowledge Assistant")

question = st.text_input(
    "Ask a question about company policies:",
    placeholder="What benefits does the company offer to employees?"
)

top_k = st.slider("Retrieved chunks", 3, 15, 10)

if st.button("🔍 Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            retrieved = retriever.retrieve(question, top_k=top_k)
            reranked = reranker.rerank(question, retrieved, top_n=4)
            answer = generator.generate_answer(question, reranked)

        st.subheader("🤖 Answer")
        st.write(answer)

        st.subheader("📚 Sources")
        for chunk in reranked:
            st.markdown(
                f"- **{chunk['metadata'].get('source_file')}** "
                f"(Section: {chunk['metadata'].get('section')})"
            )
