import os
import sys
import streamlit as st

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# RAG components
from retrieval.vector_store import FaissVectorStore
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm import Generator


INDEX_PATH = "artifacts/faiss"

# ----------------------------
# Load RAG components (once)
# ----------------------------
@st.cache_resource
def load_rag_pipeline():
    store = FaissVectorStore(384)
    store.load(INDEX_PATH)

    retriever = Retriever(store)
    reranker = Reranker()
    generator = Generator()

    return retriever, reranker, generator


retriever, reranker, generator = load_rag_pipeline()

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Enterprise Knowledge Assistant")
st.markdown(
    "Ask questions about company policies, benefits, and internal guidelines."
)

question = st.text_input(
    "Enter your question:",
    placeholder="e.g. What benefits does the company offer to employees?"
)

top_k = st.slider("Number of retrieved chunks", 3, 15, 10)

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
