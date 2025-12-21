import os
import sys
import numpy as np

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Ingestion
from ingestion.loader import load_markdown_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import HuggingFaceEmbedder

# Retrieval
from retrieval.vector_store import FaissVectorStore
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker

# Generation
from generation.llm import Generator


INDEX_PATH = "artifacts/faiss"


def build_and_save_index(data_dir: str) -> FaissVectorStore:
    print("🔹 Building FAISS index (first run)...")

    docs = load_markdown_documents(data_dir)
    chunks = chunk_documents(docs)

    embedder = HuggingFaceEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    dim = len(embedded_chunks[0]["embedding"])
    store = FaissVectorStore(dim)

    embeddings = np.array(
        [c["embedding"] for c in embedded_chunks],
        dtype="float32"
    )

    metadata = [
        {
            **c["metadata"],
            "text": c["text"]
        }
        for c in embedded_chunks
    ]

    store.add(embeddings, metadata)
    store.save(INDEX_PATH)

    print(f"✅ FAISS index saved with {len(metadata)} chunks")
    return store


def load_index() -> FaissVectorStore:
    print("🔹 Loading FAISS index from disk...")
    store = FaissVectorStore(384)
    store.load(INDEX_PATH)
    return store


def main():
    os.makedirs("artifacts", exist_ok=True)

    # 1️⃣ Load or build index
    if os.path.exists(f"{INDEX_PATH}.index"):
        vector_store = load_index()
    else:
        vector_store = build_and_save_index("data")

    # 2️⃣ Initialize pipeline
    retriever = Retriever(vector_store)
    reranker = Reranker()
    generator = Generator()

    # 3️⃣ Query
    question = "What benefits does the company offer to employees?"

    print("\n❓ QUESTION:")
    print(question)

    retrieved = retriever.retrieve(question, top_k=10)
    reranked = reranker.rerank(question, retrieved, top_n=4)
    answer = generator.generate_answer(question, reranked)

    print("\n🤖 ANSWER:")
    print(answer)


if __name__ == "__main__":
    main()
