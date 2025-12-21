import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from evaluation.eval_data import EVAL_QUESTIONS
from evaluation.metrics import precision_at_k, recall_at_k

from retrieval.vector_store import FaissVectorStore
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


INDEX_PATH = "artifacts/faiss"


def run_evaluation(k: int = 5):
    store = FaissVectorStore(384)
    store.load(INDEX_PATH)

    retriever = Retriever(store)
    reranker = Reranker()

    precision_scores = []
    recall_scores = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        relevant_sections = item["relevant_sections"]

        retrieved = retriever.retrieve(question, top_k=k)
        reranked = reranker.rerank(question, retrieved, top_n=k)

        retrieved_sections = [
            r["metadata"].get("section") for r in reranked
        ]

        p = precision_at_k(retrieved_sections, relevant_sections, k)
        r = recall_at_k(retrieved_sections, relevant_sections, k)

        precision_scores.append(p)
        recall_scores.append(r)

        print(f"\nQ: {question}")
        print(f"Retrieved sections: {retrieved_sections}")
        print(f"Precision@{k}: {p:.2f}, Recall@{k}: {r:.2f}")

    print("\n📊 AVERAGE METRICS")
    print(f"Avg Precision@{k}: {sum(precision_scores)/len(precision_scores):.2f}")
    print(f"Avg Recall@{k}: {sum(recall_scores)/len(recall_scores):.2f}")


if __name__ == "__main__":
    run_evaluation(k=5)
