from typing import List, Dict
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        top_n: int = 3
    ) -> List[Dict]:
        pairs = [
            (query, chunk["metadata"]["section"])
            for chunk in retrieved_chunks
        ]

        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(retrieved_chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in reranked[:top_n]
        ]
