from typing import List, Dict, Optional
import numpy as np

from ingestion.embedder import HuggingFaceEmbedder
from retrieval.vector_store import FaissVectorStore


class Retriever:
    def __init__(self, vector_store: FaissVectorStore):
        self.vector_store = vector_store
        self.embedder = HuggingFaceEmbedder()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        department: Optional[str] = None
    ) -> List[Dict]:
        query_embedding = self.embedder.embed_texts([query])
        query_embedding = np.array(query_embedding).astype("float32")

        results = self.vector_store.search(query_embedding, top_k=top_k)

        if department:
            results = [
                r for r in results
                if r["metadata"].get("department") == department
            ]

        return results
