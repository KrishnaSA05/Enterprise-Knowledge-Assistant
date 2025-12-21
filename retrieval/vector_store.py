import faiss
import pickle
from typing import List, Dict


class FaissVectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # cosine similarity
        self.metadata = []

    def add(self, embeddings: List[List[float]], metadatas: List[Dict]):
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding, top_k: int = 5):
        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            self.metadata = pickle.load(f)
