from typing import List, Dict
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Attach embeddings to each chunk.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append({
                "embedding": embedding,
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })

        return embedded_chunks


if __name__ == "__main__":
    from loader import load_markdown_documents
    from chunker import chunk_documents

    docs = load_markdown_documents("data")
    chunks = chunk_documents(docs)

    embedder = HuggingFaceEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    print(f"Embedded {len(embedded_chunks)} chunks")
    print(f"Embedding dimension: {len(embedded_chunks[0]['embedding'])}")
