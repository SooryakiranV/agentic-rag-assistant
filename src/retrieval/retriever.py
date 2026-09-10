from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore


class Retriever:
    def __init__(self, chunks: list[dict], vector_store: FAISSVectorStore):
        self.chunks = chunks
        self.vector_store = vector_store
        self.embedder = Embedder()

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.
        """
        query_embedding = self.embedder.embed([query])

        distances, indices = self.vector_store.search(
            query_embedding[0],
            k=k,
        )

        results = []

        for distance, index in zip(distances, indices):
            if index == -1:
                continue

            result = self.chunks[index].copy()
            result["distance"] = float(distance)

            results.append(result)

        return results