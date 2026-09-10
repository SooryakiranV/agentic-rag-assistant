import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the FAISS index.
        """
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 3):
        """
        Search the index for the k most similar vectors.

        Returns:
            distances, indices
        """
        query_embedding = np.asarray(
            query_embedding, dtype="float32"
        ).reshape(1, -1)

        k = min(k, self.size)

        distances, indices = self.index.search(query_embedding, k)

        return distances[0], indices[0]

    def save(self, file_path: str) -> None:
        """
        Save the FAISS index to disk.
        """
        faiss.write_index(self.index, file_path)

    @classmethod
    def load(cls, file_path: str):
        """
        Load a FAISS index from disk.
        """
        index = faiss.read_index(file_path)

        store = cls(index.d)
        store.index = index

        return store

    @property
    def size(self) -> int:
        """
        Return the number of vectors currently stored.
        """
        return self.index.ntotal