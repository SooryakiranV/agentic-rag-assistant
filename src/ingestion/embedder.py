from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
        """
        Convert a list of texts into numerical embeddings.

        Args:
            texts: List of text strings.

        Returns:
            A NumPy array containing the embeddings.
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )