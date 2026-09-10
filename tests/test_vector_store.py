import numpy as np

from src.retrieval.vector_store import FAISSVectorStore


def test_vector_store_add_and_search():
    embeddings = np.random.rand(5, 384).astype("float32")

    store = FAISSVectorStore()
    store.add_embeddings(embeddings)

    assert store.size == 5

    distances, indices = store.search(embeddings[0], k=3)

    assert len(distances) == 3
    assert len(indices) == 3
    assert indices[0] == 0


def test_vector_store_save_and_load(tmp_path):
    embeddings = np.random.rand(5, 384).astype("float32")

    store = FAISSVectorStore()
    store.add_embeddings(embeddings)

    index_path = tmp_path / "test.index"
    store.save(str(index_path))

    loaded_store = FAISSVectorStore.load(str(index_path))

    assert loaded_store.size == 5