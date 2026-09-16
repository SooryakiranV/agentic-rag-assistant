from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.pipeline import RAGPipeline
from unittest.mock import patch


PDF_PATH = "tests/fixtures/test_document.pdf"


def test_rag_pipeline():
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    embedder = Embedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed(texts)

    vector_store = FAISSVectorStore()
    vector_store.add_embeddings(embeddings)

    retriever = Retriever(chunks, vector_store)
    with patch("src.pipeline.Groq"):
        pipeline = RAGPipeline(retriever)

    results = pipeline.retrieve_context(
        "What programming languages are mentioned?",
        k=3,
    )

    assert len(results) == 3
    assert all("text" in result for result in results)
    assert all("page_number" in result for result in results)