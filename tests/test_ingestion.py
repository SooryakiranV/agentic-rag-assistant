from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder


PDF_PATH = "data/raw/test_document.pdf"


def test_pdf_loader():
    pages = load_pdf(PDF_PATH)

    assert len(pages) > 0
    assert all("page_number" in page for page in pages)
    assert all("text" in page for page in pages)
    assert all(page["text"] for page in pages)


def test_chunker():
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    assert len(chunks) > 0
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all("page_number" in chunk for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all(chunk["text"] for chunk in chunks)


def test_embedder():
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    texts = [chunk["text"] for chunk in chunks]

    embedder = Embedder()
    embeddings = embedder.embed(texts)

    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384