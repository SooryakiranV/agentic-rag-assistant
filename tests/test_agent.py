from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.agents.tools import document_search_tool
from src.agents.tools import data_analysis_tool
from src.agents.tools import web_search_tool
from src.agents.memory import ConversationMemory


PDF_PATH = "data/raw/test_document.pdf"
CSV_PATH = "data/raw/test_data.csv"


def test_document_search_tool():
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    embedder = Embedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed(texts)

    vector_store = FAISSVectorStore()
    vector_store.add_embeddings(embeddings)

    retriever = Retriever(chunks, vector_store)

    results = document_search_tool(
        "What programming languages are mentioned?",
        retriever,
    )

    assert len(results) == 3
    assert all("text" in result for result in results)
    assert all("page_number" in result for result in results)


def test_data_analysis_tool():
    result = data_analysis_tool(
        CSV_PATH,
        "overview",
    )

    assert "Rows: 4" in result
    assert "Columns: 3" in result
    assert "name" in result
    assert "age" in result
    assert "salary" in result


def test_web_search_tool():
    results = web_search_tool(
        "Python programming language",
        max_results=3,
    )

    assert len(results) > 0
    assert all("title" in result for result in results)


def test_conversation_memory():
    memory = ConversationMemory()

    memory.add_message(
        "user",
        "What is Python?",
    )

    memory.add_message(
        "assistant",
        "Python is a programming language.",
    )

    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is Python?"
    assert messages[1]["role"] == "assistant"

    memory.clear()

    assert memory.get_messages() == []

def test_data_analysis_duplicates():
    result = data_analysis_tool(
        CSV_PATH,
        "duplicates",
    )

    assert "Duplicate rows: 0" in result