import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.pipeline import RAGPipeline


uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)


@st.cache_resource
def build_pipeline(pdf_bytes):
    import tempfile

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:
        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)

    embedder = Embedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed(texts)

    vector_store = FAISSVectorStore()
    vector_store.add_embeddings(embeddings)

    retriever = Retriever(chunks, vector_store)

    return RAGPipeline(retriever)


st.set_page_config(
    page_title="Agentic Document Assistant",
    page_icon="📄",
)

st.title("📄 Agentic Document Assistant")

st.write("Ask questions about your uploaded documents.")

query = st.text_input("Enter your question:")

if uploaded_file and query:
    pipeline = build_pipeline(uploaded_file.getvalue())

    with st.spinner("Thinking..."):
        results = pipeline.retrieve_context(query)

answer = pipeline.generate_answer(query)

st.write("### Answer")
st.write(answer)

st.write("### Sources")

for result in results:
    st.write(
        f"**Page {result['page_number']}** "
        f"(distance: {result['distance']:.2f})"
    )
    st.write(result["text"])
    st.divider()