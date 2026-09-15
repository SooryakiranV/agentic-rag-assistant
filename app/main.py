import sys
from pathlib import Path
import hashlib
import json
import re

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.agents.agent import create_agent
from src.agents.memory import ConversationMemory


if "memory" not in st.session_state:
    st.session_state.memory = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None


uploaded_file = st.file_uploader(
    "Upload a PDF or CSV file",
    type=["pdf", "csv"],
)
if uploaded_file:
    st.caption(f"Uploaded: {uploaded_file.name}")

if uploaded_file:
    document_id = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    file_type = uploaded_file.type.split("/")[-1]
else:
    document_id = None
    file_type = None


if document_id != st.session_state.document_id:
    st.session_state.memory = None
    st.session_state.document_id = document_id


@st.cache_resource
def build_agent(file_bytes, file_type):
    import tempfile

    suffix = ".pdf" if file_type == "pdf" else ".csv"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(file_bytes)
        file_path = temp_file.name

    if file_type == "pdf":
        pages = load_pdf(file_path)
        chunks = chunk_pages(pages)
    else:
        chunks = []

    if file_type == "pdf":
        embedder = Embedder()

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed(texts)

        vector_store = FAISSVectorStore()
        vector_store.add_embeddings(embeddings)

        retriever = Retriever(
            chunks,
            vector_store,
        )
    else:
        retriever = None

    memory = ConversationMemory()

    data_file_path = file_path if file_type == "csv" else None

    agent = create_agent(
        retriever,
        memory,
        data_file_path,
    )

    return agent, memory, retriever


st.set_page_config(
    page_title="Agentic Document Assistant",
    page_icon="📄",
)

st.title("📄 Agentic Document Assistant")

st.write(
    "Ask questions about your uploaded documents."
)


query = st.chat_input(
    "Ask a question about your document..."
)


if st.session_state.memory:
    for message in st.session_state.memory.get_messages():
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])


if query:

    if uploaded_file:

        agent, built_memory, retriever = build_agent(
            uploaded_file.getvalue(),
            uploaded_file.type.split("/")[-1],
        )

        if st.session_state.memory is None:
            st.session_state.memory = built_memory

    else:

        retriever = None

        if st.session_state.memory is None:
            st.session_state.memory = ConversationMemory()

        agent = create_agent(
            None,
            st.session_state.memory,
        )

    memory = st.session_state.memory

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):

        response = agent.invoke(
            {
                "messages": (
                    memory.get_messages()
                    + [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                )
            }
        )

    answer = response["messages"][-1].content

    # Remove internal web-search markers
    answer = re.sub(
        r"\[search_web\d*\]",
        "",
        answer,
    ).strip()

    # Check whether the document search tool was actually used
    used_document_search = any(
        getattr(message, "name", None)
        == "search_document"
        for message in response["messages"]
    )

    # Retrieve document sources only when document search was used
    sources = (
        retriever.retrieve(query, k=3)
        if used_document_search and retriever
        else []
    )

    # Check whether the web search tool was actually used
    used_web_search = any(
        getattr(message, "name", None)
        == "search_web"
        for message in response["messages"]
    )

    # Extract the web-search results from the tool message
    web_sources = []

    if used_web_search:

        for message in response["messages"]:

            if getattr(message, "name", None) == "search_web":

                try:
                    web_sources = json.loads(
                        message.content
                    )
                except (json.JSONDecodeError, TypeError):
                    web_sources = []

                break

    # Save conversation history
    memory.add_message(
        "user",
        query,
    )

    memory.add_message(
        "assistant",
        answer,
    )

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(answer)

    # Display document sources only when document search was used
    if sources:

        with st.expander("Document Sources"):

            for source in sources:

                st.write(
                    f"Page {source['page_number']}"
                )

                st.write(
                    source["text"]
                )

    # Display web sources only when web search was used
    if web_sources:

        with st.expander("Web Sources"):

            for source in web_sources:

                title = source.get(
                    "title",
                    "Web Source",
                )

                href = source.get(
                    "href",
                    "",
                )

                if href:
                    st.markdown(
                        f"🔗 [{title}]({href})"
                    )
                else:
                    st.write(title)