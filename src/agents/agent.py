from dotenv import load_dotenv

import json
import re

from langchain_core.tools import tool

load_dotenv()

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from src.agents.memory import ConversationMemory
from src.agents.tools import (
    data_analysis_tool,
    document_search_tool,
    web_search_tool,
)


def create_agent(retriever, memory, data_file_path=None):

    @tool
    def search_document(query: str) -> str:
        """Search the uploaded document for relevant information."""

        results = document_search_tool(
            query,
            retriever,
        )

        # Return only the information needed by the agent.
        # Internal chunk IDs and retrieval distances are not exposed
        # in the final user-facing response.
        cleaned_results = []

        for result in results:
            cleaned_results.append(
                {
                    "page_number": result["page_number"],
                    "text": result["text"],
                }
            )

        return json.dumps(cleaned_results)

    @tool
    def analyze_data(operation: str) -> str:
        """Analyse the uploaded CSV dataset using Pandas."""

        if not data_file_path:
            return "No CSV dataset is currently uploaded."

        return data_analysis_tool(
            data_file_path,
            operation,
        )

    @tool
    def search_web(query: str) -> str:
        """Search the web for current information."""

        results = web_search_tool(query)

        formatted_results = []

        for result in results:

            cleaned_body = re.sub(
                r"\[\d+\]\[\d+-\d+\]",
                "",
                result["body"],
            )

            formatted_results.append(
                {
                    "title": result["title"],
                    "body": cleaned_body,
                    "href": result["href"],
                }
            )

        return json.dumps(formatted_results)

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    tools = [
        analyze_data,
        search_web,
    ]

    if retriever is not None:
        tools.insert(0, search_document)

    return create_react_agent(
        llm,
        tools,
        prompt=(
            "You are an agentic document and data assistant. "
            "Use the available tools whenever they are relevant. "
            "When answering questions about an uploaded document or dataset, "
            "base factual claims on the information returned by the tools. "
            "Do not invent, assume, or infer unsupported facts. "
            "Do not expose internal tool metadata, chunk IDs, retrieval "
            "distances, or other implementation details to the user. "
            "If the available tool output does not contain enough information "
            "to answer the question, clearly say that the information is not available. "
            "For current or time-sensitive information, use the web search tool."
        ),
    )