from src.retrieval.retriever import Retriever

import pandas as pd
from ddgs import DDGS


def document_search_tool(
    query: str,
    retriever: Retriever,
    k: int = 3,
) -> list[dict]:
    """
    Search the uploaded document for information relevant to a query.
    """
    return retriever.retrieve(query, k=k)


def data_analysis_tool(
    file_path: str,
    operation: str,
) -> str:
    """
    Analyse a CSV dataset using Pandas.

    Supported operations:
    - overview
    - statistics
    - missing_values
    - correlations
    - duplicates
    """

    data = pd.read_csv(file_path)

    if operation == "overview":
        return (
            f"Rows: {data.shape[0]}\n"
            f"Columns: {data.shape[1]}\n"
            f"Columns: {list(data.columns)}"
        )

    if operation == "statistics":
        return data.describe(include="all").to_string()

    if operation == "missing_values":
        return data.isnull().sum().to_string()

    if operation == "correlations":
        numeric_data = data.select_dtypes(include="number")

        if numeric_data.empty:
            return "No numeric columns available for correlation analysis."

        return numeric_data.corr().to_string()

    if operation == "duplicates":
        duplicate_count = data.duplicated().sum()

        return (
            f"Duplicate rows: {duplicate_count}"
        )

    return (
        "Unsupported operation. "
        "Use: overview, statistics, missing_values, "
        "correlations, or duplicates."
    )


def web_search_tool(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search the web using DuckDuckGo.
    """
    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            max_results=max_results,
        )

        return list(results)