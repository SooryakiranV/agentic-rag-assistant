import json
import os
from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.pipeline import RAGPipeline

import time

from groq import RateLimitError

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

QUESTIONS_PATH = "data/evaluation_questions.json"
MODEL_NAME = "openai/gpt-oss-120b"


def load_evaluation_questions(
    file_path: str = QUESTIONS_PATH,
) -> list[dict]:
    """
    Load evaluation questions from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_llm_client() -> Groq:
    """
    Create the Groq client used for evaluation.
    """
    return Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )


def generate_direct_answer(
    question: str,
    client: Groq,
    max_retries: int = 3,
) -> str:
    """
    Generate an answer directly from the LLM without document retrieval.

    Retries automatically if the Groq API rate limit is reached.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                temperature=0,
            )

            return response.choices[0].message.content

        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_time = 5 * (attempt + 1)

            print(
                f"Rate limit reached. "
                f"Waiting {wait_time} seconds before retry..."
            )

            time.sleep(wait_time)

    raise RuntimeError("Failed to generate answer.")

    return response.choices[0].message.content

def evaluate_direct_llm(
    questions: list[dict],
    client: Groq,
) -> list[dict]:
    """
    Evaluate the direct LLM baseline on all evaluation questions.
    """
    results = []

    for question in questions:
        answer = generate_direct_answer(
            question["question"],
            client,
        )

        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "direct_llm_answer": answer,
            }
        )

    return results


import csv


def save_evaluation_results(
    results: list[dict],
    file_path: str,
) -> None:
    """
    Save evaluation results to a CSV file.
    """
    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "direct_llm_answer",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

def build_rag_pipeline(pdf_path: str) -> RAGPipeline:
    """
    Build a RAG pipeline from a PDF document.
    """
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)

    embedder = Embedder()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed(texts)

    vector_store = FAISSVectorStore()
    vector_store.add_embeddings(embeddings)

    retriever = Retriever(
        chunks,
        vector_store,
    )

    return RAGPipeline(retriever)

def generate_rag_answer(
    question: str,
    pipeline: RAGPipeline,
    max_retries: int = 3,
) -> str:
    """
    Generate a RAG answer with retry handling for Groq rate limits.
    """
    for attempt in range(max_retries):
        try:
            return pipeline.generate_answer(question)

        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_time = 5 * (attempt + 1)

            print(
                f"RAG rate limit reached. "
                f"Waiting {wait_time} seconds before retry..."
            )

            time.sleep(wait_time)

    raise RuntimeError("Failed to generate RAG answer.")

def evaluate_rag(
    questions: list[dict],
    pipeline: RAGPipeline,
) -> list[dict]:
    """
    Evaluate the RAG pipeline on all evaluation questions.
    """
    results = []

    for question in questions:
        answer = generate_rag_answer(
            question["question"],
            pipeline,
        )

        time.sleep(2)

        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "rag_answer": answer,
            }
        )

    return results

def save_rag_results(
    results: list[dict],
    file_path: str,
) -> None:
    """
    Save RAG evaluation results to a CSV file.
    """
    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "rag_answer",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

def combine_evaluation_results(
    direct_results_path: str,
    rag_results_path: str,
    output_path: str,
) -> None:
    """
    Combine direct LLM and RAG evaluation results.
    """
    import pandas as pd

    direct_df = pd.read_csv(direct_results_path)
    rag_df = pd.read_csv(rag_results_path)

    combined_df = direct_df.merge(
        rag_df,
        on=["id", "question"],
        how="inner",
    )

    combined_df.to_csv(
        output_path,
        index=False,
    )

def calculate_keyword_coverage(
    answer: str,
    expected: list[str],
) -> float:
    """
    Calculate the proportion of expected facts found in an answer.

    Matching is based on normalized token sets, allowing equivalent
    wording and different word order.
    """
    if not expected:
        return 0.0

    import re

    def normalize(text: str) -> set[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return set(text.split())

    answer_tokens = normalize(answer)

    matched = 0

    for keyword in expected:
        keyword_tokens = normalize(keyword)

        if keyword_tokens.issubset(answer_tokens):
            matched += 1

    return matched / len(expected)

def score_evaluation_results(
    results_path: str,
    expected_path: str,
) -> list[dict]:
    """
    Score direct LLM and RAG answers against expected facts.
    """
    import pandas as pd

    results_df = pd.read_csv(results_path)

    with open(
        expected_path,
        "r",
        encoding="utf-8",
    ) as file:
        expected_data = json.load(file)

    expected_by_id = {
        item["id"]: item["expected"]
        for item in expected_data
    }

    scored_results = []

    for _, row in results_df.iterrows():
        expected = expected_by_id[row["id"]]

        direct_score = calculate_keyword_coverage(
            row["direct_llm_answer"],
            expected,
        )

        rag_score = calculate_keyword_coverage(
            row["rag_answer"],
            expected,
        )

        scored_results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "direct_llm_answer": row["direct_llm_answer"],
                "rag_answer": row["rag_answer"],
                "direct_llm_score": direct_score,
                "rag_score": rag_score,
            }
        )

    return scored_results

if __name__ == "__main__":
    questions = load_evaluation_questions()

    rag_pipeline = build_rag_pipeline(
        "data/raw/test_document.pdf"
    )

    rag_results = evaluate_rag(
        questions,
        rag_pipeline,
    )

    save_rag_results(
        rag_results,
        "data/rag_results.csv",
    )

    print(f"Saved {len(rag_results)} RAG evaluation results.")