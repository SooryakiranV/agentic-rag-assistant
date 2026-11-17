# Agentic RAG Assistant

A Streamlit assistant for asking questions about uploaded PDFs and CSV files with document retrieval, web search, and tool calling.

## Evaluation result

On one run, expected-fact coverage was 68.67% for RAG and 16.83% for the direct LLM baseline. The evaluation used keyword matching on 20 questions about one document: the owner's CV.

This is a single-document, single-run result. It is not a general benchmark or a measure of factual accuracy.

## Features

- Upload a PDF for retrieval-augmented answers.
- Upload a CSV for data analysis.
- Use web search for current information.
- Keep conversation context during a session.

The app uses local sentence embeddings and a FAISS index for PDF retrieval. Groq provides the language model.

## Run locally

Create an environment and install the dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate with `\.venv\Scripts\Activate.ps1`. Add your Groq API key to `.env`, then start the app:

```sh
streamlit run app/main.py
```

The app opens in your browser. Upload a PDF or CSV, then ask a question.

## Tests

Run the test suite with:

```sh
pytest
```

GitHub Actions runs the suite on pushes and pull requests to `main`.

## Topics

rag, document-search, streamlit, faiss, langgraph, groq
