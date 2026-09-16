# Agentic Document Assistant with RAG and Tool Calling

An agentic document and data assistant that combines **Retrieval-Augmented Generation (RAG)**, **semantic search**, **LLM tool calling**, **web search**, **CSV data analysis**, and **conversation memory**.

The system moves beyond a simple chatbot by allowing an LLM agent to select and use specialised tools depending on the user's request. For document-based questions, the system retrieves relevant information from an uploaded PDF before generating an answer. For structured data, it can perform analysis using Pandas, while current-information queries can be handled through web search.

---

## Overview

Large Language Models can generate fluent responses but may produce unsupported or inaccurate information when answering questions about private or domain-specific documents.

This project implements a **Retrieval-Augmented Generation pipeline** that grounds document-based responses in retrieved content rather than relying solely on the model's internal knowledge.

The system extends the RAG pipeline with an **agentic layer** that can select between multiple tools:

- Document search for uploaded PDFs
- CSV data analysis
- Web search for current information
- Conversation memory for multi-turn interactions

The application is exposed through a Streamlit interface and is supported by automated tests and GitHub Actions CI.

---

## Key Features

### 📄 PDF Document Ingestion

- Page-level text extraction using **PyMuPDF**
- Page metadata preservation
- Support for uploaded PDF documents

### ✂️ Recursive Document Chunking

- LangChain `RecursiveCharacterTextSplitter`
- 512-character chunks
- 50-character overlap

### 🧠 Local Semantic Embeddings

- `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- Runs locally without an external embedding API

### 🔎 FAISS Vector Search

- `IndexFlatL2`
- Top-K semantic retrieval
- Configurable retrieval depth

### 🤖 RAG Generation

- Retrieves relevant document context before generation
- Grounds document-based responses in retrieved evidence
- Avoids exposing internal retrieval metadata to users

### 🛠️ Agentic Tool Calling

The agent can select tools based on the user's request:

- Document search
- CSV data analysis
- Web search

### 🌐 Web Search

- DuckDuckGo-powered search
- Used for current or time-sensitive information

### 📊 CSV Data Analysis

Pandas-based analysis supporting:

- Dataset overview
- Descriptive statistics
- Missing-value analysis
- Correlation analysis
- Duplicate detection

### 💬 Conversation Memory

- Maintains user and assistant messages
- Supports multi-turn interactions
- Allows follow-up questions to reference previous conversation context

### 🖥️ Streamlit Interface

- PDF and CSV upload
- Interactive chat interface
- Tool-aware responses
- Document source display
- Web search source display

### 🧪 Automated Testing

Pytest coverage for:

- PDF ingestion
- Chunking
- Embeddings
- Vector store
- Retrieval
- Agent tools
- Data analysis
- Web search
- Conversation memory
- RAG pipeline

### ⚙️ Continuous Integration

GitHub Actions automatically runs the test suite on pushes and pull requests targeting `main`.

### 📈 RAG Evaluation

- Fixed 20-question evaluation set
- Direct LLM baseline comparison
- RAG comparison
- Expected-fact coverage metric

---

# System Architecture

```text
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Agentic Layer      │
                         │  LangChain/LangGraph │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
             ┌────────────┐  ┌────────────┐  ┌────────────┐
             │  Document  │  │    CSV     │  │    Web     │
             │   Search   │  │  Analysis  │  │   Search   │
             └─────┬──────┘  └────────────┘  └────────────┘
                   │
                   ▼
          ┌──────────────────┐
          │     Retriever    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  FAISS Vector    │
          │      Store       │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Local Embeddings │
          │  MiniLM-L6-v2    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Document Chunks │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │    PDF Loader    │
          │     PyMuPDF      │
          └──────────────────┘

          ┌──────────────────────┐
          │ Conversation Memory  │
          └──────────────────────┘

          ┌──────────────────────┐
          │      Groq LLM        │
          │ openai/gpt-oss-120b  │
          └──────────────────────┘
```

---

# RAG Pipeline

The document workflow follows:

```text
PDF
 │
 ▼
PyMuPDF
 │
 ▼
Page-level text
 │
 ▼
RecursiveCharacterTextSplitter
 │
 ▼
512-character chunks
 │
 ▼
Sentence Transformer
 │
 ▼
384-dimensional embeddings
 │
 ▼
FAISS IndexFlatL2
 │
 ▼
Top-K retrieval
 │
 ▼
Relevant document context
 │
 ▼
Groq LLM
 │
 ▼
Grounded response
```

Each retrieved chunk retains its page metadata, allowing the application to identify the document page associated with the retrieved information.

---

# Agentic Workflow

The application extends the basic RAG pipeline with tool calling.

For each user request, the agent determines whether one of its available tools is appropriate.

## 1. Document Search

The document search tool searches the uploaded PDF using semantic retrieval.

```text
User question
      │
      ▼
Query embedding
      │
      ▼
FAISS similarity search
      │
      ▼
Top-K relevant chunks
      │
      ▼
Agent
```

The tool returns relevant document text and page information to the agent.

---

## 2. CSV Data Analysis

The data analysis tool uses Pandas to analyse an uploaded CSV dataset.

### Supported operations

```text
overview
statistics
missing_values
correlations
duplicates
```

### Workflow

```text
CSV
 │
 ▼
Pandas
 │
 ├── Dataset overview
 ├── Descriptive statistics
 ├── Missing values
 ├── Correlations
 └── Duplicate rows
```

---

## 3. Web Search

For queries requiring current information, the agent can use DuckDuckGo web search.

```text
User question
      │
      ▼
Agent
      │
      ▼
Web Search Tool
      │
      ▼
Search results
      │
      ▼
LLM response
```

---

## 4. Conversation Memory

The application maintains conversation history so that follow-up questions can refer to previous interactions.

Example:

```text
User:
What programming languages are mentioned in my document?

Assistant:
Python, C++, and SQL are mentioned.

User:
What did I just ask you about?

Assistant:
You asked about the programming languages mentioned in the document.
```

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | Groq `openai/gpt-oss-120b` |
| RAG | Custom retrieval pipeline |
| PDF processing | PyMuPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Agent framework | LangChain / LangGraph |
| Data analysis | Pandas |
| Web search | DuckDuckGo |
| UI | Streamlit |
| Testing | Pytest |
| CI | GitHub Actions |
| Environment management | Conda |
| Version control | Git / GitHub |

---

## Project Structure

```text
agentic-rag-assistant/
│
├── app/
│   └── main.py
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   │
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── agents/
│   │   ├── tools.py
│   │   ├── agent.py
│   │   └── memory.py
│   │
│   ├── pipeline.py
│   └── evaluation.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│   ├── test_retriever.py
│   ├── test_agent.py
│   └── test_pipeline.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .gitignore
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/SooryakiranV/agentic-rag-assistant.git
cd agentic-rag-assistant
```

## 2. Create the Conda environment

```bash
conda create -n agentic-rag python=3.11
conda activate agentic-rag
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

The application loads the API key using `python-dotenv`.

**Never commit `.env` or expose your API key publicly.**

The repository `.gitignore` excludes `.env` and local data files.

---

# Running the Application

Start the Streamlit application:

```bash
streamlit run app/main.py
```

The application will open in your browser.

---

# Using the Application

## PDF Workflow

1. Upload a PDF document.
2. The application extracts text page by page.
3. Text is split into overlapping chunks.
4. Chunks are converted into embeddings.
5. Embeddings are stored in a FAISS index.
6. The user's question is converted into a query embedding.
7. The most relevant chunks are retrieved.
8. The agent uses the retrieved context to generate a grounded response.

## CSV Workflow

Upload a CSV file and ask questions that require dataset analysis.

The agent can use Pandas to perform:

- Dataset overview
- Statistical summaries
- Missing-value analysis
- Correlation analysis
- Duplicate detection

## Web Workflow

Questions requiring current information can be handled using the web search tool.

The agent retrieves search results and uses them as context when generating the response.

---

# Evaluation

The project includes a fixed evaluation set of **20 pre-written questions** designed to compare:

1. A direct LLM baseline
2. The RAG pipeline

The evaluation measures **expected-fact coverage**, which calculates the proportion of predefined expected facts found in the generated response.

## Results

| System | Average Expected-Fact Coverage |
|---|---:|
| Direct LLM | 16.83% |
| RAG | 68.67% |

On this 20-question document-grounding evaluation, the RAG pipeline achieved **68.67% average expected-fact coverage**, compared with **16.83% for the direct LLM baseline**.

This metric is intentionally described as **expected-fact coverage rather than accuracy**, because the evaluation measures whether predefined expected facts appear in generated responses rather than whether every response is completely correct.

---

# Testing

The project uses Pytest for automated testing.

Run the complete test suite:

```bash
pytest
```

The test suite covers:

- PDF loading
- Document chunking
- Embedding generation
- Vector store insertion and search
- Vector store persistence
- Document retrieval
- Document search tool
- CSV data analysis
- Web search tool
- Conversation memory
- RAG pipeline construction and retrieval

### Current test status

```text
12 tests
12 passed
```

---

# Continuous Integration

GitHub Actions automatically runs the test suite for pushes and pull requests targeting `main`.

The CI workflow is:

```text
GitHub Push / Pull Request
          │
          ▼
     Checkout code
          │
          ▼
    Setup Python 3.11
          │
          ▼
   Install dependencies
          │
          ▼
        Pytest
          │
          ▼
      Pass / Fail
```

The CI environment also uses synthetic test fixtures rather than private user documents or local datasets.

---

# Design Decisions

## Local Embeddings

The embedding model runs locally rather than using an external embedding API.

Benefits include:

- No embedding API cost
- Reproducible embedding generation
- Reduced external dependencies during retrieval
- 384-dimensional vectors suitable for the FAISS index

---

## FAISS `IndexFlatL2`

The current vector store uses:

```python
faiss.IndexFlatL2
```

This provides exact L2-distance search and is appropriate for the current document-scale application.

---

## Retrieval Before Generation

The system separates retrieval from generation:

```text
Question
   │
   ▼
Retriever
   │
   ▼
Relevant evidence
   │
   ▼
LLM
   │
   ▼
Answer
```

This separation makes the retrieval component independently testable and allows the RAG pipeline to be evaluated separately from the language model.

---

## Tool-Based Agent Architecture

Rather than sending every request through document retrieval, the agent has access to multiple tools.

Different request types can therefore follow different execution paths:

```text
                         User Query
                              │
                              ▼
                            Agent
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        Document           CSV/Data          Web
         Search            Analysis          Search
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                             LLM
                              │
                              ▼
                           Answer
```

---

# Security and Repository Hygiene

The repository excludes sensitive and generated local files through `.gitignore`, including:

- `.env`
- Local raw datasets
- Processed datasets
- Vector indexes
- Evaluation output files
- Python cache files
- Pytest temporary files

The test suite uses synthetic fixtures so that CI does not depend on private documents or local data.

---

# Future Improvements

Potential future extensions include:

- Persistent vector-store management
- More advanced retrieval strategies
- Hybrid keyword + semantic retrieval
- Reranking of retrieved chunks
- Streaming responses
- Authentication and multi-user sessions
- More comprehensive evaluation datasets
- Additional structured-data analysis capabilities
- Containerised deployment

These are intentionally outside the current implementation.

---

# Author

**Sooryakiran Vinod**

MSc Computing (Artificial Intelligence)

---

## License

This project is intended as a portfolio and educational project.
