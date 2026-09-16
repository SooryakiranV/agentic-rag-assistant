# Agentic Document Assistant with RAG and Tool Calling

An agentic document and data assistant that combines **Retrieval-Augmented Generation (RAG)**, **semantic search**, **LLM tool calling**, **web search**, **CSV data analysis**, and **conversation memory**.

The system is designed to move beyond a simple chatbot by allowing an LLM agent to select and use specialised tools depending on the user's request. For document-based questions, the system retrieves relevant information from an uploaded PDF before generating an answer. For structured data, it can perform analysis using Pandas, while current-information queries can be handled through web search.

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

- **PDF document ingestion**
  - Page-level text extraction using PyMuPDF
  - Metadata preservation for page-level source tracking

- **Recursive document chunking**
  - LangChain `RecursiveCharacterTextSplitter`
  - 512-character chunks
  - 50-character overlap

- **Local semantic embeddings**
  - `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dimensional embeddings
  - Runs locally without requiring an external embedding API

- **FAISS vector search**
  - `IndexFlatL2`
  - Top-K semantic retrieval
  - Configurable retrieval depth

- **RAG generation**
  - Retrieved document context is supplied to the LLM
  - Answers are constrained to available document evidence

- **Agentic tool calling**
  - The LLM decides when a tool is relevant
  - Tools are exposed through LangChain/LangGraph

- **Web search**
  - DuckDuckGo search for current information

- **CSV data analysis**
  - Pandas-based analysis
  - Dataset overview
  - Descriptive statistics
  - Missing-value analysis
  - Correlation analysis
  - Duplicate detection

- **Conversation memory**
  - Maintains user and assistant messages
  - Enables multi-turn interactions

- **Streamlit UI**
  - PDF and CSV upload
  - Interactive chat
  - Tool-aware responses
  - Source display for retrieved documents and web results

- **Automated testing**
  - Pytest test suite
  - Ingestion, retrieval, vector store, agent tools, memory and pipeline tests

- **Continuous Integration**
  - GitHub Actions
  - Automated dependency installation and test execution on pushes and pull requests

- **RAG evaluation**
  - Fixed 20-question evaluation set
  - Comparison between direct LLM responses and RAG responses
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
          │ FAISS Vector      │
          │ Store             │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Local Embeddings │
          │ MiniLM-L6-v2     │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Document Chunks  │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │    PDF Loader    │
          │     PyMuPDF      │
          └──────────────────┘

                         ┌──────────────────────┐
                         │   Conversation       │
                         │      Memory          │
                         └──────────────────────┘

                         ┌──────────────────────┐
                         │      Groq LLM        │
                         │ openai/gpt-oss-120b  │
                         └──────────────────────┘
RAG Pipeline
The document workflow follows the pipeline:
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
Each retrieved chunk retains its page metadata, allowing the application to identify the document page associated with the retrieved information.
Agentic Workflow
The application extends the basic RAG pipeline with tool calling.
For each user request, the agent determines whether one of its available tools is appropriate.
Available tools
1. Document Search
Searches the uploaded PDF using semantic retrieval.
User question
     │
     ▼
Embedding
     │
     ▼
FAISS similarity search
     │
     ▼
Top-K relevant chunks
     │
     ▼
Agent
The tool returns the relevant document text and page information to the agent.
2. CSV Data Analysis
The data analysis tool uses Pandas to perform operations on an uploaded CSV dataset.
Supported operations:
overview
statistics
missing_values
correlations
duplicates
Example workflow:
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
3. Web Search
For queries requiring current information, the agent can use DuckDuckGo web search.
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
4. Conversation Memory
The application maintains conversation history so that follow-up questions can refer to previous interactions.
Example:
User:
What programming languages are mentioned in my document?

Assistant:
Python, C++, and SQL are mentioned.

User:
What did I just ask you about?

Assistant:
You asked about the programming languages mentioned in the document.
Technology Stack
Component	Technology
Language	Python 3.11
LLM	Groq openai/gpt-oss-120b
RAG	Custom retrieval pipeline
PDF processing	PyMuPDF
Chunking	LangChain RecursiveCharacterTextSplitter
Embeddings	Sentence Transformers
Embedding model	all-MiniLM-L6-v2
Vector database	FAISS
Agent framework	LangChain / LangGraph
Data analysis	Pandas
Web search	DuckDuckGo
UI	Streamlit
Testing	Pytest
CI	GitHub Actions
API dependencies	FastAPI / Uvicorn
Environment management	Conda
Version control	Git / GitHub


Project Structure
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
│   ├── fixtures/
│   ├── test_retriever.py
│   ├── test_agent.py
│   ├── test_pipeline.py
│   ├── test_ingestion.py
│   └── test_vector_store.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
├── Dockerfile
└── README.md
Installation
1. Clone the repository
git clone https://github.com/SooryakiranV/agentic-rag-assistant.git
cd agentic-rag-assistant
2. Create the Conda environment
conda create -n agentic-rag python=3.11
conda activate agentic-rag
3. Install dependencies
pip install -r requirements.txt
Environment Configuration
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key
The API key is loaded through python-dotenv.
Do not commit .env to Git.
The repository .gitignore excludes environment files and local datasets.
Running the Application
Start the Streamlit application with:
streamlit run app/main.py
The application will open in the browser.
Using the Application
PDF workflow
1. Upload a PDF document.
2. The application extracts the text page by page.
3. Text is split into overlapping chunks.
4. Chunks are converted into embeddings.
5. Embeddings are stored in a FAISS index.
6. User questions are converted into query embeddings.
7. The most relevant chunks are retrieved.
8. The agent uses the retrieved context to generate a grounded answer.
CSV workflow
Upload a CSV file and ask questions that require dataset analysis.
The agent can use Pandas to perform:
- Dataset overview
- Statistical summaries
- Missing-value analysis
- Correlation analysis
- Duplicate detection
Web workflow
Questions requiring current information can be handled using the web search tool.
The agent can retrieve search results and use them as context when generating the response.
Evaluation
The project includes a fixed evaluation set of 20 pre-written questions designed to compare responses generated by:
1. A direct LLM baseline
2. The RAG pipeline
The evaluation measures expected-fact coverage, which calculates the proportion of predefined expected facts that are present in the generated response.
Evaluation result
System	Average Expected-Fact Coverage
Direct LLM	16.83%
RAG	68.67%


On this 20-question document-grounding evaluation, the RAG pipeline achieved 68.67% average expected-fact coverage, compared with 16.83% for the direct LLM baseline.
This metric is intentionally described as expected-fact coverage rather than accuracy, because the evaluation measures whether predefined expected facts appear in generated responses rather than whether every response is completely correct.
Testing
The project uses Pytest for automated testing.
Run the full test suite:
pytest
Current test coverage includes:
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
The current test suite contains:
12 tests
12 passed
Continuous Integration
GitHub Actions automatically runs the test suite for pushes and pull requests targeting main.
The CI workflow:
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
This helps ensure that changes to the project do not silently break the existing pipeline.
Design Decisions
Local embeddings
The embedding model runs locally rather than using an external embedding API.
Benefits include:
- No embedding API cost
- Reproducible embedding generation
- Reduced external dependencies during retrieval
- 384-dimensional vectors suitable for the FAISS index
FAISS IndexFlatL2
The current vector store uses:
faiss.IndexFlatL2
This provides a straightforward exact L2-distance search implementation and is appropriate for the current document-scale application.
Retrieval before generation
The system separates retrieval from generation.
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
This separation makes the retrieval component independently testable and allows the RAG pipeline to be evaluated separately from the underlying language model.
Tool-based agent architecture
Rather than sending every request through document retrieval, the agent has access to multiple tools.
This allows different request types to follow different execution paths:
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
Security and Repository Hygiene
The repository excludes sensitive and generated local files through .gitignore, including:
- .env
- Local raw datasets
- Processed datasets
- Vector indexes
- Evaluation output files
- Python cache files
- Pytest temporary files
The test suite uses synthetic fixtures so that CI does not depend on private documents or local data.
Future Improvements
Potential future extensions include:
- Persistent vector-store management
- More advanced retrieval strategies
- Hybrid keyword + semantic retrieval
- Reranking of retrieved chunks
- Streaming responses
- Authentication and multi-user sessions
- More comprehensive evaluation datasets
- Additional structured-data analysis capabilities
- Deployment using containerised infrastructure
These are intentionally outside the current implementation.