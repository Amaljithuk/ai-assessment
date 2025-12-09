# Agentic RAG Document QA Application

This repository contains a full-stack AI application developed for the AI Engineering Assessment. It implements a **Retrieval-Augmented Generation (RAG)** pipeline using **FastAPI**, **Docker**, **Weaviate**, and **Ollama** (Local LLMs).

## 🚀 Features
*   **Document Ingestion:** Uploads PDFs, splits text into chunks, and stores vector embeddings.
*   **Vector Database:** Uses **Weaviate** (running in Docker) for semantic search.
*   **RAG Pipeline:** Retrieves context and generates accurate answers using **Llama 3**.
*   **HyDe (Hypothetical Document Embeddings):** Enhanced retrieval using hallucinated hypothetical answers to improve search relevance.
*   **Evaluation:** Integrated **Ragas** framework to measure Context Precision and Answer Relevancy.
*   **Containerization:** Fully Dockerized application.

---

## 🛠️ Prerequisites

Before running the application, ensure you have the following installed:
1.  **Docker Desktop** (or Docker Engine + Compose plugin).
2.  **Ollama** (for local LLM inference).

---

## 📥 Setup & Installation

### 1. Setup Ollama (Local LLM)
Since the application runs in Docker but uses the host's Ollama service, you must configure Ollama to listen to all addresses.

1.  **Pull the required models:**
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ```

2.  **Start Ollama in Listening Mode:**
    *   *Linux/Mac:*
        ```bash
        OLLAMA_HOST=0.0.0.0 ollama serve
        ```
    *   *Windows (PowerShell):*
        ```powershell
        $env:OLLAMA_HOST="0.0.0.0"; ollama serve
        ```
    *(Keep this terminal window open).*

### 2. Start the Application (Docker)
Open a new terminal in the project root folder.

1.  **Build and Start Containers:**
    ```bash
    docker compose up -d --build
    ```

2.  **Verify Status:**
    ```bash
    docker compose logs -f ai-app
    ```
    Wait until you see: `Application startup complete.`

---

## 🖥️ Usage

The API is accessible via the automatic Swagger UI documentation.

**Open your browser to:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 1. Upload a Document
*   Endpoint: `POST /upload`
*   Action: Select a PDF file from your computer.
*   Result: The file is saved to `data/` and ingested into Weaviate.

### 2. Standard Chat (RAG)
*   Endpoint: `POST /chat`
*   Payload:
    ```json
    { "question": "What is the summary of this document?" }
    ```

### 3. Enhanced Chat (HyDe)
*   Endpoint: `POST /hyde-chat`
*   Description: Uses Hypothetical Document Embeddings for better retrieval performance on complex queries.

---

## 📊 Evaluation (Task 2)

The project includes an evaluation script using the **Ragas** framework. It evaluates the pipeline using **Llama 3** as the judge.

To run the evaluation:
```bash
docker compose exec ai-app python -m app.evaluation

Note: The evaluation is configured to run sequentially to prevent overloading the local CPU. It may take a few minutes.

Results: The final report will be saved to data/evaluation_report.csv.

📂 Project Structure
ai-assessment/
├── app/
│   ├── main.py          # FastAPI Server & Endpoints
│   ├── ingestion.py     # Document Loader & Weaviate Uploader
│   ├── rag_chain.py     # Standard RAG Logic (LangChain)
│   ├── hyde_chain.py    # HyDe Enhanced Logic (Task 3)
│   └── evaluation.py    # Ragas Evaluation Script (Task 2)
├── data/                # Storage for PDFs and Evaluation Reports
├── docker-compose.yml   # Orchestration for App + Weaviate
├── Dockerfile           # Python Environment Definition
├── requirements.txt     # Python Dependencies
└── README.md            # Documentation

🏗️ Architecture

Frontend: Swagger UI / HTTP Client.

Backend: FastAPI (Python 3.10).

Database: Weaviate (Vector Store).

Inference: Ollama (Host Machine) communicating via Docker Network Gateway.
