# Electricity RAG Agent

A ChatGPT-style RAG (Retrieval-Augmented Generation) application for electricity knowledge. Built with FastAPI, LangChain, FAISS, and React.

## Features

- **ChatGPT-like UI** — Session management, conversation history, markdown rendering
- **RAG Pipeline** — FAISS vector store with LangChain for accurate retrieval
- **LLM-Optimized Processing** — Uploaded documents are processed by GPT-4o to create clean, RAG-optimized text before indexing
- **Admin Panel** — Password-protected panel to upload/manage knowledge base documents
- **Multi-format Support** — PDF, DOCX, TXT, XLSX, CSV

## Setup

### 1. Backend

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Edit the `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
ADMIN_PASSWORD=Eran123
```

### 3. Frontend

```bash
cd frontend
npm install
```

### 4. Run (Development)

In two terminals:

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open http://localhost:5173

### 5. Run (Production)

```bash
cd frontend && npm run build && cd ..
uvicorn backend.main:app --port 8000
```

Open http://localhost:8000

## Architecture

```
User uploads file → Raw text extraction → GPT-4o optimization → Saved as .txt
                                                                      ↓
                                                              FAISS vector index
                                                                      ↓
                                                    User asks question → Retrieval → LLM answer
```

## Admin Panel

Navigate to `/admin` and enter the admin password. From there you can:
- Upload documents (PDF, DOCX, TXT, XLSX, CSV)
- View uploaded files and their sizes
- Delete files (automatically rebuilds the vector store)
- Monitor vector store status
