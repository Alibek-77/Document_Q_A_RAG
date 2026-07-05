# 📚 Document Q&A API

A production-ready RAG (Retrieval-Augmented Generation) API that answers questions based on your documents. Built with FastAPI, ChromaDB, and Google Gemini.

## What is RAG?

RAG combines document search with AI generation. Instead of relying on the model's training data, it finds relevant chunks from your documents and uses them as context for generating accurate answers.

```
Your Question → Search Documents → Find Relevant Chunks → Generate Answer
```

## Features

- **4 search methods** — Naive, Rewrite, Multi-Query, Hybrid
- **Semantic search** — finds relevant content even with different wording
- **BM25 + Vector hybrid search** — best of keyword and semantic search
- **Cross-encoder reranking** — reorders results for better relevance
- **Persistent storage** — ChromaDB keeps your indexed documents between restarts
- **Multiple document sources** — index and query multiple documents

## Tech Stack

- **FastAPI** — REST API framework
- **ChromaDB** — vector database for storing embeddings
- **Google Gemini** — LLM for generation + embeddings
- **sentence-transformers** — cross-encoder reranking
- **rank-bm25** — keyword search for hybrid retrieval

## Project Structure

```
document_qa/
├── main.py              # FastAPI app and endpoints
├── config.py            # Gemini client setup
├── rag/
│   ├── indexer.py       # document chunking and indexing
│   ├── retriever.py     # search methods (naive, hybrid, etc.)
│   └── generator.py     # answer generation
├── models/
│   └── schemas.py       # Pydantic schemas
├── chroma_db/           # vector database (auto-created)
├── .env
└── requirements.txt
```

## Setup

**1. Clone and install**
```bash
git clone https://github.com/your-username/document-qa-api.git
cd document-qa-api
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

**2. Create `.env`**
```
GEMINI_API_KEY=your_gemini_api_key
```

Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

**3. Run**
```bash
uvicorn main:app --reload
```

API is available at `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Index a document |
| POST | `/ask` | Ask a question |

## Usage

**1. Upload a document**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a modern web framework for building APIs with Python...",
    "source": "fastapi_docs",
    "metadata": {"category": "documentation"}
  }'
```

**2. Ask a question**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I install FastAPI?",
    "method": "hybrid",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "answer": "FastAPI can be installed using pip: pip install fastapi uvicorn...",
  "method_used": "hybrid",
  "duration_seconds": 1.243
}
```

## Search Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `naive` | Direct vector search + reranking | Simple questions |
| `rewrite` | Rewrites query before searching | Vague or short questions |
| `multi_query` | Generates 3 query variants, merges results | Complex topics |
| `hybrid` | BM25 + vector search combined | Best overall quality |

## How Hybrid Search Works

```
Query
  ├── BM25 (keyword search)    → scores for exact word matches
  └── Vector search            → scores for semantic similarity
         ↓
  Combined score = α × vector + (1-α) × BM25
         ↓
  Cross-encoder reranking
         ↓
  Top-K most relevant chunks
         ↓
  Gemini generates answer
```

## Requirements

```
fastapi
uvicorn
chromadb
google-generativeai
sentence-transformers
rank-bm25
numpy
python-dotenv
pydantic
```

## Interactive Docs

Swagger UI: `http://localhost:8000/docs`