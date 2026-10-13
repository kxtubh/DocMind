# DocMind

A full-stack Retrieval-Augmented Generation (RAG) system with a Next.js chat UI, a FastAPI ingestion/retrieval service, and Pinecone as the vector store. Built with LangChain and evaluated across multiple chunking strategies.

## Architecture

```
Next.js (chat UI)  ──►  FastAPI (RAG API)  ──►  Pinecone (vector DB)
                                │
                         fastembed (local)   ← zero-cost ONNX embeddings
```

- **Frontend** — Next.js 15, streaming chat, collapsible source citations, file upload (.txt / .md)
- **API** — FastAPI with `/ingest`, `/retrieve`, `/query`, `/metrics` endpoints
- **Embeddings** — [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) via fastembed (local ONNX, no API cost)
- **Vector store** — Pinecone (384-dim index)
- **LLM** — OpenAI gpt-4o-mini (generation only; retrieval is API-free)
- **Eval** — 20-question benchmark measuring recall@k, MRR, hit rate across 3 chunking strategies

## Retrieval eval results

| Strategy | R@1 | R@3 | R@5 | MRR | Hit Rate |
|---|---|---|---|---|---|
| recursive-500 | 0.90 | 0.95 | 0.95 | 0.917 | 95% |
| **recursive-1000** | **0.95** | **0.95** | **0.95** | **0.950** | **95%** |
| fixed-500 | 0.80 | 1.00 | 1.00 | 0.883 | 100% |

`recursive-1000` has the best MRR. `fixed-500` achieves 100% hit rate at k≥3 but ranks lower at k=1.

## Quickstart

### Prerequisites

- Node.js 18+, Python 3.12+
- [Pinecone](https://pinecone.io) account — create an index: **384 dims, cosine metric**
- OpenAI API key (for LLM generation; embeddings are local)

### 1. Environment

```bash
cp .env.example .env.local
# fill in OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX
```

### 2. Frontend

```bash
npm install
npm run dev
# http://localhost:3000
```

### 3. FastAPI service

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up
```

### 4. Run the eval

```bash
cd eval
pip install -r requirements.txt
python run_eval.py --ingest       # load sample corpus into Pinecone
python run_eval.py --retrieval    # recall@k, MRR, hit rate
python run_eval.py --compare      # compare 3 chunking strategies
```

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/ingest` | Chunk and embed text into Pinecone |
| POST | `/retrieve` | Vector search — no LLM, returns sources + latency |
| POST | `/query` | Full RAG — retrieve + generate answer |
| GET | `/metrics` | p50/p95/mean latency across all queries |

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for LLM generation |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX` | Yes | Index name (384-dim, cosine) |
| `FASTAPI_URL` | No | FastAPI base URL (default: `http://localhost:8000`) |

## Attribution

Built on [LangChain's Next.js template](https://github.com/langchain-ai/langchain-nextjs-template). Replaced the vector store with Pinecone, replaced cloud embeddings with local fastembed (BAAI/bge-small-en-v1.5), added a FastAPI ingestion/retrieval service, chunking experiments, and a retrieval evaluation harness.
