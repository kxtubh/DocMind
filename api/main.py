import os
import time
import statistics
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from fastembed import TextEmbedding

load_dotenv("../.env.local")

# ── Local embeddings (free, no API key) ──────────────────────────────────────

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embed_model


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_embed_model()
        return [v.tolist() for v in model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        model = _get_embed_model()
        return list(model.embed([text]))[0].tolist()


app = FastAPI(title="RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", os.getenv("FRONTEND_URL", "")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ponytail: global list, use Redis/DB if multi-process
_latency_log: list[dict] = []


def _get_store(namespace: str = "") -> PineconeVectorStore:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX"])
    return PineconeVectorStore(index=index, embedding=LocalEmbeddings(), namespace=namespace)


# ── Request / Response models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str
    source: str = "upload"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    strategy: str = "recursive"   # "recursive" | "fixed"
    namespace: str = ""


class QueryRequest(BaseModel):
    question: str
    namespace: str = ""
    k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency: dict


class RetrieveRequest(BaseModel):
    question: str
    namespace: str = ""
    k: int = 4


class RetrieveResponse(BaseModel):
    sources: list[dict]
    latency: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "index": os.getenv("PINECONE_INDEX", "not set")}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    if req.strategy == "fixed":
        splitter = CharacterTextSplitter(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            separator="\n",
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )

    chunks = splitter.create_documents(
        [req.text],
        metadatas=[{
            "source": req.source,
            "strategy": req.strategy,
            "chunk_size": req.chunk_size,
        }],
    )
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{req.source}_{i}"

    store = _get_store(req.namespace)
    store.add_documents(chunks)
    return {"ok": True, "chunks": len(chunks), "namespace": req.namespace or "default"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """Retrieve sources only — no LLM generation. Used by retrieval eval."""
    t0 = time.perf_counter()

    embeddings = LocalEmbeddings()
    t_embed_start = time.perf_counter()
    query_vec = embeddings.embed_query(req.question)
    t_embed = time.perf_counter() - t_embed_start

    t_retrieve_start = time.perf_counter()
    store = _get_store(req.namespace)
    docs = store.similarity_search_by_vector(query_vec, k=req.k)
    t_retrieve = time.perf_counter() - t_retrieve_start

    latency = {
        "embed_ms": round(t_embed * 1000),
        "retrieve_ms": round(t_retrieve * 1000),
        "total_ms": round((time.perf_counter() - t0) * 1000),
    }

    return RetrieveResponse(
        sources=[
            {"pageContent": d.page_content[:300], "metadata": d.metadata}
            for d in docs
        ],
        latency=latency,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.perf_counter()

    embeddings = LocalEmbeddings()
    t_embed_start = time.perf_counter()
    query_vec = embeddings.embed_query(req.question)
    t_embed = time.perf_counter() - t_embed_start

    t_retrieve_start = time.perf_counter()
    store = _get_store(req.namespace)
    docs = store.similarity_search_by_vector(query_vec, k=req.k)
    t_retrieve = time.perf_counter() - t_retrieve_start

    context = "\n\n".join(d.page_content for d in docs)

    t_gen_start = time.perf_counter()
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        response = llm.invoke(
            "Answer based only on the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {req.question}"
        )
        answer = str(response.content)
    except Exception:
        answer = "(LLM unavailable — retrieval-only mode)"
    t_gen = time.perf_counter() - t_gen_start

    total_ms = round((time.perf_counter() - t0) * 1000)
    latency = {
        "embed_ms": round(t_embed * 1000),
        "retrieve_ms": round(t_retrieve * 1000),
        "generate_ms": round(t_gen * 1000),
        "total_ms": total_ms,
    }
    _latency_log.append(latency)

    return QueryResponse(
        answer=answer,
        sources=[
            {"pageContent": d.page_content[:300], "metadata": d.metadata}
            for d in docs
        ],
        latency=latency,
    )


@app.get("/metrics")
def metrics():
    if not _latency_log:
        return {"message": "No queries logged yet. Run some queries first."}

    def pct(values: list[int], p: int) -> int:
        sorted_v = sorted(values)
        idx = max(0, int(len(sorted_v) * p / 100) - 1)
        return sorted_v[idx]

    def stats(key: str) -> dict:
        vals = [e[key] for e in _latency_log]
        return {
            "p50": pct(vals, 50),
            "p95": pct(vals, 95),
            "mean": round(statistics.mean(vals)),
        }

    return {
        "count": len(_latency_log),
        "total_ms": stats("total_ms"),
        "embed_ms": stats("embed_ms"),
        "retrieve_ms": stats("retrieve_ms"),
        "generate_ms": stats("generate_ms"),
    }
