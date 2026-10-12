# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a large language model (LLM) to generate accurate, grounded answers. Instead of relying solely on the LLM's parametric knowledge, RAG fetches relevant documents from an external knowledge base and passes them as context to the LLM.

The core idea: **retrieve first, then generate**.

## Why RAG?

LLMs have a knowledge cutoff and can hallucinate facts. RAG solves this by:

1. **Grounding answers** in real documents you control.
2. **Reducing hallucinations** — the model answers from retrieved context, not memory.
3. **Enabling private knowledge** — your internal docs, wikis, or PDFs stay out of training data.
4. **Staying up to date** — update the vector store without retraining the model.

## RAG Pipeline

A standard RAG pipeline has two phases:

### Indexing (offline)
1. Load documents (PDF, Markdown, HTML, plain text).
2. Split into chunks (e.g. 1000 tokens with 200-token overlap).
3. Embed each chunk using an embedding model (e.g. `text-embedding-3-small`).
4. Store embeddings in a vector database (Pinecone, Weaviate, Chroma, etc.).

### Querying (online)
1. Embed the user's question using the same embedding model.
2. Perform approximate nearest-neighbor (ANN) search to retrieve the top-k most similar chunks.
3. Inject the chunks into the LLM prompt as context.
4. Generate the answer, grounded in the retrieved content.

## Key Components

### Embedding Model
Converts text to a dense vector (e.g. 1536 dimensions for `text-embedding-3-small`). Semantic similarity in embedding space ≈ relevance.

### Vector Database
Stores embeddings and supports fast ANN search. Popular choices:
- **Pinecone** — managed, serverless, fast.
- **Weaviate** — open-source, hybrid search.
- **Chroma** — local, easy setup.
- **pgvector** — Postgres extension.

### Chunking Strategy
How you split documents significantly affects retrieval quality:
- **Fixed-size**: split every N tokens, simple and fast.
- **Recursive character**: respect paragraph/sentence boundaries.
- **Semantic chunking**: split at topic boundaries using embedding similarity.

### Retriever
Fetches top-k chunks from the vector store given a query embedding. Can be:
- **Dense retrieval**: pure vector similarity (cosine, dot-product).
- **Sparse retrieval**: BM25 keyword matching.
- **Hybrid**: weighted combination of dense + sparse.

## Evaluation Metrics

### Retrieval
- **Recall@k**: fraction of relevant chunks found in top-k results.
- **MRR (Mean Reciprocal Rank)**: how highly the first relevant result is ranked.
- **Hit Rate**: whether any relevant chunk appears in top-k.

### Generation
- **Faithfulness** (RAGAS): is the answer supported by the retrieved context?
- **Answer Relevancy** (RAGAS): does the answer address the question?
- **Context Precision** (RAGAS): what fraction of retrieved chunks are actually relevant?

## Common Failure Modes

1. **Chunking too small**: important context split across chunks, retriever misses it.
2. **Chunking too large**: chunks contain too much noise, diluting the signal.
3. **Query-document mismatch**: query phrased differently from document text — rephrase or use hypothetical document embeddings (HyDE).
4. **Stale index**: documents updated but embeddings not refreshed.
5. **Context window overflow**: too many chunks injected, model truncates them.

## Advanced Techniques

- **HyDE (Hypothetical Document Embeddings)**: generate a hypothetical answer, embed it, use that embedding for retrieval.
- **Reranking**: use a cross-encoder to rerank top-k results before sending to the LLM.
- **Multi-query retrieval**: generate multiple rephrased queries, merge results.
- **Parent document retrieval**: index small chunks, retrieve parent documents for context.
- **Self-query retrieval**: LLM generates structured metadata filters alongside the query.
