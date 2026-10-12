# Vector Databases

## What is a Vector Database?

A vector database stores high-dimensional vectors (embeddings) and supports efficient similarity search — finding the most similar vectors to a query vector. This is the backbone of semantic search, RAG, recommendation systems, and anomaly detection.

## How Vector Search Works

1. Every item (document chunk, image, product) is converted to an embedding vector.
2. Vectors are indexed using an Approximate Nearest Neighbor (ANN) algorithm.
3. At query time, the query is embedded and the index finds the k nearest vectors.
4. Results are returned ranked by similarity score (cosine similarity or dot product).

## ANN Algorithms

### HNSW (Hierarchical Navigable Small World)
- Graph-based algorithm. Builds a multi-layer graph where each layer is a subset of the lower layer.
- **Pros**: very fast queries, high recall.
- **Cons**: high memory usage.
- Used by: Pinecone, Weaviate, Qdrant, FAISS.

### IVF (Inverted File Index)
- Clusters vectors into Voronoi cells. At query time, only searches nearby cells.
- **Pros**: lower memory than HNSW.
- **Cons**: slower to build, lower recall without probe tuning.
- Used by: FAISS, pgvector.

### LSH (Locality-Sensitive Hashing)
- Hashes similar vectors to the same bucket.
- **Pros**: sublinear query time.
- **Cons**: lower recall, requires careful tuning.

## Similarity Metrics

- **Cosine similarity**: angle between vectors, range [-1, 1]. Best for text embeddings (magnitude irrelevant).
- **Dot product**: cosine × magnitude. Faster, but magnitude matters.
- **Euclidean (L2)**: geometric distance. Common for image embeddings.

For `text-embedding-3-small` (OpenAI), use **cosine** similarity.

## Popular Vector Databases

### Pinecone
- Fully managed, serverless.
- Automatic sharding and scaling.
- Supports namespaces (logical separation within one index).
- **Dimension limit**: depends on plan. 1536 for `text-embedding-3-small`.
- **Metadata filtering**: filter on arbitrary key-value metadata at query time.
- **When to use**: production, zero ops overhead, need reliability SLAs.

### Weaviate
- Open-source, self-hosted or managed cloud.
- **Hybrid search** built-in (BM25 + vector, configurable α weight).
- Schema-based data model.
- **When to use**: hybrid search is important, want open-source control.

### Chroma
- Open-source, embedded or client-server mode.
- Dead-simple local setup (in-memory or SQLite-backed).
- **When to use**: local dev, prototyping, small datasets.

### pgvector
- Postgres extension. Adds `vector` column type and similarity operators.
- **When to use**: already have Postgres, don't want another service.

### Qdrant
- Open-source Rust-based, high performance.
- Payload (metadata) stored alongside vectors.
- **When to use**: high-throughput production, open-source, complex metadata filtering.

## Key Concepts

### Namespaces (Pinecone) / Collections
Logical partitions within an index. Useful for multi-tenant setups (one namespace per user) or separating chunking strategies in experiments.

### Metadata Filtering
Filter vectors by metadata fields before or during ANN search. Example: `{ source: "handbook.md" }` to only retrieve from a specific document.

### Upsert vs Insert
Most vector DBs support upsert: insert if the ID doesn't exist, update if it does. Use stable IDs derived from content hash to avoid duplicates on re-ingestion.

### Dimensionality
Must match your embedding model:
- `text-embedding-3-small`: 1536 dimensions
- `text-embedding-3-large`: 3072 dimensions
- `text-embedding-ada-002`: 1536 dimensions

## Choosing a Vector Database

| Need | Choose |
|------|--------|
| Zero ops, production, fast | Pinecone |
| Hybrid search | Weaviate |
| Local dev / prototype | Chroma |
| Already on Postgres | pgvector |
| Max performance, open-source | Qdrant |

## Pinecone Concepts

### Index
The top-level container. Created with a fixed dimension and metric. Serverless indexes on the Starter plan are free.

### Namespace
A named partition within an index. Queries are scoped to a namespace. Default namespace is empty string `""`.

### Upsert
`index.upsert(vectors)` — each vector has an `id`, `values` (the float array), and optional `metadata`.

### Query
`index.query({ vector: queryEmbedding, topK: 5, includeMetadata: true })` — returns the top-k most similar vectors with their metadata.

### Metadata
Arbitrary JSON attached to each vector. Can filter at query time: `{ filter: { source: { "$eq": "manual.md" } } }`.
