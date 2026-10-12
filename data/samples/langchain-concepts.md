# LangChain Framework

## What is LangChain?

LangChain is an open-source framework for building applications powered by large language models (LLMs). It provides abstractions for chaining LLM calls, integrating tools, building agents, and connecting to external data sources like vector stores.

Available in Python (`langchain`) and JavaScript/TypeScript (`langchain`/`@langchain/core`).

## Core Abstractions

### LLMs and Chat Models
Wrappers around model providers (OpenAI, Anthropic, Cohere, etc.) with a unified interface.

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.2 });
const response = await model.invoke("What is RAG?");
```

### Prompts and PromptTemplates
Structured templates with variable substitution.

```typescript
import { PromptTemplate } from "@langchain/core/prompts";

const prompt = PromptTemplate.fromTemplate(
  "Answer the question based on the context:\n{context}\n\nQuestion: {question}"
);
```

### Document Loaders
Load data from PDFs, URLs, databases, markdown files, etc.

```typescript
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const loader = new PDFLoader("document.pdf");
const docs = await loader.load();
```

### Text Splitters
Break large documents into smaller chunks for embedding.

```typescript
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const chunks = await splitter.splitDocuments(docs);
```

### Embeddings
Convert text to vectors.

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
const vector = await embeddings.embedQuery("What is a vector database?");
```

### Vector Stores
Store and retrieve embeddings.

```typescript
import { PineconeStore } from "@langchain/pinecone";

const store = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex });
const results = await store.similaritySearch("RAG pipeline", 5);
```

### Retrievers
Fetch relevant documents given a query. Vector stores expose `.asRetriever()`.

```typescript
const retriever = store.asRetriever({ k: 5 });
const docs = await retriever.invoke("How does chunking work?");
```

## Chains

LangChain Expression Language (LCEL) composes runnables with the `|` pipe operator or `RunnableSequence.from([...])`.

```typescript
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

const chain = RunnableSequence.from([
  prompt,
  model,
  new StringOutputParser(),
]);

const answer = await chain.invoke({ context: "...", question: "..." });
```

### Streaming
```typescript
const stream = await chain.stream({ context: "...", question: "..." });
for await (const chunk of stream) {
  process.stdout.write(chunk);
}
```

## Agents

Agents use LLMs to decide which tools to call. LangGraph's `createReactAgent` implements the ReAct (Reasoning + Acting) pattern.

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { createRetrieverTool } from "@langchain/classic/tools/retriever";

const tool = createRetrieverTool(retriever, {
  name: "search_documents",
  description: "Search the knowledge base for relevant information.",
});

const agent = await createReactAgent({ llm: model, tools: [tool] });
```

### ReAct Loop
1. **Thought**: model reasons about what to do.
2. **Action**: model calls a tool.
3. **Observation**: tool returns a result.
4. Repeat until the model produces a final answer.

## LangGraph

LangGraph extends LangChain for stateful, multi-step agent workflows as graphs.

- **Nodes**: functions that process state.
- **Edges**: transitions between nodes.
- **State**: typed object passed between nodes.

LangGraph enables complex patterns: parallel tool execution, human-in-the-loop, conditional branching, memory.

## Callbacks

Callbacks intercept events in the chain lifecycle — useful for logging, tracing, and extracting intermediate results (e.g., retrieved documents).

```typescript
const retriever = store.asRetriever({
  callbacks: [{
    handleRetrieverEnd(documents) {
      console.log("Retrieved:", documents.length, "docs");
    }
  }]
});
```

## LangSmith

LangSmith is LangChain's observability platform. Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to automatically trace every chain and agent run — inputs, outputs, latency, token usage.

## Package Structure (JS)

| Package | Contains |
|---------|---------|
| `@langchain/core` | Base abstractions: Runnable, Document, Embeddings, etc. |
| `langchain` | Chains, splitters, utilities |
| `@langchain/openai` | OpenAI LLM + embeddings |
| `@langchain/pinecone` | Pinecone vector store integration |
| `@langchain/langgraph` | Stateful agent graphs |
| `@langchain/community` | Community integrations (PDF loaders, etc.) |

## Common Patterns

### Conversational RAG
1. Condense the follow-up question (with chat history) into a standalone question.
2. Retrieve docs for the standalone question.
3. Generate answer with context + history.

### Hypothetical Document Embeddings (HyDE)
1. Ask LLM to generate a hypothetical answer.
2. Embed the hypothetical answer.
3. Use that embedding for retrieval (often better than embedding the question directly).

### Multi-Query Retrieval
Generate 3-5 rephrased versions of the question, retrieve for each, deduplicate, merge results.
