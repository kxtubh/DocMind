import { ChatWindow } from "@/components/ChatWindow";

export default function RetrievalPage() {
  return (
    <ChatWindow
      endpoint="api/chat/retrieval"
      emptyStateComponent={
        <div className="p-6 max-w-xl mx-auto text-sm text-muted-foreground space-y-3">
          <h2 className="text-base font-semibold text-foreground">
            Document Q&amp;A — Powered by Pinecone + LangChain
          </h2>
          <p>
            Upload a markdown or text document, then ask questions about it.
            Answers are grounded in retrieved chunks with citations.
          </p>
          <ul className="list-disc list-inside space-y-1">
            <li>Click <strong>Upload document</strong> to ingest a file</li>
            <li>Ask any question about the document contents</li>
            <li>Sources appear below each answer</li>
          </ul>
          <p className="text-xs">
            Stack: Next.js · LangChain.js · OpenAI text-embedding-3-small · Pinecone
          </p>
        </div>
      }
      showIngestForm={true}
      placeholder="Ask a question about your uploaded documents…"
      emoji="🔍"
    />
  );
}
