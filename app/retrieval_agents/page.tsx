import { ChatWindow } from "@/components/ChatWindow";

export default function RetrievalAgentsPage() {
  return (
    <ChatWindow
      endpoint="api/chat/retrieval_agents"
      emptyStateComponent={
        <div className="p-6 max-w-xl mx-auto text-sm text-muted-foreground space-y-3">
          <h2 className="text-base font-semibold text-foreground">
            Agentic Retrieval — ReAct + Pinecone
          </h2>
          <p>
            A LangGraph ReAct agent that decides when to search your documents.
            Enable <strong>Show intermediate steps</strong> to see tool calls.
          </p>
          <ul className="list-disc list-inside space-y-1">
            <li>Agent autonomously calls the retrieval tool as needed</li>
            <li>Handles multi-hop and follow-up questions better</li>
            <li>Toggle intermediate steps to inspect reasoning</li>
          </ul>
          <p className="text-xs">
            Stack: Next.js · LangGraph.js · OpenAI gpt-4o-mini · Pinecone
          </p>
        </div>
      }
      showIngestForm={true}
      showIntermediateStepsToggle={true}
      placeholder="Ask a multi-step question about your documents…"
      emoji="🤖"
    />
  );
}
