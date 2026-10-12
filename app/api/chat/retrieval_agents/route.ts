import { NextRequest, NextResponse } from "next/server";
import { type UIMessage, type TextUIPart, createTextStreamResponse } from "ai";
import {
  AIMessage,
  BaseMessage,
  ChatMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { createRetrieverTool } from "@langchain/classic/tools/retriever";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { getPineconeStore } from "@/utils/pinecone";

export const runtime = "nodejs";

const getMessageText = (message: UIMessage) =>
  message.parts
    .filter((p): p is TextUIPart => p.type === "text")
    .map((p) => p.text)
    .join("");

const convertVercelMessageToLangChainMessage = (message: UIMessage) => {
  if (message.role === "user") {
    return new HumanMessage(getMessageText(message));
  } else if (message.role === "assistant") {
    return new AIMessage(getMessageText(message));
  } else {
    return new ChatMessage(getMessageText(message), message.role);
  }
};

const convertLangChainMessageToVercelMessage = (message: BaseMessage) => {
  if (message._getType() === "human") {
    return { content: message.content, role: "user" };
  } else if (message._getType() === "ai") {
    return {
      content: message.content,
      role: "assistant",
      tool_calls: (message as AIMessage).tool_calls,
    };
  } else {
    return { content: message.content, role: message._getType() };
  }
};

const AGENT_SYSTEM_TEMPLATE = `You are a helpful AI assistant with access to a document knowledge base. Use the search tool to find relevant information and answer questions accurately. If you cannot find relevant information, say so clearly.`;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = (body.messages ?? [])
      .filter(
        (message: UIMessage) =>
          message.role === "user" || message.role === "assistant",
      )
      .map(convertVercelMessageToLangChainMessage);
    const returnIntermediateSteps = body.show_intermediate_steps;

    const chatModel = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.2 });
    const vectorstore = await getPineconeStore();
    const retriever = vectorstore.asRetriever();

    const tool = createRetrieverTool(retriever, {
      name: "search_documents",
      description:
        "Searches the document knowledge base and returns relevant information. Use this for any question about the uploaded documents.",
    });

    const agent = await createReactAgent({
      llm: chatModel,
      tools: [tool],
      messageModifier: new SystemMessage(AGENT_SYSTEM_TEMPLATE),
    });

    if (!returnIntermediateSteps) {
      const eventStream = await agent.streamEvents(
        { messages },
        { version: "v2" },
      );

      const textEncoder = new TextEncoder();
      const transformStream = new ReadableStream({
        async start(controller) {
          for await (const { event, data } of eventStream) {
            if (event === "on_chat_model_stream") {
              if (!!data.chunk.content) {
                controller.enqueue(textEncoder.encode(data.chunk.content));
              }
            }
          }
          controller.close();
        },
      });

      return createTextStreamResponse({ stream: transformStream });
    } else {
      const result = await agent.invoke({ messages });
      return NextResponse.json(
        {
          messages: result.messages.map(convertLangChainMessageToVercelMessage),
        },
        { status: 200 },
      );
    }
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}
