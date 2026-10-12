import { NextRequest, NextResponse } from "next/server";
import { type UIMessage, type TextUIPart, createTextStreamResponse } from "ai";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { getPineconeStore } from "@/utils/pinecone";

export const runtime = "nodejs";

const combineDocumentsFn = (docs: Document[]) =>
  docs.map((doc) => doc.pageContent).join("\n\n");

const getMessageText = (message: UIMessage) =>
  message.parts
    .filter((p): p is TextUIPart => p.type === "text")
    .map((p) => p.text)
    .join("");

const formatVercelMessages = (chatHistory: UIMessage[]) =>
  chatHistory
    .map((message) => {
      if (message.role === "user") return `Human: ${getMessageText(message)}`;
      if (message.role === "assistant")
        return `Assistant: ${getMessageText(message)}`;
      return `${message.role}: ${getMessageText(message)}`;
    })
    .join("\n");

const CONDENSE_QUESTION_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

<chat_history>
  {chat_history}
</chat_history>

Follow Up Input: {question}
Standalone question:`;

const ANSWER_TEMPLATE = `You are a helpful AI assistant. Answer the question based only on the provided context and chat history. If the answer is not in the context, say you don't know.

<context>
  {context}
</context>

<chat_history>
  {chat_history}
</chat_history>

Question: {question}
Answer:`;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const previousMessages = messages.slice(0, -1);
    const currentMessageContent = getMessageText(messages[messages.length - 1]);

    const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.2 });
    const vectorstore = await getPineconeStore();

    const standaloneQuestionChain = RunnableSequence.from([
      PromptTemplate.fromTemplate(CONDENSE_QUESTION_TEMPLATE),
      model,
      new StringOutputParser(),
    ]);

    let resolveWithDocuments: (value: Document[]) => void;
    const documentPromise = new Promise<Document[]>((resolve) => {
      resolveWithDocuments = resolve;
    });

    const retriever = vectorstore.asRetriever({
      callbacks: [
        {
          handleRetrieverEnd(documents) {
            resolveWithDocuments(documents);
          },
        },
      ],
    });

    const retrievalChain = retriever.pipe(combineDocumentsFn);

    const answerChain = RunnableSequence.from([
      {
        context: RunnableSequence.from([
          (input) => input.question,
          retrievalChain,
        ]),
        chat_history: (input) => input.chat_history,
        question: (input) => input.question,
      },
      PromptTemplate.fromTemplate(ANSWER_TEMPLATE),
      model,
    ]);

    const conversationalRetrievalQAChain = RunnableSequence.from([
      {
        question: standaloneQuestionChain,
        chat_history: (input) => input.chat_history,
      },
      answerChain,
      new StringOutputParser(),
    ]);

    const stream = await conversationalRetrievalQAChain.stream({
      question: currentMessageContent,
      chat_history: formatVercelMessages(previousMessages),
    });

    const documents = await documentPromise;
    const serializedSources = Buffer.from(
      JSON.stringify(
        documents.map((doc) => ({
          pageContent: doc.pageContent.slice(0, 100) + "...",
          metadata: doc.metadata,
        })),
      ),
    ).toString("base64");

    return createTextStreamResponse({
      stream,
      headers: {
        "x-message-index": (previousMessages.length + 1).toString(),
        "x-sources": serializedSources,
      },
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}
