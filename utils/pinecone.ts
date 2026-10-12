import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";

export async function getPineconeStore() {
  const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
  const index = pinecone.Index(process.env.PINECONE_INDEX!);
  return PineconeStore.fromExistingIndex(new OpenAIEmbeddings(), {
    pineconeIndex: index,
  });
}
