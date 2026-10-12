import { NextRequest, NextResponse } from "next/server";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { getPineconeStore } from "@/utils/pinecone";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const text = body.text as string;
  const source = (body.source as string) ?? "upload";

  if (!text?.trim()) {
    return NextResponse.json({ error: "No text provided" }, { status: 400 });
  }

  try {
    const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splitDocuments = await splitter.createDocuments([text], [{ source }]);
    splitDocuments.forEach((doc, i) => {
      doc.metadata.chunk_id = `${source}_${i}`;
    });

    const vectorstore = await getPineconeStore();
    await vectorstore.addDocuments(splitDocuments);

    return NextResponse.json(
      { ok: true, chunks: splitDocuments.length },
      { status: 200 },
    );
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
