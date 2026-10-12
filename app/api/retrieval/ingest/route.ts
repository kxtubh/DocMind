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

  // Proxy to FastAPI when configured (Phase 2)
  if (process.env.FASTAPI_URL) {
    const res = await fetch(`${process.env.FASTAPI_URL}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source, strategy: "recursive", chunk_size: 1000, chunk_overlap: 200 }),
    });
    const data = await res.json();
    if (!res.ok) {
      return NextResponse.json({ error: data.detail ?? "Ingest failed" }, { status: 500 });
    }
    return NextResponse.json(data, { status: 200 });
  }

  // Fallback: direct Pinecone ingestion (Vercel / no FastAPI)
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
