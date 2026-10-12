"use client";

import { useState, useRef, type FormEvent, type ChangeEvent } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { toast } from "sonner";

export function UploadDocumentsForm() {
  const [isLoading, setIsLoading] = useState(false);
  const [text, setText] = useState("");
  const [source, setSource] = useState("upload");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSource(file.name);
    const reader = new FileReader();
    reader.onload = (ev) => setText(ev.target?.result as string);
    reader.readAsText(file);
  };

  const ingest = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!text.trim()) {
      toast.error("Paste text or upload a file first.");
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch("/api/retrieval/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, source }),
      });
      const json = await response.json();
      if (response.ok) {
        toast.success(`Ingested ${json.chunks} chunks from "${source}"`);
        setText("");
        setSource("upload");
        if (fileInputRef.current) fileInputRef.current.value = "";
      } else {
        toast.error(json.error ?? "Upload failed");
      }
    } catch {
      toast.error("Network error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={ingest} className="flex flex-col gap-4 w-full">
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium">Upload file (.txt, .md)</label>
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md"
          onChange={onFileChange}
          className="text-sm file:mr-4 file:py-1 file:px-3 file:rounded file:border-0 file:text-sm file:bg-secondary file:text-secondary-foreground hover:file:bg-secondary/80 cursor-pointer"
        />
      </div>
      <Textarea
        className="grow p-4 rounded bg-transparent min-h-[300px] text-xs font-mono"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Or paste markdown / text here…"
      />
      <Button type="submit" disabled={isLoading || !text.trim()}>
        {isLoading ? (
          <span className="flex items-center gap-2">
            <svg
              aria-hidden="true"
              className="w-4 h-4 animate-spin"
              viewBox="0 0 100 101"
              fill="none"
            >
              <path
                d="M100 50.591C100 78.205 77.614 100.591 50 100.591C22.386 100.591 0 78.205 0 50.591C0 22.977 22.386 0.591 50 0.591C77.614 0.591 100 22.977 100 50.591ZM9.081 50.591C9.081 73.19 27.401 91.509 50 91.509C72.599 91.509 90.919 73.19 90.919 50.591C90.919 27.992 72.599 9.672 50 9.672C27.401 9.672 9.081 27.992 9.081 50.591Z"
                fill="currentColor"
              />
              <path
                d="M93.968 39.041C96.393 38.404 97.862 35.912 97.008 33.554C95.293 28.823 92.871 24.369 89.817 20.348C85.845 15.119 80.883 10.724 75.212 7.413C69.542 4.102 63.275 1.940 56.770 1.051C51.767 0.368 46.698 0.447 41.735 1.279C39.261 1.693 37.813 4.198 38.450 6.623C39.087 9.049 41.569 10.472 44.051 10.107C47.851 9.549 51.719 9.527 55.540 10.049C60.864 10.777 65.993 12.546 70.633 15.255C75.274 17.965 79.335 21.562 82.585 25.841C84.918 28.912 86.800 32.291 88.181 35.876C89.083 38.216 91.542 39.678 93.968 39.041Z"
                fill="currentFill"
              />
            </svg>
            Ingesting…
          </span>
        ) : (
          "Ingest document"
        )}
      </Button>
    </form>
  );
}
