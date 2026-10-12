import { cn } from "@/utils/cn";
import type { UIMessage, TextUIPart } from "ai";

const getMessageText = (message: UIMessage) =>
  message.parts
    .filter((p): p is TextUIPart => p.type === "text")
    .map((p) => p.text)
    .join("");

export function ChatMessageBubble(props: {
  message: UIMessage;
  aiEmoji?: string;
  sources: any[];
}) {
  return (
    <div
      className={cn(
        `rounded-[24px] max-w-[80%] mb-8 flex`,
        props.message.role === "user"
          ? "bg-secondary text-secondary-foreground px-4 py-2"
          : null,
        props.message.role === "user" ? "ml-auto" : "mr-auto",
      )}
    >
      {props.message.role !== "user" && (
        <div className="mr-4 border bg-secondary -mt-2 rounded-full w-10 h-10 flex-shrink-0 flex items-center justify-center">
          {props.aiEmoji}
        </div>
      )}

      <div className="whitespace-pre-wrap flex flex-col">
        <span>{getMessageText(props.message)}</span>

        {props.sources && props.sources.length ? (
          <details className="mt-4 mr-auto">
            <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
              🔍 {props.sources.length} source
              {props.sources.length !== 1 ? "s" : ""}
            </summary>
            <div className="mt-2 space-y-2">
              {props.sources.map((source, i) => (
                <div
                  key={"source:" + i}
                  className="bg-secondary rounded px-3 py-2 text-xs"
                >
                  <div className="font-medium text-foreground mb-1">
                    {source.metadata?.source ?? "document"}
                    {source.metadata?.chunk_id
                      ? ` · chunk ${source.metadata.chunk_id.split("_").pop()}`
                      : ""}
                  </div>
                  <div className="text-muted-foreground">{source.pageContent}</div>
                </div>
              ))}
            </div>
          </details>
        ) : null}
      </div>
    </div>
  );
}
