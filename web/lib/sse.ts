// Fetch-based SSE consumer. We can't use the browser's `EventSource`
// because it only does GET — and we need POST to send the question.
//
// Frame format mirrors the Server-Sent Events spec:
//
//   event: <name>\n
//   data: <json>\n
//   <blank line>
//
// We tolerate multi-line `data:` payloads (joined with \n).

export type SSEHandler = (event: string, data: any) => void;

export type ChatStreamOpts = {
  signal?: AbortSignal;
  sessionId?: string;
  onEvent: SSEHandler;
};

export async function streamAgentAsk(
  question: string,
  { signal, sessionId, onEvent }: ChatStreamOpts,
): Promise<void> {
  const r = await fetch("/api/agent/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(sessionId ? { question, session_id: sessionId } : { question }),
    signal,
  });
  if (!r.ok || !r.body) {
    throw new Error(`agent/ask → ${r.status}`);
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let totalFrames = 0;

  console.info("[sse] open POST /api/agent/ask");

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    // Normalize CRLF → LF so the same indexOf("\n\n") split works regardless
    // of whether an intermediate proxy injects \r.
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let sep: number;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      dispatchFrame(frame, onEvent);
      totalFrames++;
    }
  }
  if (buf.trim()) { dispatchFrame(buf, onEvent); totalFrames++; }
  console.info(`[sse] closed cleanly, ${totalFrames} frames`);
}

function dispatchFrame(frame: string, onEvent: SSEHandler) {
  let event = "message";
  const dataLines: string[] = [];
  for (const raw of frame.split("\n")) {
    const line = raw.trimEnd();
    if (!line || line.startsWith(":")) continue;          // skip blank / ping
    const colon = line.indexOf(":");
    if (colon === -1) continue;
    const field = line.slice(0, colon);
    // SSE spec: a single space after the colon is part of the protocol.
    const value = line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (!dataLines.length) return;
  let data: any = dataLines.join("\n");
  try { data = JSON.parse(data); } catch { /* keep as string */ }
  try {
    onEvent(event, data);
  } catch (err) {
    // Don't let a single bad reducer kill the whole stream.
    console.error("[sse] reducer threw for event", event, err);
  }
}
