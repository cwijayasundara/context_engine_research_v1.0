"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ChatTurnEvent, ForecastGhost, GraphUpdate } from "@/lib/api";
import { streamAgentAsk } from "@/lib/sse";

const SESSION_KEY = "fctxe.session_id";

// One subagent's slice of a turn — tools it ran + tokens it produced.
type Lane = {
  name: string;
  brief?: string;
  status: "running" | "done" | "failed";
  tools: ToolEntry[];
  text: string;
};

type ToolEntry = {
  id: string;
  name: string;
  args: Record<string, unknown>;
  argsPreview: string;
  status: "running" | "done";
  // Captured from tool_result — short text we surface inline.
  preview?: string;
  rowsCount?: number | null;
  // From graph_highlight / graph_update emitted right after this tool.
  focusIds: string[];
};

type ChatTurn = {
  id: string;
  question: string;
  lanes: Lane[];
  final: string;
  done: boolean;
};

type Props = {
  open: boolean;
  onToggle: () => void;
  onHighlight: (nodeIds: string[]) => void;
  onGraphUpdate?: (update: GraphUpdate) => void;
  onTurnStart?: () => void;
  // Called after a turn finishes (or errors). hadGraphUpdate tells the
  // parent whether any subgraph context arrived during the turn, so it
  // can decide whether to leave the canvas as-is or fall back to schema.
  onTurnEnd?: (info: { hadGraphUpdate: boolean }) => void;
  onForecastGhost?: (g: ForecastGhost) => void;
  onClearGhosts?: () => void;
};

// Tool-call timeline rows surface what each subagent actually executed,
// the args it ran with, and the row-count preview that came back. The
// canvas highlight uses graph_update/graph_highlight; this is the textual
// timeline you can click to refocus the canvas on a tool call.
type TimelineRow = {
  id: string;
  ts: number;
  lane: string;
  tool: string;
  args: Record<string, unknown>;
  argsPreview: string;
  status: "running" | "done";
  preview?: string;
  rowsCount?: number | null;
  focusIds: string[];
};

const LANE_COLOR: Record<string, string> = {
  analyst:      "text-[color:var(--color-accent)]",
  wiki_browser: "text-[color:var(--color-accent-2)]",
  advisor:      "text-emerald-400",
  synthesizer:  "text-amber-400",
};


export default function ChatPanel({
  open, onToggle, onHighlight, onGraphUpdate, onTurnStart, onTurnEnd,
  onForecastGhost, onClearGhosts,
}: Props) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Boot — pull the session id from localStorage and rehydrate the transcript.
  useEffect(() => {
    const sid = window.localStorage.getItem(SESSION_KEY);
    if (!sid) return;
    setSessionId(sid);
    api.getSession(sid)
      .then((s) => {
        const rebuilt: ChatTurn[] = s.turns.map((t, i) => {
          const turn: ChatTurn = {
            id: `t-${i}-${t.ts}`,
            question: t.question,
            lanes: [],
            final: "",
            done: true,
          };
          for (const [name, data] of t.events as ChatTurnEvent[]) {
            applyEvent(turn, name, data);
          }
          return turn;
        });
        setTurns(rebuilt);
      })
      .catch(() => {
        // Stale session — drop it and start fresh on next ask.
        window.localStorage.removeItem(SESSION_KEY);
        setSessionId(null);
      });
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 9e6, behavior: "smooth" });
  }, [turns, streaming]);

  const ask = useCallback(async () => {
    const q = input.trim();
    if (!q || streaming) return;
    setInput("");

    const turnId = `t-${Date.now()}`;
    setTurns((prev) => [
      ...prev,
      { id: turnId, question: q, lanes: [], final: "", done: false },
    ]);
    // Each new ask wipes any ghost nodes the previous turn proposed.
    onClearGhosts?.();
    // Clear the graph canvas so the new turn's context starts from a
    // fresh slate (graph_update events will repopulate it).
    onTurnStart?.();

    setStreaming(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // Independent capture of the final answer so the UI can never end up
    // with nothing rendered — if the per-token reducer races or throws, the
    // finally block stamps this value into turn.final.
    let finalAnswer = "";
    let synthAccum = "";
    let eventCount = 0;
    let hadGraphUpdate = false;

    try {
      await streamAgentAsk(q, {
        signal: ctrl.signal,
        sessionId: sessionId ?? undefined,
        onEvent: (event, data) => {
          eventCount += 1;

          // === Parent-state side effects. These MUST run outside any
          // setTurns updater — React errors if a child renders by issuing
          // setState on its parent (which is exactly what setPulsed /
          // setSelected inside applyEvent would do).
          if (event === "session") {
            const sid = String(data?.session_id ?? "");
            if (sid) {
              setSessionId(sid);
              window.localStorage.setItem(SESSION_KEY, sid);
            }
            return;
          }
          if (event === "graph_highlight" && Array.isArray(data?.node_ids)) {
            onHighlight(data.node_ids);
            // fall through — reducer still records the event in the lane
          }
          if (event === "graph_update" && onGraphUpdate && data) {
            // The agent emits a full GraphUpdate for the canvas state
            // machine; defer to it over the highlight-id list when present.
            onGraphUpdate({
              nodes:         Array.isArray(data.nodes) ? data.nodes : [],
              relationships: Array.isArray(data.relationships) ? data.relationships : [],
              focus_ids:     Array.isArray(data.focus_ids) ? data.focus_ids : [],
              mode:          data.mode === "replace" ? "replace" : "merge",
            });
            hadGraphUpdate = hadGraphUpdate || (Array.isArray(data.nodes) && data.nodes.length > 0);
            // fall through
          }
          if (event === "forecast_ghost" && onForecastGhost) {
            onForecastGhost({
              merchant:       String(data?.merchant ?? ""),
              monthly_saving: Number(data?.monthly_saving ?? 0),
              action:         String(data?.action ?? ""),
            });
            // fall through
          }

          // === Defensive backstop captures for the finally block.
          if (event === "token" && data?.subagent === "synthesizer") {
            synthAccum += String(data?.text ?? "");
          }
          if (event === "result") {
            finalAnswer = String(data?.answer ?? "");
          }

          // === Pure reducer — only touches the local `turn` object and
          // returns the new turns array. NO parent-state setters allowed.
          setTurns((prev) => {
            const idx = prev.findIndex((t) => t.id === turnId);
            if (idx === -1) return prev;
            const turn = { ...prev[idx], lanes: prev[idx].lanes.map((l) => ({ ...l, tools: [...l.tools] })) };
            applyEvent(turn, event, data);
            const next = [...prev];
            next[idx] = turn;
            return next;
          });
        },
      });
    } catch (e: any) {
      if (e?.name !== "AbortError") {
        setTurns((prev) => prev.map((t) => t.id === turnId ? { ...t, final: `⚠️ ${e.message ?? e}`, done: true } : t));
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
      // Defensive backstop: whatever happened during streaming, ensure the
      // user sees the final answer if one was emitted. Prefer the explicit
      // `result` event, then the accumulated synthesizer tokens, otherwise
      // leave whatever the reducer produced.
      const fallback = finalAnswer || synthAccum;
      console.info(`[chat] turn ${turnId} closed · ${eventCount} events · ` +
                   `result=${finalAnswer.length}ch · synth=${synthAccum.length}ch · graph_update=${hadGraphUpdate}`);
      setTurns((prev) => prev.map((t) => {
        if (t.id !== turnId) return t;
        return { ...t, done: true, final: t.final || fallback };
      }));
      onTurnEnd?.({ hadGraphUpdate });
    }
  }, [input, streaming, sessionId, onHighlight, onForecastGhost]);

  const stop = () => abortRef.current?.abort();

  const reset = () => {
    abortRef.current?.abort();
    window.localStorage.removeItem(SESSION_KEY);
    setSessionId(null);
    setTurns([]);
  };

  return (
    <div className={`flex flex-col border-t border-[color:var(--color-border)] bg-[color:var(--color-panel)] transition-[height] duration-200 ${open ? "h-[38vh]" : "h-9"}`}>
      <button
        onClick={onToggle}
        className="flex items-center justify-between border-b border-[color:var(--color-border)] px-3 py-1.5 text-xs uppercase tracking-wider text-[color:var(--color-muted)] hover:text-[color:var(--color-fg)]"
      >
        <span>
          💬 Ask the agent
          {streaming && <span className="ml-2 text-[color:var(--color-accent)]">streaming…</span>}
          {sessionId && <span className="ml-3 font-mono text-[10px] opacity-60">session {sessionId.slice(0, 6)}</span>}
        </span>
        <span>{open ? "▾" : "▴"}</span>
      </button>

      {open && (
        <>
          <div ref={scrollRef} className="flex-1 overflow-auto px-4 py-3 text-sm">
            {turns.length === 0 && (
              <div className="text-[color:var(--color-muted)]">
                Try: <em>"How much did I spend on groceries in 2025?"</em> ·
                <em> "What's my biggest monthly expense?"</em> ·
                <em> "Where could I save £50 a month?"</em>
              </div>
            )}
            {turns.map((t) => (
              <Turn
                key={t.id}
                turn={t}
                onFocus={(ids) => {
                  // Reuse the existing highlight pathway so the canvas
                  // pulses + loads context exactly like a live tool call.
                  if (ids.length) onHighlight(ids);
                }}
              />
            ))}
          </div>

          <form
            onSubmit={(e) => { e.preventDefault(); ask(); }}
            className="flex gap-2 border-t border-[color:var(--color-border)] p-2"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about your spending…"
              disabled={streaming}
              className="flex-1 rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] px-3 py-1.5 text-sm outline-none focus:border-[color:var(--color-accent)]"
            />
            {streaming ? (
              <button type="button" onClick={stop}
                className="rounded border border-[color:var(--color-warn)] px-3 py-1.5 text-sm text-[color:var(--color-warn)] hover:bg-[color:var(--color-bg)]">
                Stop
              </button>
            ) : (
              <button type="submit" disabled={!input.trim()}
                className="rounded bg-[color:var(--color-accent)] px-3 py-1.5 text-sm font-medium text-black disabled:opacity-40">
                Ask
              </button>
            )}
            <button type="button" onClick={reset}
              title="Clear session"
              className="rounded border border-[color:var(--color-border)] px-2 py-1.5 text-sm text-[color:var(--color-muted)] hover:text-[color:var(--color-fg)]">
              ↺
            </button>
          </form>
        </>
      )}
    </div>
  );
}


// --- Turn rendering -------------------------------------------------------

function Turn({
  turn,
  onFocus,
}: {
  turn: ChatTurn;
  onFocus?: (ids: string[]) => void;
}) {
  return (
    <div className="mb-5">
      <div className="mb-0.5 text-xs uppercase tracking-wider text-[color:var(--color-muted)]">You</div>
      <div className="mb-3">{turn.question}</div>

      {turn.lanes.map((lane, i) => (
        <div key={i} className="mb-2 rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] p-2">
          <div className="mb-1 flex items-center gap-2 text-xs uppercase tracking-wider">
            <span className={LANE_COLOR[lane.name] ?? "text-[color:var(--color-fg)]"}>
              {lane.status === "running" ? "▶︎" : lane.status === "failed" ? "✕" : "✓"} {lane.name}
            </span>
            {lane.brief && <span className="truncate text-[color:var(--color-muted)] opacity-80">{lane.brief}</span>}
          </div>
          {lane.tools.map((t) => (
            <ToolRow key={t.id} entry={t} onFocus={onFocus} />
          ))}
          {lane.text && lane.name !== "synthesizer" && (
            <div className="mt-1 whitespace-pre-wrap text-xs text-[color:var(--color-muted)]">
              {lane.text}
            </div>
          )}
        </div>
      ))}

      {turn.final && (
        <div className="mt-2 whitespace-pre-wrap rounded border border-[color:var(--color-accent)]/40 bg-[color:var(--color-panel)] p-3">
          {turn.final}
        </div>
      )}
    </div>
  );
}


function ToolRow({
  entry,
  onFocus,
}: {
  entry: ToolEntry;
  onFocus?: (ids: string[]) => void;
}) {
  const clickable = entry.focusIds.length > 0 && !!onFocus;
  return (
    <div
      onClick={() => clickable && onFocus?.(entry.focusIds)}
      className={`ml-3 mt-1 rounded border border-transparent px-1.5 py-1 text-xs ${
        clickable
          ? "cursor-pointer hover:border-[color:var(--color-border)] hover:bg-[color:var(--color-panel)]"
          : ""
      }`}
      title={clickable ? `Click to refocus canvas on ${entry.focusIds.length} node(s)` : undefined}
    >
      <div className="flex items-center gap-2 text-[color:var(--color-muted)]">
        <span className={entry.status === "running" ? "animate-pulse text-[color:var(--color-accent)]" : "text-emerald-400"}>
          {entry.status === "running" ? "▶︎" : "✓"}
        </span>
        <span className="font-mono text-[color:var(--color-fg)]">{entry.name}</span>
        {entry.rowsCount != null && (
          <span className="rounded bg-[color:var(--color-panel)] px-1.5 py-0.5 font-mono text-[10px] text-[color:var(--color-muted)]">
            {entry.rowsCount} row{entry.rowsCount === 1 ? "" : "s"}
          </span>
        )}
        {entry.focusIds.length > 0 && (
          <span className="rounded border border-emerald-400/40 bg-emerald-400/10 px-1.5 py-0.5 font-mono text-[10px] text-emerald-200">
            🎯 {entry.focusIds.length}
          </span>
        )}
      </div>
      <div className="ml-4 truncate font-mono text-[10px] text-[color:var(--color-muted)] opacity-80">
        {entry.argsPreview}
      </div>
      {entry.preview && (
        <pre className="ml-4 mt-0.5 max-h-20 overflow-hidden truncate whitespace-pre-wrap font-mono text-[10px] text-[color:var(--color-muted)] opacity-70">
          {entry.preview.slice(0, 240)}
          {entry.preview.length > 240 ? "…" : ""}
        </pre>
      )}
    </div>
  );
}


function parseRowsCount(preview: string | undefined): number | null {
  if (!preview) return null;
  // tool results usually have shape {"rows":[...],"total":N} or
  // {"results":[...]}. Cheap-extract a length without full JSON parse.
  const total = preview.match(/"total"\s*:\s*(\d+)/);
  if (total) return Number(total[1]);
  // Fall back: count opening braces inside "rows":[ ... ]
  const rowsStart = preview.indexOf('"rows":[');
  if (rowsStart === -1) return null;
  const slice = preview.slice(rowsStart);
  const open = (slice.match(/\{/g) ?? []).length;
  return open || null;
}


// --- Pure event reducer (also used at rehydration time) -------------------

function applyEvent(
  turn: ChatTurn,
  event: string,
  data: any,
) {
  const ensureLane = (name: string, brief?: string): Lane => {
    let lane = turn.lanes.find((l) => l.name === name && l.status === "running");
    if (!lane) {
      lane = { name, brief, status: "running", tools: [], text: "" };
      turn.lanes.push(lane);
    }
    return lane;
  };

  switch (event) {
    case "plan":
      // Pre-stamp the lanes the planner committed to so the UI shows the
      // plan immediately, before any subagent actually starts.
      for (const step of (data?.steps ?? []) as { subagent: string; brief: string }[]) {
        turn.lanes.push({
          name: step.subagent,
          brief: step.brief,
          status: "running",
          tools: [],
          text: "",
        });
      }
      break;
    case "subagent_start": {
      const name = String(data?.name ?? "");
      const lane = turn.lanes.find((l) => l.name === name && l.status === "running" && !l.tools.length && !l.text)
        ?? ensureLane(name, data?.brief);
      lane.brief = lane.brief ?? data?.brief;
      lane.status = "running";
      break;
    }
    case "subagent_end": {
      const name = String(data?.name ?? "");
      const lane = [...turn.lanes].reverse().find((l) => l.name === name && l.status === "running");
      if (lane) lane.status = data?.ok === false ? "failed" : "done";
      break;
    }
    case "tool_call": {
      const lane = ensureLane(String(data?.subagent ?? "analyst"));
      const args = (data?.args ?? {}) as Record<string, unknown>;
      lane.tools.push({
        id: `tc-${lane.tools.length}-${Date.now()}`,
        name: String(data?.name ?? "tool"),
        args,
        argsPreview: JSON.stringify(args).slice(0, 140),
        status: "running",
        focusIds: [],
      });
      break;
    }
    case "tool_result": {
      const lane = ensureLane(String(data?.subagent ?? "analyst"));
      const last = lane.tools[lane.tools.length - 1];
      if (last && last.name === data?.name) {
        last.status = "done";
        last.preview = typeof data?.preview === "string" ? data.preview : undefined;
        last.rowsCount = parseRowsCount(last.preview);
      }
      break;
    }
    case "graph_highlight":
    case "graph_update": {
      // Attach the focus ids to the most recent tool entry in the lane so
      // the tool-call row in the UI becomes clickable → refocus canvas.
      const who = String(data?.subagent ?? "analyst");
      const lane = turn.lanes.find((l) => l.name === who);
      if (lane && lane.tools.length) {
        const ids: string[] = event === "graph_highlight"
          ? (Array.isArray(data?.node_ids) ? data.node_ids : [])
          : (Array.isArray(data?.focus_ids) ? data.focus_ids : []);
        const last = lane.tools[lane.tools.length - 1];
        const set = new Set([...(last.focusIds ?? []), ...ids]);
        last.focusIds = Array.from(set);
      }
      break;
    }
    case "token": {
      const who = String(data?.subagent ?? "synthesizer");
      const lane = ensureLane(who);
      lane.text += String(data?.text ?? "");
      // The synthesizer's text is the "final answer" for the user.
      if (who === "synthesizer") turn.final += String(data?.text ?? "");
      break;
    }
    case "result":
      if (!turn.final) turn.final = String(data?.answer ?? "");
      break;
    case "done":
      turn.done = true;
      break;
    case "error":
      // Mark the most recent running lane as failed; show error in final box.
      for (let i = turn.lanes.length - 1; i >= 0; i--) {
        if (turn.lanes[i].status === "running") { turn.lanes[i].status = "failed"; break; }
      }
      turn.final = `⚠️ ${data?.message ?? "error"}`;
      turn.done = true;
      break;
  }
}
