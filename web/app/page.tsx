"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import AlertsPanel from "@/components/AlertsPanel";
import ChatPanel from "@/components/ChatPanel";
import CompareView from "@/components/CompareView";
import DecisionTracePanel from "@/components/DecisionTracePanel";
import GraphCanvas from "@/components/GraphCanvas";
import MoneyClock from "@/components/MoneyClock";
import TimeScrubber from "@/components/TimeScrubber";
import WikiTree from "@/components/WikiTree";
import WikiViewer from "@/components/WikiViewer";
import {
  AlertItem, api, Decision, fetchAnomalies, ForecastGhost, GraphNode,
  PiiPair, TimelinePoint, TraceResponse, WikiPage,
} from "@/lib/api";
import { sanitizeNodeIds, useGraphView } from "@/lib/graph-view";

type Selection = { section: string; name: string } | null;
type CentreView = "graph" | "clock" | "compare";

function selectionFromNode(node: GraphNode): Selection {
  const [kind, raw] = node.id.split(":", 2);
  const name = raw ?? node.label;
  switch (kind) {
    case "merchant":  return { section: "merchants",  name };
    case "category":  return { section: "categories", name };
    case "month":     return { section: "months",     name };
    default:          return null;
  }
}

async function resolveWikilink(target: string): Promise<Selection> {
  for (const section of ["months", "merchants", "categories", "annual"]) {
    try {
      await api.wikiPage(section, target);
      return { section, name: target };
    } catch { /* try next */ }
  }
  return null;
}

export default function Workbench() {
  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);
  const [monthIdx, setMonthIdx] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const graphView = useGraphView();
  const [moneyFlow, setMoneyFlow] = useState(false);
  const [centre, setCentre] = useState<CentreView>("graph");

  const [selected, setSelected] = useState<Selection>({ section: "home", name: "Home" });
  const [page, setPage] = useState<WikiPage | null>(null);
  const [health, setHealth] = useState<{ transaction_count: number } | null>(null);
  const [chatOpen, setChatOpen] = useState(true);
  const [pulsed, setPulsed] = useState<Set<string>>(new Set());

  // --- Phase 4+ feature state ----------------------------------------------
  const [piiOn, setPiiOn] = useState(false);
  const [piiTokens, setPiiTokens] = useState<PiiPair[] | null>(null);

  const [showDecisions, setShowDecisions] = useState(false);
  const [decisions, setDecisions] = useState<Decision[]>([]);

  const [ghosts, setGhosts] = useState<ForecastGhost[]>([]);

  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [traceMonth, setTraceMonth] = useState<string | null>(null);

  const [compareLeft,  setCompareLeft]  = useState<string | null>(null);
  const [compareRight, setCompareRight] = useState<string | null>(null);

  // --- Fraud alerts --------------------------------------------------------
  const month = monthIdx != null ? timeline[monthIdx]?.month : undefined;
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  useEffect(() => {
    fetchAnomalies(month ?? undefined).then(r => setAlerts(r.alerts)).catch(() => setAlerts([]));
  }, [month]);

  const highRiskIds = useMemo(
    () => new Set(alerts.filter(a => a.fraud_score >= 0.5).map(a => `merchant:${a.merchant}`)),
    [alerts],
  );

  const [selectedAlert, setSelectedAlert] = useState<AlertItem | null>(null);

  // --- Boot fetches --------------------------------------------------------
  useEffect(() => {
    api.timeline().then(({ points }) => setTimeline(points)).catch(console.error);
    api.health().then(setHealth).catch(console.error);
    api.piiPreview().then(({ pairs }) => setPiiTokens(pairs)).catch(() => setPiiTokens([]));
  }, []);

  // When the user picks a month in the scrubber, push a one-hop month
  // context into the canvas. This replaces the "load the whole graph
  // for the month" blob with a focused slice.
  useEffect(() => {
    if (monthIdx == null) return;
    const m = timeline[monthIdx]?.month;
    if (!m) return;
    void graphView.loadContext([`month:${m}`]);
    // graphView.loadContext is stable from useCallback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [monthIdx, timeline]);

  // Decisions: poll only while the overlay is on.
  useEffect(() => {
    if (!showDecisions) return;
    const load = () => api.decisions().then(({ decisions }) => setDecisions(decisions)).catch(console.error);
    load();
    const t = window.setInterval(load, 4000);
    return () => window.clearInterval(t);
  }, [showDecisions]);

  // Seed default compare months once the timeline arrives.
  useEffect(() => {
    if (compareLeft || timeline.length === 0) return;
    setCompareLeft(timeline[Math.max(0, timeline.length - 2)].month);
    setCompareRight(timeline[timeline.length - 1].month);
  }, [timeline, compareLeft]);

  useEffect(() => {
    if (!selected) { setPage(null); return; }
    const loader = selected.section === "home"
      ? api.wikiHome()
      : api.wikiPage(selected.section, selected.name);
    loader.then(setPage).catch((e) => {
      // 404 is a benign "wiki page not generated for this entity" — log as
      // info so it doesn't read as a runtime crash in DevTools, and show a
      // friendly stub in the viewer.
      const msg = String(e?.message ?? e);
      if (msg.includes("404")) {
        console.info("[workbench] no wiki page for", selected, "(404)");
        setPage({
          type:        selected.section.replace(/s$/, ""),
          name:        selected.name,
          path:        `${selected.section}/${selected.name}.md`,
          frontmatter: {},
          markdown:
            `# ${selected.name}\n\n*No wiki page has been generated for this ` +
            `entity yet. Re-run \`python -m src.ingestion.compile_wiki\` to refresh.*`,
          outbound_links: [],
        });
      } else {
        console.error("[workbench] wiki load failed", e);
        setPage(null);
      }
    });
  }, [selected]);

  // Wiki section → graph node-id kind. NOT a generic singularizer —
  // "categories" → "categorie" silently breaks the canvas wiring, so we
  // keep it explicit. "annual" is its own beast (the year isn't a single
  // node, it's a property on Month); the backend resolves `year:<YYYY>`
  // to that year's months and their neighbourhoods.
  const SECTION_TO_KIND: Record<string, string> = {
    merchants:  "merchant",
    categories: "category",
    months:     "month",
    annual:     "year",
  };

  const selectedGraphId = useMemo(() => {
    if (!selected || selected.section === "home") return null;
    const kind = SECTION_TO_KIND[selected.section];
    if (!kind) return null;
    return `${kind}:${selected.name}`;
    // SECTION_TO_KIND is a module-level constant
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  // Pair the canvas with the wiki: when the user picks a merchant /
  // category / month in the left-nav, load its graph context so the
  // canvas and the wiki entry on the right pane describe the SAME entity.
  // We pin to selectedGraphId rather than `selected` to avoid re-firing
  // when only the wiki section ("home", "annual") changes.
  useEffect(() => {
    if (!selectedGraphId) return;
    void graphView.loadContext([selectedGraphId]);
    // graphView.loadContext is stable
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedGraphId]);

  const onNodeSelect = useCallback((node: GraphNode | null) => {
    if (!node) return;
    const next = selectionFromNode(node);
    if (next) setSelected(next);
  }, []);

  // Double-click → expand the node's neighborhood via /api/graph/context
  // (merge mode). The state machine handles deduping and focus.
  const onNodeDoubleClick = useCallback(async (node: GraphNode) => {
    // Schema nodes don't have a neighborhood — clicking one means "show
    // me a concrete slice of this type." Pick a representative.
    if (node.id.startsWith("schema:")) {
      const kindMap: Record<string, string | null> = {
        "schema:Merchant": null,
        "schema:Category": null,
        "schema:Month": timeline.length ? `month:${timeline[timeline.length - 1].month}` : null,
      };
      const seed = kindMap[node.id];
      if (seed) await graphView.loadContext([seed]);
      return;
    }
    await graphView.expandFromNode(node.id);
  }, [graphView, timeline]);

  // Right-click → cross-account trace. Only meaningful on a Month node or
  // the "Halifax Credit Card" merchant, but we accept any selection and
  // let the backend decide.
  const onNodeRightClick = useCallback(async (node: GraphNode) => {
    const [kind, name] = node.id.split(":", 2);
    let month: string | null = null;
    if (kind === "month") month = name;
    else if (timeline.length) month = timeline[timeline.length - 1].month;
    if (!month) return;
    try {
      const t = await api.trace(month);
      setTraceMonth(month);
      setTrace(t);
    } catch (e) { console.error(e); }
  }, [timeline]);

  // Agent highlight handler. Pulses the touched nodes for visual feedback
  // AND loads their one-hop context, so the canvas reflects the slice the
  // tool call actually queried — not a static topology.
  const onAgentHighlight = useCallback((rawIds: string[]) => {
    const nodeIds = sanitizeNodeIds(rawIds);
    if (nodeIds.length === 0) return;

    setPulsed((prev) => {
      const next = new Set(prev);
      nodeIds.forEach((id) => next.add(id));
      return next;
    });
    const fadeTimer = window.setTimeout(() => {
      setPulsed((prev) => {
        const next = new Set(prev);
        nodeIds.forEach((id) => next.delete(id));
        return next;
      });
    }, 4500);

    // Note: the actual subgraph is delivered via the `graph_update` event
    // (see onGraphUpdate below). graph_highlight is kept for backwards
    // compatibility — we only use it here for the pulse + side effects.

    // Jump the scrubber when the agent touches a month — preserves the
    // earlier UX of "agent's month becomes your month".
    const monthHit = nodeIds.find((id) => id.startsWith("month:"));
    if (monthHit) {
      const monthId = monthHit.slice("month:".length);
      const idx = timeline.findIndex((p) => p.month === monthId);
      if (idx >= 0 && idx !== monthIdx) setMonthIdx(idx);
    }

    // Open the right pane to the first touched merchant/category/month
    // so wiki context follows the agent's focus.
    const firstWiki = nodeIds.find((id) =>
      id.startsWith("merchant:") || id.startsWith("category:") || id.startsWith("month:"),
    );
    if (firstWiki) {
      const [kind, name] = firstWiki.split(":", 2);
      const sectionMap: Record<string, string> = {
        merchant: "merchants", category: "categories", month: "months",
      };
      const section = sectionMap[kind];
      if (section && name) setSelected({ section, name });
    }

    return () => window.clearTimeout(fadeTimer);
  }, [graphView, timeline, monthIdx]);

  // The agent's accumulated focus set lives in the graph view state machine.
  const agentFocus = graphView.focus;
  const clearAgentFocus = useCallback(() => {
    void graphView.resetToSchema();
  }, [graphView]);

  // Alert click → load Alert/Transaction/Merchant/Category/Month/Location
  // context into the canvas and open the detail panel on the right.
  const onAlertClick = useCallback((a: AlertItem) => {
    setSelectedAlert(a);
    void graphView.loadAlert(a.alert_id);
  }, [graphView]);

  const onWikilink = useCallback(async (target: string) => {
    if (/^\d{4}$/.test(target)) { setSelected({ section: "annual", name: target }); return; }
    const next = await resolveWikilink(target);
    if (next) setSelected(next);
  }, []);

  return (
    <div className="grid h-screen grid-rows-[auto_minmax(0,1fr)_auto]">
      {/* ===== Top bar ===== */}
      <header className="flex flex-wrap items-center gap-3 border-b border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-4 py-2">
        <div className="text-sm font-semibold">Finance Context Engine</div>
        <div className="text-xs text-[color:var(--color-muted)]">
          {health ? `${health.transaction_count.toLocaleString()} txns` : "…"}
          {centre === "graph" &&
            <span> · {graphView.data.nodes.length}n / {graphView.data.links.length}e</span>}
          {centre === "graph" &&
            <span className="ml-2 rounded border border-[color:var(--color-border)] px-1.5 py-0.5 uppercase tracking-wider">{graphView.mode}</span>}
        </div>

        <div className="ml-2 flex gap-1 rounded border border-[color:var(--color-border)] p-0.5 text-xs">
          {(["graph", "clock", "compare"] as CentreView[]).map((v) => (
            <button
              key={v}
              onClick={() => setCentre(v)}
              className={`rounded px-2 py-0.5 ${
                centre === v
                  ? "bg-[color:var(--color-accent)] text-black"
                  : "text-[color:var(--color-muted)] hover:text-[color:var(--color-fg)]"
              }`}
            >
              {v === "graph" ? "Graph" : v === "clock" ? "Clock" : "Compare"}
            </button>
          ))}
        </div>

        {centre === "graph" && (
          <>
            <Toggle
              on={moneyFlow}
              onClick={() => setMoneyFlow((v) => !v)}
              activeColor="var(--color-accent)"
              title="Emphasise £ flow over topology"
            >💸 Money Flow</Toggle>
            <Toggle
              on={showDecisions}
              onClick={() => setShowDecisions((v) => !v)}
              activeColor="#94a3b8"
              title="Render Decision-trace diamonds on the canvas"
            >📜 Decisions</Toggle>
            {graphView.mode !== "schema" && (
              <button
                onClick={clearAgentFocus}
                className="rounded border border-sky-400 bg-sky-400/10 px-2 py-0.5 text-xs text-sky-200"
                title="Drop the current context view and return to the schema overview"
              >
                ↺ schema view
              </button>
            )}
            {agentFocus.size > 0 && (
              <span
                className="rounded border border-emerald-400 bg-emerald-400/10 px-2 py-0.5 text-xs text-emerald-300"
                title="Nodes the latest tool call / alert focused on"
              >
                🎯 {agentFocus.size} focused
              </span>
            )}
            {trace && (
              <button
                onClick={() => { setTrace(null); setTraceMonth(null); }}
                className="rounded border border-[color:var(--color-accent)] bg-[color:var(--color-accent)]/15 px-2 py-0.5 text-xs text-[color:var(--color-accent)]"
                title="Clear cross-account trace overlay"
              >
                🔗 trace {traceMonth} · {trace.contributors.length} contribs · clear
              </button>
            )}
            {ghosts.length > 0 && (
              <button
                onClick={() => setGhosts([])}
                className="rounded border border-amber-500 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-400"
                title="Dismiss the advisor's forecast ghosts"
              >
                👻 {ghosts.length} ghost{ghosts.length === 1 ? "" : "s"} · clear
              </button>
            )}
          </>
        )}

        <Toggle
          on={piiOn}
          onClick={() => setPiiOn((v) => !v)}
          activeColor="var(--color-warn)"
          title="Show the wiki the way the LLM sees it (PII tokenized)"
        >🔒 LLM view</Toggle>

        <Link href="/canon"
          className="rounded border border-[color:var(--color-border)] px-2 py-0.5 text-xs text-[color:var(--color-muted)] hover:text-[color:var(--color-fg)]">
          ⚙ Canon
        </Link>

        <div className="ml-auto">
          <TimeScrubber
            timeline={timeline}
            monthIdx={monthIdx}
            setMonthIdx={setMonthIdx}
            playing={playing}
            setPlaying={setPlaying}
            speedMs={900}
          />
        </div>
      </header>

      {/* ===== Three-pane workbench ===== */}
      <main className="grid grid-cols-[240px_minmax(0,1fr)_minmax(360px,38ch)] overflow-hidden">
        <aside className="border-r border-[color:var(--color-border)] bg-[color:var(--color-panel)]">
          <WikiTree selected={selected} onSelect={(section, name) => setSelected({ section, name })} />
        </aside>

        <section className="relative min-h-0 bg-[color:var(--color-bg)]">
          {centre === "clock" ? (
            <MoneyClock />
          ) : centre === "compare" ? (
            (compareLeft && compareRight) ? (
              <CompareView
                timeline={timeline}
                leftMonth={compareLeft}
                rightMonth={compareRight}
                onLeftChange={setCompareLeft}
                onRightChange={setCompareRight}
              />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-[color:var(--color-muted)]">
                Need at least two months to compare.
              </div>
            )
          ) : graphView.error ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center text-sm">
              <div className="text-rose-400">Graph failed to load</div>
              <div className="font-mono text-xs text-[color:var(--color-muted)]">{graphView.error}</div>
              <div className="text-xs text-[color:var(--color-muted)]">
                Check the backend logs and that Neo4j is reachable.
              </div>
              <button
                onClick={() => graphView.resetToSchema()}
                className="mt-2 rounded border border-[color:var(--color-border)] px-3 py-1 text-xs"
              >
                Retry schema view
              </button>
            </div>
          ) : graphView.data.nodes.length > 0 ? (
            <GraphCanvas
              data={graphView.data}
              selectedId={selectedGraphId}
              pulsedIds={pulsed}
              moneyFlow={moneyFlow}
              decisions={showDecisions ? decisions : undefined}
              ghosts={ghosts}
              trace={trace}
              focus={agentFocus.size > 0 ? agentFocus : undefined}
              highRiskIds={highRiskIds}
              mode={graphView.mode}
              onSelect={onNodeSelect}
              onNodeRightClick={onNodeRightClick}
              onNodeDoubleClick={onNodeDoubleClick}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-[color:var(--color-muted)]">
              {graphView.isLoading ? "Loading graph…" : "No graph yet."}
            </div>
          )}
        </section>

        <aside className="flex flex-col border-l border-[color:var(--color-border)] bg-[color:var(--color-panel)]">
          <div className="max-h-[40%] min-h-[120px] border-b border-[color:var(--color-border)]">
            <AlertsPanel month={month ?? undefined} onAlertClick={onAlertClick} />
          </div>
          {showDecisions && (
            <div className="max-h-[30%] min-h-[100px] border-b border-[color:var(--color-border)]">
              <DecisionTracePanel
                decisions={decisions}
                onSelect={(d) => {
                  if (d.touched?.length) void graphView.loadContext(d.touched, "replace");
                }}
              />
            </div>
          )}
          {selectedAlert && (
            <div className="border-b border-[color:var(--color-border)] p-3 text-xs">
              <AlertDetail
                alert={selectedAlert}
                onClose={() => setSelectedAlert(null)}
              />
            </div>
          )}
          <div className="min-h-0 flex-1">
            <WikiViewer
              page={page}
              onWikilink={onWikilink}
              piiTokens={piiOn ? piiTokens : null}
            />
          </div>
        </aside>
      </main>

      <ChatPanel
        open={chatOpen}
        onToggle={() => setChatOpen((o) => !o)}
        onHighlight={onAgentHighlight}
        onGraphUpdate={(u) => {
          if (u.mode === "replace") graphView.replaceGraph(u, "context");
          else                       graphView.mergeGraph(u, "expanded");
        }}
        onTurnStart={() => graphView.clearGraph()}
        onTurnEnd={({ hadGraphUpdate }) => {
          // If the turn produced no graph context (zero-result queries,
          // backend driver miss, etc), the canvas would otherwise stay
          // blank. Restore the schema overview so it's never empty.
          if (!hadGraphUpdate) void graphView.resetToSchema();
        }}
        onForecastGhost={(g) => setGhosts((prev) => [...prev, g])}
        onClearGhosts={() => { setGhosts([]); clearAgentFocus(); }}
      />
    </div>
  );
}


// --- Alert detail card ----------------------------------------------------

function AlertDetail({ alert, onClose }: { alert: AlertItem; onClose: () => void }) {
  const rows: [string, string | number][] = [
    ["kind",        alert.kind],
    ["severity",    alert.severity.toFixed(2)],
    ["fraud_score", alert.fraud_score.toFixed(2)],
    ["merchant",    alert.merchant],
    ["amount",      `£${Math.abs(alert.amount).toFixed(2)}`],
    ["date",        alert.date],
    ["location",    alert.location ?? "—"],
    ["tx_id",       alert.tx_id],
  ];
  return (
    <div>
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="text-sm font-semibold text-rose-300">Alert · {alert.alert_id.slice(0, 10)}</div>
        <button
          onClick={onClose}
          className="text-[color:var(--color-muted)] hover:text-[color:var(--color-fg)]"
          title="Close detail"
        >×</button>
      </div>
      <div className="grid grid-cols-[80px_minmax(0,1fr)] gap-x-3 gap-y-1">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <div className="text-[color:var(--color-muted)]">{k}</div>
            <div className="truncate font-mono text-[color:var(--color-fg)]">{String(v)}</div>
          </div>
        ))}
      </div>
      {alert.risk_flags?.length > 0 && (
        <div className="mt-2">
          <div className="text-[color:var(--color-muted)]">risk flags</div>
          <div className="mt-0.5 flex flex-wrap gap-1">
            {alert.risk_flags.map((f) => (
              <span key={f} className="rounded border border-rose-300/40 bg-rose-400/10 px-1.5 py-0.5 font-mono text-[10px] text-rose-200">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
      {alert.rationale && (
        <div className="mt-2 italic text-[color:var(--color-muted)]">{alert.rationale}</div>
      )}
    </div>
  );
}


// --- Tiny on/off pill -----------------------------------------------------

function Toggle({
  on, onClick, activeColor, title, children,
}: {
  on: boolean;
  onClick: () => void;
  activeColor: string;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="rounded border px-2 py-0.5 text-xs transition-colors"
      style={
        on
          ? { borderColor: activeColor, color: activeColor, backgroundColor: `${activeColor}1f` }
          : { borderColor: "var(--color-border)", color: "var(--color-muted)" }
      }
    >
      {children} {on ? "on" : "off"}
    </button>
  );
}
