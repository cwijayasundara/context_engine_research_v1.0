"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  Decision, ForecastGhost, GraphNode, GraphResponse, TraceResponse,
} from "@/lib/api";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const FILL: Record<string, string> = {
  Merchant: "#38bdf8",
  Category: "#a78bfa",
  Month: "#f59e0b",
  Decision: "#94a3b8",
  Ghost: "#fbbf24",
  Transaction: "#22d3ee",
  Alert: "#ef4444",
  Statement: "#64748b",
  Account: "#475569",
  Day: "#0ea5e9",
  Location: "#10b981",
};

const NODE_RADIUS: Record<string, number> = {
  Merchant: 5,
  Category: 9,
  Month: 11,
  Decision: 5,
  Ghost: 7,
  Transaction: 4,
  Alert: 7,
  Statement: 8,
  Account: 10,
  Day: 4,
  Location: 6,
};

// In schema view the same ids carry a marker in their properties so we can
// blow them up to overview-size circles.
const SCHEMA_RADIUS_BOOST = 14;

type Props = {
  data: GraphResponse;
  selectedId?: string | null;
  pulsedIds?: Set<string>;
  moneyFlow?: boolean;
  decisions?: Decision[];
  ghosts?: ForecastGhost[];
  trace?: TraceResponse | null;
  focus?: Set<string>;
  highRiskIds?: Set<string>;
  // 'schema' tells the canvas the nodes are type-level (big circles),
  // 'context' is an instance slice driven by chat/alert, 'expanded' is
  // a user-grown view. The canvas itself doesn't care for layout, but
  // some downstream UI does (Hud caption, default Legend, click hint).
  mode?: "schema" | "context" | "expanded";
  onSelect?: (node: GraphNode | null) => void;
  onNodeRightClick?: (node: GraphNode) => void;
  onNodeDoubleClick?: (node: GraphNode) => void;
};

type VizNode = {
  id: string;
  label: string;
  type: GraphNode["type"] | "Decision" | "Ghost";
  color: string;
  radius: number;
  category?: string | null;
  total_spend?: number | null;
  visits?: number | null;
  income?: number | null;
  expense?: number | null;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
  dim?: boolean;
};

type VizLink = {
  id: string;
  source: string | VizNode;
  target: string | VizNode;
  type: string;
  color: string;
  width: number;
  weight?: number | null;
  visits?: number | null;
  dim?: boolean;
};

export default function GraphCanvas({
  data, selectedId, pulsedIds, moneyFlow, decisions, ghosts, trace, focus, highRiskIds,
  mode = "context",
  onSelect, onNodeRightClick, onNodeDoubleClick,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(undefined);
  const clickRef = useRef<{ id: string; ts: number } | null>(null);
  const pinnedRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [selectedNode, setSelectedNode] = useState<VizNode | null>(null);
  const isFocused = !!focus && focus.size > 0;

  useEffect(() => {
    if (!containerRef.current) return;
    const measure = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const availableHeight = Math.max(280, window.innerHeight - rect.top - 8);
      const visibleHeight = Math.max(280, Math.min(rect.height || availableHeight, availableHeight));
      if (rect.width > 50 && visibleHeight > 50) {
        setSize({ w: Math.round(rect.width), h: Math.round(visibleHeight) });
      }
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(containerRef.current);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);

  const { graphData, fullCounts } = useMemo(() => {
    const allNodes: VizNode[] = data.nodes.map((node) => {
      const pinned = pinnedRef.current.get(node.id);
      return {
        ...node,
        id: node.id,
        label: node.label,
        type: node.type,
        color: FILL[node.type] ?? "#64748b",
        radius: nodeRadius(node, focus),
        ...(pinned ? { x: pinned.x, y: pinned.y, fx: pinned.x, fy: pinned.y } : {}),
      };
    });

    const allLinks: VizLink[] = data.links.map((link, index) => {
      const source = typeof link.source === "string" ? link.source : (link.source as any).id;
      const target = typeof link.target === "string" ? link.target : (link.target as any).id;
      const spendWeight = link.type === "ACTIVE_IN" ? Math.log10((link.weight ?? 0) + 1) : 0;
      return {
        id: `rel-${index}`,
        source,
        target,
        type: link.type,
        color: moneyFlow && link.type === "ACTIVE_IN" ? "#38bdf8" : "#334155",
        width: link.type === "ACTIVE_IN" ? Math.min(2.4, 0.45 + spendWeight * 0.35) : 0.35,
        weight: link.weight,
        visits: link.visits,
      };
    });

    if (decisions?.length) {
      const inGraph = new Set(allNodes.map((node) => node.id));
      for (const decision of decisions) {
        const touched = decision.touched.filter((id) => inGraph.has(id));
        if (!touched.length) continue;
        const id = `decision:${decision.id}`;
        const pinned = pinnedRef.current.get(id);
        allNodes.push({
          id,
          label: "decision",
          type: "Decision",
          color: FILL.Decision,
          radius: nodeRadius({ id, type: "Decision" } as VizNode, focus),
          ...(pinned ? { x: pinned.x, y: pinned.y, fx: pinned.x, fy: pinned.y } : {}),
        });
        for (const touchedId of touched) {
          allLinks.push({
            id: `dec-${id}-${touchedId}`,
            source: id,
            target: touchedId,
            type: "TOUCHED",
            color: "rgba(148,163,184,0.35)",
            width: 0.5,
          });
        }
      }
    }

    if (ghosts?.length) {
      const inGraph = new Set(allNodes.map((node) => node.id));
      for (const ghost of ghosts) {
        const anchor = `merchant:${ghost.merchant}`;
        if (!inGraph.has(anchor)) continue;
        const id = `ghost:${ghost.merchant}`;
        const pinned = pinnedRef.current.get(id);
        allNodes.push({
          id,
          label: `save £${Math.round(ghost.monthly_saving)}/mo`,
          type: "Ghost",
          color: FILL.Ghost,
          radius: nodeRadius({ id, type: "Ghost" } as VizNode, focus),
          ...(pinned ? { x: pinned.x, y: pinned.y, fx: pinned.x, fy: pinned.y } : {}),
        });
        allLinks.push({
          id: `gho-${id}`,
          source: anchor,
          target: id,
          type: "GHOST",
          color: "rgba(251,191,36,0.55)",
          width: 0.8,
        });
      }
    }

    const fullCounts = { nodes: allNodes.length, links: allLinks.length };
    if (!focus || focus.size === 0) {
      return { graphData: { nodes: allNodes, links: allLinks }, fullCounts };
    }

    const visibleIds = new Set(focus);
    const links = allLinks.filter((link) => {
      const source = endpointId(link.source);
      const target = endpointId(link.target);
      const keep = focus.has(source) || focus.has(target);
      if (keep) {
        visibleIds.add(source);
        visibleIds.add(target);
      }
      return keep;
    });
    const nodes = allNodes.filter((node) => visibleIds.has(node.id));

    return { graphData: { nodes, links }, fullCounts };
  }, [data, decisions, ghosts, focus, moneyFlow]);

  const fit = useCallback((duration = 450) => {
    window.setTimeout(() => fgRef.current?.zoomToFit(duration, isFocused ? 34 : 72), 80);
  }, [isFocused]);

  const centerGraph = useCallback((duration = 350) => {
    const positioned = graphData.nodes.filter(
      (node) => typeof node.x === "number" && typeof node.y === "number",
    );
    if (!fgRef.current || positioned.length === 0) return;
    const minX = Math.min(...positioned.map((node) => node.x as number));
    const maxX = Math.max(...positioned.map((node) => node.x as number));
    const minY = Math.min(...positioned.map((node) => node.y as number));
    const maxY = Math.max(...positioned.map((node) => node.y as number));
    fgRef.current.centerAt((minX + maxX) / 2, (minY + maxY) / 2, duration);
  }, [graphData.nodes]);

  const focusedIdsKey = useMemo(
    () => focus ? [...focus].sort().join("|") : "",
    [focus],
  );

  const selectedLinks = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.links.filter((link) => (
      endpointId(link.source) === selectedNode.id || endpointId(link.target) === selectedNode.id
    ));
  }, [graphData.links, selectedNode]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedNode(null);
      return;
    }
    const next = graphData.nodes.find((node) => node.id === selectedId);
    if (next) setSelectedNode(next);
  }, [selectedId, graphData.nodes]);

  useEffect(() => {
    if (!fgRef.current || graphData.nodes.length === 0) return;
    const linkForce = fgRef.current.d3Force("link") as any;
    linkForce?.distance?.((link: VizLink) => (
      isFocused ? (link.type === "IN_CATEGORY" ? 70 : 96) : (link.type === "IN_CATEGORY" ? 82 : 128)
    ));
    linkForce?.strength?.(isFocused ? 0.16 : 0.08);
    const chargeForce = fgRef.current.d3Force("charge") as any;
    chargeForce?.strength?.(isFocused ? -260 : -95);
    fgRef.current.d3Force("center");
    fgRef.current.d3ReheatSimulation();
    fit(650);
    window.setTimeout(() => centerGraph(400), 900);
  }, [centerGraph, fit, focusedIdsKey, graphData.nodes.length, graphData.links.length, isFocused]);

  const asGraphNode = (node: VizNode): GraphNode => ({
    id: node.id,
    label: node.label,
    type: node.type === "Decision" || node.type === "Ghost" ? "Merchant" : node.type,
  } as GraphNode);

  const handleNodeClick = (node: VizNode) => {
    const now = Date.now();
    if (clickRef.current?.id === node.id && now - clickRef.current.ts < 320) {
      clickRef.current = null;
      onNodeDoubleClick?.(asGraphNode(node));
      return;
    }
    clickRef.current = { id: node.id, ts: now };
    setSelectedNode(node);
    onSelect?.(asGraphNode(node));
  };

  const handleBackgroundClick = () => {
    setSelectedNode(null);
    onSelect?.(null);
  };

  const handleNodeDragEnd = (node: any) => {
    if (typeof node.x !== "number" || typeof node.y !== "number") return;
    node.fx = node.x;
    node.fy = node.y;
    pinnedRef.current.set(node.id, { x: node.x, y: node.y });
  };

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 overflow-hidden"
      style={{ background: "#0b0d10" }}
    >
      {size ? (
        <ForceGraph2D
          ref={fgRef as any}
          width={size.w}
          height={size.h}
          graphData={graphData}
          backgroundColor="#0b0d10"
          nodeId="id"
          nodeLabel={(node: any) => node.label}
          minZoom={0.08}
          maxZoom={7}
          cooldownTicks={90}
          cooldownTime={2400}
          d3VelocityDecay={0.45}
          enableNodeDrag
          enablePanInteraction
          enableZoomInteraction
          linkColor={(link: any) => link.color}
          linkWidth={(link: any) => link.width}
          linkDirectionalParticles={(link: any) => moneyFlow && link.type === "ACTIVE_IN" ? 1 : 0}
          linkDirectionalParticleWidth={1.4}
          linkDirectionalParticleSpeed={0.004}
          nodeCanvasObjectMode={() => "replace"}
          nodeCanvasObject={(node: any, ctx, globalScale) => {
            drawNode(node, ctx, globalScale, {
              selected: node.id === selectedId,
              pulsed: !!pulsedIds?.has(node.id),
            });
            if (highRiskIds?.has(node.id)) {
              ctx.save();
              ctx.strokeStyle = "#dc2626";
              ctx.lineWidth   = 2 / globalScale;
              ctx.beginPath();
              ctx.arc(node.x ?? 0, node.y ?? 0, (node.radius ?? 6) + 3 / globalScale, 0, Math.PI * 2);
              ctx.stroke();
              ctx.restore();
            }
          }}
          nodePointerAreaPaint={(node: any, color, ctx) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(node.x ?? 0, node.y ?? 0, Math.max(8, node.radius + 5), 0, 2 * Math.PI);
            ctx.fill();
          }}
          onNodeClick={(node) => handleNodeClick(node as VizNode)}
          onNodeDragEnd={handleNodeDragEnd}
          onNodeRightClick={(node, event) => {
            event.preventDefault();
            onNodeRightClick?.(asGraphNode(node as VizNode));
          }}
          onBackgroundClick={handleBackgroundClick}
        />
      ) : (
        <div className="flex h-full items-center justify-center text-xs text-[color:var(--color-muted)]">
          Measuring canvas…
        </div>
      )}

      <Hud
        size={size ?? { w: 0, h: 0 }}
        nodes={graphData.nodes.length}
        rels={graphData.links.length}
        fullNodes={fullCounts.nodes}
        fullRels={fullCounts.links}
        focused={!!focus && focus.size > 0}
        ready={!!size}
        mode={mode}
      />
      <Toolbar
        onZoomIn={() => fgRef.current?.zoom((fgRef.current.zoom() ?? 1) * 1.25, 180)}
        onZoomOut={() => fgRef.current?.zoom((fgRef.current.zoom() ?? 1) / 1.25, 180)}
        onFit={() => fit(300)}
        onReset={() => fit(300)}
      />
      {selectedNode && (
        <NodeDetails
          node={selectedNode}
          links={selectedLinks}
          onClose={() => setSelectedNode(null)}
        />
      )}
      <Legend />
    </div>
  );
}

function drawNode(
  node: VizNode,
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  state: { selected: boolean; pulsed: boolean },
) {
  const x = node.x ?? 0;
  const y = node.y ?? 0;
  const r = node.radius;

  if (state.selected || state.pulsed) {
    ctx.beginPath();
    ctx.arc(x, y, r + (state.pulsed ? 5 : 3), 0, 2 * Math.PI);
    ctx.strokeStyle = state.pulsed ? "#fbbf24" : "#22d3ee";
    ctx.lineWidth = Math.max(1.2, 2.2 / globalScale);
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = node.color;
  ctx.fill();
  ctx.strokeStyle = "#020617";
  ctx.lineWidth = Math.max(0.5, 1.2 / globalScale);
  ctx.stroke();

  const shouldLabel = state.selected || state.pulsed || node.radius >= 7 || node.type === "Category" || node.type === "Month" || globalScale > 1.35;
  if (!shouldLabel) return;

  const fontSize = Math.max(8, 11 / globalScale);
  ctx.font = `${fontSize}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.lineWidth = Math.max(2, 3 / globalScale);
  ctx.strokeStyle = "#0b0d10";
  ctx.fillStyle = "#e5e7eb";
  const label = node.label.length > 26 ? `${node.label.slice(0, 25)}…` : node.label;
  ctx.strokeText(label, x, y + r + 3);
  ctx.fillText(label, x, y + r + 3);
}

function nodeRadius(node: Pick<VizNode, "id" | "type">, focus?: Set<string>): number {
  // Schema-overview nodes always render as big anchor circles.
  if (node.id.startsWith("schema:")) return SCHEMA_RADIUS_BOOST;
  const base = NODE_RADIUS[node.type] ?? 4;
  if (!focus || focus.size === 0) return base;
  return focus.has(node.id) ? Math.max(12, base * 2.1) : Math.max(7, base * 1.45);
}

function endpointId(endpoint: string | VizNode): string {
  return typeof endpoint === "string" ? endpoint : endpoint.id;
}

function NodeDetails({
  node, links, onClose,
}: {
  node: VizNode;
  links: VizLink[];
  onClose: () => void;
}) {
  const rows = [
    ["type", node.type],
    ["category", node.category],
    ["total spend", formatMoney(node.total_spend)],
    ["visits", node.visits],
    ["income", formatMoney(node.income)],
    ["expense", formatMoney(node.expense)],
    ["pinned", node.fx != null && node.fy != null ? "yes" : "no"],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");

  return (
    <div className="absolute right-14 top-3 z-10 w-[min(320px,calc(100%-5rem))] rounded border border-[color:var(--color-border)] bg-[color:var(--color-panel)]/95 p-3 text-xs shadow-xl">
      <div className="mb-2 flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-[color:var(--color-fg)]">{node.label}</div>
          <div className="font-mono text-[10px] text-[color:var(--color-muted)]">{node.id}</div>
        </div>
        <button
          onClick={onClose}
          className="flex h-6 w-6 items-center justify-center rounded border border-[color:var(--color-border)] text-[color:var(--color-muted)] hover:text-[color:var(--color-fg)]"
          title="Close details"
        >
          ×
        </button>
      </div>
      <div className="grid grid-cols-[88px_minmax(0,1fr)] gap-x-3 gap-y-1">
        {rows.map(([key, value]) => (
          <div key={String(key)} className="contents">
            <div className="text-[color:var(--color-muted)]">{key}</div>
            <div className="truncate font-mono text-[color:var(--color-fg)]">{String(value)}</div>
          </div>
        ))}
      </div>
      {links.length > 0 && (
        <div className="mt-3 border-t border-[color:var(--color-border)] pt-2">
          <div className="mb-1 text-[color:var(--color-muted)]">links</div>
          <div className="max-h-28 overflow-auto font-mono text-[10px] text-[color:var(--color-muted)]">
            {links.map((link) => {
              const source = endpointId(link.source);
              const target = endpointId(link.target);
              const other = source === node.id ? target : source;
              return (
                <div key={link.id} className="truncate">
                  {link.type} → {other}{link.weight != null ? ` · ${formatMoney(link.weight)}` : ""}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function formatMoney(value: number | null | undefined): string | null {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0,
  }).format(Math.abs(value));
}

function Hud({
  size, nodes, rels, fullNodes, fullRels, focused, ready, mode,
}: {
  size: { w: number; h: number };
  nodes: number;
  rels: number;
  fullNodes: number;
  fullRels: number;
  focused: boolean;
  ready: boolean;
  mode: "schema" | "context" | "expanded";
}) {
  const MODE_COLOR: Record<string, string> = {
    schema:   "text-sky-400",
    context:  "text-emerald-300",
    expanded: "text-amber-300",
  };
  const hint =
    mode === "schema"
      ? "schema · ask the agent or click an alert to see actual data"
      : "scroll = zoom · drag = pan · double-click = expand";
  return (
    <div className="absolute left-2 top-2 z-10 flex items-center gap-2 rounded border border-[color:var(--color-border)] bg-[color:var(--color-panel)]/85 px-2 py-1 text-[10px] font-mono text-[color:var(--color-muted)]">
      <span>{size.w}×{size.h}</span>
      <span className={MODE_COLOR[mode] ?? ""}>{mode}</span>
      <span className="text-[color:var(--color-fg)]">
        {nodes}n / {rels}r
        {focused && <span className="text-[color:var(--color-muted)]"> of {fullNodes}n / {fullRels}r</span>}
      </span>
      <span className={ready ? "text-emerald-400" : "text-amber-400 animate-pulse"}>● {ready ? "force" : "boot"}</span>
      <span className="opacity-60">{hint}</span>
    </div>
  );
}

function Toolbar({
  onZoomIn, onZoomOut, onFit, onReset,
}: {
  onZoomIn: () => void; onZoomOut: () => void; onFit: () => void; onReset: () => void;
}) {
  const btn = "flex h-8 w-8 items-center justify-center border-b border-[color:var(--color-border)] " +
              "text-[color:var(--color-muted)] hover:bg-[color:var(--color-bg)] hover:text-[color:var(--color-fg)] " +
              "last:border-b-0";
  return (
    <div className="absolute bottom-3 right-3 z-10 flex flex-col rounded border border-[color:var(--color-border)] bg-[color:var(--color-panel)]/90 text-base font-mono">
      <button className={btn} title="Zoom in" onClick={onZoomIn}>＋</button>
      <button className={btn} title="Zoom out" onClick={onZoomOut}>−</button>
      <button className={btn} title="Fit graph" onClick={onFit}>⊡</button>
      <button className={btn} title="Reset view" onClick={onReset}>⟲</button>
    </div>
  );
}

function Legend({ types }: { types?: readonly string[] }) {
  const kinds = types && types.length
    ? types
    : (["Merchant", "Category", "Month", "Transaction", "Alert"] as const);
  return (
    <div className="absolute bottom-3 left-3 z-10 flex flex-wrap gap-3 text-xs text-[color:var(--color-muted)]">
      {kinds.map((kind) => (
        <span key={kind} className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-full" style={{ background: FILL[kind] ?? "#64748b" }} />
          {kind}
        </span>
      ))}
    </div>
  );
}
