"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api, GraphResponse, TimelinePoint } from "@/lib/api";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

type Props = {
  timeline: TimelinePoint[];
  leftMonth: string;
  rightMonth: string;
  onLeftChange: (m: string) => void;
  onRightChange: (m: string) => void;
};

// ----- Diff colours -------------------------------------------------------
// Same merchant on both sides → grey (unchanged).
// Only-on-left  (disappeared)    → rose.
// Only-on-right (appeared)       → emerald.
// Spend changed > 20%            → tinted by the same colour.

const COLORS = {
  unchanged: "#64748b",
  only:      "#f43f5e",
  newly:     "#10b981",
  month:     "#f59e0b",
  category:  "#a78bfa",
};

export default function CompareView({
  timeline, leftMonth, rightMonth, onLeftChange, onRightChange,
}: Props) {
  const [left, setLeft] = useState<GraphResponse | null>(null);
  const [right, setRight] = useState<GraphResponse | null>(null);

  useEffect(() => {
    api.graph({ month: leftMonth }).then(setLeft).catch(console.error);
  }, [leftMonth]);
  useEffect(() => {
    api.graph({ month: rightMonth }).then(setRight).catch(console.error);
  }, [rightMonth]);

  const diff = useMemo(() => {
    if (!left || !right) return null;
    const leftMerchants = new Map(
      left.nodes.filter((n) => n.type === "Merchant").map((n) => [n.label, n.total_spend ?? 0]),
    );
    const rightMerchants = new Map(
      right.nodes.filter((n) => n.type === "Merchant").map((n) => [n.label, n.total_spend ?? 0]),
    );
    const labelStatus = (label: string, side: "left" | "right") => {
      const inOther = (side === "left" ? rightMerchants : leftMerchants).has(label);
      if (inOther) return "unchanged";
      return side === "left" ? "only" : "newly";
    };
    return { leftMerchants, rightMerchants, labelStatus };
  }, [left, right]);

  return (
    <div className="grid h-full grid-rows-[auto_1fr]">
      <div className="flex items-center gap-3 border-b border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-3 py-2 text-xs">
        <Selector label="A" value={leftMonth} timeline={timeline} onChange={onLeftChange} />
        <span className="text-[color:var(--color-muted)]">vs</span>
        <Selector label="B" value={rightMonth} timeline={timeline} onChange={onRightChange} />
        <span className="ml-auto flex items-center gap-3 text-[color:var(--color-muted)]">
          <Legend swatch={COLORS.unchanged} label="unchanged" />
          <Legend swatch={COLORS.only}      label="only in A" />
          <Legend swatch={COLORS.newly}     label="new in B" />
        </span>
      </div>
      <div className="grid grid-cols-2">
        <Pane title={leftMonth}  graph={left}  side="left"  diff={diff} />
        <Pane title={rightMonth} graph={right} side="right" diff={diff} />
      </div>
    </div>
  );
}


function Selector({
  label, value, timeline, onChange,
}: {
  label: string;
  value: string;
  timeline: TimelinePoint[];
  onChange: (m: string) => void;
}) {
  return (
    <label className="flex items-center gap-1">
      <span className="text-[color:var(--color-muted)]">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] px-2 py-0.5 font-mono text-xs"
      >
        {timeline.map((p) => <option key={p.month} value={p.month}>{p.month}</option>)}
      </select>
    </label>
  );
}


function Legend({ swatch, label }: { swatch: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="inline-block h-2 w-2 rounded-full" style={{ background: swatch }} />
      {label}
    </span>
  );
}


function Pane({
  title, graph, side, diff,
}: {
  title: string;
  graph: GraphResponse | null;
  side: "left" | "right";
  diff: { labelStatus: (label: string, side: "left" | "right") => string } | null;
}) {
  if (!graph) {
    return <div className="flex h-full items-center justify-center text-xs text-[color:var(--color-muted)]">Loading {title}…</div>;
  }
  const data = useMemo(() => ({
    nodes: graph.nodes.map((n) => ({ ...n })),
    links: graph.links.map((l) => ({ ...l })),
  }), [graph]);

  return (
    <div className="relative border-l border-[color:var(--color-border)] first:border-l-0">
      <div className="absolute left-2 top-2 z-10 rounded bg-[color:var(--color-panel)]/80 px-2 py-0.5 text-xs">
        {title}
      </div>
      <ForceGraph2D
        graphData={data}
        backgroundColor="#0b0d10"
        nodeRelSize={4}
        cooldownTime={1500}
        linkColor={() => "#1f242b"}
        linkWidth={(l: any) => l.type === "ACTIVE_IN" ? 1.2 : 0.6}
        nodeCanvasObjectMode={() => "after"}
        nodeCanvasObject={(node: any, ctx, scale) => {
          let fill = COLORS.unchanged;
          let r = 3.5;
          if (node.type === "Month")        { fill = COLORS.month;    r = 7; }
          else if (node.type === "Category") { fill = COLORS.category; r = 5; }
          else if (node.type === "Merchant" && diff) {
            const status = diff.labelStatus(node.label, side);
            fill = status === "unchanged" ? COLORS.unchanged
                 : status === "only"      ? COLORS.only
                 :                          COLORS.newly;
            r = 4;
          }
          ctx.fillStyle = fill;
          ctx.beginPath();
          ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
          ctx.fill();
          if (scale > 1.6) {
            ctx.fillStyle = "#e6e9ee";
            ctx.font = `${11 / scale}px sans-serif`;
            ctx.fillText(node.label, node.x + r + 2, node.y + 3);
          }
        }}
      />
    </div>
  );
}
