"use client";

import { Decision } from "@/lib/api";

type Props = {
  decisions: Decision[];
  onSelect: (d: Decision) => void;
};

// Lists recent agent decisions. Clicking one loads the touched-node
// context into the canvas — turning historical traces into navigable
// state. Polling/data fetching lives in the parent; this is presentational.
export default function DecisionTracePanel({ decisions, onSelect }: Props) {
  if (!decisions.length) {
    return (
      <div className="flex h-full flex-col">
        <Header count={0} />
        <div className="px-3 py-3 text-xs text-[color:var(--color-muted)]">
          No decisions yet. Ask the agent something — its traces will show up here.
        </div>
      </div>
    );
  }
  return (
    <div className="flex h-full flex-col">
      <Header count={decisions.length} />
      <ul className="flex-1 divide-y divide-[color:var(--color-border)] overflow-y-auto text-xs">
        {decisions.map((d) => (
          <li
            key={d.id}
            onClick={() => onSelect(d)}
            className="cursor-pointer px-3 py-2 hover:bg-[color:var(--color-bg)]"
            title="Click to load this decision's touched nodes into the graph"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-medium text-[color:var(--color-fg)]">
                {d.question}
              </span>
              <span className="shrink-0 font-mono text-[10px] text-[color:var(--color-muted)]">
                {formatTime(d.ts)}
              </span>
            </div>
            {d.summary && (
              <div className="mt-1 line-clamp-2 text-[color:var(--color-muted)]">
                {d.summary}
              </div>
            )}
            {d.touched?.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-1">
                {d.touched.slice(0, 6).map((id) => (
                  <span
                    key={id}
                    className="rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] px-1.5 py-0.5 font-mono text-[10px] text-[color:var(--color-muted)]"
                  >
                    {id}
                  </span>
                ))}
                {d.touched.length > 6 && (
                  <span className="text-[10px] text-[color:var(--color-muted)]">
                    +{d.touched.length - 6}
                  </span>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Header({ count }: { count: number }) {
  return (
    <header className="flex items-center justify-between border-b border-[color:var(--color-border)] px-3 py-1.5 text-xs uppercase tracking-wider text-[color:var(--color-muted)]">
      <span>📜 Decisions</span>
      <span>{count}</span>
    </header>
  );
}

function formatTime(ts: string): string {
  // The backend sends Neo4j datetime → string. Cheap render: take the
  // tail "HH:mm" if we can, otherwise return as-is.
  const m = ts.match(/T(\d{2}:\d{2})/);
  return m ? m[1] : ts.slice(0, 16);
}
