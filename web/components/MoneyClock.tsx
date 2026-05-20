"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";

type DayPoint = {
  day: number;
  spend: number;
  transactions: number;
  by_category: { category: string; spend: number; transactions: number }[];
};

// 31 evenly-spaced spokes around the clock face. Bar height = spend on
// that day-of-month across every month in the dataset. Long bars on the
// 1st (utilities/council), 7th (mortgage), 25th (savings), end-of-month
// (salary) are the load-bearing patterns the user is here to see.

export default function MoneyClock() {
  const [points, setPoints] = useState<DayPoint[]>([]);
  const [hover, setHover] = useState<DayPoint | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState(560);

  useEffect(() => {
    api.dayOfMonth().then((d) => setPoints(d.points)).catch(console.error);
  }, []);

  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize(Math.max(280, Math.min(width, height) - 24));
    });
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  const { center, innerR, maxOuterR, scale } = useMemo(() => {
    const center = size / 2;
    const innerR = size * 0.22;
    const maxOuterR = size * 0.45;
    const max = Math.max(1, ...points.map((p) => p.spend));
    return {
      center, innerR, maxOuterR,
      scale: (v: number) => innerR + (maxOuterR - innerR) * (v / max),
    };
  }, [points, size]);

  // Hour-spoke labels every 5 days so it actually reads like a dial.
  const spokeLabels = [1, 5, 10, 15, 20, 25, 31];

  return (
    <div ref={wrapRef} className="relative flex h-full w-full items-center justify-center overflow-hidden p-3">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Ring of guide circles */}
        <circle cx={center} cy={center} r={innerR}     fill="none" stroke="#1f242b" />
        <circle cx={center} cy={center} r={maxOuterR}  fill="none" stroke="#1f242b" />
        <circle
          cx={center} cy={center}
          r={innerR + (maxOuterR - innerR) * 0.5}
          fill="none" stroke="#1f242b" strokeDasharray="2 4"
        />

        {/* Day spokes */}
        {points.map((p) => {
          const days = 31;
          const angle = (p.day - 1) / days * Math.PI * 2 - Math.PI / 2; // 1st at the top
          const outer = scale(p.spend);
          const x1 = center + innerR    * Math.cos(angle);
          const y1 = center + innerR    * Math.sin(angle);
          const x2 = center + outer     * Math.cos(angle);
          const y2 = center + outer     * Math.sin(angle);
          return (
            <g
              key={p.day}
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)}
              style={{ cursor: "pointer" }}
            >
              <line
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={hover?.day === p.day ? "#38bdf8" : "#a78bfa"}
                strokeWidth={Math.max(3, (size / 31) * 0.55)}
                strokeLinecap="round"
                opacity={hover && hover.day !== p.day ? 0.35 : 1}
              />
            </g>
          );
        })}

        {/* Spoke labels */}
        {spokeLabels.map((d) => {
          const angle = (d - 1) / 31 * Math.PI * 2 - Math.PI / 2;
          const r = maxOuterR + 14;
          const x = center + r * Math.cos(angle);
          const y = center + r * Math.sin(angle);
          return (
            <text
              key={d}
              x={x} y={y}
              fill="#8b94a3"
              fontSize={11}
              textAnchor="middle"
              dominantBaseline="central"
            >
              {d}
            </text>
          );
        })}

        {/* Centre label */}
        <text x={center} y={center - 8} fill="#e6e9ee" fontSize={13} textAnchor="middle">
          Money Clock
        </text>
        <text x={center} y={center + 10} fill="#8b94a3" fontSize={10} textAnchor="middle">
          spend by day-of-month
        </text>
      </svg>

      {hover && (
        <div className="pointer-events-none absolute bottom-4 left-4 rounded border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-3 text-xs shadow-lg">
          <div className="mb-1 font-semibold text-[color:var(--color-fg)]">
            Day {hover.day} · £{hover.spend.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <div className="mb-2 text-[color:var(--color-muted)]">{hover.transactions} transactions</div>
          <ul>
            {hover.by_category.slice(0, 5).map((c) => (
              <li key={c.category} className="flex justify-between gap-4">
                <span className="text-[color:var(--color-accent-2)]">{c.category}</span>
                <span className="font-mono">£{c.spend.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
