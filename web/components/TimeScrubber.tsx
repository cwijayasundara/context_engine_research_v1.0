"use client";

import { useEffect, useRef } from "react";
import type { TimelinePoint } from "@/lib/api";

type Props = {
  timeline: TimelinePoint[];
  monthIdx: number | null;     // null = "all months"
  setMonthIdx: (idx: number | null) => void;
  playing: boolean;
  setPlaying: (v: boolean) => void;
  speedMs: number;             // ms per month while playing
};

export default function TimeScrubber({
  timeline, monthIdx, setMonthIdx, playing, setPlaying, speedMs,
}: Props) {
  const playingRef = useRef(playing);
  playingRef.current = playing;

  // Auto-advance loop. We use an interval rather than rAF so the speed
  // setting is honoured exactly regardless of frame rate.
  useEffect(() => {
    if (!playing || timeline.length === 0) return;
    const tick = window.setInterval(() => {
      if (!playingRef.current) return;
      setMonthIdx(
        monthIdx == null
          ? 0
          : monthIdx >= timeline.length - 1
            ? 0
            : monthIdx + 1,
      );
    }, speedMs);
    return () => window.clearInterval(tick);
  }, [playing, monthIdx, timeline.length, speedMs, setMonthIdx]);

  const active = monthIdx != null ? timeline[monthIdx] : null;

  return (
    <div className="flex items-center gap-3">
      <span className="min-w-[6.5rem] text-right text-xs font-mono text-[color:var(--color-muted)]">
        {active ? active.month : "all months"}
      </span>

      <button
        onClick={() => setPlaying(!playing)}
        disabled={timeline.length === 0}
        title={playing ? "Pause" : "Play"}
        className="rounded border border-[color:var(--color-border)] px-2 py-0.5 text-xs hover:bg-[color:var(--color-bg)]"
      >
        {playing ? "❚❚" : "▶︎"}
      </button>

      <button
        onClick={() => { setPlaying(false); setMonthIdx(null); }}
        title="Show every month at once"
        className="rounded border border-[color:var(--color-border)] px-2 py-0.5 text-xs hover:bg-[color:var(--color-bg)]"
      >
        All
      </button>

      <input
        type="range"
        min={0}
        max={Math.max(0, timeline.length - 1)}
        value={monthIdx ?? 0}
        onChange={(e) => { setPlaying(false); setMonthIdx(Number(e.target.value)); }}
        disabled={timeline.length === 0}
        className="w-64 accent-[color:var(--color-accent)]"
      />

      {active && (
        <span className="text-xs text-[color:var(--color-muted)]">
          in £{active.income.toLocaleString()} ·
          out £{active.expense.toLocaleString()} ·
          net <span className={active.net >= 0 ? "text-emerald-400" : "text-rose-400"}>
            £{active.net.toLocaleString()}
          </span>
        </span>
      )}
    </div>
  );
}
