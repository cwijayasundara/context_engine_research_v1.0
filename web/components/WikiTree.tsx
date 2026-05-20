"use client";

import { useEffect, useState } from "react";
import { api, WikiTree as WikiTreeT } from "@/lib/api";

type Props = {
  selected?: { section: string; name: string } | null;
  onSelect: (section: string, name: string) => void;
};

export default function WikiTree({ selected, onSelect }: Props) {
  const [tree, setTree] = useState<WikiTreeT | null>(null);

  useEffect(() => {
    api.wikiTree().then(setTree).catch(console.error);
  }, []);

  if (!tree) return <div className="p-3 text-sm text-[color:var(--color-muted)]">Loading…</div>;

  return (
    <nav className="h-full overflow-auto p-2 text-sm">
      <button
        onClick={() => onSelect("home", "Home")}
        className={`mb-3 block w-full rounded px-2 py-1 text-left
          ${selected?.section === "home"
            ? "bg-[color:var(--color-panel)] text-[color:var(--color-accent)]"
            : "hover:bg-[color:var(--color-panel)]"}`}
      >
        🏠 Home
      </button>

      {tree.sections.map((s) => (
        <details key={s.section} open className="mb-2">
          <summary className="cursor-pointer select-none px-2 py-1 text-xs uppercase tracking-wider text-[color:var(--color-muted)]">
            {s.section} <span className="opacity-50">({s.pages.length})</span>
          </summary>
          <ul className="mt-1">
            {s.pages.map((p) => {
              const active = selected?.section === s.section && selected?.name === p;
              return (
                <li key={p}>
                  <button
                    onClick={() => onSelect(s.section, p)}
                    className={`block w-full truncate rounded px-3 py-1 text-left
                      ${active
                        ? "bg-[color:var(--color-panel)] text-[color:var(--color-accent)]"
                        : "hover:bg-[color:var(--color-panel)] text-[color:var(--color-fg)]"}`}
                  >
                    {p}
                  </button>
                </li>
              );
            })}
          </ul>
        </details>
      ))}
    </nav>
  );
}
