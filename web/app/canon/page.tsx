"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { api, CanonCache } from "@/lib/api";

export default function CanonPage() {
  const [data, setData] = useState<CanonCache | null>(null);
  const [categories, setCategories] = useState<string[]>([]);
  const [filter, setFilter] = useState("");
  const [aliasFor, setAliasFor] = useState<string | null>(null);
  const [aliasTo, setAliasTo] = useState("");
  const [lockFor, setLockFor] = useState<string | null>(null);
  const [lockCat, setLockCat] = useState("");
  const [pendingError, setPendingError] = useState<string | null>(null);

  const refresh = () => {
    setPendingError(null);
    api.canonCache().then(setData).catch((e) => setPendingError(String(e)));
  };
  useEffect(() => {
    refresh();
    api.canonCategories().then(({ categories }) => setCategories(categories)).catch(console.error);
  }, []);

  const filtered = useMemo(() => {
    if (!data) return [];
    const needle = filter.trim().toLowerCase();
    if (!needle) return data.cache;
    return data.cache.filter((e) =>
      e.raw.toLowerCase().includes(needle)
      || e.effective.canonical_name.toLowerCase().includes(needle)
      || e.effective.category.toLowerCase().includes(needle),
    );
  }, [data, filter]);

  const submitAlias = async (variant: string) => {
    if (!aliasTo.trim()) return;
    try {
      await api.addAlias(variant, aliasTo.trim());
      setAliasFor(null);
      setAliasTo("");
      refresh();
    } catch (e: any) { setPendingError(String(e)); }
  };

  const submitLock = async (canonical: string) => {
    if (!lockCat) return;
    try {
      await api.lockCategory(canonical, lockCat);
      setLockFor(null);
      setLockCat("");
      refresh();
    } catch (e: any) { setPendingError(String(e)); }
  };

  const evict = async (raw: string) => {
    try {
      await api.evictCache(raw);
      refresh();
    } catch (e: any) { setPendingError(String(e)); }
  };

  return (
    <div className="mx-auto max-w-6xl p-6 text-sm">
      <div className="mb-4 flex items-baseline gap-3">
        <h1 className="text-lg font-semibold">Canonicalizer Cache</h1>
        <Link href="/" className="text-xs text-[color:var(--color-accent)] hover:underline">
          ← back to workbench
        </Link>
      </div>

      {pendingError && (
        <div className="mb-3 rounded border border-rose-700/50 bg-rose-950/30 p-2 text-rose-300">
          {pendingError}
        </div>
      )}

      {data && (
        <div className="mb-4 grid grid-cols-3 gap-3 text-xs">
          <Stat label="Cache entries"   value={data.cache.length} />
          <Stat label="Stylistic aliases" value={Object.keys(data.aliases).length} />
          <Stat label="Category locks"  value={Object.keys(data.category_lock).length} />
        </div>
      )}

      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="filter by raw description, canonical or category…"
        className="mb-3 w-full rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] px-3 py-1.5 outline-none focus:border-[color:var(--color-accent)]"
      />

      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="text-left text-[color:var(--color-muted)]">
            <th className="border-b border-[color:var(--color-border)] py-2 pr-2">Raw description</th>
            <th className="border-b border-[color:var(--color-border)] py-2 pr-2">Cached as</th>
            <th className="border-b border-[color:var(--color-border)] py-2 pr-2">→ Effective</th>
            <th className="border-b border-[color:var(--color-border)] py-2 pr-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((e) => {
            const aliased = e.canonical_name !== e.effective.canonical_name;
            const locked  = e.category       !== e.effective.category;
            return (
              <tr key={e.raw} className="align-top">
                <td className="border-b border-[color:var(--color-border)]/60 py-1.5 pr-2 font-mono">
                  {e.raw}
                </td>
                <td className="border-b border-[color:var(--color-border)]/60 py-1.5 pr-2 text-[color:var(--color-muted)]">
                  {e.canonical_name}
                  <div className="text-[10px] opacity-60">[{e.category}]</div>
                </td>
                <td className="border-b border-[color:var(--color-border)]/60 py-1.5 pr-2">
                  <span className={aliased ? "text-[color:var(--color-accent)]" : ""}>
                    {e.effective.canonical_name}
                  </span>
                  <div className="text-[10px]">
                    <span className={locked ? "text-emerald-400" : "text-[color:var(--color-muted)]"}>
                      [{e.effective.category}]
                    </span>
                  </div>
                </td>
                <td className="border-b border-[color:var(--color-border)]/60 py-1.5 pr-2 text-right">
                  {aliasFor === e.canonical_name ? (
                    <span className="inline-flex items-center gap-1">
                      <input
                        value={aliasTo} onChange={(ev) => setAliasTo(ev.target.value)}
                        placeholder="canonical name"
                        className="w-40 rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] px-1 py-0.5 text-xs"
                      />
                      <button onClick={() => submitAlias(e.canonical_name)}
                        className="rounded bg-[color:var(--color-accent)] px-2 py-0.5 text-xs text-black">Save</button>
                      <button onClick={() => { setAliasFor(null); setAliasTo(""); }}
                        className="text-[color:var(--color-muted)]">✕</button>
                    </span>
                  ) : lockFor === e.effective.canonical_name ? (
                    <span className="inline-flex items-center gap-1">
                      <select
                        value={lockCat} onChange={(ev) => setLockCat(ev.target.value)}
                        className="rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] px-1 py-0.5 text-xs"
                      >
                        <option value="">—</option>
                        {categories.map((c) => <option key={c} value={c}>{c}</option>)}
                      </select>
                      <button onClick={() => submitLock(e.effective.canonical_name)}
                        className="rounded bg-[color:var(--color-accent)] px-2 py-0.5 text-xs text-black">Save</button>
                      <button onClick={() => { setLockFor(null); setLockCat(""); }}
                        className="text-[color:var(--color-muted)]">✕</button>
                    </span>
                  ) : (
                    <span className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => { setAliasFor(e.canonical_name); setAliasTo(e.effective.canonical_name); }}
                        title="Fold this LLM-emitted name onto a canonical spelling"
                        className="text-[color:var(--color-accent)] hover:underline">alias</button>
                      <button
                        onClick={() => { setLockFor(e.effective.canonical_name); setLockCat(e.effective.category); }}
                        title="Pin this merchant's category so the LLM can't drift"
                        className="text-emerald-400 hover:underline">lock category</button>
                      <button
                        onClick={() => evict(e.raw)}
                        title="Drop this row so the next ask re-classifies it"
                        className="text-rose-400 hover:underline">evict</button>
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {data && (
        <div className="mt-6 grid grid-cols-2 gap-6 text-xs">
          <div>
            <h2 className="mb-2 font-semibold text-[color:var(--color-accent)]">Active aliases</h2>
            <pre className="overflow-auto rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] p-2">{JSON.stringify(data.aliases, null, 2)}</pre>
          </div>
          <div>
            <h2 className="mb-2 font-semibold text-emerald-400">Active category locks</h2>
            <pre className="overflow-auto rounded border border-[color:var(--color-border)] bg-[color:var(--color-bg)] p-2">{JSON.stringify(data.category_lock, null, 2)}</pre>
          </div>
        </div>
      )}

      <div className="mt-4 text-xs text-[color:var(--color-muted)]">
        Edits mutate the live agent's in-memory tables instantly. They are
        not yet persisted back to <code>src/ingestion/llm_normalize.py</code>;
        commit them there to survive a restart.
      </div>
    </div>
  );
}


function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-2">
      <div className="text-[color:var(--color-muted)]">{label}</div>
      <div className="text-base font-semibold">{value}</div>
    </div>
  );
}
