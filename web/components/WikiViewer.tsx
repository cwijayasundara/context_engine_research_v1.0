"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { WikiPage } from "@/lib/api";

type PiiPair = { real: string; token: string; kind: string };

type Props = {
  page: WikiPage | null;
  onWikilink: (target: string) => void;
  piiTokens?: PiiPair[] | null;     // when present, swap `real` → `token` in the body
};

// Type-keyed accents — match the canvas FILL palette so the wiki entry on
// the right visually pairs with the node colour on the canvas.
const TYPE_ACCENT: Record<string, string> = {
  merchant: "#38bdf8",
  category: "#a78bfa",
  month:    "#f59e0b",
  annual:   "#10b981",
};

// Match [[Page]] and [[Page|display]] — strip wrappers so they don't
// reach react-markdown as plain `[[…]]` tokens.
const WIKILINK = /\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g;

function applyPiiMask(text: string, pairs: PiiPair[] | null | undefined): string {
  if (!pairs?.length) return text;
  // Longest first so "14 Avengers Close" beats just "Avengers".
  const sorted = [...pairs].sort((a, b) => b.real.length - a.real.length);
  let out = text;
  for (const { real, token } of sorted) {
    if (!real) continue;
    // Escape the needle so e.g. "5286 83** **** 1588" doesn't blow up regex.
    const safe = real.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    out = out.replace(new RegExp(safe, "g"), token);
  }
  return out;
}

export default function WikiViewer({ page, onWikilink, piiTokens }: Props) {
  if (!page) {
    return (
      <div className="p-6 text-sm text-[color:var(--color-muted)]">
        Click a node on the graph (or a page in the sidebar) to load its wiki entry.
      </div>
    );
  }

  // Apply PII redaction FIRST so wikilink text and table cells are masked.
  const masked = applyPiiMask(page.markdown, piiTokens);
  // Rewrite [[Target]] into an HTML anchor we can intercept. We can't use
  // a markdown link directly because GFM doesn't know about [[…]].
  const rewritten = masked.replace(
    WIKILINK,
    (_, target: string, display: string | undefined) =>
      `[${(display || target).trim()}](#wikilink:${encodeURIComponent(target.trim())})`,
  );

  const accent = TYPE_ACCENT[page.type] ?? "#94a3b8";

  return (
    <article className="prose-vault h-full overflow-auto p-6">
      <div className="mb-3 flex items-center justify-between text-xs uppercase tracking-wider text-[color:var(--color-muted)]">
        <span className="flex items-center gap-2">
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: accent }} />
          {page.type} · {page.path}
        </span>
        {piiTokens && piiTokens.length > 0 && (
          <span className="rounded bg-[color:var(--color-warn)]/15 px-1.5 py-0.5 text-[10px] text-[color:var(--color-warn)]">
            🔒 LLM view
          </span>
        )}
      </div>

      <StatsStrip page={page} accent={accent} />

      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("#wikilink:")) {
              const target = decodeURIComponent(href.slice("#wikilink:".length));
              return (
                <button
                  onClick={(e) => { e.preventDefault(); onWikilink(target); }}
                  className="text-[color:var(--color-accent)] hover:underline"
                >
                  {children}
                </button>
              );
            }
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
          },
        }}
      >
        {rewritten}
      </ReactMarkdown>
    </article>
  );
}


// ---------- KPI strip ----------------------------------------------------
//
// Reads the wiki page's frontmatter — which is the authoritative, already-
// aggregated copy of the headline figures — and renders them as a compact
// stat-tile row above the markdown. This is the bit that makes the wiki
// readable at a glance in a narrow side pane: numbers up top, prose below.

type StatTile = { label: string; value: string; sub?: string };

function StatsStrip({ page, accent }: { page: WikiPage; accent: string }) {
  const tiles = tilesFor(page);
  if (tiles.length === 0) return null;
  return (
    <div
      className="mb-4 grid gap-2"
      style={{ gridTemplateColumns: `repeat(${Math.min(tiles.length, 4)}, minmax(0, 1fr))` }}
    >
      {tiles.map((t) => (
        <div
          key={t.label}
          className="rounded border bg-[color:var(--color-panel)] p-2"
          style={{ borderColor: `${accent}40` }}
        >
          <div className="text-[10px] uppercase tracking-wider text-[color:var(--color-muted)]">
            {t.label}
          </div>
          <div className="mt-0.5 truncate text-base font-semibold" style={{ color: accent }}>
            {t.value}
          </div>
          {t.sub && (
            <div className="truncate text-[10px] text-[color:var(--color-muted)]">{t.sub}</div>
          )}
        </div>
      ))}
    </div>
  );
}

function tilesFor(page: WikiPage): StatTile[] {
  const fm = page.frontmatter as Record<string, unknown>;
  const num = (k: string) => (typeof fm[k] === "number" ? (fm[k] as number) : null);
  const str = (k: string) => (typeof fm[k] === "string" ? (fm[k] as string) : null);

  switch (page.type) {
    case "merchant": {
      const spend  = num("total_spend");
      const visits = num("visits");
      const avg = spend != null && visits ? spend / visits : null;
      return compact([
        spend  != null && tile("Total spend", money(spend)),
        visits != null && tile("Visits", visits.toLocaleString()),
        avg    != null && tile("Avg / visit", money(avg)),
        !!str("category") && tile("Category", str("category")!),
      ]);
    }
    case "category": {
      const spend = num("total_spend");
      const txs   = num("transactions");
      const avg = spend != null && txs ? spend / txs : null;
      return compact([
        spend != null && tile("Total spend", money(spend)),
        txs   != null && tile("Transactions", txs.toLocaleString()),
        avg   != null && tile("Avg / tx", money(avg)),
        !!str("kind") && tile("Kind", str("kind")!),
      ]);
    }
    case "month": {
      const income  = num("income");
      const expense = num("expense");
      const net     = num("net");
      const txs     = num("transactions");
      return compact([
        income  != null && tile("Income", money(income)),
        expense != null && tile("Expense", money(expense)),
        net     != null && tile("Net", money(net), net < 0 ? "deficit" : net > 0 ? "surplus" : undefined),
        txs     != null && tile("Transactions", txs.toLocaleString()),
      ]);
    }
    case "annual": {
      const income  = num("income");
      const expense = num("expense");
      // The annual wiki writes `savings` (= income − expense), not `net`.
      const savings = num("savings") ?? num("net");
      return compact([
        income  != null && tile("Income", money(income)),
        expense != null && tile("Expense", money(expense)),
        savings != null && tile("Savings", money(savings), savings < 0 ? "deficit" : savings > 0 ? "surplus" : undefined),
      ]);
    }
    default:
      return [];
  }
}

function tile(label: string, value: string, sub?: string): StatTile {
  return { label, value, sub };
}

function compact<T>(arr: (T | false | null | undefined)[]): T[] {
  return arr.filter(Boolean) as T[];
}

function money(value: number): string {
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0,
  }).format(Math.abs(value));
}
