// Typed client for the FastAPI surface. Kept tiny on purpose — Phase 1
// only needs the read-only endpoints. Add more here as later phases wire
// up agent SSE, canon edits, etc.

export type GraphNode = {
  id: string;
  label: string;
  type: "Merchant" | "Category" | "Month";
  category?: string | null;
  total_spend?: number | null;
  visits?: number | null;
  income?: number | null;
  expense?: number | null;
};

export type GraphLink = {
  source: string;
  target: string;
  type: "IN_CATEGORY" | "ACTIVE_IN";
  weight?: number | null;
  visits?: number | null;
};

export type GraphResponse = {
  nodes: GraphNode[];
  links: GraphLink[];
  range: string[];
};

// ---------- Context-graph contract ---------------------------------------
// Mirrors src/api/models.py. Every flow that mutates the canvas (chat tool
// calls, alert clicks, expand-on-double-click) speaks GraphUpdate so the
// canvas state machine has one shape to consume.

export type GraphViewNode = {
  id: string;
  label: string;
  type: string;
  properties?: Record<string, unknown>;
};

export type GraphViewRel = {
  id: string;
  source: string;
  target: string;
  type: string;
  properties?: Record<string, unknown>;
};

export type GraphUpdate = {
  nodes: GraphViewNode[];
  relationships: GraphViewRel[];
  focus_ids: string[];
  mode: "replace" | "merge";
};

export type SchemaNode = {
  id: string;
  label: string;
  type: string;
  count?: number | null;
  description?: string | null;
};

export type SchemaRel = {
  id: string;
  source: string;
  target: string;
  type: string;
  description?: string | null;
};

export type SchemaResponse = {
  nodes: SchemaNode[];
  relationships: SchemaRel[];
};

export type WikiPage = {
  type: string;
  name: string;
  path: string;
  frontmatter: Record<string, unknown>;
  markdown: string;
  outbound_links: string[];
};

export type WikiTree = {
  sections: { section: string; pages: string[] }[];
};

export type TimelinePoint = {
  month: string;
  income: number;
  expense: number;
  net: number;
  transactions: number;
};

async function jsonGet<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json() as Promise<T>;
}

export type ChatTurnEvent = [string, any];
export type ChatSession = {
  id: string;
  created_at: number;
  turns: { ts: number; question: string; events: ChatTurnEvent[] }[];
};

export const api = {
  health: () => jsonGet<{ status: string; transaction_count: number }>("/api/health"),
  timeline: () => jsonGet<{ points: TimelinePoint[] }>("/api/timeline"),
  graph: (opts: { month?: string; range?: string; merchant?: string; category?: string } = {}) => {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(opts)) if (v) qs.set(k, v);
    return jsonGet<GraphResponse>(`/api/graph${qs.size ? `?${qs}` : ""}`);
  },
  wikiTree: () => jsonGet<WikiTree>("/api/wiki/tree"),
  wikiHome: () => jsonGet<WikiPage>("/api/wiki/home"),
  wikiPage: (section: string, name: string) =>
    jsonGet<WikiPage>(`/api/wiki/page?section=${section}&name=${encodeURIComponent(name)}`),
  getSession: (sid: string) => jsonGet<ChatSession>(`/api/agent/sessions/${sid}`),
  dayOfMonth: () => jsonGet<{
    points: {
      day: number;
      spend: number;
      transactions: number;
      by_category: { category: string; spend: number; transactions: number }[];
    }[];
  }>("/api/timeline/day_of_month"),

  // ---- PII ----------------------------------------------------------------
  piiPreview: () => jsonGet<{ pairs: PiiPair[] }>("/api/pii/preview"),

  // ---- Canon cache editor ------------------------------------------------
  canonCache: () => jsonGet<CanonCache>("/api/canon/cache"),
  canonCategories: () => jsonGet<{ categories: string[] }>("/api/canon/categories"),
  addAlias: (variant: string, canonical: string) =>
    jsonPost<{ ok: boolean }>("/api/canon/aliases", { variant, canonical }),
  lockCategory: (canonical: string, category: string) =>
    jsonPost<{ ok: boolean }>("/api/canon/category-lock", { canonical, category }),
  evictCache: (raw: string) =>
    jsonDelete<{ ok: boolean }>(`/api/canon/cache/${encodeURIComponent(raw)}`),

  // ---- Decision trace overlay --------------------------------------------
  decisions: () => jsonGet<{ decisions: Decision[] }>("/api/graph/decisions"),

  // ---- Cross-account trace -----------------------------------------------
  trace: (month: string) => jsonGet<TraceResponse>(`/api/graph/trace?month=${month}`),

  // ---- Expand-on-click (Neo4j context-graph pattern) ---------------------
  expand: (nodeId: string) =>
    jsonGet<GraphResponse & { center: string }>(
      `/api/graph/expand?id=${encodeURIComponent(nodeId)}`,
    ),

  // ---- Context-graph contract -------------------------------------------
  graphSchema: () => jsonGet<SchemaResponse>("/api/graph/schema"),
  graphContext: (ids: string[], mode: "replace" | "merge" = "replace") =>
    jsonGet<GraphUpdate>(
      `/api/graph/context?ids=${encodeURIComponent(ids.join(","))}&mode=${mode}`,
    ),
  alertContext: (alertId: string) =>
    jsonGet<GraphUpdate>(`/api/graph/alert/${encodeURIComponent(alertId)}`),
};

async function jsonPost<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json() as Promise<T>;
}

async function jsonDelete<T>(url: string): Promise<T> {
  const r = await fetch(url, { method: "DELETE" });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json() as Promise<T>;
}

export type PiiPair = { real: string; token: string; kind: string };

export type CanonCache = {
  cache: {
    raw: string;
    canonical_name: string;
    category: string;
    kind: string;
    effective: { canonical_name: string; category: string; kind: string };
  }[];
  aliases: Record<string, string>;
  category_lock: Record<string, string>;
  cache_path: string;
};

export type Decision = {
  id: string;
  question: string;
  ts: string;
  summary: string;
  touched: string[];
};

export type TraceResponse = {
  settlement: {
    month: string;
    amount: number;
    date: string;
    savings_stmt: string;
    cc_stmt: string;
  } | null;
  contributors: { merchant: string; spend: number; visits: number }[];
  node_ids: string[];
  links: { source: string; target: string; kind: "spend" | "settle" }[];
};

export type ForecastGhost = {
  merchant: string;
  monthly_saving: number;
  action: string;
};

// ---------- Fraud / alerts ----------

export type AlertItem = {
  alert_id: string;
  tx_id: string;
  kind: string;
  severity: number;
  fraud_score: number;
  risk_flags: string[];
  rationale: string;
  merchant: string;
  amount: number;
  date: string;
  description: string;
  location: string | null;
};

export type AlertsResponse = {
  month: string | null;
  alerts: AlertItem[];
};

export async function fetchAnomalies(month?: string): Promise<AlertsResponse> {
  const q = month ? `?month=${encodeURIComponent(month)}` : '';
  const r = await fetch(`/api/fraud/anomalies${q}`);
  if (!r.ok) throw new Error(`anomalies ${r.status}`);
  return r.json();
}

export async function recomputeFraud(opts: { skipGds?: boolean } = {}): Promise<{ scored: number; alerts: number }> {
  const q = opts.skipGds ? '?skip_gds=true' : '';
  const r = await fetch(`/api/fraud/recompute${q}`, { method: 'POST' });
  if (!r.ok) throw new Error(`recompute ${r.status}`);
  return r.json();
}
