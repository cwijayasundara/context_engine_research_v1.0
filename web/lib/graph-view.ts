"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  GraphLink,
  GraphNode,
  GraphResponse,
  GraphUpdate,
  GraphViewNode,
  GraphViewRel,
  SchemaResponse,
} from "@/lib/api";

// The canvas state machine.
//
//   mode='schema'    → high-level type/topology overview (boot state).
//   mode='context'   → showing a slice driven by chat / alert / decision.
//   mode='expanded'  → user double-clicked to grow the view by 1 hop.
//
// Every mutation goes through one of the actions below; the hook projects
// the internal store into the GraphResponse shape the current canvas
// already consumes, so we don't have to rewrite ForceGraph2D wiring.

export type GraphViewMode = "schema" | "context" | "expanded";

type Store = {
  nodes: Map<string, GraphViewNode>;
  rels: Map<string, GraphViewRel>;
};

const empty = (): Store => ({ nodes: new Map(), rels: new Map() });

export type GraphViewState = {
  mode: GraphViewMode;
  data: GraphResponse;
  focus: Set<string>;
  isLoading: boolean;
  error: string | null;
  replaceGraph: (update: GraphUpdate, mode?: GraphViewMode) => void;
  mergeGraph: (update: GraphUpdate, mode?: GraphViewMode) => void;
  focusIds: (ids: string[]) => void;
  clearGraph: () => void;
  resetToSchema: () => Promise<void>;
  loadContext: (ids: string[], mode?: "replace" | "merge") => Promise<void>;
  loadAlert: (alertId: string) => Promise<void>;
  expandFromNode: (id: string) => Promise<void>;
};

export function useGraphView(): GraphViewState {
  const [store, setStore] = useState<Store>(empty);
  const [mode, setMode] = useState<GraphViewMode>("schema");
  const [focus, setFocus] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Cached schema response — we only need to fetch it once per session.
  const schemaCache = useRef<SchemaResponse | null>(null);

  // ---- Actions ------------------------------------------------------------

  const applyUpdate = useCallback((
    update: GraphUpdate,
    intendedMode: GraphViewMode,
    merge: boolean,
  ) => {
    setStore((prev) => {
      const nodes = merge ? new Map(prev.nodes) : new Map<string, GraphViewNode>();
      const rels  = merge ? new Map(prev.rels)  : new Map<string, GraphViewRel>();
      for (const n of update.nodes) nodes.set(n.id, n);
      for (const r of update.relationships) rels.set(r.id, r);
      return { nodes, rels };
    });
    setMode(intendedMode);
    setFocus(new Set(update.focus_ids));
  }, []);

  const replaceGraph = useCallback((update: GraphUpdate, m: GraphViewMode = "context") => {
    applyUpdate(update, m, false);
  }, [applyUpdate]);

  const mergeGraph = useCallback((update: GraphUpdate, m: GraphViewMode = "expanded") => {
    applyUpdate(update, m, true);
  }, [applyUpdate]);

  const focusIds = useCallback((ids: string[]) => {
    setFocus(new Set(ids));
  }, []);

  const clearGraph = useCallback(() => {
    setStore(empty());
    setFocus(new Set());
    setMode("context");
  }, []);

  const resetToSchema = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const schema = schemaCache.current ?? (await api.graphSchema());
      schemaCache.current = schema;
      // Schema → GraphUpdate. We keep schema:* ids as-is and map labels.
      const update: GraphUpdate = {
        nodes: schema.nodes.map((n) => ({
          id: n.id,
          label: n.count != null ? `${n.label} · ${n.count.toLocaleString()}` : n.label,
          type: n.type,
          properties: {
            kind: "schema",
            count: n.count,
            description: n.description,
          },
        })),
        relationships: schema.relationships.map((r) => ({
          id: r.id,
          source: r.source,
          target: r.target,
          type: r.type,
          properties: { description: r.description },
        })),
        focus_ids: [],
        mode: "replace",
      };
      applyUpdate(update, "schema", false);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setIsLoading(false);
    }
  }, [applyUpdate]);

  const loadContext = useCallback(async (ids: string[], m: "replace" | "merge" = "replace") => {
    if (ids.length === 0) return;
    setIsLoading(true);
    setError(null);
    try {
      const update = await api.graphContext(ids, m);
      applyUpdate(update, m === "merge" ? "expanded" : "context", m === "merge");
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setIsLoading(false);
    }
  }, [applyUpdate]);

  const loadAlert = useCallback(async (alertId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const update = await api.alertContext(alertId);
      applyUpdate(update, "context", false);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setIsLoading(false);
    }
  }, [applyUpdate]);

  const expandFromNode = useCallback(async (id: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const update = await api.graphContext([id], "merge");
      applyUpdate(update, "expanded", true);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setIsLoading(false);
    }
  }, [applyUpdate]);

  // Boot: schema view by default.
  useEffect(() => {
    void resetToSchema();
    // run-once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Projection to GraphResponse for the existing canvas ---------------

  const data: GraphResponse = useMemo(() => {
    const nodes: GraphNode[] = Array.from(store.nodes.values()).map(toCanvasNode);
    const linkKeys = new Set<string>();
    const links: GraphLink[] = [];
    const nodeIds = new Set(nodes.map((n) => n.id));
    for (const r of store.rels.values()) {
      if (!nodeIds.has(r.source) || !nodeIds.has(r.target)) continue;
      const key = `${r.source}->${r.target}:${r.type}`;
      if (linkKeys.has(key)) continue;
      linkKeys.add(key);
      links.push(toCanvasLink(r));
    }
    return { nodes, links, range: [] };
  }, [store]);

  return {
    mode,
    data,
    focus,
    isLoading,
    error,
    replaceGraph,
    mergeGraph,
    focusIds,
    clearGraph,
    resetToSchema,
    loadContext,
    loadAlert,
    expandFromNode,
  };
}

// ---- Mapping helpers -----------------------------------------------------

function toCanvasNode(n: GraphViewNode): GraphNode {
  const props = (n.properties ?? {}) as Record<string, unknown>;
  return {
    id: n.id,
    label: n.label,
    // GraphCanvas keys colors and radii off this. We pass the label
    // through; unknown labels fall back to a default in the canvas.
    type: n.type as GraphNode["type"],
    category: typeof props.category === "string" ? props.category : null,
    total_spend: typeof props.spend === "number" ? props.spend :
                  typeof props.total_spend === "number" ? props.total_spend : null,
    visits: typeof props.visits === "number" ? props.visits : null,
    income: typeof props.income === "number" ? props.income : null,
    expense: typeof props.expense === "number" ? props.expense : null,
  };
}

function toCanvasLink(r: GraphViewRel): GraphLink {
  const props = (r.properties ?? {}) as Record<string, unknown>;
  return {
    source: r.source,
    target: r.target,
    type: r.type as GraphLink["type"],
    weight: typeof props.spend === "number" ? props.spend :
            typeof props.weight === "number" ? props.weight : null,
    visits: typeof props.visits === "number" ? props.visits : null,
  };
}

// Helper: derive a node-id list for a chat highlight payload.
// The agent's graph_highlight events already send canonical ids; we
// just guard against bad/empty strings.
export function sanitizeNodeIds(ids: unknown): string[] {
  if (!Array.isArray(ids)) return [];
  const out: string[] = [];
  for (const id of ids) {
    if (typeof id !== "string") continue;
    const trimmed = id.trim();
    if (!trimmed || trimmed.indexOf(":") === -1) continue;
    out.push(trimmed);
  }
  return out;
}
