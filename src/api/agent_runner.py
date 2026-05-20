"""Streaming agent loop — planner + scoped subagents + synthesizer.

Design
------
1. **Planner** (cheap, non-streaming): looks at the question and picks one or
   more subagents to run, with a short brief for each. The model output is
   strict JSON so the UI can show the plan before any expensive work fires.

2. **Subagents** (streaming, tool-using): each has its own system prompt and
   a *subset* of the available tools — keeps prompts focused, prevents the
   ``analyst`` from poking at wiki pages it doesn't need.

3. **Synthesizer** (streaming, no tools): re-reads the subagent outputs and
   writes one concise final answer for the user.

Every stage emits SSE-friendly ``(event, data)`` tuples. The vocabulary is
backwards-compatible with Phase 2 — we only *added* events:

  started · plan · subagent_start · subagent_end ·
  tool_call · tool_result · graph_highlight · token · result · done · error

``token`` events carry an optional ``subagent`` field so the UI can route
them into the right swimlane.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from src.api.deps import get_driver, get_wiki_root
from src.api.models import GraphViewNode, GraphViewRel
from src.ontology import load_ontology

log = logging.getLogger(__name__)

# OpenAI calls that don't return within this many seconds are aborted so the
# SSE stream surfaces an error rather than hanging the browser.
OPENAI_TIMEOUT_SECS = float(os.getenv("FCE_OPENAI_TIMEOUT_SECS", "60"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").removeprefix("openai:")
# Optional OpenAI-compatible base URL. Set this to point at Google's Gemini
# OpenAI-compat endpoint (or any other compatible provider). The SDK still
# uses the OPENAI_API_KEY env var — for Gemini, that's a Google API key.
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
MAX_TURNS_PER_SUBAGENT = 4
TOOL_RESULT_PREVIEW_CHARS = 600


# --------------------------------------------------------------------------
# Ontology-derived prompt fragment
# --------------------------------------------------------------------------
#
# We compile a compact entity/property/relationship summary from the YAML
# ontology and stitch it into the analyst's prompt. This is what stops the
# model inventing properties — adding a new field to the ontology
# automatically advertises it here, and removing one stops the model
# referencing it in Cypher.


def _ontology_prompt_fragment() -> str:
    """Compact, model-readable summary of the ontology.

    Format is deliberately terse so we don't spend tokens on prose. Skips
    properties used only by background ML (FastRP embedding, etc.) and
    every PII-flagged field.
    """
    try:
        o = load_ontology()
    except Exception as exc:
        log.warning("ontology unavailable for analyst prompt: %s", exc)
        return ""

    skip_props = {"embedding", "desc_embedding", "aliases"}
    lines: list[str] = ["Graph schema (from data/ontology/finance.yaml):"]
    for entity in o.entities:
        props: list[str] = []
        for p in entity.properties:
            if p.pii or p.name in skip_props or p.vector:
                continue
            props.append(p.name)
        lines.append(f"  ({entity.name}): {', '.join(props) or '(no public props)'}")

    lines.append("Relationships:")
    for r in o.relationships:
        lines.append(f"  ({r.source})-[:{r.type}]->({r.target})")
    return "\n".join(lines)


_ONTOLOGY_FRAGMENT = _ontology_prompt_fragment()


# --------------------------------------------------------------------------
# Subagent registry — prompt + tool subset
# --------------------------------------------------------------------------

ANALYST_PROMPT = f"""You are the ANALYST subagent. Your job is to compute
aggregations, trends and rankings against the Neo4j context graph. Use the
graph_query tool with Cypher.

{_ONTOLOGY_FRAGMENT}

Useful conventions:
* Transaction.amount is signed — negative = expense, positive = income.
* Transaction.month is 'YYYY-MM' for fast filtering; year is int.
* fraud_score is a per-tx float; risk_flags is a list of short tags.
* For fraud or alert questions, prefer the fraud_alerts tool over Cypher.
* Do not invent properties or labels. The list above is exhaustive.

Reply with the numbers, sources (tool or Cypher), and a one-sentence
interpretation.
"""

WIKI_BROWSER_PROMPT = """You are the WIKI_BROWSER subagent. Pull facts from
the pre-aggregated Obsidian-style wiki vault before doing live graph work.
Use wiki_search to find pages then wiki_read to load them. Quote relevant
numbers and links; don't synthesise advice."""

ADVISOR_PROMPT = """You are the ADVISOR subagent. Produce 1–3 grounded,
actionable savings recommendations based on the analyst's findings + the
wiki context the orchestrator gives you.

For EACH recommendation you propose, call the ``propose_saving`` tool with
the concrete merchant, the action, and the monthly saving — this lets the
UI render a "ghost node" on the graph next to that merchant so the user
sees the suggestion in context. After the tool calls, write a short prose
summary citing the same figures. Stay short and practical."""


SUBAGENTS: dict[str, dict] = {
    "analyst":      {"prompt": ANALYST_PROMPT,      "tools": {"graph_query", "fraud_alerts"}},
    "wiki_browser": {"prompt": WIKI_BROWSER_PROMPT, "tools": {"wiki_search", "wiki_read"}},
    "advisor":      {"prompt": ADVISOR_PROMPT,      "tools": {"wiki_read", "propose_saving"}},
}


# --------------------------------------------------------------------------
# Tool schemas (kept as one dict; we filter per-subagent below)
# --------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict] = {
    "graph_query": {
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": "Run a read-only Cypher query against Neo4j and return rows as JSON.",
            "parameters": {
                "type": "object",
                "properties": {"cypher": {"type": "string"}},
                "required": ["cypher"],
            },
        },
    },
    "wiki_search": {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Substring search the wiki vault. Returns up to 20 hits.",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    },
    "fraud_alerts": {
        "type": "function",
        "function": {
            "name": "fraud_alerts",
            "description": (
                "Return the highest fraud/anomaly alerts from the graph. "
                "Use this for questions about fraud alerts, suspicious "
                "transactions, highest risk, alert triage, or anomaly review."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "string",
                        "description": "Optional month filter in YYYY-MM format.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum alerts to return. Defaults to 10.",
                    },
                },
            },
        },
    },
    "wiki_read": {
        "type": "function",
        "function": {
            "name": "wiki_read",
            "description": "Read a wiki page by section + name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string",
                                "enum": ["merchants", "categories", "months", "annual"]},
                    "name": {"type": "string"},
                },
                "required": ["section", "name"],
            },
        },
    },
    "propose_saving": {
        "type": "function",
        "function": {
            "name": "propose_saving",
            "description": ("Emit ONE concrete savings recommendation tied to a merchant. "
                            "The UI renders this as a ghost node on the graph."),
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant":       {"type": "string",
                                       "description": "Canonical merchant name."},
                    "monthly_saving": {"type": "number",
                                       "description": "Estimated £ saved per month."},
                    "action":         {"type": "string",
                                       "description": "What the user should do — keep it short."},
                },
                "required": ["merchant", "monthly_saving", "action"],
            },
        },
    },
}


# --------------------------------------------------------------------------
# Tool implementations (unchanged from Phase 2)
# --------------------------------------------------------------------------

_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH|CALL\s+apoc\.)\b", re.IGNORECASE,
)


def _tool_graph_query(cypher: str) -> dict:
    if _WRITE_RE.search(cypher):
        log.warning("graph_query rejected (write op): %s", cypher[:140].replace("\n", " "))
        return {"error": "Only read-only queries are allowed.", "rows": []}
    driver = get_driver()
    t0 = time.perf_counter()
    try:
        with driver.session() as s:
            rows = s.run(cypher).data()
        dur_ms = (time.perf_counter() - t0) * 1000
        log.info("graph_query → %d rows in %.0fms · cypher=%s",
                 len(rows), dur_ms, cypher[:160].replace("\n", " "))
        return {"rows": rows[:50], "truncated": len(rows) > 50, "total": len(rows)}
    except Exception as exc:
        log.warning("graph_query FAILED in %.0fms: %s · cypher=%s",
                    (time.perf_counter() - t0) * 1000, exc, cypher[:160].replace("\n", " "))
        return {"error": f"{exc.__class__.__name__}: {exc}", "rows": []}


def _tool_wiki_search(q: str) -> dict:
    needle = q.strip().lower()
    if not needle:
        return {"results": []}
    root = get_wiki_root()
    t0 = time.perf_counter()
    hits: list[dict] = []
    for md_path in root.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        if needle not in text.lower():
            continue
        section = md_path.parent.name if md_path.parent != root else "root"
        hits.append({"section": section, "name": md_path.stem})
        if len(hits) >= 20:
            break
    log.info("wiki_search %r → %d hits in %.0fms",
             q, len(hits), (time.perf_counter() - t0) * 1000)
    return {"results": hits}


def _tool_wiki_read(section: str, name: str) -> dict:
    root = get_wiki_root()
    md_path = root / section / f"{name}.md"
    if not md_path.exists():
        log.info("wiki_read MISS %s/%s", section, name)
        return {"error": f"{section}/{name} not found"}
    body = md_path.read_text(encoding="utf-8")
    log.info("wiki_read %s/%s → %d chars", section, name, len(body))
    return {"markdown": body}


def _tool_propose_saving(args: dict) -> dict:
    """No-op tool — the value lives in the SSE event the runner emits."""
    return {
        "ok": True,
        "merchant": str(args.get("merchant", "")),
        "monthly_saving": float(args.get("monthly_saving", 0) or 0),
        "action": str(args.get("action", "")),
    }


def _tool_fraud_alerts(args: dict) -> dict:
    month = str(args.get("month") or "").strip() or None
    limit = int(args.get("limit") or 10)
    limit = max(1, min(limit, 50))
    query = """
    MATCH (a:Alert)-[:FLAGS]->(t:Transaction)-[:AT]->(m:Merchant)
    OPTIONAL MATCH (t)-[:AT_LOCATION]->(l:Location)
    WHERE $month IS NULL OR t.month = $month
    WITH t, m, l, collect(a) AS alerts
    WITH t, m, l, alerts,
         reduce(best = alerts[0], a IN alerts |
           CASE WHEN coalesce(a.severity, 0) > coalesce(best.severity, 0) THEN a ELSE best END
         ) AS a
    RETURN a.id AS alert_id,
           a.kind AS kind,
           a.severity AS severity,
           a.rationale AS rationale,
           t.id AS tx_id,
           t.fraud_score AS fraud_score,
           coalesce(t.risk_flags, []) AS risk_flags,
           t.amount AS amount,
           toString(t.date) AS date,
           t.month AS month,
           t.description AS description,
           m.canonical_name AS merchant,
           coalesce(l.name + ' (' + l.country + ')', null) AS location
    ORDER BY t.fraud_score DESC, abs(t.amount) DESC, t.date DESC
    LIMIT $limit
    """
    t0 = time.perf_counter()
    try:
        with get_driver().session() as s:
            rows = s.run(query, month=month, limit=limit).data()
        log.info("fraud_alerts → %d rows in %.0fms · month=%s limit=%d",
                 len(rows), (time.perf_counter() - t0) * 1000, month, limit)
        return {"rows": rows, "total": len(rows), "month": month}
    except Exception as exc:
        log.warning("fraud_alerts FAILED in %.0fms: %s",
                    (time.perf_counter() - t0) * 1000, exc)
        return {"error": f"{exc.__class__.__name__}: {exc}", "rows": []}


def _run_tool(name: str, args: dict) -> dict:
    if name == "graph_query":     return _tool_graph_query(str(args.get("cypher", "")))
    if name == "fraud_alerts":    return _tool_fraud_alerts(args)
    if name == "wiki_search":     return _tool_wiki_search(str(args.get("q", "")))
    if name == "wiki_read":       return _tool_wiki_read(str(args.get("section", "")),
                                                         str(args.get("name", "")))
    if name == "propose_saving":  return _tool_propose_saving(args)
    return {"error": f"unknown tool {name!r}"}


# --------------------------------------------------------------------------
# Graph-highlight derivation
# --------------------------------------------------------------------------

_MERCHANT_KEYS = {"canonical_name", "merchant", "name"}
_CATEGORY_KEYS = {"category", "category_name"}
_MONTH_KEYS = {"month"}
_MONTH_VALUE_RE = re.compile(r"\b\d{4}-\d{2}\b")
_QUOTED_VALUE_RE = re.compile(r"""['"]([^'"]{3,80})['"]""")
_IGNORED_CONTEXT_TERMS = {
    "active_in",
    "category",
    "canonical_name",
    "description",
    "expense",
    "merchant",
    "month",
    "name",
    "transaction",
}


def _highlights_from_rows(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows[:25]:
        for k, v in row.items():
            if not isinstance(v, str):
                continue
            if k in _MERCHANT_KEYS:
                node_id = f"merchant:{v}"
            elif k in _CATEGORY_KEYS:
                node_id = f"category:{v}"
            elif k in _MONTH_KEYS and re.match(r"^\d{4}-\d{2}$", v):
                node_id = f"month:{v}"
            else:
                continue
            if node_id not in seen:
                seen.add(node_id)
                ids.append(node_id)
    return ids


def _dedupe_ids(ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for node_id in ids:
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


def _context_terms_from_cypher(cypher: str) -> tuple[list[str], list[str]]:
    """Extract likely entity filters from Cypher for graph visualization.

    Analyst queries often return only numeric aggregates. In that case the
    rows do not contain node ids, but the Cypher usually still contains the
    user's concrete context, for example ``t.month = '2026-04'`` and
    ``toLower(c.name) CONTAINS 'coffee'``. This mirrors the context-graph
    pattern: use the retrieved context to drive the visible subgraph.
    """
    months = sorted(set(_MONTH_VALUE_RE.findall(cypher)))
    terms: list[str] = []
    for raw in _QUOTED_VALUE_RE.findall(cypher):
        term = raw.strip().lower()
        if (
            term in months
            or term in _IGNORED_CONTEXT_TERMS
            or _MONTH_VALUE_RE.fullmatch(term)
            or len(term) < 3
        ):
            continue
        terms.append(term)
    return _dedupe_ids(terms), months


def _highlights_from_graph_query(cypher: str, rows: list[dict]) -> list[str]:
    ids = _highlights_from_rows(rows)
    terms, months = _context_terms_from_cypher(cypher)
    ids.extend(f"month:{month}" for month in months)
    if not terms:
        return _dedupe_ids(ids)

    # If the LLM returned only a scalar aggregate, recover the graph context
    # from the query predicates: matching merchants/categories/descriptions,
    # optionally constrained to the explicit months in the Cypher.
    try:
        with get_driver().session() as s:
            lookup_rows = s.run(
                """
                MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category)
                WHERE t.amount < 0
                  AND (size($months) = 0 OR t.month IN $months)
                  AND any(term IN $terms WHERE
                    toLower(m.canonical_name) CONTAINS term OR
                    toLower(c.name) CONTAINS term OR
                    toLower(t.description) CONTAINS term)
                WITH m, c, t.month AS month, sum(-t.amount) AS spend
                ORDER BY spend DESC
                RETURN collect(DISTINCT m.canonical_name)[0..8] AS merchants,
                       collect(DISTINCT c.name)[0..8] AS categories,
                       collect(DISTINCT month)[0..8] AS months
                """,
                terms=terms,
                months=months,
            ).single()
    except Exception as exc:
        log.info("graph highlight context lookup skipped: %s", exc)
        return _dedupe_ids(ids)

    if lookup_rows:
        ids.extend(f"merchant:{name}" for name in (lookup_rows["merchants"] or []) if name)
        ids.extend(f"category:{name}" for name in (lookup_rows["categories"] or []) if name)
        ids.extend(f"month:{month}" for month in (lookup_rows["months"] or []) if month)
    return _dedupe_ids(ids)


def _highlights_from_alert_rows(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in rows[:25]:
        merchant = row.get("merchant")
        month = row.get("month")
        if isinstance(merchant, str) and merchant:
            ids.append(f"merchant:{merchant}")
        if isinstance(month, str) and _MONTH_VALUE_RE.fullmatch(month):
            ids.append(f"month:{month}")
    return _dedupe_ids(ids)


def _graph_update_payload(node_ids: list[str]) -> dict | None:
    """Resolve highlighted node ids → an actual GraphUpdate the canvas can apply.

    Mirrors GET /api/graph/context but stays in-process so the SSE stream
    doesn't have to round-trip through HTTP. Returns ``None`` if nothing
    resolves — the caller should just skip emitting ``graph_update``.
    """
    from src.api.routes.graph import _context_for_id  # local import → no cycle

    if not node_ids:
        return None
    nodes: dict[str, GraphViewNode] = {}
    rels:  dict[str, GraphViewRel]  = {}
    try:
        driver = get_driver()
        for node_id in node_ids:
            try:
                _context_for_id(driver, node_id, nodes, rels)
            except Exception as exc:
                log.info("graph_update: context for %s failed: %s", node_id, exc)
    except Exception as exc:
        log.info("graph_update: driver unavailable: %s", exc)
        return None

    if not nodes:
        return None
    return {
        "nodes":         [n.model_dump() for n in nodes.values()],
        "relationships": [r.model_dump() for r in rels.values()],
        "focus_ids":     node_ids,
        "mode":          "merge",
    }


def _highlights_from_wiki_hits(hits: list[dict]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for h in hits:
        section, name = h.get("section"), h.get("name")
        if not name:
            continue
        key = {"merchants": "merchant", "categories": "category", "months": "month"}.get(section)
        if not key:
            continue
        node_id = f"{key}:{name}"
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

PLANNER_PROMPT = """You are the orchestrator. Given a user question about
their personal finance context, pick which subagents to run and brief each
one. Output STRICT JSON of the shape:

{
  "plan": [
    {"subagent": "analyst" | "wiki_browser" | "advisor", "brief": "<short instruction>"}
  ]
}

Heuristics:
* Numeric / ranking / trend questions → analyst.
* "What is X?" or "show me X's page" → wiki_browser.
* "How can I save / cut costs / what should I do?" → analyst then advisor.
* Most questions need just one subagent; some chain two; rarely three.
* Order matters — earlier subagents' findings inform later ones."""


async def _plan(client: Any, question: str) -> list[dict]:
    """Ask the model for a strict-JSON plan. On failure, fall back to analyst-only."""
    t0 = time.perf_counter()
    log.info("plan: requesting plan from %s for %r", MODEL, question[:80])
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0.1,
            timeout=OPENAI_TIMEOUT_SECS,
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        steps = payload.get("plan", [])
        valid = [
            {"subagent": s.get("subagent"), "brief": str(s.get("brief", "")).strip()}
            for s in steps
            if s.get("subagent") in SUBAGENTS and s.get("brief")
        ]
        result = valid or [{"subagent": "analyst", "brief": question}]
        log.info("plan: %d step(s) in %.0fms — %s",
                 len(result), (time.perf_counter() - t0) * 1000,
                 [s["subagent"] for s in result])
        return result
    except Exception as exc:
        log.warning("plan FAILED after %.0fms (%s); falling back to analyst-only",
                    (time.perf_counter() - t0) * 1000, exc)
        return [{"subagent": "analyst", "brief": question}]


# --------------------------------------------------------------------------
# Subagent loop — streams a single subagent's run and yields events
# --------------------------------------------------------------------------

async def _run_subagent(
    client: Any,
    *,
    name: str,
    brief: str,
    context: str,
) -> AsyncIterator[tuple[str, dict, str]]:
    """Drive one subagent. Yields (event, data, output_text_so_far)."""
    cfg = SUBAGENTS[name]
    tools = [TOOL_SCHEMAS[t] for t in cfg["tools"]]

    user_msg = brief if not context else (
        f"{brief}\n\n--- prior subagent findings ---\n{context.strip()}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": cfg["prompt"]},
        {"role": "user",   "content": user_msg},
    ]

    final_text = ""
    log.info("subagent %s: starting (tools=%s, brief=%r)",
             name, sorted(cfg["tools"]), brief[:80])
    subagent_t0 = time.perf_counter()

    for turn_idx in range(MAX_TURNS_PER_SUBAGENT):
        turn_t0 = time.perf_counter()
        log.debug("subagent %s turn %d: opening stream", name, turn_idx + 1)
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools or None,
            tool_choice="auto" if tools else "none",
            stream=True,
            temperature=0.2,
            timeout=OPENAI_TIMEOUT_SECS,
        )

        chunk_text = ""
        tool_calls: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice or not choice.delta:
                continue
            delta = choice.delta

            if delta.content:
                chunk_text += delta.content
                yield "token", {"text": delta.content, "subagent": name}, final_text + chunk_text

            for tc in delta.tool_calls or []:
                slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:                          slot["id"] = tc.id
                if tc.function and tc.function.name:      slot["name"] = tc.function.name
                if tc.function and tc.function.arguments: slot["args"] += tc.function.arguments

        final_text += chunk_text

        if not tool_calls:
            log.info("subagent %s: done after %d turn(s), %d chars, %.1fs",
                     name, turn_idx + 1, len(final_text),
                     time.perf_counter() - subagent_t0)
            return  # subagent finished

        log.info("subagent %s turn %d: %d tool call(s) in %.1fs",
                 name, turn_idx + 1, len(tool_calls),
                 time.perf_counter() - turn_t0)

        messages.append({
            "role": "assistant",
            "content": chunk_text or None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["args"] or "{}"},
                }
                for call in tool_calls.values()
            ],
        })

        for call in tool_calls.values():
            try:
                args = json.loads(call["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield "tool_call", {"name": call["name"], "args": args, "subagent": name}, final_text
            result = _run_tool(call["name"], args)

            highlights: list[str] = []
            if call["name"] == "graph_query":
                highlights = _highlights_from_graph_query(
                    str(args.get("cypher", "")),
                    result.get("rows") or [],
                )
            elif call["name"] == "fraud_alerts":
                highlights = _highlights_from_alert_rows(result.get("rows") or [])
            elif call["name"] == "wiki_search":
                highlights = _highlights_from_wiki_hits(result.get("results") or [])
            elif call["name"] == "wiki_read":
                key = {"merchants": "merchant", "categories": "category",
                       "months": "month"}.get(str(args.get("section")))
                if key:
                    highlights = [f"{key}:{args.get('name')}"]

            preview = json.dumps(result, default=str)[:TOOL_RESULT_PREVIEW_CHARS]
            yield "tool_result", {"name": call["name"], "preview": preview, "subagent": name}, final_text
            if highlights:
                yield "graph_highlight", {"node_ids": highlights, "subagent": name}, final_text
                # Resolve highlighted ids → actual subgraph for the canvas.
                # We MERGE so the canvas accumulates context across tool
                # calls inside one turn rather than flicker between slices.
                update = _graph_update_payload(highlights)
                if update is not None:
                    yield "graph_update", {**update, "subagent": name}, final_text

            # Forecast ghosts — promote the advisor's propose_saving calls
            # into a dedicated SSE event the canvas reads directly.
            if call["name"] == "propose_saving" and not result.get("error"):
                yield "forecast_ghost", {
                    "merchant":       result["merchant"],
                    "monthly_saving": result["monthly_saving"],
                    "action":         result["action"],
                    "subagent":       name,
                }, final_text

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, default=str),
            })

    log.warning("subagent %s: HIT MAX_TURNS (%d) after %.1fs",
                name, MAX_TURNS_PER_SUBAGENT,
                time.perf_counter() - subagent_t0)


# --------------------------------------------------------------------------
# Synthesizer
# --------------------------------------------------------------------------

SYNTHESIZER_PROMPT = """You are the final synthesizer. The user asked a
question; subagents gathered the relevant facts. Write ONE concise answer
for the user — 2-4 sentences plus a small table if helpful. Use £ for all
amounts. Don't recap the subagents' process; just answer."""


async def _synthesize(
    client: Any, question: str, subagent_outputs: list[dict],
) -> AsyncIterator[tuple[str, dict]]:
    context = "\n\n".join(
        f"### {o['subagent']}\n{o['text']}".strip()
        for o in subagent_outputs if o.get("text")
    )
    user = f"Question:\n{question}\n\nFindings:\n{context}"
    log.info("synthesizer: opening stream over %d findings (ctx=%d chars)",
             len(subagent_outputs), len(context))
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user",   "content": user},
        ],
        stream=True,
        temperature=0.2,
        timeout=OPENAI_TIMEOUT_SECS,
    )
    final = ""
    async for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice or not choice.delta or not choice.delta.content:
            continue
        delta = choice.delta.content
        final += delta
        yield "token", {"text": delta, "subagent": "synthesizer"}
    log.info("synthesizer: done in %.1fs, %d chars",
             time.perf_counter() - t0, len(final))
    yield "result", {"answer": final}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

async def run_agent_stream(
    question: str,
    *,
    history: list[dict] | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Plan → subagents → synthesize, yielding SSE-friendly events.

    ``history`` is an optional list of prior {role, content} turns used to
    give the planner conversational context (not the tool history)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield "error", {"message": "OPENAI_API_KEY missing"}
        return
    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield "error", {"message": "openai SDK not installed"}
        return

    overall_t0 = time.perf_counter()
    log.info("=== ASK: %r (history=%d turns) === model=%s base_url=%s",
             question[:80], len(history or []),
             MODEL, LLM_BASE_URL or "default")
    client = AsyncOpenAI(
        api_key=api_key,
        **({"base_url": LLM_BASE_URL} if LLM_BASE_URL else {}),
    )
    yield "started", {"question": question}

    # ----- Plan ----------------------------------------------------------
    planner_input = question
    if history:
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
        planner_input = f"Recent conversation:\n{recent}\n\nNew question:\n{question}"

    plan = await _plan(client, planner_input)
    yield "plan", {"steps": plan}

    # ----- Run subagents in sequence, threading findings forward ---------
    subagent_outputs: list[dict] = []
    context = ""
    for step in plan:
        name = step["subagent"]
        brief = step["brief"]
        yield "subagent_start", {"name": name, "brief": brief}
        text_so_far = ""
        try:
            async for event, data, snapshot in _run_subagent(
                client, name=name, brief=brief, context=context,
            ):
                text_so_far = snapshot
                yield event, data
        except Exception as exc:
            yield "error", {"message": f"{name}: {exc.__class__.__name__}: {exc}"}
            yield "subagent_end", {"name": name, "ok": False}
            return
        subagent_outputs.append({"subagent": name, "text": text_so_far})
        context = (
            f"{context}\n\n### {name}\n{text_so_far}".strip()
            if text_so_far else context
        )
        yield "subagent_end", {"name": name, "ok": True}

    # ----- Synthesize final answer ---------------------------------------
    yield "subagent_start", {"name": "synthesizer", "brief": "compose final answer"}
    try:
        async for event, data in _synthesize(client, question, subagent_outputs):
            yield event, data
    except Exception as exc:
        log.exception("synthesizer failed")
        yield "error", {"message": f"synthesizer: {exc.__class__.__name__}: {exc}"}
    yield "subagent_end", {"name": "synthesizer", "ok": True}
    log.info("=== DONE %r in %.1fs ===", question[:80],
             time.perf_counter() - overall_t0)
    yield "done", {}
