"""Graph slicing for the force-graph canvas.

The endpoint emits an aggregated graph (Merchant ↔ Category ↔ Month) rather
than the raw 1,500-transaction edge cloud — that's the shape react-force-graph
can render fluidly. Edges are weighted by total spend in the requested window
so the force layout naturally clusters frequently-used merchants near their
busiest months.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import Driver

from src.api.deps import get_driver
from src.api.models import (
    GraphLink,
    GraphNode,
    GraphResponse,
    GraphUpdate,
    GraphViewNode,
    GraphViewRel,
    SchemaNode,
    SchemaRel,
    SchemaResponse,
)
from src.ontology import load_ontology

router = APIRouter(prefix="/graph", tags=["graph"])

_RANGE_RE = re.compile(r"^\d{4}-\d{2}:\d{4}-\d{2}$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


# ---------- Context-graph helpers ----------------------------------------

# Map raw Neo4j node labels onto the prefixed ids the canvas uses.
# Order matters — the first label we recognise on a node wins.
_LABEL_TO_KIND: dict[str, str] = {
    "Merchant":    "merchant",
    "Category":    "category",
    "Month":       "month",
    "Transaction": "transaction",
    "Alert":       "alert",
    "Statement":   "statement",
    "Account":     "account",
    "Day":         "day",
    "Location":    "location",
    "Decision":    "decision",
}

# "annual:2025" isn't a single graph node — the year is a property on each
# Month. We resolve it to that year's months so the canvas shows the full
# year's shape (12 month nodes + their merchant/category neighbourhoods).
_YEAR_CONTEXT = """
MATCH (mo:Month) WHERE mo.year = $year
OPTIONAL MATCH (t:Transaction)-[im:IN_MONTH]->(mo)
WITH mo, t, im
ORDER BY abs(t.amount) DESC
WITH mo, collect(t)[..6] AS topTxs, collect(im)[..6] AS topIms
UNWIND range(0, size(topTxs) - 1) AS i
WITH mo, topTxs[i] AS t, topIms[i] AS im
OPTIONAL MATCH (t)-[at:AT]->(m:Merchant)
OPTIONAL MATCH (m)-[ic:IN_CATEGORY]->(c:Category)
RETURN mo, t, im, m, at, c, ic
"""

def _node_id(record: dict, label: str) -> str | None:
    kind = _LABEL_TO_KIND.get(label)
    if not kind:
        return None
    # Per-label primary key. We mirror the prefixed ids the rest of the
    # codebase already uses ("merchant:Tesco", "month:2025-04", …).
    if label == "Merchant":
        name = record.get("canonical_name") or record.get("name")
        return f"merchant:{name}" if name else None
    if label == "Category":
        name = record.get("name")
        return f"category:{name}" if name else None
    if label == "Month":
        m = record.get("id")
        return f"month:{m}" if m else None
    if label == "Day":
        d = record.get("id")
        return f"day:{d}" if d else None
    if label == "Location":
        loc_id = record.get("id")
        return f"location:{loc_id}" if loc_id else None
    if label == "Alert":
        a_id = record.get("id")
        return f"alert:{a_id}" if a_id else None
    if label == "Transaction":
        t_id = record.get("id")
        return f"transaction:{t_id}" if t_id else None
    if label == "Statement":
        s_id = record.get("id")
        return f"statement:{s_id}" if s_id else None
    if label == "Account":
        a_id = record.get("id")
        return f"account:{a_id}" if a_id else None
    return None


def _node_label(record: dict, label: str) -> str:
    # The short, human-readable text on the canvas. Falls back to id.
    if label == "Merchant":    return str(record.get("canonical_name") or record.get("name") or "merchant")
    if label == "Category":    return str(record.get("name") or "category")
    if label == "Month":       return str(record.get("id") or "month")
    if label == "Transaction":
        desc = (record.get("description") or "").strip()
        return desc[:40] or str(record.get("id"))
    if label == "Alert":       return str(record.get("kind") or "alert")
    if label == "Statement":   return str(record.get("id") or "statement")
    if label == "Account":     return str(record.get("id") or "account")
    if label == "Day":         return str(record.get("id") or "day")
    if label == "Location":
        name = record.get("name") or record.get("id")
        country = record.get("country")
        return f"{name} ({country})" if name and country else str(name)
    return str(record.get("name") or record.get("id") or label.lower())


def _parse_node_id(node_id: str) -> tuple[str, str]:
    kind, _, name = node_id.partition(":")
    return kind, name


# ---------- Schema view ---------------------------------------------------

# A pre-baked "this is what's in the graph" overview. We don't query for it
# every request — the topology is stable and the per-label counts only need
# to be ballpark accurate. Counts come from a single fast aggregation query.
#
# Schema topology is read from the ontology YAML (data/ontology/finance.yaml)
# so adding a new entity / relationship there makes it appear here too. The
# only hardcoded extra is Decision — it's a runtime artifact that the agent
# writes during execution rather than an ontology entity.

SCHEMA_COUNTS_QUERY = """
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS c
"""

# Runtime-only labels that aren't in the ontology but exist in the graph
# (the agent writes Decision nodes during execution).
_RUNTIME_NODE_TYPES: list[tuple[str, str | None]] = [
    ("Decision", "An agent decision trace — the question + the nodes it touched."),
]
_RUNTIME_RELS: list[tuple[str, str, str, str]] = [
    ("Decision", "Merchant", "TOUCHED", "agent decision touched this node"),
    ("Decision", "Category", "TOUCHED", "agent decision touched this node"),
    ("Decision", "Month",    "TOUCHED", "agent decision touched this node"),
]


@router.get("/schema", response_model=SchemaResponse)
def get_schema(driver: Driver = Depends(get_driver)) -> SchemaResponse:
    """High-level "what kinds of things are in the graph" view.

    Returns the type/type relationship topology so the canvas can start
    with a tidy 6–8 node overview instead of dumping 1500 transactions
    into a force layout. Driven by the ontology YAML.
    """
    counts: dict[str, int] = {}
    try:
        with driver.session() as s:
            for row in s.run(SCHEMA_COUNTS_QUERY).data():
                if row.get("label"):
                    counts[row["label"]] = int(row["c"] or 0)
    except Exception:
        # The schema view should never fail the boot. Empty counts are
        # acceptable — the canvas still shows the topology.
        counts = {}

    ontology = load_ontology()
    nodes: list[SchemaNode] = [
        SchemaNode(
            id=f"schema:{entity.name}",
            label=entity.name,
            type=entity.name,
            count=counts.get(entity.name),
            description=entity.description or None,
        )
        for entity in ontology.entities
    ]
    nodes.extend(
        SchemaNode(
            id=f"schema:{name}",
            label=name,
            type=name,
            count=counts.get(name),
            description=desc,
        )
        for name, desc in _RUNTIME_NODE_TYPES
    )

    rels: list[SchemaRel] = [
        SchemaRel(
            id=f"schema-rel:{r.source}-{r.type}-{r.target}",
            source=f"schema:{r.source}",
            target=f"schema:{r.target}",
            type=r.type,
            description=r.description or None,
        )
        for r in ontology.relationships
    ]
    rels.extend(
        SchemaRel(
            id=f"schema-rel:{src}-{rtype}-{tgt}",
            source=f"schema:{src}",
            target=f"schema:{tgt}",
            type=rtype,
            description=desc,
        )
        for src, tgt, rtype, desc in _RUNTIME_RELS
    )
    return SchemaResponse(nodes=nodes, relationships=rels)


# ---------- Context (one-hop neighborhood for an id-set) -----------------

CONTEXT_MERCHANT_HEAD = """
MATCH (m:Merchant {canonical_name: $name})
OPTIONAL MATCH (m)-[ic:IN_CATEGORY]->(c:Category)
RETURN m, c, ic
"""

CONTEXT_MERCHANT_TXS = """
MATCH (t:Transaction)-[at:AT]->(m:Merchant {canonical_name: $name})
WITH t, at ORDER BY t.date DESC LIMIT 15
RETURN t, at
"""

CONTEXT_CATEGORY_PAIRS = """
MATCH (c:Category {name: $name})
OPTIONAL MATCH (m:Merchant)-[ic:IN_CATEGORY]->(c)
RETURN c, m, ic
"""

CONTEXT_MONTH_TXS = """
MATCH (t:Transaction)-[im:IN_MONTH]->(mo:Month {id: $name})
WITH mo, t, im
ORDER BY abs(t.amount) DESC LIMIT 20
OPTIONAL MATCH (t)-[at:AT]->(m:Merchant)
OPTIONAL MATCH (m)-[ic:IN_CATEGORY]->(c:Category)
RETURN mo, t, im, m, at, c, ic
"""

CONTEXT_TRANSACTION = """
MATCH (t:Transaction {id: $name})
OPTIONAL MATCH (t)-[at:AT]->(m:Merchant)
OPTIONAL MATCH (m)-[ic:IN_CATEGORY]->(c:Category)
OPTIONAL MATCH (t)-[im:IN_MONTH]->(mo:Month)
OPTIONAL MATCH (a:Alert)-[fl:FLAGS]->(t)
OPTIONAL MATCH (t)-[atl:AT_LOCATION]->(l:Location)
OPTIONAL MATCH (s:Statement)-[cn:CONTAINS]->(t)
RETURN t, m, c, mo, a, l, s, at, ic, im, fl, atl, cn
"""

CONTEXT_ALERT_BY_ID = """
MATCH (a:Alert {id: $name})-[fl:FLAGS]->(t:Transaction)
OPTIONAL MATCH (t)-[at:AT]->(m:Merchant)
OPTIONAL MATCH (m)-[ic:IN_CATEGORY]->(c:Category)
OPTIONAL MATCH (t)-[im:IN_MONTH]->(mo:Month)
OPTIONAL MATCH (t)-[atl:AT_LOCATION]->(l:Location)
RETURN a, t, m, c, mo, l, fl, at, ic, im, atl
"""


def _row_to_props(row: Any) -> dict[str, Any]:
    """Turn a Neo4j node into a plain dict, coercing date/datetime to str."""
    if row is None:
        return {}
    props: dict[str, Any] = {}
    try:
        items = dict(row).items()
    except TypeError:
        # Could already be a dict
        items = (row or {}).items()
    for k, v in items:
        if hasattr(v, "iso_format"):
            props[k] = v.iso_format()
        elif hasattr(v, "isoformat"):
            props[k] = v.isoformat()
        else:
            props[k] = v
    return props


def _first_label(node: Any) -> str | None:
    if node is None:
        return None
    labels = getattr(node, "labels", None)
    if labels:
        for label in labels:
            if label in _LABEL_TO_KIND:
                return label
        for label in labels:
            return label
    return None


def _view_node(node: Any) -> GraphViewNode | None:
    if node is None:
        return None
    label = _first_label(node)
    if not label:
        return None
    props = _row_to_props(node)
    node_id = _node_id(props, label)
    if not node_id:
        return None
    return GraphViewNode(
        id=node_id,
        label=_node_label(props, label),
        type=label,
        properties=props,
    )


def _view_rel(rel: Any, source_id: str, target_id: str) -> GraphViewRel | None:
    if rel is None or not source_id or not target_id:
        return None
    rtype = getattr(rel, "type", None) or "RELATED"
    rel_id = f"{source_id}|{rtype}|{target_id}"
    return GraphViewRel(
        id=rel_id,
        source=source_id,
        target=target_id,
        type=rtype,
        properties=_row_to_props(rel),
    )


def _accumulate(
    nodes: dict[str, GraphViewNode],
    rels: dict[str, GraphViewRel],
    node: Any,
) -> str | None:
    view = _view_node(node)
    if view is None:
        return None
    nodes.setdefault(view.id, view)
    return view.id


def _add_rel(
    rels: dict[str, GraphViewRel],
    rel: Any,
    source_id: str | None,
    target_id: str | None,
) -> None:
    if not source_id or not target_id:
        return
    view = _view_rel(rel, source_id, target_id)
    if view is None:
        return
    rels.setdefault(view.id, view)


def _context_for_id(driver: Driver, node_id: str,
                    nodes: dict[str, GraphViewNode],
                    rels: dict[str, GraphViewRel]) -> None:
    kind, name = _parse_node_id(node_id)
    if not name:
        return
    with driver.session() as s:
        if kind == "merchant":
            head = s.run(CONTEXT_MERCHANT_HEAD, name=name).single()
            if not head:
                return
            m_id = _accumulate(nodes, rels, head["m"])
            c_id = _accumulate(nodes, rels, head["c"])
            _add_rel(rels, head["ic"], m_id, c_id)
            for rec in s.run(CONTEXT_MERCHANT_TXS, name=name):
                t_id = _accumulate(nodes, rels, rec["t"])
                _add_rel(rels, rec["at"], t_id, m_id)
        elif kind == "category":
            c_id: str | None = None
            for rec in s.run(CONTEXT_CATEGORY_PAIRS, name=name):
                c_id = _accumulate(nodes, rels, rec["c"]) or c_id
                m_id = _accumulate(nodes, rels, rec["m"])
                _add_rel(rels, rec["ic"], m_id, c_id)
        elif kind == "month":
            records = list(s.run(CONTEXT_MONTH_TXS, name=name))
            for record in records:
                mo_id = _accumulate(nodes, rels, record["mo"])
                t_id  = _accumulate(nodes, rels, record["t"])
                m_id  = _accumulate(nodes, rels, record["m"])
                c_id  = _accumulate(nodes, rels, record["c"])
                _add_rel(rels, record["im"], t_id, mo_id)
                _add_rel(rels, record["at"], t_id, m_id)
                _add_rel(rels, record["ic"], m_id, c_id)
            if not records:
                stub = s.run(
                    "MATCH (mo:Month {id: $name}) RETURN mo", name=name,
                ).single()
                if stub:
                    _accumulate(nodes, rels, stub["mo"])
        elif kind == "transaction":
            row = s.run(CONTEXT_TRANSACTION, name=name).single()
            if not row:
                return
            t_id  = _accumulate(nodes, rels, row["t"])
            m_id  = _accumulate(nodes, rels, row["m"])
            c_id  = _accumulate(nodes, rels, row["c"])
            mo_id = _accumulate(nodes, rels, row["mo"])
            a_id  = _accumulate(nodes, rels, row["a"])
            l_id  = _accumulate(nodes, rels, row["l"])
            s_id  = _accumulate(nodes, rels, row["s"])
            _add_rel(rels, row["at"], t_id, m_id)
            _add_rel(rels, row["ic"], m_id, c_id)
            _add_rel(rels, row["im"], t_id, mo_id)
            _add_rel(rels, row["fl"], a_id, t_id)
            _add_rel(rels, row["atl"], t_id, l_id)
            _add_rel(rels, row["cn"], s_id, t_id)
        elif kind == "alert":
            row = s.run(CONTEXT_ALERT_BY_ID, name=name).single()
            if not row:
                return
            a_id  = _accumulate(nodes, rels, row["a"])
            t_id  = _accumulate(nodes, rels, row["t"])
            m_id  = _accumulate(nodes, rels, row["m"])
            c_id  = _accumulate(nodes, rels, row["c"])
            mo_id = _accumulate(nodes, rels, row["mo"])
            l_id  = _accumulate(nodes, rels, row["l"])
            _add_rel(rels, row["fl"], a_id, t_id)
            _add_rel(rels, row["at"], t_id, m_id)
            _add_rel(rels, row["ic"], m_id, c_id)
            _add_rel(rels, row["im"], t_id, mo_id)
            _add_rel(rels, row["atl"], t_id, l_id)
        elif kind == "year" or kind == "annual":
            try:
                year_int = int(name)
            except ValueError:
                return
            for rec in s.run(_YEAR_CONTEXT, year=year_int):
                mo_id = _accumulate(nodes, rels, rec["mo"])
                t_id  = _accumulate(nodes, rels, rec["t"])
                m_id  = _accumulate(nodes, rels, rec["m"])
                c_id  = _accumulate(nodes, rels, rec["c"])
                _add_rel(rels, rec["im"], t_id, mo_id)
                _add_rel(rels, rec["at"], t_id, m_id)
                _add_rel(rels, rec["ic"], m_id, c_id)


@router.get("/context", response_model=GraphUpdate)
def get_context(
    ids: str = Query(..., description="Comma-separated node ids, e.g. 'merchant:Tesco,month:2025-04'"),
    mode: str = Query("replace", pattern="^(replace|merge)$"),
    driver: Driver = Depends(get_driver),
) -> GraphUpdate:
    """One-hop neighborhood for one or more nodes.

    This is the workhorse endpoint behind chat and alert-click flows —
    "given that the agent touched X and Y, here's the slice of the graph
    that matters." Returns a GraphUpdate the canvas can apply directly.
    """
    raw_ids = [s.strip() for s in ids.split(",") if s.strip()]
    if not raw_ids:
        raise HTTPException(400, "ids must contain at least one node id")

    nodes: dict[str, GraphViewNode] = {}
    rels: dict[str, GraphViewRel] = {}
    for node_id in raw_ids:
        try:
            _context_for_id(driver, node_id, nodes, rels)
        except Exception as exc:
            # Skip but don't fail the batch — return what we have.
            log_msg = f"context lookup failed for {node_id}: {exc}"
            import logging as _logging
            _logging.getLogger(__name__).info(log_msg)

    return GraphUpdate(
        nodes=list(nodes.values()),
        relationships=list(rels.values()),
        focus_ids=raw_ids,
        mode="merge" if mode == "merge" else "replace",
    )


@router.get("/alert/{alert_id}", response_model=GraphUpdate)
def get_alert_context(alert_id: str, driver: Driver = Depends(get_driver)) -> GraphUpdate:
    """Alert detail context: Alert → Transaction → Merchant → Category / Month / Location."""
    nodes: dict[str, GraphViewNode] = {}
    rels: dict[str, GraphViewRel] = {}
    _context_for_id(driver, f"alert:{alert_id}", nodes, rels)
    if not nodes:
        raise HTTPException(404, f"alert not found: {alert_id}")
    return GraphUpdate(
        nodes=list(nodes.values()),
        relationships=list(rels.values()),
        focus_ids=[f"alert:{alert_id}"],
        mode="replace",
    )


# ---------- Cypher --------------------------------------------------------

# Merchant / category nodes participating in the requested window.
NODES_QUERY = """
MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE t.month IN $months AND t.amount < 0 AND c.name <> 'System'
WITH m, c, count(t) AS visits, sum(-t.amount) AS spend
RETURN m.canonical_name AS merchant,
       c.name           AS category,
       visits,
       spend
"""

MONTH_TOTALS_QUERY = """
MATCH (t:Transaction)
WHERE t.month IN $months
WITH t.month AS month,
     sum(CASE WHEN t.amount > 0 THEN  t.amount ELSE 0 END) AS income,
     sum(CASE WHEN t.amount < 0 THEN -t.amount ELSE 0 END) AS expense
RETURN month, income, expense
ORDER BY month
"""

MERCHANT_MONTH_EDGES_QUERY = """
MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE t.month IN $months AND t.amount < 0 AND c.name <> 'System'
WITH m, t.month AS month, count(t) AS visits, sum(-t.amount) AS spend
RETURN m.canonical_name AS merchant, month, visits, spend
"""


# ---------- Helpers -------------------------------------------------------

def _resolve_months(driver: Driver,
                    month: str | None,
                    range_: str | None) -> list[str]:
    """Translate the query params into a concrete list of YYYY-MM keys."""
    if month and range_:
        raise HTTPException(400, "pass either month= or range=, not both")
    if month:
        if not _MONTH_RE.match(month):
            raise HTTPException(400, f"month must be YYYY-MM, got {month!r}")
        return [month]
    if range_:
        if not _RANGE_RE.match(range_):
            raise HTTPException(400, f"range must be YYYY-MM:YYYY-MM, got {range_!r}")
        start, end = range_.split(":")
        with driver.session() as s:
            rows = s.run(
                "MATCH (mo:Month) WHERE mo.id >= $a AND mo.id <= $b "
                "RETURN mo.id AS id ORDER BY id",
                a=start, b=end,
            ).data()
        return [r["id"] for r in rows]

    # Default: every month present in the graph.
    with driver.session() as s:
        rows = s.run("MATCH (mo:Month) RETURN mo.id AS id ORDER BY id").data()
    return [r["id"] for r in rows]


def _month_display(month_id: str) -> str:
    y, m = month_id.split("-")
    names = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
    return f"{names[int(m) - 1]} {y}"


# ---------- Endpoint ------------------------------------------------------

# Cross-account chain: credit-card transactions → CC payments → savings DD.
# We don't have an Account-typed schema for the CC (we treat it as a Statement
# scope), so the chain is reconstructed by matching the same canonical merchant
# string "Halifax Credit Card" (savings outflow) with the credit-card statement
# whose total it settles. This is data-driven — no schema migration needed.

TRACE_CC_SETTLEMENT = """
MATCH (cc:Statement {id: $cc_statement_id})-[:CONTAINS]->(t:Transaction)-[:AT]->(m:Merchant)
WHERE t.amount < 0
RETURN m.canonical_name AS merchant,
       sum(-t.amount)   AS spend,
       count(t)         AS visits
ORDER BY spend DESC
"""

TRACE_SAVINGS_DD = """
MATCH (s:Statement)-[:CONTAINS]->(t:Transaction)-[:AT]->(:Merchant {canonical_name: 'Halifax Credit Card'})
WHERE s.id STARTS WITH $savings_account_id
  AND t.month = $month
RETURN t.month AS month, -t.amount AS amount, t.date AS date,
       s.id AS statement_id
ORDER BY amount DESC LIMIT 1
"""

CC_STATEMENTS_FOR_MONTH = """
MATCH (a:Account)-[:HAS_STATEMENT]->(s:Statement)
WHERE a.account_type = 'credit_card' AND s.id STARTS WITH '1588-'
RETURN s.id AS id ORDER BY id
"""


@router.get("/trace")
def trace(
    month: str = Query(..., description="Savings month containing the DD (YYYY-MM)"),
    driver: Driver = Depends(get_driver),
) -> dict:
    """Trace the cross-account chain CC purchases → CC bill → savings DD.

    Returns ``{settlement, contributors, links}``:
      * ``settlement`` — the savings-account DD that paid the CC bill.
      * ``contributors`` — every merchant whose CC spend made up that bill.
      * ``links`` — node ids for the UI to render as a glowing path.
    """
    if not month or len(month) != 7:
        raise HTTPException(400, "month must be YYYY-MM")

    with driver.session() as s:
        dd = s.run(TRACE_SAVINGS_DD, savings_account_id="12345678",
                   month=month).single()
        if not dd:
            return {"settlement": None, "contributors": [], "links": []}

        # The savings DD on the Nth of the month settles the previous
        # month's credit-card statement (which ran from mid-prev to mid-this).
        y, m = month.split("-")
        prev_m = int(m) - 1 or 12
        prev_y = int(y) - 1 if prev_m == 12 else int(y)
        cc_month_short = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"][prev_m - 1]
        cc_stmt_id = f"1588-{prev_y}-{prev_m:02d}-13"  # statement issued mid-prev-month

        # Fall back: just find the most recent CC statement before this month.
        if not s.run("MATCH (st:Statement {id: $id}) RETURN st", id=cc_stmt_id).single():
            stmts = s.run(
                "MATCH (st:Statement) WHERE st.id STARTS WITH '1588-' "
                "AND st.id < $cap RETURN st.id AS id ORDER BY id DESC LIMIT 1",
                cap=f"1588-{y}-{m}-99",
            ).single()
            if stmts:
                cc_stmt_id = stmts["id"]

        contributors = s.run(TRACE_CC_SETTLEMENT, cc_statement_id=cc_stmt_id).data()

    # Build the link manifest the canvas overlay needs.
    settlement_node = f"merchant:Halifax Credit Card"
    month_node      = f"month:{month}"
    contributor_ids = [f"merchant:{c['merchant']}" for c in contributors]
    links: list[dict] = [
        {"source": cid, "target": settlement_node, "kind": "spend"}
        for cid in contributor_ids
    ] + [{"source": settlement_node, "target": month_node, "kind": "settle"}]

    return {
        "settlement": {
            "month":         dd["month"],
            "amount":        float(dd["amount"]),
            "date":          str(dd["date"]),
            "savings_stmt":  dd["statement_id"],
            "cc_stmt":       cc_stmt_id,
        },
        "contributors": [
            {"merchant": c["merchant"], "spend": float(c["spend"]), "visits": c["visits"]}
            for c in contributors
        ],
        "node_ids": [*contributor_ids, settlement_node, month_node],
        "links":    links,
    }


DECISIONS_QUERY = """
MATCH (d:Decision)
OPTIONAL MATCH (d)-[:TOUCHED]->(n)
WITH d, collect({label: labels(n)[0], name:
       coalesce(n.canonical_name, n.name, n.id)}) AS touched
RETURN d.id AS id, d.question AS question, d.ts AS ts,
       d.summary AS summary, touched
ORDER BY d.ts DESC LIMIT 50
"""


# Expand-on-click: return the 1-hop neighborhood of a single node.
# Adopted from neo4j-labs/create-context-graph and johnymontana's
# context-graph-demo — lets the user grow the visible canvas
# interactively rather than fetching the whole topology upfront.
EXPAND_MERCHANT = """
MATCH (m:Merchant {canonical_name: $name})
OPTIONAL MATCH (m)-[:IN_CATEGORY]->(c:Category)
OPTIONAL MATCH (t:Transaction)-[:AT]->(m), (t)-[:IN_MONTH]->(mo:Month)
WITH m, c, mo, count(t) AS visits, sum(-t.amount) AS spend
WHERE mo IS NOT NULL
RETURN m.canonical_name AS me, c.name AS category, mo.id AS month, visits, spend
"""

EXPAND_CATEGORY = """
MATCH (c:Category {name: $name})<-[:IN_CATEGORY]-(m:Merchant)
OPTIONAL MATCH (t:Transaction)-[:AT]->(m)
WHERE t.amount < 0
WITH c, m, sum(-t.amount) AS spend, count(t) AS visits
RETURN c.name AS me, m.canonical_name AS merchant, spend, visits
ORDER BY spend DESC LIMIT 20
"""

EXPAND_MONTH = """
MATCH (mo:Month {id: $name})
MATCH (t:Transaction)-[:IN_MONTH]->(mo)
WHERE t.amount < 0
MATCH (t)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE c.name <> 'System'
WITH mo, m, c, sum(-t.amount) AS spend, count(t) AS visits
RETURN mo.id AS me, m.canonical_name AS merchant, c.name AS category, spend, visits
ORDER BY spend DESC LIMIT 20
"""


@router.get("/expand")
def expand(
    id: str = Query(..., description="Node id, e.g. 'merchant:Tesco'"),
    driver: Driver = Depends(get_driver),
) -> dict:
    """Return the immediate neighborhood of one node as graph data."""
    kind, _, name = id.partition(":")
    if kind not in {"merchant", "category", "month"} or not name:
        raise HTTPException(400, f"id must be 'merchant:<name>' / 'category:<name>' / 'month:<YYYY-MM>', got {id!r}")

    nodes_by_id: dict[str, GraphNode] = {}
    links: list[GraphLink] = []

    def add_node(node: GraphNode) -> None:
        nodes_by_id.setdefault(node.id, node)

    with driver.session() as s:
        if kind == "merchant":
            rows = s.run(EXPAND_MERCHANT, name=name).data()
            for r in rows:
                add_node(GraphNode(id=f"merchant:{r['me']}", label=r["me"], type="Merchant",
                                   category=r["category"]))
                if r["category"]:
                    add_node(GraphNode(id=f"category:{r['category']}", label=r["category"], type="Category"))
                    links.append(GraphLink(source=f"merchant:{r['me']}",
                                           target=f"category:{r['category']}", type="IN_CATEGORY"))
                if r["month"]:
                    add_node(GraphNode(id=f"month:{r['month']}", label=r["month"], type="Month"))
                    links.append(GraphLink(source=f"merchant:{r['me']}",
                                           target=f"month:{r['month']}", type="ACTIVE_IN",
                                           weight=round(r["spend"] or 0, 2),
                                           visits=r["visits"]))
        elif kind == "category":
            rows = s.run(EXPAND_CATEGORY, name=name).data()
            add_node(GraphNode(id=f"category:{name}", label=name, type="Category"))
            for r in rows:
                add_node(GraphNode(id=f"merchant:{r['merchant']}", label=r["merchant"],
                                   type="Merchant", category=name))
                links.append(GraphLink(source=f"merchant:{r['merchant']}",
                                       target=f"category:{name}", type="IN_CATEGORY"))
        elif kind == "month":
            rows = s.run(EXPAND_MONTH, name=name).data()
            add_node(GraphNode(id=f"month:{name}", label=name, type="Month"))
            for r in rows:
                add_node(GraphNode(id=f"merchant:{r['merchant']}", label=r["merchant"],
                                   type="Merchant", category=r["category"]))
                add_node(GraphNode(id=f"category:{r['category']}", label=r["category"], type="Category"))
                links.append(GraphLink(source=f"merchant:{r['merchant']}",
                                       target=f"category:{r['category']}", type="IN_CATEGORY"))
                links.append(GraphLink(source=f"merchant:{r['merchant']}",
                                       target=f"month:{name}", type="ACTIVE_IN",
                                       weight=round(r["spend"] or 0, 2),
                                       visits=r["visits"]))

    return {"nodes": list(nodes_by_id.values()), "links": links, "center": id}


@router.get("/decisions")
def get_decisions(driver: Driver = Depends(get_driver)) -> dict:
    """Recent agent decisions for the trace-overlay UI."""
    with driver.session() as s:
        rows = s.run(DECISIONS_QUERY).data()
    out: list[dict] = []
    for r in rows:
        touched_ids: list[str] = []
        for t in r["touched"] or []:
            label, name = t.get("label"), t.get("name")
            if not label or not name:
                continue
            key = {"Merchant": "merchant", "Category": "category", "Month": "month"}.get(label)
            if key:
                touched_ids.append(f"{key}:{name}")
        out.append({
            "id":       r["id"],
            "question": r["question"],
            "ts":       str(r["ts"]),
            "summary":  r["summary"],
            "touched":  touched_ids,
        })
    return {"decisions": out}


@router.get("", response_model=GraphResponse)
def get_graph(
    month: str | None = Query(None, description="One month: YYYY-MM"),
    range: str | None = Query(None, description="Inclusive range: YYYY-MM:YYYY-MM"),
    merchant: str | None = Query(None, description="Filter to a single merchant"),
    category: str | None = Query(None, description="Filter to a single category"),
    driver: Driver = Depends(get_driver),
) -> GraphResponse:
    months = _resolve_months(driver, month, range)
    if not months:
        return GraphResponse(nodes=[], links=[], range=[])

    with driver.session() as s:
        rows = s.run(NODES_QUERY, months=months).data()
        month_totals = {r["month"]: r for r in s.run(MONTH_TOTALS_QUERY, months=months).data()}
        merchant_month_rows = s.run(MERCHANT_MONTH_EDGES_QUERY, months=months).data()

    if merchant:
        rows = [r for r in rows if r["merchant"] == merchant]
        merchant_month_rows = [r for r in merchant_month_rows if r["merchant"] == merchant]
    if category:
        rows = [r for r in rows if r["category"] == category]
        keep = {r["merchant"] for r in rows}
        merchant_month_rows = [r for r in merchant_month_rows if r["merchant"] in keep]

    # ----- Nodes ----------------------------------------------------------
    nodes: list[GraphNode] = []
    seen_merchants: dict[str, dict] = {}
    seen_categories: dict[str, float] = {}

    for r in rows:
        m = seen_merchants.setdefault(r["merchant"], {"visits": 0, "spend": 0.0, "category": r["category"]})
        m["visits"] += r["visits"]
        m["spend"] += r["spend"]
        seen_categories[r["category"]] = seen_categories.get(r["category"], 0) + r["spend"]

    for name, info in seen_merchants.items():
        nodes.append(GraphNode(
            id=f"merchant:{name}",
            label=name,
            type="Merchant",
            category=info["category"],
            total_spend=round(info["spend"], 2),
            visits=info["visits"],
        ))
    for name, spend in seen_categories.items():
        nodes.append(GraphNode(
            id=f"category:{name}",
            label=name,
            type="Category",
            total_spend=round(spend, 2),
        ))
    for m_id in months:
        totals = month_totals.get(m_id, {})
        nodes.append(GraphNode(
            id=f"month:{m_id}",
            label=_month_display(m_id),
            type="Month",
            income=round(totals.get("income") or 0, 2),
            expense=round(totals.get("expense") or 0, 2),
        ))

    # ----- Links ----------------------------------------------------------
    links: list[GraphLink] = []
    for name, info in seen_merchants.items():
        links.append(GraphLink(
            source=f"merchant:{name}",
            target=f"category:{info['category']}",
            type="IN_CATEGORY",
        ))
    for r in merchant_month_rows:
        if r["merchant"] not in seen_merchants:
            continue
        links.append(GraphLink(
            source=f"merchant:{r['merchant']}",
            target=f"month:{r['month']}",
            type="ACTIVE_IN",
            weight=round(r["spend"], 2),
            visits=r["visits"],
        ))

    return GraphResponse(nodes=nodes, links=links, range=months)
