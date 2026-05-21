"""Shared graph context derivation for agent query results."""
from __future__ import annotations

import logging
import re

from src.api.deps import get_driver
from src.api.models import GraphViewNode, GraphViewRel

log = logging.getLogger(__name__)

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


def node_ids_for_graph_query(cypher: str, rows: list[dict]) -> list[str]:
    ids = _highlights_from_rows(rows)
    terms, months = _context_terms_from_cypher(cypher)
    ids.extend(f"month:{month}" for month in months)
    if not terms:
        return _dedupe_ids(ids)

    try:
        with get_driver().session() as session:
            lookup_rows = session.run(
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


def node_ids_for_alert_rows(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in rows[:25]:
        merchant = row.get("merchant")
        month = row.get("month")
        if isinstance(merchant, str) and merchant:
            ids.append(f"merchant:{merchant}")
        if isinstance(month, str) and _MONTH_VALUE_RE.fullmatch(month):
            ids.append(f"month:{month}")
    return _dedupe_ids(ids)


def graph_update_payload(node_ids: list[str]) -> dict | None:
    from src.api.routes.graph import _context_for_id

    if not node_ids:
        return None
    nodes: dict[str, GraphViewNode] = {}
    relationships: dict[str, GraphViewRel] = {}
    try:
        driver = get_driver()
        for node_id in node_ids:
            try:
                _context_for_id(driver, node_id, nodes, relationships)
            except Exception as exc:
                log.info("graph_update: context for %s failed: %s", node_id, exc)
    except Exception as exc:
        log.info("graph_update: driver unavailable: %s", exc)
        return None

    if not nodes:
        return None
    return {
        "nodes": [node.model_dump() for node in nodes.values()],
        "relationships": [rel.model_dump() for rel in relationships.values()],
        "focus_ids": node_ids,
        "mode": "merge",
    }


def _highlights_from_rows(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows[:25]:
        for key, value in row.items():
            if not isinstance(value, str):
                continue
            key_l = key.lower()
            if key_l in _MERCHANT_KEYS:
                node_id = f"merchant:{value}"
            elif key_l in _CATEGORY_KEYS:
                node_id = f"category:{value}"
            elif key_l in _MONTH_KEYS and _MONTH_VALUE_RE.fullmatch(value):
                node_id = f"month:{value}"
            else:
                continue
            if node_id not in seen:
                seen.add(node_id)
                ids.append(node_id)
    return ids


def _context_terms_from_cypher(cypher: str) -> tuple[list[str], list[str]]:
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


def _dedupe_ids(ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for node_id in ids:
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out
