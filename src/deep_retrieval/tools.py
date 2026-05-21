"""Deep Agents retrieval tools for graph, wiki, and fraud context."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.api.deps import get_driver, get_wiki_root
from src.api.graph_context import graph_update_payload, node_ids_for_graph_query
from src.deep_retrieval.cypher_guard import CypherGuard
from src.deep_retrieval.schema import build_schema_context, finance_cypher_examples
from src.ontology import load_ontology

try:
    from langchain_core.tools import tool
except Exception:  # pragma: no cover - only used when optional deps are absent
    def tool(name: str | None = None, parse_docstring: bool = False):  # type: ignore
        def decorate(fn: Callable):
            fn.name = name or fn.__name__
            return fn
        return decorate


@tool("schema_context", parse_docstring=False)
def schema_context() -> dict:
    """Return schema text and few-shot Cypher examples for finance graph queries."""
    return {
        "schema": build_schema_context(),
        "examples": finance_cypher_examples(),
    }


@tool("graph_query", parse_docstring=False)
def graph_query(
    cypher: str,
    params: dict[str, Any] | None = None,
    purpose: str = "",
    expected_columns: list[str] | None = None,
    limit: int = 50,
) -> dict:
    """Run a validated read-only Cypher query against Neo4j."""
    guard = CypherGuard.from_ontology(load_ontology())
    result = guard.validate(
        cypher,
        params=params or {},
        add_missing_limit=True,
        default_limit=limit,
    )
    driver = get_driver()
    with driver.session() as session:
        rows = [record.data() for record in session.run(result.cypher, **(params or {}))]
    node_ids = node_ids_for_graph_query(result.cypher, rows)
    output = {
        "purpose": purpose,
        "cypher": result.cypher,
        "params": params or {},
        "expected_columns": expected_columns or [],
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
        "row_count": len(rows),
        "node_ids": node_ids,
    }
    update = graph_update_payload(node_ids)
    if update:
        output["graph_update"] = update
    return output


@tool("wiki_list", parse_docstring=False)
def wiki_list(prefix: str = "") -> list[str]:
    """List compiled wiki markdown paths under the wiki root."""
    root = get_wiki_root()
    target = _safe_join(root, prefix) if prefix else root
    if not target.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in target.rglob("*.md"))


@tool("wiki_read", parse_docstring=False)
def wiki_read(path: str) -> dict:
    """Read one compiled wiki markdown artifact by relative path."""
    root = get_wiki_root()
    target = _safe_join(root, path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"wiki entry not found: {path}")
    return {"path": str(target.relative_to(root)), "markdown": target.read_text(encoding="utf-8")}


@tool("fraud_alerts", parse_docstring=False)
def fraud_alerts(month: str | None = None, limit: int = 20) -> dict:
    """Return fraud alerts from the graph, optionally filtered by month."""
    limit = max(1, min(int(limit), 100))
    where = "WHERE t.month = $month" if month else ""
    cypher = f"""
    MATCH (a:Alert)-[:FLAGS]->(t:Transaction)-[:AT]->(m:Merchant)
    {where}
    RETURN a.id AS alert_id, a.kind AS kind, a.severity AS severity,
           a.rationale AS rationale, t.id AS tx_id, t.month AS month,
           t.amount AS amount, m.canonical_name AS merchant
    ORDER BY severity DESC
    LIMIT {limit}
    """
    return graph_query(cypher, {"month": month} if month else {}, "retrieve fraud alerts")


def retrieval_tools() -> list:
    return [schema_context, graph_query, wiki_list, wiki_read, fraud_alerts]


def _safe_join(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and not str(target).startswith(str(root_resolved) + "/"):
        raise PermissionError(f"path escapes wiki root: {rel}")
    return target
