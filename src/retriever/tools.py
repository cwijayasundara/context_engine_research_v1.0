"""Tools the agent (and subagents) call.

We keep the surface small and typed:

  - ``graph_query(cypher, params)`` — read-only Cypher over Neo4j
  - ``wiki_read(path)``             — read a compiled markdown artifact
  - ``wiki_list(prefix)``           — list available wiki entries
  - ``python_exec(code)``           — execute Python in the REPL middleware
                                     (the agent uses this for PTC; tools above
                                      are callable from inside the code too)

Read-only Cypher is enforced by a static pattern check. The full guard would
use Neo4j's read-only role + per-tool auth; this is the dev-time minimum.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from neo4j import GraphDatabase

from src.config import SETTINGS

log = logging.getLogger(__name__)

_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH|CALL\s+apoc\.cypher\.run\w*)\b",
    re.IGNORECASE,
)


@tool("graph_query", parse_docstring=True)
def graph_query(cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
    """Run a read-only Cypher query against the finance context graph.

    Use this for structured questions: aggregations, time-windowed sums,
    merchant lookups, "top N" rankings, joins across the ontology.

    Args:
        cypher: The Cypher query. Must be read-only — write keywords
            (CREATE, MERGE, DELETE, SET, REMOVE, DROP, DETACH) are rejected.
        params: Optional parameter dict bound at execution time.

    Returns:
        A list of result rows, each row is a dict of column → value.
    """
    if _WRITE_KEYWORDS.search(cypher):
        raise PermissionError(
            "graph_query is read-only; write operations are not permitted."
        )
    driver = GraphDatabase.driver(
        SETTINGS.neo4j_uri, auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password)
    )
    try:
        with driver.session(database=SETTINGS.neo4j_database) as session:
            return [r.data() for r in session.run(cypher, **(params or {}))]
    finally:
        driver.close()


@tool("wiki_read", parse_docstring=True)
def wiki_read(path: str) -> str:
    """Read a compiled markdown wiki artifact.

    Args:
        path: Path relative to the wiki root (e.g. ``merchants/Costco.md``).

    Returns:
        The raw markdown content.
    """
    safe = _safe_join(SETTINGS.wiki_dir, path)
    if not safe.exists():
        raise FileNotFoundError(f"wiki entry not found: {path}")
    return safe.read_text()


@tool("wiki_list", parse_docstring=True)
def wiki_list(prefix: str = "") -> list[str]:
    """List available wiki entries.

    Args:
        prefix: Optional subdirectory prefix (e.g. ``merchants``, ``categories``).
    """
    root = SETTINGS.wiki_dir
    if prefix:
        root = _safe_join(root, prefix)
    if not root.exists():
        return []
    return sorted(str(p.relative_to(SETTINGS.wiki_dir)) for p in root.rglob("*.md"))


def _safe_join(root: Path, rel: str) -> Path:
    """Defend against path traversal: resolve and confirm it stays under root."""
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    if not str(candidate).startswith(str(root_resolved) + "/") and candidate != root_resolved:
        raise PermissionError(f"path escapes wiki root: {rel}")
    return candidate


# Convenient export for the agent constructor
TOOLS = [graph_query, wiki_read, wiki_list]
