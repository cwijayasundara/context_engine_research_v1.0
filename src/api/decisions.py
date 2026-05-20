"""Decision-trace recording.

Every agent turn writes a ``Decision`` node into Neo4j linked to the graph
nodes it touched. The wiki compiler's "System" category trick keeps these
out of the wiki, and the UI exposes them via an optional layer.

Schema:
    (:Decision {id, question, ts, tools, summary})
    (:Decision)-[:TOUCHED]->(:Merchant | :Category | :Month)
"""
from __future__ import annotations

import logging
import uuid
from typing import Iterable

from src.api.deps import get_driver

log = logging.getLogger(__name__)

UPSERT = """
MERGE (d:Decision {id: $id})
  ON CREATE SET d.question = $question,
                d.ts       = datetime($ts),
                d.tools    = $tools,
                d.summary  = $summary
WITH d
UNWIND $node_ids AS nid
  WITH d, split(nid, ':')[0] AS kind, split(nid, ':')[1] AS name
  // Three label-scoped lookups; coalesce picks the non-null one. Avoids
  // the pre-Cypher-25 `CALL { ... UNION ... }` subquery shape that Neo4j
  // 5.23+ warns about (the new form requires an explicit variable scope).
  OPTIONAL MATCH (m:Merchant  {canonical_name: name}) WHERE kind = 'merchant'
  OPTIONAL MATCH (c:Category  {name:           name}) WHERE kind = 'category'
  OPTIONAL MATCH (mo:Month    {id:             name}) WHERE kind = 'month'
  WITH d, coalesce(m, c, mo) AS target
  WHERE target IS NOT NULL
  MERGE (d)-[:TOUCHED]->(target)
RETURN d.id AS id
"""


def write_decision(
    *,
    question: str,
    ts_iso: str,
    summary: str,
    tools: list[str],
    node_ids: Iterable[str],
) -> str | None:
    """Persist one decision. Best-effort — never raises into the agent loop."""
    decision_id = f"dec-{uuid.uuid4().hex[:10]}"
    driver = get_driver()
    try:
        with driver.session() as s:
            s.run(
                UPSERT,
                id=decision_id,
                question=question[:500],
                ts=ts_iso,
                tools=list(tools),
                summary=summary[:1000],
                node_ids=[nid for nid in set(node_ids) if isinstance(nid, str) and ":" in nid],
            )
        return decision_id
    except Exception as exc:                       # pragma: no cover
        log.warning("decision write failed: %s", exc)
        return None
