"""Combine rule findings + GDS features into a per-transaction fraud_score
and persist them back into Neo4j (Transaction.fraud_score, Transaction.risk_flags,
plus one :Alert node per flagged transaction).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from neo4j import Driver

from src.fraud.rules import run_all_rules

log = logging.getLogger(__name__)

W_RULES = 0.6
W_GDS   = 0.4
ALERT_THRESHOLD = 0.50


def combine(*, rule_findings: list[dict], is_outlier: bool, emb_dist: float) -> dict:
    rule_score = max((f["severity"] for f in rule_findings), default=0.0)
    gds_score  = 0.5 * (1.0 if is_outlier else 0.0) + 0.5 * max(0.0, min(1.0, emb_dist))
    score = W_RULES * rule_score + W_GDS * gds_score
    score = max(0.0, min(1.0, score))
    flags = sorted({f["rule"] for f in rule_findings})
    max_rule = max(rule_findings, key=lambda f: f["severity"])["rule"] if rule_findings else None
    rationale = " | ".join(f["rationale"] for f in rule_findings) or (
        "Merchant is a graph outlier" if is_outlier else "" )
    return {
        "fraud_score": round(score, 4),
        "risk_flags":  flags,
        "max_rule":    max_rule,
        "rationale":   rationale,
    }


def score_all(driver: Driver) -> dict[str, dict]:
    rule_findings = run_all_rules(driver)
    with driver.session() as s:
        centroid_row = s.run("""
            MATCH (m:Merchant) WHERE m.embedding IS NOT NULL
            WITH collect(m.embedding) AS embs
            RETURN embs
        """).single()
    centroid = _centroid(centroid_row["embs"]) if centroid_row and centroid_row["embs"] else None

    with driver.session() as s:
        rows = s.run("""
            MATCH (t:Transaction)-[:AT]->(m:Merchant)
            RETURN t.id AS tx_id,
                   coalesce(m.is_outlier, false) AS is_outlier,
                   m.embedding AS embedding
        """).data()

    per_tx: dict[str, dict] = {}
    for r in rows:
        tx_id   = r["tx_id"]
        emb     = r["embedding"]
        dist    = _emb_distance(emb, centroid) if (emb and centroid) else 0.0
        bundle  = combine(
            rule_findings=rule_findings.get(tx_id, []),
            is_outlier=r["is_outlier"],
            emb_dist=dist,
        )
        per_tx[tx_id] = bundle
    return per_tx


_WRITE_BACK = """
UNWIND $rows AS row
MATCH (t:Transaction {id: row.tx_id})
SET t.fraud_score = row.fraud_score,
    t.risk_flags  = row.risk_flags
WITH t, row
WHERE row.fraud_score >= $threshold AND row.max_rule IS NOT NULL
MERGE (a:Alert {id: row.alert_id})
  ON CREATE SET a.kind = row.max_rule,
                a.severity = row.fraud_score,
                a.created_at = datetime($now),
                a.rationale  = row.rationale
MERGE (a)-[:FLAGS]->(t)
"""


def write_back(driver: Driver, per_tx: dict[str, dict]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            **bundle,
            "tx_id": tx_id,
            "alert_id": f"{tx_id}:{bundle['max_rule']}" if bundle.get("max_rule") else str(uuid4()),
        }
        for tx_id, bundle in per_tx.items()
    ]
    with driver.session() as s:
        s.run("MATCH (a:Alert) DETACH DELETE a")
        s.run(_WRITE_BACK, rows=rows, threshold=ALERT_THRESHOLD, now=now)
    flagged = sum(1 for b in per_tx.values() if b["fraud_score"] >= ALERT_THRESHOLD)
    log.info("wrote %d scores, created %d alerts", len(per_tx), flagged)
    return flagged


def _centroid(embs: list[list[float]]) -> list[float]:
    n = len(embs)
    if n == 0:
        return []
    dim = len(embs[0])
    out = [0.0] * dim
    for e in embs:
        for i, v in enumerate(e):
            out[i] += v
    return [v / n for v in out]


def _emb_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance, scaled to [0, 1]."""
    import math
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a)) or 1.0
    nb = math.sqrt(sum(bi * bi for bi in b)) or 1.0
    cos = dot / (na * nb)
    return max(0.0, min(1.0, (1.0 - cos) / 2.0))
