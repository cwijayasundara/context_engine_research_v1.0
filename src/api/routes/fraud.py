"""HTTP surface for the fraud / anomaly layer.

GET  /api/fraud/anomalies        — list alerts (optionally filtered by month)
GET  /api/fraud/score/{tx_id}    — return fraud_score + risk flags for one tx
POST /api/fraud/recompute        — re-run the pipeline (rules + optionally GDS)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import Driver

from src.api.deps import get_driver
from src.api.models import AlertItem, AlertsResponse, FraudScoreResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/fraud", tags=["fraud"])


_ANOMALIES_QUERY = """
MATCH (a:Alert)-[:FLAGS]->(t:Transaction)-[:AT]->(m:Merchant)
OPTIONAL MATCH (t)-[:AT_LOCATION]->(l:Location)
WHERE $month IS NULL OR t.month = $month
WITH t, m, l, collect(a) AS alerts
WITH t, m, l, alerts,
     reduce(best = alerts[0], a IN alerts |
       CASE WHEN coalesce(a.severity, 0) > coalesce(best.severity, 0) THEN a ELSE best END
     ) AS a
RETURN a.id        AS alert_id,
       t.id        AS tx_id,
       a.kind      AS kind,
       a.severity  AS severity,
       t.fraud_score AS fraud_score,
       t.risk_flags  AS risk_flags,
       a.rationale AS rationale,
       m.canonical_name AS merchant,
       t.amount    AS amount,
       toString(t.date) AS date,
       t.description AS description,
       coalesce(l.name + ' (' + l.country + ')', null) AS location
ORDER BY t.fraud_score DESC, t.date DESC
LIMIT 200
"""


@router.get("/anomalies", response_model=AlertsResponse)
def get_anomalies(
    month: str | None = Query(None, description="Filter to YYYY-MM"),
    driver: Driver = Depends(get_driver),
) -> AlertsResponse:
    with driver.session() as s:
        rows = s.run(_ANOMALIES_QUERY, month=month).data()
    return AlertsResponse(month=month, alerts=[AlertItem(**r) for r in rows])


@router.get("/score/{tx_id}", response_model=FraudScoreResponse)
def get_score(tx_id: str, driver: Driver = Depends(get_driver)) -> FraudScoreResponse:
    with driver.session() as s:
        row = s.run(
            "MATCH (t:Transaction {id:$id}) "
            "OPTIONAL MATCH (a:Alert)-[:FLAGS]->(t) "
            "RETURN t.fraud_score AS fraud_score, "
            "       coalesce(t.risk_flags, []) AS risk_flags, "
            "       coalesce(a.rationale, '') AS rationale",
            id=tx_id,
        ).single()
    if row is None:
        raise HTTPException(404, f"transaction not found: {tx_id}")
    return FraudScoreResponse(
        tx_id=tx_id,
        fraud_score=float(row["fraud_score"] or 0.0),
        risk_flags=list(row["risk_flags"] or []),
        rationale=row["rationale"] or "",
    )


@router.post("/recompute")
def recompute(skip_gds: bool = Query(False), driver: Driver = Depends(get_driver)) -> dict:
    """Re-run the fraud pipeline. Synchronous — returns counts when done."""
    from src.fraud.gds import GdsClient
    from src.fraud.score import score_all, write_back

    if not skip_gds:
        gds = GdsClient(driver)
        gds.project_merchant_coincidence()
        gds.run_pagerank(); gds.run_louvain(); gds.run_fastrp()
        gds.run_knn();      gds.run_node_similarity(); gds.mark_outliers()
    per_tx  = score_all(driver)
    flagged = write_back(driver, per_tx)
    return {"scored": len(per_tx), "alerts": flagged}
