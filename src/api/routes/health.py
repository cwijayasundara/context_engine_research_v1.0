"""Liveness + a one-shot graph sanity probe."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from neo4j import Driver

from src.api.deps import get_driver, get_wiki_root
from src.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(driver: Driver = Depends(get_driver)) -> HealthResponse:
    try:
        with driver.session() as s:
            tx_count = s.run("MATCH (t:Transaction) RETURN count(t) AS c").single()["c"]
        neo4j = "ok"
    except Exception as exc:  # surface failure rather than 500
        tx_count = 0
        neo4j = f"unreachable: {exc.__class__.__name__}"
    return HealthResponse(
        status="ok",
        neo4j=neo4j,
        wiki_root_exists=get_wiki_root().exists(),
        transaction_count=tx_count,
    )
