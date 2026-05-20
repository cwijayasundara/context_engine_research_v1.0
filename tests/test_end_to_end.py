"""End-to-end: load fixtures → run pipeline → assert each crafted fraud case
shows up in /api/fraud/anomalies."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from neo4j import Driver

# Pin the regex normalizer so the e2e doesn't accidentally call OpenAI.
os.environ.setdefault("NORMALIZER", "regex")

from src.api.main import app  # noqa: E402
from src.fraud.run import main as run_fraud  # noqa: E402
from src.ingestion.load_to_graph import UPSERT_TX, _to_cypher_params  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.neo4j
@pytest.mark.skip(
    reason="src.fraud.run.main does not yet accept a target database — "
           "running it during tests would touch the dev `neo4j` DB. "
           "Enable once the fraud runner honors NEO4J_DATABASE."
)
def test_pipeline_flags_each_crafted_fraud_case(
    clean_graph: Driver, fixtures_dir, client: TestClient
) -> None:
    # First apply the project schema so constraints + indexes exist.
    schema_path = Path("src/ontology/schema.cypher")
    schema_statements = [
        s.strip()
        for s in schema_path.read_text().split(";")
        if s.strip() and not s.strip().startswith("//")
    ]
    with clean_graph.session() as s:
        for stmt in schema_statements:
            s.run(stmt)

    # 1. Load both fixture files via the same params + Cypher used by the loader.
    with clean_graph.session() as s:
        for fname in ("normal_txs.jsonl", "fraud_injections.jsonl"):
            for line in (fixtures_dir / fname).read_text().splitlines():
                rec = json.loads(line)
                params = _to_cypher_params(rec)
                s.run(UPSERT_TX, **params)

    # 2. Run the full fraud pipeline (rules + GDS) end-to-end.
    rc = run_fraud(argv=[])
    assert rc == 0

    # 3. Query the anomalies endpoint and verify each crafted fraud case is represented.
    resp = client.get("/api/fraud/anomalies?month=2025-06")
    assert resp.status_code == 200
    alerts = resp.json()["alerts"]
    descriptions = {a["description"] for a in alerts}

    assert any("[FRAUD-CASE-1]" in d for d in descriptions), (
        "Case 1 (duplicate-charge) not flagged. Alerts seen: "
        + "; ".join(sorted(descriptions))
    )
    assert any("[FRAUD-CASE-2]" in d for d in descriptions), (
        "Case 2 (card-testing) not flagged. Alerts seen: "
        + "; ".join(sorted(descriptions))
    )
    assert any("[FRAUD-CASE-3]" in d for d in descriptions), (
        "Case 3 (new-merchant high-amount) not flagged. Alerts seen: "
        + "; ".join(sorted(descriptions))
    )
    assert any("[FRAUD-CASE-4]" in d for d in descriptions), (
        "Case 4 (geo-mismatch) not flagged. Alerts seen: "
        + "; ".join(sorted(descriptions))
    )
