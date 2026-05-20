import pytest
from fastapi.testclient import TestClient
from neo4j import Driver

from src.api.deps import get_driver
from src.api.main import app
from tests.conftest import TEST_DATABASE, _TestDbDriver


@pytest.fixture()
def client(neo4j_driver: Driver):
    # Force the FastAPI app to query the isolated test DB during these
    # tests, otherwise the route handlers default to `neo4j` and find
    # populated dev data instead of the fixture rows the test just seeded.
    app.dependency_overrides[get_driver] = lambda: _TestDbDriver(neo4j_driver)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_driver, None)


@pytest.mark.neo4j
def test_get_anomalies_empty_returns_empty_list(client, clean_graph: Driver):
    resp = client.get("/api/fraud/anomalies")
    assert resp.status_code == 200
    assert resp.json() == {"month": None, "alerts": []}


@pytest.mark.neo4j
def test_get_score_404_for_missing_tx(client, clean_graph: Driver):
    resp = client.get("/api/fraud/score/nope")
    assert resp.status_code == 404


@pytest.mark.neo4j
def test_get_anomalies_returns_seeded_alert(client, clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (m:Merchant {id:'foo', canonical_name:'Foo'})
            MERGE (l:Location {id:'rome', name:'Rome', country:'IT'})
            CREATE (t:Transaction {
              id:'tx-1', amount:-100.0, description:'FOO ROME IT',
              date:date('2025-06-25'), month:'2025-06',
              fraud_score:0.7, risk_flags:['round_fx']
            })
            CREATE (a:Alert {
              id:'al-1', kind:'round_fx', severity:0.7,
              created_at:datetime(), rationale:'round in italy'
            })
            MERGE (t)-[:AT]->(m) MERGE (t)-[:AT_LOCATION]->(l)
            MERGE (a)-[:FLAGS]->(t)
        """)
    resp = client.get("/api/fraud/anomalies?month=2025-06")
    assert resp.status_code == 200
    body = resp.json()
    assert body["month"] == "2025-06"
    assert len(body["alerts"]) == 1
    alert = body["alerts"][0]
    assert alert["kind"] == "round_fx"
    assert alert["merchant"] == "Foo"
    assert alert["location"] == "Rome (IT)"
