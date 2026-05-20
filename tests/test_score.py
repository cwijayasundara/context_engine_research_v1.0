import pytest
from neo4j import Driver

from src.fraud.score import combine, write_back


def test_combine_picks_max_rule_severity_and_blends_gds():
    findings = [
        {"rule": "duplicate_charge", "severity": 0.9, "rationale": "x"},
        {"rule": "velocity",         "severity": 0.8, "rationale": "y"},
    ]
    result = combine(rule_findings=findings, is_outlier=True, emb_dist=0.4)
    assert result["fraud_score"] == pytest.approx(0.6 * 0.9 + 0.4 * (0.5 * 1 + 0.5 * 0.4), rel=1e-3)
    assert set(result["risk_flags"]) == {"duplicate_charge", "velocity"}


def test_combine_no_findings_and_no_outlier_is_zero():
    result = combine(rule_findings=[], is_outlier=False, emb_dist=0.0)
    assert result["fraud_score"] == 0.0
    assert result["risk_flags"] == []


@pytest.mark.neo4j
def test_write_back_writes_score_and_creates_alert(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("CREATE (t:Transaction {id:'tx-1', amount:-50.0, description:'X'})")
    write_back(
        clean_graph,
        per_tx={
            "tx-1": {
                "fraud_score": 0.85,
                "risk_flags": ["duplicate_charge"],
                "rationale": "double-post",
                "max_rule":  "duplicate_charge",
            }
        },
    )
    with clean_graph.session() as s:
        row = s.run("""
            MATCH (a:Alert)-[:FLAGS]->(t:Transaction {id:'tx-1'})
            RETURN t.fraud_score AS score, t.risk_flags AS flags,
                   a.kind AS kind, a.severity AS sev
        """).single()
    assert row["score"] == pytest.approx(0.85)
    assert row["flags"] == ["duplicate_charge"]
    assert row["kind"] == "duplicate_charge"
    assert row["sev"]  == pytest.approx(0.85)
