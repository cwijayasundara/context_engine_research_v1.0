from __future__ import annotations

import pytest

from src.deep_retrieval.cypher_guard import CypherGuard, CypherValidationError
from src.deep_retrieval.schema import build_schema_context, finance_cypher_examples
from src.ontology import load_ontology


def guard() -> CypherGuard:
    return CypherGuard.from_ontology(load_ontology())


def test_accepts_parameterized_read_query_with_limit() -> None:
    result = guard().validate(
        """
        MATCH (t:Transaction)-[:AT]->(m:Merchant)
        WHERE m.canonical_name = $merchant
        RETURN t.date AS date, t.amount AS amount
        ORDER BY date
        LIMIT 20
        """,
        params={"merchant": "Costco"},
    )

    assert result.cypher.startswith("MATCH")
    assert result.limit == 20
    assert result.labels == {"Transaction", "Merchant"}
    assert result.relationships == {"AT"}


def test_rejects_write_keywords() -> None:
    with pytest.raises(CypherValidationError, match="write operation"):
        guard().validate(
            "MATCH (t:Transaction) SET t.flagged = true RETURN t LIMIT 1",
            params={},
        )


def test_rejects_unknown_labels_and_properties() -> None:
    with pytest.raises(CypherValidationError, match="unknown label"):
        guard().validate("MATCH (x:Invoice) RETURN x LIMIT 10", params={})

    with pytest.raises(CypherValidationError, match="unknown property"):
        guard().validate(
            "MATCH (m:Merchant) RETURN m.swift_code AS swift_code LIMIT 10",
            params={},
        )


def test_rejects_unknown_relationship_and_wrong_direction() -> None:
    with pytest.raises(CypherValidationError, match="unknown relationship"):
        guard().validate(
            "MATCH (t:Transaction)-[:PAID_TO]->(m:Merchant) RETURN m.name LIMIT 10",
            params={},
        )

    with pytest.raises(CypherValidationError, match="wrong direction"):
        guard().validate(
            "MATCH (m:Merchant)-[:AT]->(t:Transaction) RETURN m.name LIMIT 10",
            params={},
        )


def test_requires_limit_for_row_returning_queries_but_not_aggregates() -> None:
    with pytest.raises(CypherValidationError, match="LIMIT"):
        guard().validate("MATCH (t:Transaction) RETURN t.id", params={})

    result = guard().validate(
        "MATCH (t:Transaction) RETURN count(t) AS tx_count",
        params={},
    )

    assert result.limit is None


def test_can_append_default_limit_for_deep_agent_row_queries() -> None:
    result = guard().validate(
        "MATCH (t:Transaction) RETURN t.id AS id ORDER BY id",
        params={},
        add_missing_limit=True,
        default_limit=25,
    )

    assert result.limit == 25
    assert result.cypher.endswith("ORDER BY id LIMIT 25")


def test_default_limit_repair_preserves_aggregate_queries() -> None:
    result = guard().validate(
        "MATCH (t:Transaction) RETURN sum(t.amount) AS spend",
        params={},
        add_missing_limit=True,
        default_limit=25,
    )

    assert result.limit is None
    assert "LIMIT" not in result.cypher


def test_default_limit_repair_clamps_to_safe_range() -> None:
    too_high = guard().validate(
        "MATCH (t:Transaction) RETURN t.id AS id",
        params={},
        add_missing_limit=True,
        default_limit=500,
    )
    too_low = guard().validate(
        "MATCH (t:Transaction) RETURN t.id AS id",
        params={},
        add_missing_limit=True,
        default_limit=-5,
    )

    assert too_high.cypher.endswith("LIMIT 100")
    assert too_low.cypher.endswith("LIMIT 1")


def test_rejects_literal_values_in_where_clause() -> None:
    with pytest.raises(CypherValidationError, match="parameter"):
        guard().validate(
            "MATCH (m:Merchant) WHERE m.canonical_name = 'Costco' RETURN m LIMIT 5",
            params={},
        )


def test_schema_context_and_examples_are_grounded() -> None:
    schema = build_schema_context(load_ontology())
    examples = finance_cypher_examples()

    assert "(Transaction)" in schema
    assert "(Transaction)-[:AT]->(Merchant)" in schema
    assert "Use only these labels" in schema
    assert any("merchant spend" in ex["name"].lower() for ex in examples)
    assert all("cypher" in ex and "params" in ex for ex in examples)
