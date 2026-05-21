from __future__ import annotations

from src.deep_retrieval.tools import graph_query


def test_graph_query_appends_limit_before_execution(monkeypatch) -> None:
    executed: dict[str, object] = {}

    class FakeRecord:
        def data(self) -> dict:
            return {"id": "tx-1"}

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def run(self, cypher: str, **params):
            executed["cypher"] = cypher
            executed["params"] = params
            return [FakeRecord()]

    class FakeDriver:
        def session(self):
            return FakeSession()

    monkeypatch.setattr("src.deep_retrieval.tools.get_driver", lambda: FakeDriver())

    result = graph_query.func(
        "MATCH (t:Transaction) RETURN t.id AS id ORDER BY id",
        params={},
        limit=10,
    )

    assert executed["cypher"] == "MATCH (t:Transaction) RETURN t.id AS id ORDER BY id LIMIT 10"
    assert result["cypher"] == executed["cypher"]
    assert result["rows"] == [{"id": "tx-1"}]


def test_graph_query_returns_graph_update_for_scalar_context(monkeypatch) -> None:
    executed: dict[str, object] = {}

    class FakeRecord:
        def data(self) -> dict:
            return {"total_spend": 9262.02, "transaction_count": 144}

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def run(self, cypher: str, **params):
            executed["cypher"] = cypher
            executed["params"] = params
            return [FakeRecord()]

    class FakeDriver:
        def session(self):
            return FakeSession()

    monkeypatch.setattr("src.deep_retrieval.tools.get_driver", lambda: FakeDriver())
    monkeypatch.setattr(
        "src.deep_retrieval.tools.node_ids_for_graph_query",
        lambda cypher, rows: ["category:Groceries", "merchant:Tesco"],
    )
    monkeypatch.setattr(
        "src.deep_retrieval.tools.graph_update_payload",
        lambda node_ids: {
            "nodes": [{"id": node_ids[0], "label": "Groceries", "type": "Category", "properties": {}}],
            "relationships": [],
            "focus_ids": node_ids,
            "mode": "merge",
        },
    )

    result = graph_query.func(
        """
        MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category {name: "Groceries"})
        WHERE t.year = $year AND t.amount < 0
        RETURN sum(-t.amount) AS total_spend, count(t) AS transaction_count
        """,
        params={"year": 2025},
    )

    assert result["node_ids"] == ["category:Groceries", "merchant:Tesco"]
    assert result["graph_update"]["focus_ids"] == ["category:Groceries", "merchant:Tesco"]
