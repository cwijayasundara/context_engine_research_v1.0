from __future__ import annotations

from src.api.graph_context import node_ids_for_graph_query


def test_node_ids_for_scalar_category_query_use_cypher_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResult:
        def single(self):
            return {
                "merchants": ["Tesco", "Sainsbury's"],
                "categories": ["Groceries"],
                "months": ["2025-01", "2025-02"],
            }

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def run(self, cypher: str, **params):
            captured["params"] = params
            return FakeResult()

    class FakeDriver:
        def session(self):
            return FakeSession()

    monkeypatch.setattr("src.api.graph_context.get_driver", lambda: FakeDriver())

    ids = node_ids_for_graph_query(
        """
        MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category {name: "Groceries"})
        WHERE t.year = $year AND t.amount < 0
        RETURN sum(-t.amount) AS total_spend
        """,
        [{"total_spend": 9262.02}],
    )

    assert captured["params"] == {"terms": ["groceries"], "months": []}
    assert ids == [
        "merchant:Tesco",
        "merchant:Sainsbury's",
        "category:Groceries",
        "month:2025-01",
        "month:2025-02",
    ]
