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
