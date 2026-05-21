from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_get_stale_agent_session_returns_empty_session() -> None:
    client = TestClient(app)

    response = client.get("/api/agent/sessions/stale-session")

    assert response.status_code == 200
    assert response.json()["id"] == "stale-session"
    assert response.json()["turns"] == []
