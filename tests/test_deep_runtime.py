from __future__ import annotations

import pytest

from src.deep_retrieval.runtime import run_deep_agent_stream


@pytest.mark.asyncio
async def test_deep_runtime_requires_google_key(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    events = [
        event async for event in run_deep_agent_stream("How much did I spend?")
    ]

    assert events == [
        ("error", {"message": "GOOGLE_API_KEY missing for Deep Agents Gemini runtime"}),
    ]


@pytest.mark.asyncio
async def test_deep_runtime_streams_adapted_agent_events(monkeypatch) -> None:
    class FakeAgent:
        async def astream_events(self, payload, version):
            assert version == "v3"
            assert payload["messages"][-1]["content"] == "question"
            yield {"event": "messages", "name": "graph_analyst", "data": {"delta": {"content": "hi"}}}
            yield {"event": "final", "data": {"content": "answer"}}
            yield {"event": "end", "data": {}}

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr("src.deep_retrieval.runtime.build_deep_agent", lambda: FakeAgent())

    events = [event async for event in run_deep_agent_stream("question", history=[{"role": "user", "content": "old"}])]

    assert events == [
        ("started", {"question": "question", "runtime": "deepagents"}),
        ("token", {"text": "hi", "subagent": "graph_analyst"}),
        ("result", {"answer": "answer"}),
        ("done", {}),
    ]
