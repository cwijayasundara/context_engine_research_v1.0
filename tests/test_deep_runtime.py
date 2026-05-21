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


@pytest.mark.asyncio
async def test_deep_runtime_awaits_stream_coroutine(monkeypatch) -> None:
    async def stream_events():
        yield {"event": "final", "data": {"content": "answer"}}
        yield {"event": "end", "data": {}}

    class FakeAgent:
        async def astream_events(self, payload, version):
            assert version == "v3"
            return stream_events()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr("src.deep_retrieval.runtime.build_deep_agent", lambda: FakeAgent())

    events = [event async for event in run_deep_agent_stream("question")]

    assert events == [
        ("started", {"question": "question", "runtime": "deepagents"}),
        ("result", {"answer": "answer"}),
        ("done", {}),
    ]


@pytest.mark.asyncio
async def test_deep_runtime_disables_langsmith_tracing_without_key(monkeypatch) -> None:
    observed: dict[str, str | None] = {}

    async def stream_events():
        yield {"event": "end", "data": {}}

    class FakeAgent:
        def astream_events(self, payload, version):
            observed["LANGSMITH_TRACING"] = __import__("os").environ.get("LANGSMITH_TRACING")
            observed["LANGCHAIN_TRACING_V2"] = __import__("os").environ.get("LANGCHAIN_TRACING_V2")
            return stream_events()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("DEEP_AGENT_LANGSMITH_TRACING", raising=False)
    monkeypatch.setattr("src.deep_retrieval.runtime.build_deep_agent", lambda: FakeAgent())

    events = [event async for event in run_deep_agent_stream("question")]

    assert events == [
        ("started", {"question": "question", "runtime": "deepagents"}),
        ("done", {}),
    ]
    assert observed == {
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
    }
    assert __import__("os").environ["LANGSMITH_TRACING"] == "true"
    assert __import__("os").environ["LANGCHAIN_TRACING_V2"] == "true"


@pytest.mark.asyncio
async def test_deep_runtime_keeps_langsmith_tracing_with_key(monkeypatch) -> None:
    observed: dict[str, str | None] = {}

    async def stream_events():
        yield {"event": "end", "data": {}}

    class FakeAgent:
        def astream_events(self, payload, version):
            observed["LANGSMITH_TRACING"] = __import__("os").environ.get("LANGSMITH_TRACING")
            return stream_events()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-valid-looking-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("DEEP_AGENT_LANGSMITH_TRACING", raising=False)
    monkeypatch.setattr("src.deep_retrieval.runtime.build_deep_agent", lambda: FakeAgent())

    events = [event async for event in run_deep_agent_stream("question")]

    assert events == [
        ("started", {"question": "question", "runtime": "deepagents"}),
        ("done", {}),
    ]
    assert observed == {"LANGSMITH_TRACING": "true"}


@pytest.mark.asyncio
async def test_deep_runtime_allows_explicit_langsmith_opt_out(monkeypatch) -> None:
    observed: dict[str, str | None] = {}

    async def stream_events():
        yield {"event": "end", "data": {}}

    class FakeAgent:
        def astream_events(self, payload, version):
            observed["LANGSMITH_TRACING"] = __import__("os").environ.get("LANGSMITH_TRACING")
            return stream_events()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-valid-looking-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("DEEP_AGENT_LANGSMITH_TRACING", "false")
    monkeypatch.setattr("src.deep_retrieval.runtime.build_deep_agent", lambda: FakeAgent())

    events = [event async for event in run_deep_agent_stream("question")]

    assert events == [
        ("started", {"question": "question", "runtime": "deepagents"}),
        ("done", {}),
    ]
    assert observed == {"LANGSMITH_TRACING": "false"}
