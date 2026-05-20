from __future__ import annotations

from src.api.routes import agent


def test_selects_deepagents_runner_by_default(monkeypatch) -> None:
    async def current(question: str, *, history=None):
        yield "result", {"answer": "current"}

    async def deep(question: str, *, history=None):
        yield "result", {"answer": "deep"}

    monkeypatch.delenv("AGENT_RUNTIME", raising=False)
    selected = agent._select_agent_stream(current_runner=current, deep_runner=deep)

    assert selected is deep


def test_selects_current_runner_when_explicitly_requested(monkeypatch) -> None:
    async def current(question: str, *, history=None):
        yield "result", {"answer": "current"}

    async def deep(question: str, *, history=None):
        yield "result", {"answer": "deep"}

    monkeypatch.setenv("AGENT_RUNTIME", "current")
    selected = agent._select_agent_stream(current_runner=current, deep_runner=deep)

    assert selected is current


def test_selects_current_runner_for_legacy_alias(monkeypatch) -> None:
    async def current(question: str, *, history=None):
        yield "result", {"answer": "current"}

    async def deep(question: str, *, history=None):
        yield "result", {"answer": "deep"}

    monkeypatch.setenv("AGENT_RUNTIME", "legacy")
    selected = agent._select_agent_stream(current_runner=current, deep_runner=deep)

    assert selected is current


def test_selects_deepagents_runner_when_enabled(monkeypatch) -> None:
    async def current(question: str, *, history=None):
        yield "result", {"answer": "current"}

    async def deep(question: str, *, history=None):
        yield "result", {"answer": "deep"}

    monkeypatch.setenv("AGENT_RUNTIME", "deepagents")
    selected = agent._select_agent_stream(current_runner=current, deep_runner=deep)

    assert selected is deep


def test_runtime_selection_is_case_and_space_insensitive(monkeypatch) -> None:
    async def current(question: str, *, history=None):
        yield "result", {"answer": "current"}

    async def deep(question: str, *, history=None):
        yield "result", {"answer": "deep"}

    monkeypatch.setenv("AGENT_RUNTIME", " DeepAgents ")
    selected = agent._select_agent_stream(current_runner=current, deep_runner=deep)

    assert selected is deep
