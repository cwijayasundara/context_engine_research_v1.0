from __future__ import annotations

import warnings

from langchain_core._api import LangChainBetaWarning


def test_load_repl_middleware_from_installed_quickjs_package() -> None:
    from src.deep_retrieval.builder import _load_repl_middleware

    middleware_cls = _load_repl_middleware()

    assert middleware_cls.__name__ in {"REPLMiddleware", "CodeInterpreterMiddleware"}


def test_build_deep_agent_constructs_compiled_graph(monkeypatch) -> None:
    from src.deep_retrieval.builder import build_deep_agent

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = build_deep_agent()

    assert type(agent).__name__ == "CompiledStateGraph"
    assert not [
        warning for warning in caught
        if issubclass(warning.category, LangChainBetaWarning)
    ]
