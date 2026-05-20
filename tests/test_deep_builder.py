from __future__ import annotations


def test_load_repl_middleware_from_installed_quickjs_package() -> None:
    from src.deep_retrieval.builder import _load_repl_middleware

    middleware_cls = _load_repl_middleware()

    assert middleware_cls.__name__ in {"REPLMiddleware", "CodeInterpreterMiddleware"}


def test_build_deep_agent_constructs_compiled_graph(monkeypatch) -> None:
    from src.deep_retrieval.builder import build_deep_agent

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    agent = build_deep_agent()

    assert type(agent).__name__ == "CompiledStateGraph"
