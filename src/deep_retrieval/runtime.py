"""Public Deep Agents runtime entry point."""
from __future__ import annotations

import os
import inspect
import warnings
from contextlib import contextmanager
from collections.abc import AsyncIterator

from src.deep_retrieval.builder import build_deep_agent
from src.deep_retrieval.stream_adapter import adapt_stream_event, extract_final_answer


async def run_deep_agent_stream(
    question: str,
    *,
    history: list[dict] | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Stream Deep Agents retrieval events through the existing SSE contract."""
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        yield "error", {"message": "GOOGLE_API_KEY missing for Deep Agents Gemini runtime"}
        return

    yield "started", {"question": question, "runtime": "deepagents"}
    try:
        emitted_result = False
        emitted_done = False
        emitted_visible = False
        final_answer = ""
        with _deep_agent_langsmith_env():
            agent = build_deep_agent()
            payload = {"messages": _messages(question, history or [])}
            with _suppress_langchain_beta_warnings():
                stream = agent.astream_events(payload, version="v3")
                if inspect.isawaitable(stream):
                    stream = await stream
                async for raw in stream:
                    final_answer = extract_final_answer(raw) or final_answer
                    for event in adapt_stream_event(raw):
                        emitted_visible = True
                        emitted_result = emitted_result or event[0] == "result"
                        emitted_done = emitted_done or event[0] == "done"
                        yield event
        if not emitted_result and final_answer:
            emitted_visible = True
            yield "result", {"answer": final_answer}
        if not emitted_done:
            if not emitted_visible:
                yield "error", {"message": "Deep Agents completed without a visible response"}
            else:
                yield "done", {}
    except Exception as exc:
        yield "error", {"message": f"deepagents: {exc.__class__.__name__}: {exc}"}


def _messages(question: str, history: list[dict]) -> list[dict]:
    messages = [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
        for item in history[-6:]
        if item.get("content")
    ]
    messages.append({"role": "user", "content": question})
    return messages


@contextmanager
def _deep_agent_langsmith_env():
    """Disable hosted LangSmith uploads when tracing is not configured."""
    enabled = os.getenv("DEEP_AGENT_LANGSMITH_TRACING", "").strip().lower()
    if enabled in {"1", "true", "yes", "on"}:
        yield
        return
    if enabled not in {"0", "false", "no", "off"} and os.getenv("LANGSMITH_API_KEY"):
        yield
        return

    keys = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "false"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _suppress_langchain_beta_warnings():
    try:
        from langchain_core._api import LangChainBetaWarning
    except Exception:  # pragma: no cover - compatibility with older LangChain
        LangChainBetaWarning = Warning  # type: ignore

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LangChainBetaWarning)
        yield
