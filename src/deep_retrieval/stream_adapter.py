"""Adapt Deep Agents stream events to the existing SSE event contract."""
from __future__ import annotations

from typing import Any

StreamEvent = tuple[str, dict[str, Any]]


def adapt_stream_event(raw: Any) -> list[StreamEvent]:
    """Convert one raw Deep Agents-like event into frontend SSE events.

    The adapter is intentionally permissive because Deep Agents v3 events may
    carry provider-specific payload shapes. Unknown metadata events are ignored.
    """
    if _is_normalized(raw):
        name, data = raw
        return [(str(name), dict(data))]
    if not isinstance(raw, dict):
        return []

    event = str(raw.get("event") or raw.get("type") or "")
    name = raw.get("name") or raw.get("subagent")
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}

    if event in {"messages", "message", "on_chat_model_stream"}:
        text = _extract_text_delta(data)
        return [("token", {"text": text, "subagent": str(name)})] if text else []

    if event in {"tool_call", "on_tool_start"}:
        return [("tool_call", {"name": str(name), "args": data.get("args", {})})]

    if event in {"tool_result", "on_tool_end"}:
        output = data.get("output", data.get("result", data))
        events: list[StreamEvent] = [
            ("tool_result", {"name": str(name), "result": output}),
        ]
        if isinstance(output, dict):
            node_ids = output.get("node_ids")
            if isinstance(node_ids, list) and node_ids:
                events.append(("graph_highlight", {"node_ids": node_ids}))
            graph_update = output.get("graph_update")
            if isinstance(graph_update, dict):
                events.append(("graph_update", graph_update))
        return events

    if event in {"subagent_start", "on_subagent_start"}:
        return [("subagent_start", {"name": str(name), "brief": data.get("brief", "")})]

    if event in {"subagent_end", "on_subagent_end"}:
        return [("subagent_end", {"name": str(name), "ok": bool(data.get("ok", True))})]

    if event in {"final", "result"}:
        answer = data.get("answer", data.get("content", ""))
        return [("result", {"answer": str(answer)})]

    if event in {"end", "done"}:
        return [("done", {})]

    if event == "error":
        return [("error", {"message": str(data.get("message", "unknown error"))})]

    return []


def _is_normalized(raw: Any) -> bool:
    return (
        isinstance(raw, tuple)
        and len(raw) == 2
        and isinstance(raw[0], str)
        and isinstance(raw[1], dict)
    )


def _extract_text_delta(data: dict[str, Any]) -> str:
    delta = data.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content") or delta.get("text")
        if content:
            return str(content)
    message = data.get("message")
    content = getattr(message, "content", None)
    if content:
        return str(content)
    for key in ("content", "text"):
        if data.get(key):
            return str(data[key])
    return ""
