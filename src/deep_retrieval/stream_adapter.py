"""Adapt Deep Agents stream events to the existing SSE event contract."""
from __future__ import annotations

import json
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

    if raw.get("type") == "event" and "method" in raw and isinstance(raw.get("params"), dict):
        return _adapt_protocol_event(raw)

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
            (
                "tool_result",
                {
                    "name": str(name),
                    "result": output,
                    "preview": _tool_preview(output),
                },
            ),
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


def extract_final_answer(raw: Any) -> str:
    """Extract the latest assistant answer from Deep Agents/LangGraph state."""
    if isinstance(raw, dict) and raw.get("type") == "event":
        params = raw.get("params")
        if isinstance(params, dict):
            return _extract_answer_from_value(params.get("data"))
    if isinstance(raw, dict):
        return _extract_answer_from_value(raw.get("data", raw))
    return _extract_answer_from_value(raw)


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


def _adapt_protocol_event(raw: dict[str, Any]) -> list[StreamEvent]:
    method = str(raw.get("method") or "")
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    data = params.get("data")

    if method == "messages":
        payload = data[0] if isinstance(data, tuple) and data else data
        if not _is_ai_message(payload):
            return []
        text = _extract_text_from_message(payload)
        return [("token", {"text": text, "subagent": "synthesizer"})] if text else []

    if method == "custom":
        return adapt_stream_event(data)

    return []


def _extract_answer_from_value(value: Any) -> str:
    if value is None:
        return ""
    if _is_tool_message(value) or _is_human_message(value):
        return ""
    message_text = _extract_text_from_message(value)
    if message_text and _is_ai_message(value):
        return message_text
    if isinstance(value, dict):
        if _is_tool_message(value) or _is_human_message(value):
            return ""
        for key in ("answer", "output"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key]
        messages = value.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                text = _extract_answer_from_value(message)
                if text:
                    return text
        for nested in reversed(list(value.values())):
            text = _extract_answer_from_value(nested)
            if text:
                return text
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            text = _extract_answer_from_value(item)
            if text:
                return text
    return ""


def _extract_text_from_message(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content") or message.get("text")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _is_ai_message(message: Any) -> bool:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "")
        return role in {"ai", "assistant"}
    msg_type = str(getattr(message, "type", ""))
    if msg_type in {"ai", "assistant"}:
        return True
    return message.__class__.__name__.lower().startswith("ai")


def _is_tool_message(message: Any) -> bool:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "")
        return role == "tool"
    return str(getattr(message, "type", "")) == "tool"


def _is_human_message(message: Any) -> bool:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "")
        return role in {"human", "user"}
    return str(getattr(message, "type", "")) in {"human", "user"}


def _tool_preview(output: Any) -> str:
    if not isinstance(output, dict):
        return _compact_json(output)
    if "schema" in output and "examples" in output:
        examples = output.get("examples")
        count = len(examples) if isinstance(examples, list) else 0
        return f"Finance graph schema loaded ({count} examples)"
    rows = output.get("rows")
    if isinstance(rows, list):
        header = f"{len(rows)} row{'s' if len(rows) != 1 else ''}"
        if not rows:
            return header
        lines = [header]
        for row in rows[:5]:
            if isinstance(row, dict):
                parts = [
                    f"{key}: {_format_cell(value)}"
                    for key, value in list(row.items())[:4]
                ]
                lines.append(" | ".join(parts))
            else:
                lines.append(_format_cell(row))
        if len(rows) > 5:
            lines.append(f"... {len(rows) - 5} more")
        return "\n".join(lines)
    return _compact_json(output)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if value is None:
        return "null"
    return str(value)


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False)[:500]
    except (TypeError, ValueError):
        return str(value)[:500]
