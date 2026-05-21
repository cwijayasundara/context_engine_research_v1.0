from __future__ import annotations

from langchain_core.messages import AIMessage

from src.deep_retrieval.stream_adapter import adapt_stream_event, extract_final_answer


def test_preserves_already_normalized_events() -> None:
    assert adapt_stream_event(("token", {"text": "hello"})) == [
        ("token", {"text": "hello"}),
    ]


def test_maps_message_deltas_to_token_events() -> None:
    adapted = adapt_stream_event({
        "event": "messages",
        "data": {"delta": {"content": "£42"}},
        "name": "graph_analyst",
    })

    assert adapted == [("token", {"text": "£42", "subagent": "graph_analyst"})]


def test_maps_tool_events() -> None:
    call = adapt_stream_event({
        "event": "tool_call",
        "name": "graph_query",
        "data": {"args": {"cypher": "MATCH (n) RETURN n LIMIT 1"}},
    })
    result = adapt_stream_event({
        "event": "tool_result",
        "name": "graph_query",
        "data": {"output": {"rows": [{"x": 1}]}},
    })

    assert call == [("tool_call", {"name": "graph_query", "args": {"cypher": "MATCH (n) RETURN n LIMIT 1"}})]
    assert result == [("tool_result", {"name": "graph_query", "result": {"rows": [{"x": 1}]}})]


def test_maps_subagent_lifecycle_and_final_events() -> None:
    assert adapt_stream_event({"event": "subagent_start", "name": "wiki_retriever", "data": {"brief": "read wiki"}}) == [
        ("subagent_start", {"name": "wiki_retriever", "brief": "read wiki"}),
    ]
    assert adapt_stream_event({"event": "subagent_end", "name": "wiki_retriever", "data": {"ok": True}}) == [
        ("subagent_end", {"name": "wiki_retriever", "ok": True}),
    ]
    assert adapt_stream_event({"event": "final", "data": {"content": "answer"}}) == [
        ("result", {"answer": "answer"}),
    ]
    assert adapt_stream_event({"event": "end", "data": {}}) == [("done", {})]


def test_maps_errors_and_ignores_unknown_events() -> None:
    assert adapt_stream_event({"event": "error", "data": {"message": "bad"}}) == [
        ("error", {"message": "bad"}),
    ]
    assert adapt_stream_event({"event": "metadata", "data": {"x": 1}}) == []


def test_graph_tool_result_emits_highlight_and_update() -> None:
    adapted = adapt_stream_event({
        "event": "tool_result",
        "name": "graph_query",
        "data": {
            "output": {
                "rows": [],
                "node_ids": ["merchant:Costco"],
                "graph_update": {
                    "nodes": [{"id": "merchant:Costco", "label": "Costco", "type": "Merchant"}],
                    "relationships": [],
                    "focus_ids": ["merchant:Costco"],
                    "mode": "merge",
                },
            }
        },
    })

    assert ("tool_result", {"name": "graph_query", "result": adapted[0][1]["result"]}) in adapted
    assert ("graph_highlight", {"node_ids": ["merchant:Costco"]}) in adapted
    assert adapted[-1][0] == "graph_update"


def test_ignores_protocol_values_but_extracts_final_answer() -> None:
    raw = {
        "type": "event",
        "method": "values",
        "params": {
            "data": {
                "messages": [
                    {"role": "user", "content": "question"},
                    AIMessage(content="The answer is GBP 42."),
                ],
            },
        },
    }

    assert adapt_stream_event(raw) == []
    assert extract_final_answer(raw) == "The answer is GBP 42."


def test_maps_protocol_message_events_to_tokens() -> None:
    raw = {
        "type": "event",
        "method": "messages",
        "params": {"data": (AIMessage(content="partial"), {"node": "model"})},
    }

    assert adapt_stream_event(raw) == [
        ("token", {"text": "partial", "subagent": "synthesizer"}),
    ]
