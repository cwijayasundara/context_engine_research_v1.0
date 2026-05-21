"""Chat surface — POST /api/agent/ask (SSE) + session GETs."""
from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import logging
import time
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from src.api.agent_runner import run_agent_stream
from src.api.decisions import write_decision
from src.api.sessions import STORE
from src.deep_retrieval.runtime import run_deep_agent_stream

log = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, description="Existing session to append to; "
                                                     "omit to create a new one.")


def _select_agent_stream(
    *,
    current_runner=run_agent_stream,
    deep_runner=run_deep_agent_stream,
):
    runtime = os.getenv("AGENT_RUNTIME", "deepagents").strip().lower()
    return current_runner if runtime in {"current", "legacy"} else deep_runner


@router.post("/sessions")
def create_session() -> dict:
    return {"session_id": STORE.create()}


@router.get("/sessions/{sid}")
def get_session(sid: str) -> dict:
    return STORE.get_or_create(sid)


@router.post("/ask")
async def ask(req: AskRequest) -> EventSourceResponse:
    sid = req.session_id or STORE.create()
    history = STORE.history_for_planner(sid)
    log.info("POST /api/agent/ask sid=%s history=%d turns Q=%r",
             sid, len(history), req.question[:120])

    async def stream() -> AsyncIterator[ServerSentEvent]:
        # First event tells the client the canonical session id (handy when
        # the request didn't include one — the UI then persists it locally).
        captured: list[tuple[str, dict]] = []
        yield ServerSentEvent(event="session", data=json.dumps({"session_id": sid}))
        captured.append(("session", {"session_id": sid}))

        wire_t0 = time.perf_counter()
        bytes_out = 0
        runner = _select_agent_stream()
        async for name, data in runner(req.question, history=history):
            captured.append((name, data))
            try:
                payload = json.dumps(data, default=str)
            except (TypeError, ValueError):
                log.warning("dropping unserializable %s event", name)
                continue
            bytes_out += len(payload)
            yield ServerSentEvent(event=name, data=payload)

        # Persist *after* the stream so a disconnected client still records
        # whatever events the agent emitted before they vanished.
        STORE.append_turn(sid, req.question, captured)

        counts = Counter(ev for ev, _ in captured)
        log.info(
            "sse complete sid=%s · %.1fs · %d events (%s) · %d bytes",
            sid, time.perf_counter() - wire_t0, len(captured),
            ", ".join(f"{k}={v}" for k, v in counts.most_common()),
            bytes_out,
        )

        # Record a Decision node linking the agent's reasoning to every
        # graph node it touched. Best-effort — never breaks the response.
        node_ids: list[str] = []
        tools: list[str] = []
        summary = ""
        for ev, data in captured:
            if ev == "graph_highlight" and isinstance(data.get("node_ids"), list):
                node_ids.extend(data["node_ids"])
            elif ev == "tool_call" and data.get("name"):
                tools.append(str(data["name"]))
            elif ev == "result" and data.get("answer"):
                summary = str(data["answer"])
        dec_id = write_decision(
            question=req.question,
            ts_iso=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            tools=tools,
            node_ids=node_ids,
        )
        if dec_id:
            log.info("decision_recorded sid=%s id=%s touched=%d tools=%s",
                     sid, dec_id, len(set(node_ids)), tools)
            yield ServerSentEvent(event="decision_recorded",
                                  data=json.dumps({"decision_id": dec_id}))
        else:
            log.warning("decision NOT recorded sid=%s (write_decision returned None)", sid)

    return EventSourceResponse(stream(), ping=15)
