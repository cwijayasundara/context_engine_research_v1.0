"""Tiny in-memory session store for the chat panel.

A session is a chronological list of turns. Each turn records the user
question and an envelope of the events that fired (plan, tool_call,
graph_highlight, …, result) so the frontend can rehydrate the transcript
on page reload.

In a real deployment swap this for SQLite or a tiny KV. For Phase 3 the
in-memory version is enough — agent state is cheap to recompute.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from threading import Lock

# Keep at most this many sessions in memory. Anything older drops off the
# LRU. The chat is single-user dev tooling, so the cap is tiny.
_MAX_SESSIONS = 128
_MAX_EVENTS_PER_TURN = 800


class _SessionStore:
    """Thread-safe LRU map. ``get`` resurrects an entry to the front."""

    def __init__(self, max_size: int = _MAX_SESSIONS) -> None:
        self._lock = Lock()
        self._max = max_size
        self._data: "OrderedDict[str, dict]" = OrderedDict()

    def create(self) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._data[sid] = {
                "id": sid,
                "created_at": time.time(),
                "turns": [],
            }
            self._data.move_to_end(sid)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
        return sid

    def get(self, sid: str) -> dict | None:
        with self._lock:
            entry = self._data.get(sid)
            if entry is not None:
                self._data.move_to_end(sid)
            return entry

    def append_turn(self, sid: str, question: str, events: list[tuple[str, dict]]) -> dict:
        """Append a turn. Auto-creates the session if it doesn't exist yet."""
        with self._lock:
            entry = self._data.get(sid)
            if entry is None:
                entry = self._data[sid] = {
                    "id": sid,
                    "created_at": time.time(),
                    "turns": [],
                }
                while len(self._data) > self._max:
                    self._data.popitem(last=False)
            else:
                self._data.move_to_end(sid)
            turn = {
                "ts": time.time(),
                "question": question,
                # Cap events so a runaway agent can't blow memory.
                "events": events[:_MAX_EVENTS_PER_TURN],
            }
            entry["turns"].append(turn)
            return turn

    def history_for_planner(self, sid: str) -> list[dict]:
        """A flat list of {role, content} pairs for the planner's context."""
        entry = self.get(sid)
        if not entry:
            return []
        out: list[dict] = []
        for turn in entry["turns"]:
            out.append({"role": "user", "content": turn["question"]})
            # Find the final synthesised answer for this turn, if any.
            answer = ""
            for name, data in turn["events"]:
                if name == "result":
                    answer = data.get("answer", "") or ""
            if answer:
                out.append({"role": "assistant", "content": answer})
        return out


STORE = _SessionStore()
