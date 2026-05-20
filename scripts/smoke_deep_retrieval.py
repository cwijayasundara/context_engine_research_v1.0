"""Smoke test the Deep Agents retrieval runtime.

Run from repo root:

    GOOGLE_API_KEY=... python scripts/smoke_deep_retrieval.py
"""
from __future__ import annotations

import asyncio
import os
import sys

from src.deep_retrieval.runtime import run_deep_agent_stream


QUESTION = "How much did I spend at Costco in 2025?"


async def main() -> int:
    runtime = os.getenv("AGENT_RUNTIME", "deepagents").strip().lower()
    if runtime in {"current", "legacy"}:
        print("Unset AGENT_RUNTIME or set AGENT_RUNTIME=deepagents to smoke the Deep Agents runtime.", file=sys.stderr)
        return 2
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is required for the Gemini Deep Agents runtime.", file=sys.stderr)
        return 2

    saw_result = False
    saw_tool = False
    async for event, data in run_deep_agent_stream(QUESTION):
        print(event, data)
        saw_result = saw_result or event == "result"
        saw_tool = saw_tool or event == "tool_call"
        if event == "error":
            return 1

    if not saw_result:
        print("No final result event emitted.", file=sys.stderr)
        return 1
    if not saw_tool:
        print("No tool_call event emitted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
