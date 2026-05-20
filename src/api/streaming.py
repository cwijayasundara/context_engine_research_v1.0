"""SSE helpers — bridge the agent runner's ``(event, data)`` tuples into
``sse-starlette`` ``ServerSentEvent`` objects.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from sse_starlette.sse import ServerSentEvent

log = logging.getLogger(__name__)


async def to_sse(events: AsyncIterator[tuple[str, dict]]) -> AsyncIterator[ServerSentEvent]:
    """Wrap the runner's tuples into SSE frames."""
    async for name, data in events:
        try:
            payload = json.dumps(data, default=str)
        except (TypeError, ValueError) as exc:
            log.warning("dropping unserializable SSE event %r: %s", name, exc)
            continue
        yield ServerSentEvent(event=name, data=payload)
