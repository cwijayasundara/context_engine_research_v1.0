"""DeepAgents middleware that enforces the PII boundary.

DeepAgents middleware sits between the harness and the LLM. Each turn:

  1. ``modify_model_input``   — runs on the way OUT to the LLM. We walk
     every message and every pending tool result, tokenize any leaked
     PII, and substitute the tokens. After this method returns the LLM
     should never see real names, account numbers, addresses, etc.

  2. ``after_model``          — runs on the way BACK from the LLM. The
     model may emit tokens like ``<ACCT_01>`` in its reasoning or final
     answer. We do NOT detokenize here — the agent loop continues
     reasoning over tokens. Detokenization happens once, at the very
     end, before returning to the user (handled by `main.run_agent`).

This split keeps tokens stable through the agent loop while still
guaranteeing real values only re-appear at the outer boundary.

Reference for middleware shape:
    https://docs.langchain.com/oss/python/deepagents/middleware
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# DeepAgents exposes middleware via a small base class. If the import path
# changes across point releases, adjust here.
from deepagents.middleware import AgentMiddleware  # type: ignore[import-not-found]

from src.config import SETTINGS
from src.pii.filter import PIIFilter
from src.pii.vault import get_default_vault

log = logging.getLogger(__name__)


class PIIMiddleware(AgentMiddleware):
    """Tokenize every message body, tool name, and tool args on the way to
    the LLM. Leaves tokens intact on the way back."""

    name = "pii"

    def __init__(self, filter_: PIIFilter | None = None) -> None:
        self._filter = filter_ or PIIFilter(
            vault=get_default_vault(salt=SETTINGS.pii_token_salt)
        )

    # -- egress: state → model ---------------------------------------

    def modify_model_input(self, state: dict[str, Any]) -> dict[str, Any]:
        messages: list[BaseMessage] = state.get("messages", [])
        sanitized: list[BaseMessage] = []
        for msg in messages:
            sanitized.append(self._sanitize_message(msg))
        return {**state, "messages": sanitized}

    # -- ingress: model → state --------------------------------------

    def after_model(self, state: dict[str, Any]) -> dict[str, Any]:
        # Deliberate no-op. Tokens stay in agent state so subsequent
        # reasoning is consistent. Final detokenization happens once,
        # at the user-visible output boundary.
        return state

    # -- helpers ------------------------------------------------------

    def _sanitize_message(self, msg: BaseMessage) -> BaseMessage:
        if isinstance(msg, HumanMessage):
            scrubbed = self._filter.tokenize(_as_text(msg.content)).text
            return HumanMessage(content=scrubbed, additional_kwargs=msg.additional_kwargs)
        if isinstance(msg, AIMessage):
            content = self._filter.tokenize(_as_text(msg.content)).text
            tool_calls = [self._sanitize_tool_call(tc) for tc in (msg.tool_calls or [])]
            return AIMessage(
                content=content,
                tool_calls=tool_calls,
                additional_kwargs=msg.additional_kwargs,
            )
        if isinstance(msg, ToolMessage):
            content = self._filter.tokenize(_as_text(msg.content)).text
            return ToolMessage(
                content=content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
                additional_kwargs=msg.additional_kwargs,
            )
        # Unknown message types pass through but we still scrub content if textual.
        if hasattr(msg, "content"):
            try:
                msg.content = self._filter.tokenize(_as_text(msg.content)).text  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                log.exception("failed to sanitize unknown message type %s", type(msg))
        return msg

    def _sanitize_tool_call(self, tc: dict[str, Any]) -> dict[str, Any]:
        args = tc.get("args") or {}
        scrubbed_args = {k: self._scrub_any(v) for k, v in args.items()}
        return {**tc, "args": scrubbed_args}

    def _scrub_any(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._filter.tokenize(value).text
        if isinstance(value, list):
            return [self._scrub_any(v) for v in value]
        if isinstance(value, dict):
            return {k: self._scrub_any(v) for k, v in value.items()}
        return value


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multi-part content (e.g. text + tool-use blocks). Join text segments.
        out: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return str(content)
