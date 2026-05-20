"""The DeepAgents 0.6 harness.

This is where all four pillars come together:

  - Model               openai:gpt-5.4-mini
  - PTC + recursive     langchain_quickjs.REPLMiddleware
  - PII enforcement     PIIMiddleware (custom)
  - Tools               graph_query / wiki_read / wiki_list
  - Subagents           categorizer / analyst / advisor
  - Backend             ContextHubBackend (versioned wiki + memories)

Use ``run_agent(question)`` from your application code, or invoke the
examples in ``examples/``.
"""
from __future__ import annotations

import logging

from deepagents import create_deep_agent
from deepagents.backends import ContextHubBackend  # type: ignore[import-not-found]
from langchain_quickjs import REPLMiddleware  # type: ignore[import-not-found]

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.subagents import SUBAGENTS
from src.config import SETTINGS
from src.pii.filter import PIIFilter
from src.pii.vault import get_default_vault
from src.retriever.pii_middleware import PIIMiddleware
from src.retriever.tools import TOOLS

log = logging.getLogger(__name__)


def build_agent():
    """Construct the DeepAgents harness. Cached for reuse if you wrap this."""
    log.info("building DeepAgents harness with model=%s", SETTINGS.model)

    middleware = [
        PIIMiddleware(),    # tokenize on egress (MUST be before REPL so tool args are scrubbed)
        REPLMiddleware(),   # PTC + recursive workflows
    ]

    backend = None
    if SETTINGS.langsmith_api_key:
        backend = ContextHubBackend("finance-context-engine")

    return create_deep_agent(
        model=SETTINGS.model,
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        subagents=SUBAGENTS,
        middleware=middleware,
        backend=backend,
    )


def run_agent(question: str) -> str:
    """End-to-end execution: tokenize question, run agent, detokenize answer.

    The middleware tokenizes again internally; tokenizing here too is
    deterministic thanks to the vault — same value, same token.
    """
    agent = build_agent()
    pii = PIIFilter(vault=get_default_vault(salt=SETTINGS.pii_token_salt))
    tokenized = pii.tokenize(question).text

    result = agent.invoke({"messages": [{"role": "user", "content": tokenized}]})
    final = _extract_final_text(result)
    return pii.detokenize(final)


def _extract_final_text(result: dict) -> str:
    """Grab the last AI message content as a string."""
    messages = result.get("messages") or []
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or msg.get("role", "")        # type: ignore[union-attr]
        content = getattr(msg, "content", None) or msg.get("content", "")   # type: ignore[union-attr]
        if role in {"ai", "assistant"} and content:
            if isinstance(content, list):
                return "\n".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            return str(content)
    return ""
