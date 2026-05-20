"""Subagent definitions.

DeepAgents 0.6 lets you register named subagents that the parent invokes
via the built-in ``task`` tool. Each subagent has its own system prompt,
its own scoped tool set, and runs inside the same code interpreter — so
the parent agent can fire several in parallel from Python and aggregate
the results without a model round-trip per call.
"""
from __future__ import annotations

from src.agent.prompts import (
    ADVISOR_PROMPT,
    ANALYST_PROMPT,
    CATEGORIZER_PROMPT,
    FRAUD_ANALYST_PROMPT,
)
from src.retriever.tools import graph_query, wiki_list, wiki_read

SUBAGENTS = [
    {
        "name": "categorizer",
        "description": "Propose canonical_name + category + kind for a raw transaction description.",
        "prompt": CATEGORIZER_PROMPT,
        "tools": [],   # pure-LLM classifier, no graph access
    },
    {
        "name": "analyst",
        "description": "Compute aggregations, trends, and outliers from the context graph and wiki.",
        "prompt": ANALYST_PROMPT,
        "tools": [graph_query, wiki_read, wiki_list],
    },
    {
        "name": "advisor",
        "description": "Generate 1–3 grounded, actionable savings recommendations from analyst findings.",
        "prompt": ADVISOR_PROMPT,
        "tools": [wiki_read],
    },
    {
        "name": "fraud_analyst",
        "description": "Investigate flagged transactions and risk patterns in the context graph.",
        "prompt": FRAUD_ANALYST_PROMPT,
        "tools": [graph_query, wiki_read, wiki_list],
    },
]
