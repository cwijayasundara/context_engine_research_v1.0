"""Build the Deep Agents retrieval harness."""
from __future__ import annotations

import os
from typing import Any

from src.deep_retrieval.profiles import register_gemini_flash_profile
from src.deep_retrieval.prompts import (
    ADVISOR_PROMPT,
    FRAUD_INVESTIGATOR_PROMPT,
    SYSTEM_PROMPT,
    WIKI_RETRIEVER_PROMPT,
    graph_analyst_prompt,
)
from src.deep_retrieval.tools import retrieval_tools


DEFAULT_DEEP_AGENT_MODEL = "google_genai:gemini-3.5-flash"


def build_deep_agent() -> Any:
    """Construct the Deep Agents 0.6 retrieval agent with lazy imports."""
    try:
        from deepagents import create_deep_agent  # type: ignore
        from langchain_quickjs import REPLMiddleware  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Deep Agents runtime requires deepagents[quickjs] and langchain-quickjs"
        ) from exc

    register_gemini_flash_profile()
    model = os.getenv("DEEP_AGENT_MODEL", DEFAULT_DEEP_AGENT_MODEL)
    tools = retrieval_tools()
    subagents = [
        {
            "name": "graph_analyst",
            "description": "Generate safe Cypher and retrieve grounded Neo4j facts.",
            "prompt": graph_analyst_prompt(),
            "tools": tools,
        },
        {
            "name": "wiki_retriever",
            "description": "Read compiled markdown context artifacts.",
            "prompt": WIKI_RETRIEVER_PROMPT,
            "tools": tools,
        },
        {
            "name": "fraud_investigator",
            "description": "Inspect fraud alerts, scores, and anomaly rationale.",
            "prompt": FRAUD_INVESTIGATOR_PROMPT,
            "tools": tools,
        },
        {
            "name": "advisor",
            "description": "Synthesize recommendations from retrieved evidence.",
            "prompt": ADVISOR_PROMPT,
            "tools": tools,
        },
    ]
    kwargs = {
        "model": model,
        "tools": tools,
        "subagents": subagents,
        "middleware": [REPLMiddleware()],
    }
    try:
        return create_deep_agent(instructions=SYSTEM_PROMPT, **kwargs)
    except TypeError:
        return create_deep_agent(system_prompt=SYSTEM_PROMPT, **kwargs)
