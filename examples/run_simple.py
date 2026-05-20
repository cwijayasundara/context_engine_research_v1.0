"""Simple factual query — one-shot PTC trace.

  $ python examples/run_simple.py "How much did I spend at Costco in 2025?"

The agent will:
  1. Read data/wiki/merchants/Costco.md (if present) for a fast answer.
  2. Otherwise, write a single Cypher query inside the interpreter:
        MATCH (t:Transaction)-[:AT]->(m:Merchant {canonical_name:'Costco Wholesale'})
        WHERE t.year = 2025
        RETURN sum(-t.amount) AS total, count(t) AS visits
  3. Render the answer with PII detokenization applied at the boundary.
"""
from __future__ import annotations

import logging
import sys

from rich.console import Console

from src.agent.main import run_agent


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    argv = argv or sys.argv[1:]
    question = " ".join(argv) or "How much did I spend at Costco in 2025?"

    console = Console()
    console.rule(f"[bold]Question[/bold]")
    console.print(question)
    console.rule(f"[bold]Answer[/bold]")
    console.print(run_agent(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
