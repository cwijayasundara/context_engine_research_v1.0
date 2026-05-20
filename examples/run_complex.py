"""Complex multi-step analysis — recursive subagent workflow.

  $ python examples/run_complex.py

This kicks off the full pipeline:
  1. Main agent spawns three subagents in parallel from inside the interpreter:
        - income_analyzer
        - expense_categorizer
        - savings_advisor
  2. Analyst subagents read wiki entries and run Cypher.
  3. Advisor subagent waits on the analyst findings, then writes
     a savings plan that cites specific wiki entries.
  4. Final report is saved to data/wiki/reports/<year>-savings-plan.md.
"""
from __future__ import annotations

import logging

from rich.console import Console

from src.agent.main import run_agent

QUESTION = """\
Do a complete analysis on my income and expenditure for 2025. Identify
where I spend most money, surface any unusual trends or anomalies, and
recommend 3 concrete actions I could take to maximize my savings — each
with an estimated annual impact in dollars.

Save the final write-up to data/wiki/reports/2025-savings-plan.md and
cite the wiki entries you relied on.
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    console = Console()
    console.rule("[bold]Question[/bold]")
    console.print(QUESTION)
    console.rule("[bold]Answer[/bold]")
    console.print(run_agent(QUESTION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
