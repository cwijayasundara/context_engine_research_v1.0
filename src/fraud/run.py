"""End-to-end fraud detection runner.

Usage::

    python -m src.fraud.run                # score the whole graph
    python -m src.fraud.run --skip-gds     # rules only (fast iteration)
"""
from __future__ import annotations

import argparse
import logging

from neo4j import GraphDatabase

from src.config import SETTINGS
from src.fraud.gds import GdsClient
from src.fraud.score import score_all, write_back


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Score every transaction in the graph.")
    parser.add_argument("--skip-gds", action="store_true",
                        help="Skip GDS projection + algorithms; rules only.")
    parser.add_argument("--fastrp-dim", type=int, default=64)
    parser.add_argument("--knn-top-k",  type=int, default=5)
    args = parser.parse_args(argv)

    driver = GraphDatabase.driver(
        SETTINGS.neo4j_uri, auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password)
    )
    try:
        if not args.skip_gds:
            gds = GdsClient(driver)
            gds.project_merchant_coincidence()
            gds.run_pagerank()
            gds.run_louvain()
            gds.run_fastrp(dim=args.fastrp_dim)
            gds.run_knn(top_k=args.knn_top_k)
            gds.run_node_similarity()
            gds.mark_outliers()
        per_tx = score_all(driver)
        flagged = write_back(driver, per_tx)
        print(f"scored {len(per_tx)} transactions, created {flagged} alerts")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
