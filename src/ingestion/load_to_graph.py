"""Load normalized transactions into Neo4j.

Idempotent: re-running this on the same input upserts nodes/relationships
via `MERGE`. Uses periodic batching for sane memory behaviour on large
statement archives.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase, Session

from src.config import SETTINGS
from src.ingestion.normalize import canonicalize

log = logging.getLogger(__name__)

BATCH_SIZE = 500


# Cypher templates -------------------------------------------------------------

UPSERT_TX = """
MERGE (acct:Account {id: $account_token})
  ON CREATE SET acct.institution = $institution,
                acct.account_type = $account_type
MERGE (stmt:Statement {id: $statement_id})
MERGE (acct)-[:HAS_STATEMENT]->(stmt)

MERGE (mo:Month {id: $month})
  ON CREATE SET mo.year = $year
MERGE (cat:Category {id: $category})
  ON CREATE SET cat.name = $category, cat.kind = $kind
MERGE (m:Merchant {id: $merchant_id})
  ON CREATE SET m.name = $merchant_name,
                m.canonical_name = $merchant_canonical,
                m.aliases = [$description]
  ON MATCH  SET m.aliases = CASE
                WHEN $description IN m.aliases THEN m.aliases
                ELSE m.aliases + $description END
MERGE (m)-[:IN_CATEGORY]->(cat)

MERGE (t:Transaction {id: $tx_id})
  ON CREATE SET t.date = date($date),
                t.amount = $amount,
                t.description = $description,
                t.balance = $balance,
                t.year = $year,
                t.month = $month
MERGE (stmt)-[:CONTAINS]->(t)
MERGE (t)-[:FROM_ACCOUNT]->(acct)
MERGE (t)-[:AT]->(m)
MERGE (t)-[:IN_MONTH]->(mo)

MERGE (day:Day {id: $day_id})
  ON CREATE SET day.weekday = $weekday, day.month = $month
MERGE (t)-[:ON_DAY]->(day)

MERGE (tpl:DescriptionTemplate {id: $template_id})
  ON CREATE SET tpl.template = $template_str
MERGE (t)-[:MATCHES_TEMPLATE]->(tpl)

FOREACH (_ IN CASE WHEN $location_id IS NULL THEN [] ELSE [1] END |
  MERGE (loc:Location {id: $location_id})
    ON CREATE SET loc.name = $location_name, loc.country = $location_country
  MERGE (t)-[:AT_LOCATION]->(loc)
)
"""


def load_directory(normalized_dir: Path) -> int:
    # If the LLM normalizer is selected, warm its cache with every unique
    # description first — that turns ~1.5k per-row LLM calls into ~3 batches.
    if os.getenv("NORMALIZER", "regex").lower() == "llm":
        from src.ingestion.llm_normalize import prime_cache
        descs = (
            json.loads(line)["description"]
            for path in sorted(normalized_dir.glob("*.jsonl"))
            for line in path.open()
        )
        stats = prime_cache(descs)
        log.info("LLM canon cache primed: %s", stats)

    driver = GraphDatabase.driver(
        SETTINGS.neo4j_uri,
        auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password),
    )
    count = 0
    try:
        with driver.session(database=SETTINGS.neo4j_database) as session:
            _apply_schema(session)
            for path in sorted(normalized_dir.glob("*.jsonl")):
                count += _load_file(session, path)
    finally:
        driver.close()
    log.info("loaded %d transactions in total", count)
    return count


def _apply_schema(session: Session) -> None:
    schema_path = Path(__file__).parent.parent / "ontology" / "schema.cypher"
    statements = [
        s.strip() for s in schema_path.read_text().split(";") if s.strip() and not s.strip().startswith("//")
    ]
    for stmt in statements:
        session.run(stmt)
    log.info("schema applied (%d statements)", len(statements))


def _load_file(session: Session, path: Path) -> int:
    log.info("loading %s", path.name)
    batch: list[dict] = []
    n = 0
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            batch.append(_to_cypher_params(record))
            if len(batch) >= BATCH_SIZE:
                _flush(session, batch)
                n += len(batch)
                batch.clear()
    if batch:
        _flush(session, batch)
        n += len(batch)
    log.info("  → %d rows", n)
    return n


def _flush(session: Session, batch: list[dict]) -> None:
    # Each batch runs in one tx for write efficiency.
    with session.begin_transaction() as tx:
        for params in batch:
            tx.run(UPSERT_TX, **params)
        tx.commit()


def _to_cypher_params(record: dict) -> dict:
    from datetime import date as _date
    from src.ingestion.locations import parse_location
    from src.ingestion.templates import fingerprint

    info = canonicalize(record["description"])
    merchant_id = info.canonical_name.lower().replace(" ", "-")
    d = _date.fromisoformat(record["date"])
    loc = parse_location(record["description"])
    tpl = fingerprint(record["description"])
    return {
        "tx_id": f"{record['statement_id']}-{record['date']}-{record['amount']}-{merchant_id}",
        "account_token": record["account_id_token"],
        "institution": record["institution"],
        "account_type": "checking",
        "statement_id": record["statement_id"],
        "month": record["month"],
        "year": record["year"],
        "category": info.category,
        "kind": info.kind,
        "merchant_id": merchant_id,
        "merchant_name": info.canonical_name,
        "merchant_canonical": info.canonical_name,
        "description": record["description"],
        "date": record["date"],
        "amount": record["amount"],
        "balance": record.get("balance"),
        "day_id":   record["date"],
        "weekday":  d.weekday(),
        "template_id":  tpl.id,
        "template_str": tpl.template,
        "location_id":      loc.id      if loc else None,
        "location_name":    loc.name    if loc else None,
        "location_country": loc.country if loc else None,
    }


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    argv = argv or sys.argv[1:]
    normalized = Path(argv[0]) if argv else SETTINGS.statements_dir / "_normalized"
    if not normalized.exists():
        log.error("normalized dir not found: %s", normalized)
        log.error("run `python -m src.ingestion.parse_statements` first")
        return 1
    load_directory(normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
