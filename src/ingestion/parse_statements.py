"""Parse markdown statements into normalized Transaction records.

Expected statement shape (see README for details):

  ---
  account_id: "1234"
  account_type: "checking"
  institution: "Chase"
  period_start: "2025-01-01"
  period_end:   "2025-01-31"
  ---
  # Bank Statement — January 2025

  | Date       | Description           | Amount  | Balance  |
  |------------|-----------------------|---------|----------|
  | 2025-01-02 | COSTCO WHOLESALE ...  |  -84.32 |  4815.68 |
  | ...

The parser:
  1. Reads frontmatter for account metadata.
  2. Finds the first markdown table and parses each row as a transaction.
  3. Runs the PII filter over the *description* field before persisting.
  4. Writes a JSONL file to data/statements/_normalized/<name>.jsonl
     for the graph loader to consume.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import frontmatter

from src.config import SETTINGS
from src.pii.filter import PIIFilter
from src.pii.vault import get_default_vault

log = logging.getLogger(__name__)


@dataclass
class ParsedTransaction:
    statement_id: str
    account_id: str
    account_id_token: str        # what the LLM will see
    institution: str
    date: str                    # ISO YYYY-MM-DD
    description: str             # PII-scrubbed
    description_raw_len: int     # for sanity checks
    amount: float
    balance: float | None
    year: int
    month: str                   # YYYY-MM


# Match a markdown table row:  | col | col | ... |
_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_SEP = re.compile(r"^\s*\|[-\s|:]+\|\s*$")


def parse_statement(path: Path, pii: PIIFilter) -> list[ParsedTransaction]:
    post = frontmatter.load(path)
    meta = post.metadata
    if not meta.get("account_id"):
        raise ValueError(f"{path}: missing required frontmatter `account_id`")

    account_id = str(meta["account_id"])
    statement_id = f"{account_id}-{meta.get('period_start', 'unknown')}"
    institution = str(meta.get("institution", "unknown"))

    # Tokenize the raw account id once; transactions reference the token.
    vault = get_default_vault(salt=SETTINGS.pii_token_salt)
    account_token = vault.tokenize_value("ACCT", account_id)

    rows = _extract_table_rows(post.content)
    out: list[ParsedTransaction] = []
    for row in rows:
        tx = _row_to_transaction(
            row=row,
            statement_id=statement_id,
            account_id=account_id,
            account_token=account_token,
            institution=institution,
            pii=pii,
        )
        if tx is not None:
            out.append(tx)
    log.info("%s → %d transactions", path.name, len(out))
    return out


def _extract_table_rows(content: str) -> list[list[str]]:
    """Return body rows of the first markdown table in `content`."""
    lines = content.splitlines()
    rows: list[list[str]] = []
    in_table = False
    header_seen = False
    for line in lines:
        if _SEP.match(line):
            in_table = True
            continue
        m = _ROW.match(line)
        if not m:
            if in_table:
                break       # table ended
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if not header_seen:
            header_seen = True
            continue        # skip header row
        if in_table:
            rows.append(cells)
    return rows


def _row_to_transaction(
    row: list[str],
    statement_id: str,
    account_id: str,
    account_token: str,
    institution: str,
    pii: PIIFilter,
) -> ParsedTransaction | None:
    if len(row) < 3:
        return None
    try:
        d = _parse_date(row[0])
        amount = _parse_money(row[2])
        balance = _parse_money(row[3]) if len(row) >= 4 else None
    except ValueError as e:
        log.warning("skipping malformed row %r: %s", row, e)
        return None

    raw_desc = row[1]
    scrubbed = pii.tokenize(raw_desc).text

    return ParsedTransaction(
        statement_id=statement_id,
        account_id=account_id,                       # stays local, never crosses the LLM boundary
        account_id_token=account_token,              # what the LLM sees
        institution=institution,
        date=d.isoformat(),
        description=scrubbed,
        description_raw_len=len(raw_desc),
        amount=amount,
        balance=balance,
        year=d.year,
        month=f"{d.year:04d}-{d.month:02d}",
    )


def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date: {s!r}")


def _parse_money(s: str) -> float:
    cleaned = s.replace("$", "").replace(",", "").replace("£", "").replace("€", "").strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    return float(cleaned)


# ---------- CLI entrypoint ----------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    argv = argv or sys.argv[1:]
    src = Path(argv[0]) if argv else SETTINGS.statements_dir
    out_dir = src / "_normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    pii = PIIFilter()
    for path in sorted(src.glob("*.md")):
        txs = parse_statement(path, pii=pii)
        out_path = out_dir / (path.stem + ".jsonl")
        with out_path.open("w") as f:
            for tx in txs:
                f.write(json.dumps(asdict(tx)) + "\n")
        log.info("wrote %s (%d txs)", out_path, len(txs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
