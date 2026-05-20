"""LLM Wiki Compiler — graph → curated markdown vault.

For each merchant / category / month / year we render a dense markdown
"wiki entry" that summarises the relevant facts. Agents read THESE rather
than raw transactions — they're cheaper to bring into context, already
aggregated, and written in a shape an LLM can act on.

The output is also a valid **Obsidian vault**:

* Every page carries YAML frontmatter (``type``, ``tags``, key totals) so
  Obsidian's Properties panel and Dataview queries work out of the box.
* Cross-references use Obsidian ``[[wikilinks]]`` so the graph view shows
  the merchant ↔ category ↔ month ↔ year structure.
* A top-level ``Home.md`` is a Map-of-Content (MoC) listing every page.

This is the open-source analog of the Pinecone Nexus Context Compiler:
do the expensive aggregation once, when data changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from neo4j import GraphDatabase

from src.config import SETTINGS

log = logging.getLogger(__name__)

# All numbers in this codebase are GBP; the original implementation rendered
# them with ``$`` which was misleading for the UK statement source.
CCY = "£"


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

YEARS_AVAILABLE = """
MATCH (t:Transaction)
RETURN DISTINCT t.year AS year ORDER BY year
"""

MERCHANTS_FOR_YEAR = """
MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE t.year = $year AND c.name <> 'System'
RETURN DISTINCT m.canonical_name AS name, c.name AS category
ORDER BY name
"""

MERCHANT_LIFETIME_TOTALS = """
MATCH (t:Transaction)-[:AT]->(m:Merchant {canonical_name: $name})
RETURN count(t) AS visits, sum(abs(t.amount)) AS amount
"""

MERCHANT_BY_YEAR = """
MATCH (t:Transaction)-[:AT]->(m:Merchant {canonical_name: $name})
RETURN t.year AS year, count(t) AS visits, sum(abs(t.amount)) AS amount
ORDER BY year
"""

MERCHANT_BY_MONTH_ALL = """
MATCH (t:Transaction)-[:AT]->(m:Merchant {canonical_name: $name})
RETURN t.month AS month, count(t) AS visits, sum(abs(t.amount)) AS amount
ORDER BY month
"""

ALL_MERCHANTS = """
MATCH (m:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE c.name <> 'System'
RETURN DISTINCT m.canonical_name AS name, c.name AS category
ORDER BY name
"""

CATEGORIES_FOR_YEAR = """
MATCH (t:Transaction)-[:AT]->(:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE t.year = $year AND c.name <> 'System' AND t.amount < 0
RETURN c.name AS name, sum(-t.amount) AS spend, count(t) AS tx
ORDER BY spend DESC
"""

MERCHANTS_IN_CATEGORY_YEAR = """
MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category {name: $name})
WHERE t.year = $year AND t.amount < 0
RETURN m.canonical_name AS merchant, sum(-t.amount) AS spend, count(t) AS visits
ORDER BY spend DESC
"""

MERCHANTS_IN_CATEGORY_LIFETIME = """
MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category {name: $name})
// abs() so income categories (where amounts are positive) are picked up too.
RETURN m.canonical_name AS merchant,
       sum(abs(t.amount)) AS amount,
       count(t)           AS visits
ORDER BY amount DESC
"""

CATEGORY_KIND = """
MATCH (c:Category {name: $name}) RETURN c.kind AS kind
"""

ALL_CATEGORIES = """
MATCH (c:Category) WHERE c.name <> 'System'
RETURN DISTINCT c.name AS name ORDER BY name
"""

MONTHS_FOR_YEAR = """
MATCH (mo:Month) WHERE mo.year = $year
RETURN mo.id AS month ORDER BY month
"""

MONTH_TOTALS = """
MATCH (t:Transaction)-[:IN_MONTH]->(:Month {id: $month})
RETURN sum(CASE WHEN t.amount > 0 THEN  t.amount ELSE 0 END) AS income,
       sum(CASE WHEN t.amount < 0 THEN -t.amount ELSE 0 END) AS expense,
       count(t) AS tx_count
"""

MONTH_TOP_MERCHANTS = """
MATCH (t:Transaction)-[:IN_MONTH]->(:Month {id: $month})
WHERE t.amount < 0
MATCH (t)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE c.name <> 'System'
RETURN m.canonical_name AS merchant, c.name AS category,
       sum(-t.amount) AS spend, count(t) AS visits
ORDER BY spend DESC LIMIT 10
"""

MONTH_CATEGORIES = """
MATCH (t:Transaction)-[:IN_MONTH]->(:Month {id: $month})
WHERE t.amount < 0
MATCH (t)-[:AT]->(:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE c.name <> 'System'
RETURN c.name AS category, sum(-t.amount) AS spend, count(t) AS tx
ORDER BY spend DESC
"""

ANNUAL_TOTALS = """
MATCH (t:Transaction) WHERE t.year = $year
RETURN sum(CASE WHEN t.amount > 0 THEN  t.amount ELSE 0 END) AS income,
       sum(CASE WHEN t.amount < 0 THEN -t.amount ELSE 0 END) AS expense
"""

CATEGORY_YEARLY_TOTALS = """
MATCH (t:Transaction)-[:AT]->(:Merchant)-[:IN_CATEGORY]->(c:Category {name: $name})
RETURN t.year AS year, sum(abs(t.amount)) AS amount, count(t) AS tx
ORDER BY year
"""


# ---------------------------------------------------------------------------
# Helpers — escaping, link rendering, frontmatter
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Strip filesystem-hostile chars; preserve everything Obsidian accepts."""
    return name.replace("/", "_").replace(":", "-")


def _link(name: str, display: str | None = None) -> str:
    """Render an Obsidian wikilink, optionally with an aliased display string."""
    if display and display != name:
        return f"[[{name}|{display}]]"
    return f"[[{name}]]"


def _frontmatter(props: dict) -> str:
    """Render a minimal YAML frontmatter block. Values stay flat."""
    lines = ["---"]
    for k, v in props.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
        elif isinstance(v, str):
            # Quote strings that contain colons / brackets / quotes so YAML
            # round-trips cleanly.
            needs_quote = any(c in v for c in ':[]{}#,&*!|>%@`"\'')
            lines.append(f'{k}: "{v}"' if needs_quote else f"{k}: {v}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---\n")
    return "\n".join(lines)


def _month_display(month_id: str) -> str:
    """``2025-04`` → ``April 2025`` for nicer link text."""
    try:
        y, m = month_id.split("-")
        names = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        return f"{names[int(m) - 1]} {y}"
    except (ValueError, IndexError):
        return month_id


def _tag_for_category(category: str) -> str:
    """Obsidian tags can't contain spaces — slugify ``Council Tax`` → ``council-tax``."""
    return category.lower().replace(" ", "-").replace("&", "and")


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _write_merchant_page(root: Path, session, name: str, category: str) -> Path | None:
    """Lifetime merchant page — aggregates every year present in the graph.

    Works for both expense and income merchants (e.g. salary, interest)
    because the underlying queries use ``abs(t.amount)``. The label flips
    to "Received" for income categories so the page reads correctly.
    """
    totals = session.run(MERCHANT_LIFETIME_TOTALS, name=name).single()
    if not totals or not totals["visits"]:
        return None
    by_year = session.run(MERCHANT_BY_YEAR, name=name).data()
    by_month = session.run(MERCHANT_BY_MONTH_ALL, name=name).data()
    top = max(by_month, key=lambda r: r["amount"]) if by_month else None
    years = [r["year"] for r in by_year]

    kind_row = session.run(CATEGORY_KIND, name=category).single()
    kind = (kind_row or {}).get("kind") or "expense"
    flow_label = "Received" if kind == "income" else "Spend"

    fm = _frontmatter({
        "type": "merchant",
        "canonical": name,
        "category": category,
        "kind": kind,
        f"total_{'received' if kind == 'income' else 'spend'}": round(totals["amount"], 2),
        "visits": totals["visits"],
        "tags": (
            ["merchant", f"category/{_tag_for_category(category)}", f"kind/{kind}"]
            + [f"year/{y}" for y in years]
        ),
    })

    avg = totals["amount"] / totals["visits"]
    body = [
        fm,
        f"# {name}",
        "",
        f"**Category:** {_link(category)}",
        "",
        "## Lifetime",
        f"- **Total {flow_label.lower()}:** {CCY}{totals['amount']:,.2f}",
        f"- **Visits:** {totals['visits']}",
        f"- **Avg per visit:** {CCY}{avg:,.2f}",
    ]
    if top:
        body.append(
            f"- **Top month:** {_link(top['month'], _month_display(top['month']))} "
            f"({CCY}{top['amount']:,.2f})"
        )

    if len(by_year) > 1:
        body += [
            "",
            "## By year",
            "",
            f"| Year | Visits | {flow_label} |",
            "|------|--------|-------|",
        ]
        for r in by_year:
            body.append(
                f"| {_link(str(r['year']))} | {r['visits']} | {CCY}{r['amount']:,.2f} |"
            )

    body += [
        "",
        "## Monthly trend",
        "",
        f"| Month | Visits | {flow_label} |",
        "|-------|--------|-------|",
    ]
    for r in by_month:
        body.append(
            f"| {_link(r['month'], _month_display(r['month']))} | {r['visits']} "
            f"| {CCY}{r['amount']:,.2f} |"
        )

    path = root / "merchants" / f"{_safe_filename(name)}.md"
    path.write_text("\n".join(body) + "\n")
    return path


def _write_category_page(root: Path, session, name: str) -> Path | None:
    """Lifetime category page with one merchant table + per-year breakdown.

    Handles both expense categories (negative-amount transactions) and
    income / transfer categories (positive amounts). The flow column is
    labelled "Received" for income, "Spend" otherwise.
    """
    merchants = session.run(MERCHANTS_IN_CATEGORY_LIFETIME, name=name).data()
    if not merchants:
        return None
    yearly = session.run(CATEGORY_YEARLY_TOTALS, name=name).data()
    kind_row = session.run(CATEGORY_KIND, name=name).single()
    kind = (kind_row or {}).get("kind") or "expense"

    flow_label = "Received" if kind == "income" else "Spend"
    lifetime_total = sum(r["amount"] for r in merchants)
    lifetime_tx = sum(r["visits"] for r in merchants)

    fm = _frontmatter({
        "type": "category",
        "name": name,
        "kind": kind,
        f"total_{'received' if kind == 'income' else 'spend'}": round(lifetime_total, 2),
        "transactions": lifetime_tx,
        "tags": ["category", f"category/{_tag_for_category(name)}", f"kind/{kind}"],
    })

    body = [
        fm,
        f"# {name}",
        "",
        "## Lifetime",
        f"- **{flow_label}:** {CCY}{lifetime_total:,.2f} across {lifetime_tx} transactions, "
        f"{len(merchants)} merchants.",
        "",
        "## Merchants",
        "",
        f"| Merchant | {flow_label} | Visits |",
        "|----------|-------|--------|",
    ]
    for r in merchants:
        body.append(
            f"| {_link(r['merchant'])} | {CCY}{r['amount']:,.2f} | {r['visits']} |"
        )

    if yearly:
        body += [
            "",
            "## By year",
            "",
            f"| Year | {flow_label} | Transactions |",
            "|------|-------|--------------|",
        ]
        for r in yearly:
            body.append(
                f"| {_link(str(r['year']))} | {CCY}{r['amount']:,.2f} | {r['tx']} |"
            )

    path = root / "categories" / f"{_safe_filename(name)}.md"
    path.write_text("\n".join(body) + "\n")
    return path


def _write_month_page(root: Path, session, month_id: str) -> Path | None:
    totals = session.run(MONTH_TOTALS, month=month_id).single()
    if not totals or not totals["tx_count"]:
        return None
    top_merchants = session.run(MONTH_TOP_MERCHANTS, month=month_id).data()
    categories = session.run(MONTH_CATEGORIES, month=month_id).data()
    year = int(month_id.split("-")[0])
    display = _month_display(month_id)

    fm = _frontmatter({
        "type": "month",
        "month": month_id,
        "year": year,
        "income": round(totals["income"], 2),
        "expense": round(totals["expense"], 2),
        "net": round(totals["income"] - totals["expense"], 2),
        "transactions": totals["tx_count"],
        "tags": ["month", f"year/{year}"],
    })

    body = [
        fm,
        f"# {display}",
        "",
        f"Part of {_link(str(year))}.",
        "",
        "## Totals",
        f"- **Income:**   {CCY}{totals['income']:,.2f}",
        f"- **Expense:**  {CCY}{totals['expense']:,.2f}",
        f"- **Net:**      {CCY}{(totals['income'] - totals['expense']):,.2f}",
        f"- **Transactions:** {totals['tx_count']}",
        "",
        "## Top merchants",
        "",
        "| Merchant | Category | Spend | Visits |",
        "|----------|----------|-------|--------|",
    ]
    for r in top_merchants:
        body.append(
            f"| {_link(r['merchant'])} | {_link(r['category'])} "
            f"| {CCY}{r['spend']:,.2f} | {r['visits']} |"
        )

    body += [
        "",
        "## Categories",
        "",
        "| Category | Spend | Transactions |",
        "|----------|-------|--------------|",
    ]
    for r in categories:
        body.append(
            f"| {_link(r['category'])} | {CCY}{r['spend']:,.2f} | {r['tx']} |"
        )

    path = root / "months" / f"{month_id}.md"
    path.write_text("\n".join(body) + "\n")
    return path


def _write_annual_page(root: Path, session, year: int) -> Path | None:
    totals = session.run(ANNUAL_TOTALS, year=year).single()
    if not totals or (totals["income"] == 0 and totals["expense"] == 0):
        return None
    cats = session.run(CATEGORIES_FOR_YEAR, year=year).data()
    months = session.run(MONTHS_FOR_YEAR, year=year).data()

    fm = _frontmatter({
        "type": "annual",
        "year": year,
        "income": round(totals["income"], 2),
        "expense": round(totals["expense"], 2),
        "savings": round(totals["income"] - totals["expense"], 2),
        "tags": ["annual", f"year/{year}"],
    })

    body = [
        fm,
        f"# {year} — Annual Summary",
        "",
        "## Totals",
        f"- **Income:**   {CCY}{totals['income']:,.2f}",
        f"- **Expense:**  {CCY}{totals['expense']:,.2f}",
        f"- **Savings:**  {CCY}{(totals['income'] - totals['expense']):,.2f}",
        "",
        "## Spending by category",
        "",
        "| # | Category | Spend | Transactions |",
        "|---|----------|-------|--------------|",
    ]
    for i, r in enumerate(cats, start=1):
        body.append(
            f"| {i} | {_link(r['name'])} | {CCY}{r['spend']:,.2f} | {r['tx']} |"
        )

    if months:
        body += [
            "",
            "## Months",
            "",
            "| Month | |",
            "|-------|--|",
        ]
        for r in months:
            body.append(f"| {_link(r['month'], _month_display(r['month']))} | |")

    path = root / "annual" / f"{year}.md"
    path.write_text("\n".join(body) + "\n")
    return path


def _write_home_page(
    root: Path,
    years: list[int],
    categories: list[str],
    months: list[str],
    merchants: list[str],
) -> Path:
    """Map-of-Content. Open this in Obsidian first."""
    fm = _frontmatter({"type": "index", "tags": ["index"]})
    body = [
        fm,
        "# Personal Finance — Map of Content",
        "",
        "Generated by `src/ingestion/compile_wiki.py` from the Neo4j context graph.",
        "Open this file in Obsidian to browse the vault — the graph view is the",
        "best way to see how merchants, categories, months and years connect.",
        "",
        "## Years",
        "",
    ]
    body += [f"- {_link(str(y))}" for y in years]
    body += ["", "## Categories", ""]
    body += [f"- {_link(c)}" for c in categories]
    body += ["", "## Months", ""]
    body += [f"- {_link(m, _month_display(m))}" for m in months]
    body += ["", "## Merchants", ""]
    body += [f"- {_link(m)}" for m in merchants]

    path = root / "Home.md"
    path.write_text("\n".join(body) + "\n")
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compile_wiki(year: int | None = None) -> None:
    """Compile the wiki for one year, or for every year present in the graph."""
    driver = GraphDatabase.driver(
        SETTINGS.neo4j_uri,
        auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password),
    )
    wiki_root = SETTINGS.wiki_dir
    for sub in ("merchants", "categories", "months", "annual"):
        (wiki_root / sub).mkdir(parents=True, exist_ok=True)

    try:
        with driver.session(database=SETTINGS.neo4j_database) as session:
            years = (
                [year] if year is not None
                else [r["year"] for r in session.run(YEARS_AVAILABLE)]
            )

            # Merchants and categories are lifetime pages — one per node,
            # aggregating every year present in the graph. We always render
            # the full set so a re-run never leaves stale per-year files.
            merchants = session.run(ALL_MERCHANTS).data()
            for record in merchants:
                _write_merchant_page(
                    wiki_root, session, record["name"], record["category"],
                )

            categories = [r["name"] for r in session.run(ALL_CATEGORIES)]
            for name in categories:
                _write_category_page(wiki_root, session, name)

            # Months + annual pages are per-year by definition.
            all_months: set[str] = set()
            for y in years:
                for mo in session.run(MONTHS_FOR_YEAR, year=y):
                    all_months.add(mo["month"])
                    _write_month_page(wiki_root, session, mo["month"])
                _write_annual_page(wiki_root, session, y)

            _write_home_page(
                wiki_root,
                years=sorted(years),
                categories=sorted(categories),
                months=sorted(all_months),
                merchants=sorted(r["name"] for r in merchants),
            )
            log.info(
                "wiki: %d years, %d categories, %d months, %d merchants",
                len(years), len(categories), len(all_months), len(merchants),
            )

        compile_alerts(driver, wiki_root)
    finally:
        driver.close()


def compile_alerts(driver, wiki_root):
    """Emit one `alerts/<YYYY-MM>.md` per month containing flagged transactions."""
    from pathlib import Path

    out_dir = Path(wiki_root) / "alerts"
    out_dir.mkdir(parents=True, exist_ok=True)
    with driver.session() as s:
        rows = s.run("""
            MATCH (a:Alert)-[:FLAGS]->(t:Transaction)-[:AT]->(m:Merchant)
            OPTIONAL MATCH (t)-[:AT_LOCATION]->(l:Location)
            RETURN t.month AS month, a.kind AS kind, a.severity AS severity,
                   t.id AS tx_id, t.description AS description, t.amount AS amount,
                   m.canonical_name AS merchant,
                   coalesce(l.name + ' (' + l.country + ')', '—') AS location,
                   a.rationale AS rationale
            ORDER BY month, severity DESC
        """).data()

    by_month: dict[str, list[dict]] = {}
    for r in rows:
        by_month.setdefault(r["month"], []).append(r)

    for month, items in by_month.items():
        lines = [f"# Alerts — {month}", ""]
        for it in items:
            lines.append(
                f"- **{it['kind']}** (severity {it['severity']:.2f}) — "
                f"`{it['merchant']}` £{abs(it['amount']):.2f} · {it['location']}  \n"
                f"  > {it['rationale']}"
            )
        (out_dir / f"{month}.md").write_text("\n".join(lines) + "\n")


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    argv = argv or sys.argv[1:]
    year = int(argv[0]) if argv else None
    compile_wiki(year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
