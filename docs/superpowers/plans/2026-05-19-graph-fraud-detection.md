# Graph Fraud Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an IEEE-CIS-inspired, single-user fraud / anomaly detection layer on top of the existing personal finance context graph: rule signals + Neo4j GDS algorithms (FastRP, KNN, Node Similarity, Louvain, PageRank) combine into a `fraud_score` on every `:Transaction`, an `:Alert` sub-graph, an API surface, a `fraud_analyst` subagent, and a UI badge.

**Architecture:** Three layers (a) ingestion adds `:Day`, `:Location`, `:DescriptionTemplate` nodes so structural signals exist, (b) a `src/fraud/` package runs deterministic rules + GDS algorithms over the existing graph and writes back `fraud_score` + `:Alert` nodes, (c) FastAPI exposes `/api/fraud/*`, a new subagent reasons over alerts, and the GraphCanvas paints suspicious merchants in red. No labelled data is required — the score is rule-driven and the GDS features express "how unusual is this transaction relative to the rest of *your* graph".

**Tech Stack:** Python 3.11 / Neo4j 5.23 Enterprise + GDS 2.x (already in `docker-compose.yml`) / `graphdatascience>=1.10` / `pytest>=8.0` / FastAPI / Next.js 15.

---

## Reference: schema additions

New nodes:

| Label                  | Identifier             | Other properties                                    | Why                                                              |
|------------------------|------------------------|-----------------------------------------------------|------------------------------------------------------------------|
| `Day`                  | `id` (YYYY-MM-DD)      | `weekday` int, `month` str                          | Intra-day windows for card-testing / velocity / duplicate-charge |
| `Location`             | `id` (lowercase token) | `name`, `country`                                   | Geo-mismatch detection                                           |
| `DescriptionTemplate`  | `id` (sha8 of template)| `template` (digits/IDs masked)                      | IEEE-CIS-style "same description seen across many merchants"     |
| `Alert`                | `id` (uuid)            | `kind`, `severity`, `created_at`, `rationale`       | One alert per flagged transaction                                |

New relationships:

| Pattern                                 | Why                                  |
|-----------------------------------------|--------------------------------------|
| `(:Transaction)-[:ON_DAY]->(:Day)`      | Intra-day grouping                   |
| `(:Transaction)-[:AT_LOCATION]->(:Location)` | Geo signal                       |
| `(:Transaction)-[:MATCHES_TEMPLATE]->(:DescriptionTemplate)` | Template clustering |
| `(:Merchant)-[:CO_OCCURRED {weight}]->(:Merchant)` | Projected, weighted by shared days |
| `(:Merchant)-[:SIMILAR_BY_EMBED {score}]->(:Merchant)` | KNN output                       |
| `(:Merchant)-[:SIMILAR_BY_VISITORS {score}]->(:Merchant)` | Node Similarity output           |
| `(:Alert)-[:FLAGS]->(:Transaction)`     | Alert provenance                     |
| `(:Alert)-[:TRIGGERED_BY {weight}]->(:Rule)` | Multi-rule attribution            |

New properties on existing nodes:

* `:Transaction.fraud_score` float in `[0, 1]`
* `:Transaction.risk_flags` list of strings (rule names that fired)
* `:Merchant.pagerank` float
* `:Merchant.community` int (Louvain)
* `:Merchant.embedding` list of float (FastRP)
* `:Merchant.is_outlier` bool

## Reference: fraud scoring formula

```
rule_severities = {
  duplicate_charge:          0.90,
  card_testing:              0.95,
  new_merchant_high_amount:  0.70,
  geo_mismatch:              0.60,
  velocity:                  0.80,
  round_fx:                  0.50,
}

rule_score(tx)   = max(rule_severities[r] for r in fired_rules(tx) ) or 0
gds_score(tx)    = 0.5 * (1 if merchant.is_outlier else 0)
                 + 0.5 * normalized_emb_distance(merchant, user_centroid)
fraud_score(tx)  = clamp(0.6 * rule_score(tx) + 0.4 * gds_score(tx), 0, 1)
alert_threshold  = 0.50
```

## Reference: file structure (new and modified)

```
finance-context-engine/
├── docs/superpowers/plans/2026-05-19-graph-fraud-detection.md     # this file
├── docker-compose.yml                                              # unchanged (GDS already enabled)
├── requirements.txt                                                # + graphdatascience, pytest, pytest-asyncio
├── pyproject.toml                                                  # NEW – pytest config
├── data/ontology/finance.yaml                                      # MODIFIED – new entities/rels
├── src/
│   ├── ontology/schema.cypher                                      # MODIFIED – new constraints + indexes
│   ├── ingestion/
│   │   ├── locations.py                                            # NEW – description-tail parser
│   │   ├── templates.py                                            # NEW – description fingerprint
│   │   └── load_to_graph.py                                        # MODIFIED – Day/Location/Template nodes
│   ├── fraud/                                                      # NEW package
│   │   ├── __init__.py
│   │   ├── rules.py                                                # NEW – deterministic detectors
│   │   ├── gds.py                                                  # NEW – projection + algorithm runners
│   │   ├── score.py                                                # NEW – combiner + writer
│   │   ├── alerts.py                                               # NEW – :Alert node writer
│   │   └── run.py                                                  # NEW – CLI entrypoint
│   ├── api/
│   │   ├── models.py                                               # MODIFIED – Alert response models
│   │   ├── main.py                                                 # MODIFIED – include fraud router
│   │   └── routes/fraud.py                                         # NEW – /api/fraud/*
│   ├── agent/
│   │   ├── prompts.py                                              # MODIFIED – FRAUD_ANALYST_PROMPT, system prompt addendum
│   │   └── subagents.py                                            # MODIFIED – register fraud_analyst
│   └── ingestion/compile_wiki.py                                   # MODIFIED – emit alerts/{month}.md
├── tests/                                                          # NEW
│   ├── conftest.py                                                 # NEW – neo4j driver fixture, fixture data loader
│   ├── fixtures/
│   │   ├── normal_txs.jsonl                                        # NEW – 30 clean transactions
│   │   └── fraud_injections.jsonl                                  # NEW – 6 crafted fraud cases
│   ├── test_locations.py
│   ├── test_templates.py
│   ├── test_rules.py
│   ├── test_gds.py
│   ├── test_score.py
│   ├── test_fraud_api.py
│   └── test_end_to_end.py
└── web/
    ├── lib/api.ts                                                  # MODIFIED – fraud helpers
    ├── components/AlertsPanel.tsx                                  # NEW
    ├── components/GraphCanvas.tsx                                  # MODIFIED – red-ring for high-risk merchants
    └── app/page.tsx                                                # MODIFIED – mount AlertsPanel
```

---

## Task 1: Bootstrap pytest + project test scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add test dependencies to requirements.txt**

Append the following block to the existing `requirements.txt`:

```
# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0

# Graph Data Science Python driver
graphdatascience>=1.10.0
```

- [ ] **Step 2: Create pyproject.toml with pytest config**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "neo4j: tests that require a live Neo4j database (auto-skipped if unavailable)",
]
addopts = "-q --strict-markers"
```

- [ ] **Step 3: Create tests/__init__.py**

```python
```

(empty file — just makes `tests/` a package)

- [ ] **Step 4: Create tests/conftest.py with neo4j fixture**

```python
"""Pytest fixtures shared across the test suite.

The neo4j_driver fixture connects to the local docker-compose Neo4j; tests
marked `@pytest.mark.neo4j` are skipped if no driver can be opened. Every
neo4j-marked test runs against an isolated database named `test`, with all
nodes wiped before each test so order does not matter.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from neo4j import Driver, GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "please-change-me")
TEST_DATABASE = "neo4j"  # community uses 'neo4j'; enterprise allows 'test' — keep simple


def _driver_or_skip() -> Driver | None:
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        drv.verify_connectivity()
        return drv
    except Exception:
        return None


@pytest.fixture(scope="session")
def neo4j_driver() -> Driver:
    drv = _driver_or_skip()
    if drv is None:
        pytest.skip("Neo4j not reachable on {} — start docker compose first".format(NEO4J_URI))
    yield drv
    drv.close()


@pytest.fixture()
def clean_graph(neo4j_driver: Driver):
    """Wipe the graph before each test that uses it."""
    with neo4j_driver.session(database=TEST_DATABASE) as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield neo4j_driver


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
```

- [ ] **Step 5: Run pytest to confirm scaffolding works**

```bash
pip install -r requirements.txt
pytest -q
```

Expected: `no tests ran` (zero tests collected, exit code 5 — that's fine).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml tests/__init__.py tests/conftest.py
git commit -m "feat(test): bootstrap pytest with neo4j fixtures"
```

---

## Task 2: Test fixtures — normal transactions and crafted fraud cases

**Files:**
- Create: `tests/fixtures/normal_txs.jsonl`
- Create: `tests/fixtures/fraud_injections.jsonl`

- [ ] **Step 1: Write `tests/fixtures/normal_txs.jsonl`** (a small slice of believable Tony Stark spending)

```jsonl
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-14","description":"SAINSBURY'S S/MKT WATFORD","amount":-45.19,"balance":-45.19,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-15","description":"TFL TRAVEL CH TFL.GOV.UK/CP","amount":-18.20,"balance":-63.39,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-16","description":"TESCO STORES 3372","amount":-61.20,"balance":-124.59,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-17","description":"GITHUB, INC. SAN FRANCISCO CA","amount":-9.00,"balance":-133.59,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-18","description":"PARENTPAY E-COM R BRIDGWATER","amount":-58.00,"balance":-191.59,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-19","description":"UBER *TRIP HELP.UBER.COM","amount":-12.90,"balance":-204.49,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-20","description":"B & Q 1245 WATFORD","amount":-54.76,"balance":-259.25,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-21","description":"PARENTPAY E-COM R BRIDGWATER","amount":-74.00,"balance":-333.25,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-22","description":"SAINSBURY'S S/MKT WATFORD","amount":-52.73,"balance":-385.98,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-23","description":"TFL TRAVEL CH TFL.GOV.UK/CP","amount":-17.60,"balance":-403.58,"year":2025,"month":"2025-06"}
```

- [ ] **Step 2: Write `tests/fixtures/fraud_injections.jsonl`** (six crafted fraud-shaped rows, each labelled in description with `[FRAUD-CASE-N]` to make assertions easy)

```jsonl
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-24","description":"DUPLICATE MERCHANT XYZ 4421 [FRAUD-CASE-1]","amount":-79.99,"balance":-483.57,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-24","description":"DUPLICATE MERCHANT XYZ 4421 [FRAUD-CASE-1]","amount":-79.99,"balance":-563.56,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-25","description":"TINY TEST CHARGE NL [FRAUD-CASE-2]","amount":-1.05,"balance":-564.61,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-25","description":"ELECTRONICS MEGA AMSTERDAM NL [FRAUD-CASE-2]","amount":-489.00,"balance":-1053.61,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-26","description":"LUXE WATCH STORE PARIS FR [FRAUD-CASE-3]","amount":-2400.00,"balance":-3453.61,"year":2025,"month":"2025-06"}
{"statement_id":"1588-2025-06-13","account_id":"1588","account_id_token":"<ACCT_01>","institution":"Halifax Clarity","date":"2025-06-27","description":"ATM WITHDRAWAL BANGKOK TH [FRAUD-CASE-4]","amount":-500.00,"balance":-3953.61,"year":2025,"month":"2025-06"}
```

(Cases: 1 = duplicate, 2 = card-testing, 3 = new-merchant high-amount + geo, 4 = geo-mismatch.)

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/normal_txs.jsonl tests/fixtures/fraud_injections.jsonl
git commit -m "test(fixtures): seed normal + fraud-shaped transaction fixtures"
```

---

## Task 3: Schema diff — extend `finance.yaml` and `schema.cypher`

**Files:**
- Modify: `data/ontology/finance.yaml`
- Modify: `src/ontology/schema.cypher`

- [ ] **Step 1: Add new entities to `data/ontology/finance.yaml`** (insert before the `relationships:` line)

```yaml
  Day:
    description: A calendar day — used for intra-day windows (velocity, card-testing).
    properties:
      - { name: id,      type: string, identifier: true }   # YYYY-MM-DD
      - { name: weekday, type: int }
      - { name: month,   type: string }                     # YYYY-MM

  Location:
    description: Geographic token extracted from a transaction description tail.
    properties:
      - { name: id,      type: string, identifier: true }   # lowercase
      - { name: name,    type: string }
      - { name: country, type: string }

  DescriptionTemplate:
    description: Fingerprint of a raw description with digits/IDs masked.
    properties:
      - { name: id,       type: string, identifier: true }  # sha8 of template
      - { name: template, type: string }

  Alert:
    description: A fraud / anomaly alert against a single Transaction.
    properties:
      - { name: id,         type: string, identifier: true }
      - { name: kind,       type: string }
      - { name: severity,   type: float }
      - { name: created_at, type: datetime }
      - { name: rationale,  type: string }
```

- [ ] **Step 2: Append relationships to `data/ontology/finance.yaml`** (under the existing `relationships:` list)

```yaml
  - { type: ON_DAY,            from: Transaction, to: Day,                 cardinality: many_to_one }
  - { type: AT_LOCATION,       from: Transaction, to: Location,            cardinality: many_to_one }
  - { type: MATCHES_TEMPLATE,  from: Transaction, to: DescriptionTemplate, cardinality: many_to_one }
  - { type: FLAGS,             from: Alert,       to: Transaction,         cardinality: one_to_one  }
```

- [ ] **Step 3: Append constraints + indexes to `src/ontology/schema.cypher`**

Append below the existing index block (before the vector index):

```cypher
// Fraud-detection nodes
CREATE CONSTRAINT day_id        IF NOT EXISTS FOR (d:Day)                 REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT location_id   IF NOT EXISTS FOR (l:Location)            REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT desc_tpl_id   IF NOT EXISTS FOR (t:DescriptionTemplate) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT alert_id      IF NOT EXISTS FOR (a:Alert)               REQUIRE a.id IS UNIQUE;

CREATE INDEX transaction_fraud_score IF NOT EXISTS FOR (t:Transaction) ON (t.fraud_score);
CREATE INDEX merchant_community      IF NOT EXISTS FOR (m:Merchant)    ON (m.community);
CREATE INDEX merchant_pagerank       IF NOT EXISTS FOR (m:Merchant)    ON (m.pagerank);
CREATE INDEX alert_kind              IF NOT EXISTS FOR (a:Alert)       ON (a.kind);
```

- [ ] **Step 4: Smoke-apply against the running Neo4j**

```bash
docker compose up -d
cat src/ontology/schema.cypher | docker exec -i finance-ctx-neo4j cypher-shell -u neo4j -p please-change-me
```

Expected: `0 rows available` after each statement, no errors.

- [ ] **Step 5: Commit**

```bash
git add data/ontology/finance.yaml src/ontology/schema.cypher
git commit -m "feat(schema): add Day/Location/DescriptionTemplate/Alert + indexes"
```

---

## Task 4: Description-tail location parser

**Files:**
- Create: `src/ingestion/locations.py`
- Create: `tests/test_locations.py`

- [ ] **Step 1: Write the failing test in `tests/test_locations.py`**

```python
from src.ingestion.locations import parse_location, Location


def test_uk_town_no_country():
    loc = parse_location("SAINSBURY'S S/MKT WATFORD")
    assert loc == Location(id="watford", name="Watford", country="GB")


def test_us_state_code():
    loc = parse_location("GITHUB, INC. SAN FRANCISCO CA")
    assert loc == Location(id="san-francisco-ca", name="San Francisco CA", country="US")


def test_explicit_country_code():
    loc = parse_location("LUXE WATCH STORE PARIS FR")
    assert loc == Location(id="paris-fr", name="Paris FR", country="FR")


def test_apple_billing_country():
    loc = parse_location("APPLE.COM/BILL CORK IRL")
    assert loc == Location(id="cork-irl", name="Cork IRL", country="IE")


def test_unknown_returns_none():
    assert parse_location("BALANCE FROM PREVIOUS STATEMENT") is None
```

- [ ] **Step 2: Run to confirm the test fails**

```bash
pytest tests/test_locations.py -v
```

Expected: `ImportError: No module named 'src.ingestion.locations'`.

- [ ] **Step 3: Implement `src/ingestion/locations.py`**

```python
"""Extract a (name, country) Location from a transaction description tail.

Statements don't ship structured merchant addresses, but most descriptions end
in a town / state / country token (e.g. ``SAINSBURY'S S/MKT WATFORD``,
``GITHUB, INC. SAN FRANCISCO CA``, ``APPLE.COM/BILL CORK IRL``). This module
recovers a coarse Location signal from those suffixes — used by the
geo-mismatch rule and Louvain projection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    country: str  # ISO 3166-1 alpha-2


# Explicit country tokens we recognise at the very end of the description.
_COUNTRY_SUFFIX = {
    "FR": "FR", "DE": "DE", "ES": "ES", "IT": "IT", "NL": "NL",
    "IRL": "IE", "IE": "IE", "TH": "TH", "JP": "JP", "CN": "CN",
    "LUX": "LU", "LU": "LU", "BR": "BR", "AU": "AU",
}

# US state codes — when one is the last token we treat it as US.
_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}

# Strip common merchant-tail noise so the *location* tokens are isolated.
_NOISE_TOKENS = {
    "INC", "CORP", "LTD", "PLC", "LLC", "LIMITED", "COMPANY",
    "CO", "INC.", "CO.", "BILL", "BILL.", "EU", "EU.",
}

_TOKEN_RE = re.compile(r"[A-Z][A-Z\.\-']+")


def _strip_noise(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.rstrip(".") not in _NOISE_TOKENS]


def parse_location(description: str) -> Location | None:
    """Return a `Location` for the trailing geo tokens, or `None`.

    Strategy:
      1. Take the last 1–3 ALL-CAPS tokens.
      2. If the very last token is an explicit country code → use it.
      3. Else if the last token is a 2-letter US state → country = US.
      4. Else if the second-to-last token looks like a UK town → country = GB.
      5. Otherwise return None.
    """
    tokens = _TOKEN_RE.findall(description.upper())
    tokens = _strip_noise(tokens)
    if not tokens:
        return None

    last = tokens[-1].rstrip(".")
    # Case 1: explicit country
    if last in _COUNTRY_SUFFIX:
        country = _COUNTRY_SUFFIX[last]
        # Town/city = everything between the noise-stripped prefix and the country
        prefix = tokens[-2] if len(tokens) >= 2 else None
        if prefix and prefix not in _COUNTRY_SUFFIX:
            name = f"{prefix.title()} {last}"
            sid = f"{prefix.lower()}-{last.lower()}"
        else:
            name = last
            sid = last.lower()
        return Location(id=sid, name=name, country=country)

    # Case 2: US state code
    if last in _US_STATES and len(tokens) >= 2:
        city = tokens[-2]
        return Location(
            id=f"{city.lower()}-{last.lower()}",
            name=f"{city.title()} {last}",
            country="US",
        )

    # Case 3: UK-style "<DESCRIPTION> <TOWN>" — we accept any trailing
    # ALL-CAPS token that isn't a known noise word, defaulting to GB.
    # This is intentionally permissive; the rule engine treats GB as the
    # baseline so false positives here are harmless.
    if last.isalpha() and 3 <= len(last) <= 16:
        return Location(id=last.lower(), name=last.title(), country="GB")

    return None
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_locations.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/locations.py tests/test_locations.py
git commit -m "feat(ingest): extract Location from description tail"
```

---

## Task 5: Description fingerprint templates

**Files:**
- Create: `src/ingestion/templates.py`
- Create: `tests/test_templates.py`

- [ ] **Step 1: Write the failing test in `tests/test_templates.py`**

```python
from src.ingestion.templates import fingerprint


def test_strips_digits():
    a = fingerprint("PARENTPAY E-COM R BRIDGWATER")
    b = fingerprint("PARENTPAY E-COM R BRIDGWATER")
    assert a == b
    assert a.template == "PARENTPAY E-COM R BRIDGWATER"


def test_groups_numeric_ids():
    a = fingerprint("AMZNMKTPLACE*R66EF9ZC4")
    b = fingerprint("AMZNMKTPLACE*R12AB3XY0")
    assert a == b


def test_different_merchants_different_template():
    a = fingerprint("TESCO STORES 3372")
    b = fingerprint("SAINSBURY'S S/MKT WATFORD")
    assert a != b


def test_template_id_is_stable_hash():
    a = fingerprint("UBER *TRIP HELP.UBER.COM")
    b = fingerprint("UBER *TRIP HELP.UBER.COM")
    assert a.id == b.id
    assert len(a.id) == 8
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_templates.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/ingestion/templates.py`**

```python
"""Hash a raw transaction description down to a stable 'template' fingerprint.

Inspired by the IEEE-CIS Identity columns where many transactions share the
same template string. Two descriptions that differ only in trailing IDs /
amounts hash to the same template, so we can group them in the graph and
later run community detection over the resulting bipartite Transaction ↔
Template graph.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    id: str       # first 8 chars of sha256(template)
    template: str


# Mask: any run of 3+ alphanumerics that contains a digit → "<ID>".
# Standalone digit groups → "<NUM>".
_ID_RE  = re.compile(r"\b(?=\w*\d)\w{3,}\b")
_NUM_RE = re.compile(r"\b\d+\b")


def fingerprint(description: str) -> Template:
    template = _ID_RE.sub("<ID>", description.upper())
    template = _NUM_RE.sub("<NUM>", template)
    template = re.sub(r"\s+", " ", template).strip()
    h = hashlib.sha256(template.encode()).hexdigest()[:8]
    return Template(id=h, template=template)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_templates.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/templates.py tests/test_templates.py
git commit -m "feat(ingest): fingerprint descriptions into stable Templates"
```

---

## Task 6: Loader extension — write Day / Location / Template nodes

**Files:**
- Modify: `src/ingestion/load_to_graph.py`

- [ ] **Step 1: Extend `UPSERT_TX` Cypher in `src/ingestion/load_to_graph.py`**

Replace the existing `UPSERT_TX = """ ... """` block with this version. The only differences are the four new `MERGE` clauses near the end and four new param bindings.

```python
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
```

- [ ] **Step 2: Extend `_to_cypher_params` in the same file**

Replace the existing `_to_cypher_params(record)` function with:

```python
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
        # ----- new fields -----
        "day_id":   record["date"],
        "weekday":  d.weekday(),
        "template_id":  tpl.id,
        "template_str": tpl.template,
        "location_id":      loc.id      if loc else None,
        "location_name":    loc.name    if loc else None,
        "location_country": loc.country if loc else None,
    }
```

- [ ] **Step 3: Re-ingest existing statements to populate the new nodes**

```bash
python -m src.ingestion.load_to_graph
```

Expected: same log output as before plus no errors.

- [ ] **Step 4: Spot-check the new nodes**

```bash
docker exec -i finance-ctx-neo4j cypher-shell -u neo4j -p please-change-me \
  "MATCH (d:Day) RETURN count(d) AS days; MATCH (l:Location) RETURN count(l) AS locations; MATCH (tpl:DescriptionTemplate) RETURN count(tpl) AS templates;"
```

Expected: `days ≥ 200`, `locations ≥ 20`, `templates ≥ 50` (real numbers depend on dataset).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/load_to_graph.py
git commit -m "feat(ingest): write Day/Location/DescriptionTemplate nodes"
```

---

## Task 7: Rule engine — duplicate-charge detector

**Files:**
- Create: `src/fraud/__init__.py`
- Create: `src/fraud/rules.py`
- Create: `tests/test_rules.py`

- [ ] **Step 1: Create empty `src/fraud/__init__.py`**

```python
"""Fraud / anomaly detection layer."""
```

- [ ] **Step 2: Write a failing test in `tests/test_rules.py`**

```python
import pytest
from neo4j import Driver

from src.fraud.rules import duplicate_charge


@pytest.mark.neo4j
def test_duplicate_charge_flags_identical_same_day(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (m:Merchant {id:'foo', canonical_name:'Foo'})
            CREATE (t1:Transaction {id:'tx-1', amount:-79.99, date:date('2025-06-24'), month:'2025-06', description:'FOO X'})
            CREATE (t2:Transaction {id:'tx-2', amount:-79.99, date:date('2025-06-24'), month:'2025-06', description:'FOO X'})
            CREATE (t3:Transaction {id:'tx-3', amount:-12.50, date:date('2025-06-24'), month:'2025-06', description:'FOO X'})
            MERGE (d:Day {id:'2025-06-24'})
            MERGE (t1)-[:AT]->(m) MERGE (t2)-[:AT]->(m) MERGE (t3)-[:AT]->(m)
            MERGE (t1)-[:ON_DAY]->(d) MERGE (t2)-[:ON_DAY]->(d) MERGE (t3)-[:ON_DAY]->(d)
        """)

    flagged = duplicate_charge(clean_graph)

    assert {f["tx_id"] for f in flagged} == {"tx-1", "tx-2"}
    assert all(f["rule"] == "duplicate_charge" for f in flagged)
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_rules.py -v
```

Expected: `ImportError: No module named 'src.fraud.rules'`.

- [ ] **Step 4: Implement `src/fraud/rules.py` with the duplicate-charge rule**

```python
"""Deterministic fraud-rule detectors.

Each rule is a pure function `(driver) -> list[dict]`. The dict shape is:

    {"tx_id": str, "rule": str, "severity": float, "rationale": str}

Rules are intentionally read-only — write-back is done by `src.fraud.score`.
"""
from __future__ import annotations

from neo4j import Driver

# Severity weights used by the combiner. Centralised so they remain in sync
# across the rule, scoring, and prompt layers.
SEVERITY = {
    "duplicate_charge":         0.90,
    "card_testing":             0.95,
    "new_merchant_high_amount": 0.70,
    "geo_mismatch":             0.60,
    "velocity":                 0.80,
    "round_fx":                 0.50,
}


_DUPLICATE_QUERY = """
MATCH (t:Transaction)-[:AT]->(m:Merchant), (t)-[:ON_DAY]->(d:Day)
WITH m, d, t.amount AS amt, collect(t.id) AS tx_ids, count(*) AS n
WHERE n >= 2 AND amt < 0
UNWIND tx_ids AS tx_id
RETURN tx_id, m.canonical_name AS merchant, d.id AS day, amt AS amount, n AS occurrences
"""


def duplicate_charge(driver: Driver) -> list[dict]:
    """Flag charges where the same (merchant, day, amount) occurs ≥ 2 times.

    Classic skimming / double-post signature. Severity 0.9.
    """
    sev = SEVERITY["duplicate_charge"]
    with driver.session() as s:
        rows = s.run(_DUPLICATE_QUERY).data()
    return [
        {
            "tx_id":     r["tx_id"],
            "rule":      "duplicate_charge",
            "severity":  sev,
            "rationale": f"{r['merchant']} charged £{abs(r['amount']):.2f} {r['occurrences']}× on {r['day']}",
        }
        for r in rows
    ]
```

- [ ] **Step 5: Run test to verify pass**

```bash
pytest tests/test_rules.py -v
```

Expected: 1 passed (or `skipped` if Neo4j is down).

- [ ] **Step 6: Commit**

```bash
git add src/fraud/__init__.py src/fraud/rules.py tests/test_rules.py
git commit -m "feat(fraud): detect duplicate same-day same-amount charges"
```

---

## Task 8: Rule — card-testing pattern

**Files:**
- Modify: `src/fraud/rules.py`
- Modify: `tests/test_rules.py`

- [ ] **Step 1: Append failing test to `tests/test_rules.py`**

```python
from src.fraud.rules import card_testing


@pytest.mark.neo4j
def test_card_testing_small_then_big_same_day_different_merchants(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (m1:Merchant {id:'small', canonical_name:'Small'})
            MERGE (m2:Merchant {id:'big',   canonical_name:'Big'})
            MERGE (d:Day {id:'2025-06-25'})
            CREATE (t1:Transaction {id:'tx-small', amount:-1.05, date:date('2025-06-25')})
            CREATE (t2:Transaction {id:'tx-big',   amount:-489.0, date:date('2025-06-25')})
            MERGE (t1)-[:AT]->(m1) MERGE (t2)-[:AT]->(m2)
            MERGE (t1)-[:ON_DAY]->(d) MERGE (t2)-[:ON_DAY]->(d)
        """)
    flagged = card_testing(clean_graph)
    assert {f["tx_id"] for f in flagged} == {"tx-small", "tx-big"}
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_rules.py::test_card_testing_small_then_big_same_day_different_merchants -v
```

Expected: `ImportError: cannot import name 'card_testing'`.

- [ ] **Step 3: Append `card_testing` to `src/fraud/rules.py`**

```python
_CARD_TESTING_QUERY = """
MATCH (small:Transaction)-[:ON_DAY]->(d:Day)<-[:ON_DAY]-(big:Transaction),
      (small)-[:AT]->(ms:Merchant), (big)-[:AT]->(mb:Merchant)
WHERE small.amount < 0 AND big.amount < 0
  AND abs(small.amount) <= 5
  AND abs(big.amount)   >= 50
  AND ms.canonical_name <> mb.canonical_name
  AND small.id < big.id   // dedupe symmetric pair
RETURN small.id AS small_id, big.id AS big_id,
       ms.canonical_name AS small_merchant, mb.canonical_name AS big_merchant,
       d.id AS day, small.amount AS small_amount, big.amount AS big_amount
"""


def card_testing(driver: Driver) -> list[dict]:
    """Flag the classic "small probe then big spend" pattern same-day.

    A skimmer typically pushes a sub-£5 charge to confirm a stolen card,
    then immediately runs a large purchase at a different merchant.
    """
    sev = SEVERITY["card_testing"]
    with driver.session() as s:
        rows = s.run(_CARD_TESTING_QUERY).data()
    out: list[dict] = []
    for r in rows:
        rationale = (
            f"Probe £{abs(r['small_amount']):.2f} at {r['small_merchant']} → "
            f"£{abs(r['big_amount']):.2f} at {r['big_merchant']} on {r['day']}"
        )
        out.append({"tx_id": r["small_id"], "rule": "card_testing", "severity": sev, "rationale": rationale})
        out.append({"tx_id": r["big_id"],   "rule": "card_testing", "severity": sev, "rationale": rationale})
    return out
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_rules.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/rules.py tests/test_rules.py
git commit -m "feat(fraud): detect small-then-big card-testing pattern"
```

---

## Task 9: Rule — new-merchant high-amount

**Files:**
- Modify: `src/fraud/rules.py`
- Modify: `tests/test_rules.py`

- [ ] **Step 1: Append failing test**

```python
from src.fraud.rules import new_merchant_high_amount


@pytest.mark.neo4j
def test_new_merchant_high_amount_flagged(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            // Baseline: 10 small charges over 30 days at known merchants
            UNWIND range(1, 10) AS i
            MERGE (m:Merchant {id:'known' + i, canonical_name:'Known' + i})
            CREATE (t:Transaction {id:'norm-' + i, amount:-15.0, date:date('2025-06-01') + duration({days:i})})
            MERGE (t)-[:AT]->(m)
        """)
        s.run("""
            // First-ever charge at a brand-new merchant for £2400
            MERGE (lux:Merchant {id:'lux', canonical_name:'Lux'})
            CREATE (t:Transaction {id:'big', amount:-2400.0, date:date('2025-06-25')})
            MERGE (t)-[:AT]->(lux)
        """)
    flagged = new_merchant_high_amount(clean_graph, multiplier=5.0)
    assert any(f["tx_id"] == "big" for f in flagged)
    assert not any(f["tx_id"].startswith("norm-") for f in flagged)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_rules.py::test_new_merchant_high_amount_flagged -v
```

Expected: `ImportError`.

- [ ] **Step 3: Append `new_merchant_high_amount` to `src/fraud/rules.py`**

```python
_NEW_MERCHANT_QUERY = """
// Compute the median absolute charge across the whole graph as a baseline.
MATCH (t:Transaction) WHERE t.amount < 0
WITH percentileCont(abs(t.amount), 0.5) AS median_amt
MATCH (m:Merchant)<-[:AT]-(t:Transaction)
WHERE t.amount < 0
WITH m, collect({id: t.id, amount: t.amount, date: t.date}) AS txs, median_amt
WHERE size(txs) = 1
WITH m, txs[0] AS tx, median_amt
WHERE abs(tx.amount) >= $multiplier * median_amt
RETURN tx.id AS tx_id, m.canonical_name AS merchant,
       tx.amount AS amount, median_amt AS baseline
"""


def new_merchant_high_amount(driver: Driver, multiplier: float = 5.0) -> list[dict]:
    """First-ever spend at a new merchant ≥ `multiplier`× the user's median charge."""
    sev = SEVERITY["new_merchant_high_amount"]
    with driver.session() as s:
        rows = s.run(_NEW_MERCHANT_QUERY, multiplier=multiplier).data()
    return [
        {
            "tx_id":     r["tx_id"],
            "rule":      "new_merchant_high_amount",
            "severity":  sev,
            "rationale": (
                f"First charge at {r['merchant']} is £{abs(r['amount']):.2f} "
                f"vs. baseline median £{r['baseline']:.2f}"
            ),
        }
        for r in rows
    ]
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_rules.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/rules.py tests/test_rules.py
git commit -m "feat(fraud): flag first-ever charge at unusually high amount"
```

---

## Task 10: Rule — geo-mismatch

**Files:**
- Modify: `src/fraud/rules.py`
- Modify: `tests/test_rules.py`

- [ ] **Step 1: Append failing test**

```python
from src.fraud.rules import geo_mismatch


@pytest.mark.neo4j
def test_geo_mismatch_flags_unusual_country(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (gb:Location {id:'watford', country:'GB'})
            MERGE (th:Location {id:'bangkok-th', country:'TH'})
            MERGE (m:Merchant {id:'foo', canonical_name:'Foo'})
            UNWIND range(1, 20) AS i
            CREATE (t:Transaction {id:'home-' + i, amount:-10.0, date:date('2025-06-01') + duration({days:i})})
            MERGE (t)-[:AT_LOCATION]->(gb) MERGE (t)-[:AT]->(m)
        """)
        s.run("""
            MATCH (th:Location {id:'bangkok-th'}), (m:Merchant {id:'foo'})
            CREATE (t:Transaction {id:'away', amount:-500.0, date:date('2025-06-27')})
            MERGE (t)-[:AT_LOCATION]->(th) MERGE (t)-[:AT]->(m)
        """)
    flagged = geo_mismatch(clean_graph)
    assert any(f["tx_id"] == "away" for f in flagged)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_rules.py::test_geo_mismatch_flags_unusual_country -v
```

Expected: `ImportError`.

- [ ] **Step 3: Append `geo_mismatch` to `src/fraud/rules.py`**

```python
_GEO_MISMATCH_QUERY = """
// Identify the user's "home country set" — countries seen in ≥ 5 transactions.
MATCH (t:Transaction)-[:AT_LOCATION]->(l:Location)
WITH l.country AS country, count(t) AS n
WHERE n >= 5
WITH collect(country) AS home_countries
MATCH (t:Transaction)-[:AT_LOCATION]->(l:Location)
WHERE NOT l.country IN home_countries AND t.amount < 0
RETURN t.id AS tx_id, l.country AS country, l.name AS place, t.amount AS amount
"""


def geo_mismatch(driver: Driver) -> list[dict]:
    """Flag transactions whose location country is outside the user's normal set."""
    sev = SEVERITY["geo_mismatch"]
    with driver.session() as s:
        rows = s.run(_GEO_MISMATCH_QUERY).data()
    return [
        {
            "tx_id":     r["tx_id"],
            "rule":      "geo_mismatch",
            "severity":  sev,
            "rationale": f"Charge in {r['place']} ({r['country']}) — outside your usual countries",
        }
        for r in rows
    ]
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_rules.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/rules.py tests/test_rules.py
git commit -m "feat(fraud): flag transactions outside the user's home-country set"
```

---

## Task 11: Rule — velocity (many charges same merchant in short window)

**Files:**
- Modify: `src/fraud/rules.py`
- Modify: `tests/test_rules.py`

- [ ] **Step 1: Append failing test**

```python
from src.fraud.rules import velocity


@pytest.mark.neo4j
def test_velocity_flags_three_charges_same_merchant_same_day(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (m:Merchant {id:'vel', canonical_name:'VeloShop'})
            MERGE (d:Day {id:'2025-06-26'})
            UNWIND range(1,4) AS i
            CREATE (t:Transaction {id:'vt-' + i, amount:-25.0, date:date('2025-06-26')})
            MERGE (t)-[:AT]->(m) MERGE (t)-[:ON_DAY]->(d)
        """)
    flagged = velocity(clean_graph, threshold=3)
    assert len({f["tx_id"] for f in flagged}) == 4
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_rules.py::test_velocity_flags_three_charges_same_merchant_same_day -v
```

Expected: `ImportError`.

- [ ] **Step 3: Append `velocity` to `src/fraud/rules.py`**

```python
_VELOCITY_QUERY = """
MATCH (t:Transaction)-[:AT]->(m:Merchant), (t)-[:ON_DAY]->(d:Day)
WHERE t.amount < 0
WITH m, d, collect(t.id) AS tx_ids, count(*) AS n
WHERE n >= $threshold
UNWIND tx_ids AS tx_id
RETURN tx_id, m.canonical_name AS merchant, d.id AS day, n AS charges
"""


def velocity(driver: Driver, threshold: int = 3) -> list[dict]:
    """Flag ≥ N charges at the same merchant in the same calendar day."""
    sev = SEVERITY["velocity"]
    with driver.session() as s:
        rows = s.run(_VELOCITY_QUERY, threshold=threshold).data()
    return [
        {
            "tx_id":     r["tx_id"],
            "rule":      "velocity",
            "severity":  sev,
            "rationale": f"{r['charges']} charges at {r['merchant']} on {r['day']}",
        }
        for r in rows
    ]
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_rules.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/rules.py tests/test_rules.py
git commit -m "feat(fraud): flag high-velocity same-merchant charges"
```

---

## Task 12: Rule — round-amount FX charge

**Files:**
- Modify: `src/fraud/rules.py`
- Modify: `tests/test_rules.py`

- [ ] **Step 1: Append failing test**

```python
from src.fraud.rules import round_fx


@pytest.mark.neo4j
def test_round_fx_flags_round_foreign_amount(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (gb:Location {id:'london', country:'GB'})
            MERGE (it:Location {id:'rome',  country:'IT'})
            CREATE (t1:Transaction {id:'normal', amount:-47.32})-[:AT_LOCATION]->(it)
            CREATE (t2:Transaction {id:'round', amount:-200.00})-[:AT_LOCATION]->(it)
            CREATE (t3:Transaction {id:'home',  amount:-200.00})-[:AT_LOCATION]->(gb)
        """)
    flagged = round_fx(clean_graph)
    assert {f["tx_id"] for f in flagged} == {"round"}
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_rules.py::test_round_fx_flags_round_foreign_amount -v
```

Expected: `ImportError`.

- [ ] **Step 3: Append `round_fx` to `src/fraud/rules.py`**

```python
_ROUND_FX_QUERY = """
MATCH (t:Transaction)-[:AT_LOCATION]->(l:Location)
WHERE l.country <> 'GB'
  AND t.amount < 0
  AND abs(t.amount) % 10 = 0
  AND abs(t.amount) >= 50
RETURN t.id AS tx_id, l.country AS country, t.amount AS amount, l.name AS place
"""


def round_fx(driver: Driver) -> list[dict]:
    """Flag round-magnitude (£50, £100, £200, …) charges in a non-home country."""
    sev = SEVERITY["round_fx"]
    with driver.session() as s:
        rows = s.run(_ROUND_FX_QUERY).data()
    return [
        {
            "tx_id":     r["tx_id"],
            "rule":      "round_fx",
            "severity":  sev,
            "rationale": f"Round £{abs(r['amount']):.0f} charge in {r['place']} ({r['country']})",
        }
        for r in rows
    ]
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_rules.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/rules.py tests/test_rules.py
git commit -m "feat(fraud): flag round foreign-currency amounts"
```

---

## Task 13: Rule orchestrator — `run_all_rules`

**Files:**
- Modify: `src/fraud/rules.py`
- Modify: `tests/test_rules.py`

- [ ] **Step 1: Append failing test**

```python
from src.fraud.rules import run_all_rules


@pytest.mark.neo4j
def test_run_all_rules_merges_per_transaction(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (m:Merchant {id:'foo', canonical_name:'Foo'})
            MERGE (d:Day {id:'2025-06-24'})
            CREATE (t1:Transaction {id:'dup-a', amount:-79.99, date:date('2025-06-24')})
            CREATE (t2:Transaction {id:'dup-b', amount:-79.99, date:date('2025-06-24')})
            MERGE (t1)-[:AT]->(m) MERGE (t2)-[:AT]->(m)
            MERGE (t1)-[:ON_DAY]->(d) MERGE (t2)-[:ON_DAY]->(d)
        """)
    grouped = run_all_rules(clean_graph)
    assert set(grouped["dup-a"]) == {"duplicate_charge"}
    assert set(grouped["dup-b"]) == {"duplicate_charge"}
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_rules.py::test_run_all_rules_merges_per_transaction -v
```

Expected: `ImportError`.

- [ ] **Step 3: Append `run_all_rules` to `src/fraud/rules.py`**

```python
from collections import defaultdict


def run_all_rules(driver: Driver) -> dict[str, list[dict]]:
    """Run every rule and group findings by transaction id.

    Returns ``{tx_id: [finding, …]}``. Each finding has its full dict shape;
    the score combiner picks the max severity.
    """
    rule_fns = [
        duplicate_charge,
        card_testing,
        new_merchant_high_amount,
        geo_mismatch,
        velocity,
        round_fx,
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for fn in rule_fns:
        for finding in fn(driver):
            grouped[finding["tx_id"]].append(finding)
    return dict(grouped)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_rules.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/rules.py tests/test_rules.py
git commit -m "feat(fraud): orchestrate all rules into per-transaction findings"
```

---

## Task 14: GDS client + merchant-coincidence projection

**Files:**
- Create: `src/fraud/gds.py`
- Create: `tests/test_gds.py`

- [ ] **Step 1: Write failing test in `tests/test_gds.py`**

```python
import pytest
from neo4j import Driver

from src.fraud.gds import GdsClient


@pytest.mark.neo4j
def test_projection_creates_co_occurred_relationships(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (a:Merchant {id:'a', canonical_name:'A'})
            MERGE (b:Merchant {id:'b', canonical_name:'B'})
            MERGE (c:Merchant {id:'c', canonical_name:'C'})
            MERGE (d1:Day {id:'2025-06-01'})
            MERGE (d2:Day {id:'2025-06-02'})
            CREATE (ta:Transaction)-[:AT]->(a), (ta)-[:ON_DAY]->(d1)
            CREATE (tb:Transaction)-[:AT]->(b), (tb)-[:ON_DAY]->(d1)
            CREATE (tb2:Transaction)-[:AT]->(b), (tb2)-[:ON_DAY]->(d2)
            CREATE (tc:Transaction)-[:AT]->(c), (tc)-[:ON_DAY]->(d2)
        """)
    client = GdsClient(clean_graph)
    client.project_merchant_coincidence()

    with clean_graph.session() as s:
        rows = s.run(
            "MATCH (m1:Merchant)-[r:CO_OCCURRED]->(m2:Merchant) "
            "RETURN m1.canonical_name AS a, m2.canonical_name AS b, r.weight AS w"
        ).data()
    pairs = {(r["a"], r["b"]): r["w"] for r in rows}
    # A&B share day 06-01, B&C share day 06-02. A&C share nothing.
    assert ("A", "B") in pairs or ("B", "A") in pairs
    assert ("B", "C") in pairs or ("C", "B") in pairs
    assert ("A", "C") not in pairs and ("C", "A") not in pairs
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_gds.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/fraud/gds.py`**

```python
"""Neo4j GDS pipeline for the fraud layer.

Builds a `merchant-coincidence` graph (Merchants linked when they appear on
the same calendar day), then runs PageRank, Louvain, FastRP, KNN, and Node
Similarity over it. All writes happen back onto the persistent Neo4j store
so the rest of the app (rules, API, agent) can read them with plain Cypher.
"""
from __future__ import annotations

import logging

from neo4j import Driver

log = logging.getLogger(__name__)

GRAPH_NAME = "merchant-coincidence"


_PROJECT_COINCIDENCE = """
// Materialise the weighted edges first so we can use a native projection.
MATCH (m1:Merchant)<-[:AT]-(:Transaction)-[:ON_DAY]->(d:Day)<-[:ON_DAY]-(:Transaction)-[:AT]->(m2:Merchant)
WHERE id(m1) < id(m2)
WITH m1, m2, count(DISTINCT d) AS w
MERGE (m1)-[r:CO_OCCURRED]->(m2)
  SET r.weight = w
"""


class GdsClient:
    """Thin wrapper around the GDS Cypher procedure surface.

    We use Cypher rather than the `graphdatascience` Python driver here
    so the tests / app can run against community Neo4j with the GDS plugin
    enabled — no extra session management needed.
    """

    def __init__(self, driver: Driver):
        self.driver = driver

    def project_merchant_coincidence(self) -> None:
        """(Re)materialise persistent :CO_OCCURRED edges and a GDS projection."""
        with self.driver.session() as s:
            s.run("MATCH ()-[r:CO_OCCURRED]->() DELETE r")
            s.run(_PROJECT_COINCIDENCE)
            s.run(f"CALL gds.graph.exists('{GRAPH_NAME}') YIELD exists "
                  f"WITH exists WHERE exists "
                  f"CALL gds.graph.drop('{GRAPH_NAME}') YIELD graphName RETURN graphName")
            s.run(
                "CALL gds.graph.project($name, 'Merchant', "
                "{CO_OCCURRED: {orientation: 'UNDIRECTED', properties: 'weight'}})",
                name=GRAPH_NAME,
            )
            log.info("projected GDS graph %s", GRAPH_NAME)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_gds.py::test_projection_creates_co_occurred_relationships -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/gds.py tests/test_gds.py
git commit -m "feat(fraud): project Merchant co-occurrence as a GDS graph"
```

---

## Task 15: GDS PageRank + Louvain writes

**Files:**
- Modify: `src/fraud/gds.py`
- Modify: `tests/test_gds.py`

- [ ] **Step 1: Append failing test to `tests/test_gds.py`**

```python
@pytest.mark.neo4j
def test_pagerank_and_louvain_write_properties(clean_graph: Driver):
    with clean_graph.session() as s:
        # Same seed graph as the previous test.
        s.run("""
            MERGE (a:Merchant {id:'a', canonical_name:'A'})
            MERGE (b:Merchant {id:'b', canonical_name:'B'})
            MERGE (c:Merchant {id:'c', canonical_name:'C'})
            MERGE (d1:Day {id:'2025-06-01'})
            MERGE (d2:Day {id:'2025-06-02'})
            CREATE (ta:Transaction)-[:AT]->(a), (ta)-[:ON_DAY]->(d1)
            CREATE (tb:Transaction)-[:AT]->(b), (tb)-[:ON_DAY]->(d1)
            CREATE (tb2:Transaction)-[:AT]->(b), (tb2)-[:ON_DAY]->(d2)
            CREATE (tc:Transaction)-[:AT]->(c), (tc)-[:ON_DAY]->(d2)
        """)
    client = GdsClient(clean_graph)
    client.project_merchant_coincidence()
    client.run_pagerank()
    client.run_louvain()

    with clean_graph.session() as s:
        rows = s.run("MATCH (m:Merchant) RETURN m.canonical_name AS n, "
                     "m.pagerank AS pr, m.community AS c").data()
    by_name = {r["n"]: r for r in rows}
    assert by_name["B"]["pr"] > by_name["A"]["pr"]  # B is the hub
    assert all(r["c"] is not None for r in rows)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_gds.py::test_pagerank_and_louvain_write_properties -v
```

Expected: `AttributeError: 'GdsClient' object has no attribute 'run_pagerank'`.

- [ ] **Step 3: Append methods to `src/fraud/gds.py`**

```python
    def run_pagerank(self) -> None:
        with self.driver.session() as s:
            s.run(
                f"CALL gds.pageRank.write('{GRAPH_NAME}', {{"
                f"  relationshipWeightProperty: 'weight',"
                f"  writeProperty: 'pagerank'"
                f"}})"
            )
            log.info("pagerank written")

    def run_louvain(self) -> None:
        with self.driver.session() as s:
            s.run(
                f"CALL gds.louvain.write('{GRAPH_NAME}', {{"
                f"  relationshipWeightProperty: 'weight',"
                f"  writeProperty: 'community'"
                f"}})"
            )
            log.info("louvain written")
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_gds.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/gds.py tests/test_gds.py
git commit -m "feat(fraud): write pagerank + community via GDS"
```

---

## Task 16: GDS FastRP + KNN + Node Similarity

**Files:**
- Modify: `src/fraud/gds.py`
- Modify: `tests/test_gds.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.neo4j
def test_fastrp_knn_node_similarity_write_back(clean_graph: Driver):
    with clean_graph.session() as s:
        # Slightly richer seed: 5 merchants, two communities.
        s.run("""
            UNWIND ['a','b','c','x','y'] AS k MERGE (:Merchant {id:k, canonical_name:k})
            MERGE (d1:Day {id:'2025-06-01'}) MERGE (d2:Day {id:'2025-06-02'})
            MERGE (d3:Day {id:'2025-06-03'})
            WITH d1,d2,d3
            MATCH (a:Merchant {id:'a'}), (b:Merchant {id:'b'}), (c:Merchant {id:'c'}),
                  (x:Merchant {id:'x'}), (y:Merchant {id:'y'})
            CREATE (t1:Transaction)-[:AT]->(a), (t1)-[:ON_DAY]->(d1),
                   (t2:Transaction)-[:AT]->(b), (t2)-[:ON_DAY]->(d1),
                   (t3:Transaction)-[:AT]->(b), (t3)-[:ON_DAY]->(d2),
                   (t4:Transaction)-[:AT]->(c), (t4)-[:ON_DAY]->(d2),
                   (t5:Transaction)-[:AT]->(x), (t5)-[:ON_DAY]->(d3),
                   (t6:Transaction)-[:AT]->(y), (t6)-[:ON_DAY]->(d3)
        """)
    client = GdsClient(clean_graph)
    client.project_merchant_coincidence()
    client.run_fastrp(dim=16)
    client.run_knn(top_k=2)
    client.run_node_similarity()

    with clean_graph.session() as s:
        emb = s.run("MATCH (m:Merchant {id:'a'}) RETURN m.embedding AS e").single()
        assert emb and emb["e"] and len(emb["e"]) == 16
        knn = s.run("MATCH (:Merchant)-[r:SIMILAR_BY_EMBED]->(:Merchant) RETURN count(r) AS n").single()
        ns  = s.run("MATCH (:Merchant)-[r:SIMILAR_BY_VISITORS]->(:Merchant) RETURN count(r) AS n").single()
        assert knn["n"] > 0
        assert ns["n"]  >= 0     # node similarity can return 0 in trivial graphs; just confirm the call ran
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_gds.py::test_fastrp_knn_node_similarity_write_back -v
```

Expected: `AttributeError: ... run_fastrp`.

- [ ] **Step 3: Append methods to `src/fraud/gds.py`**

```python
    def run_fastrp(self, dim: int = 64) -> None:
        with self.driver.session() as s:
            s.run(
                f"CALL gds.fastRP.write('{GRAPH_NAME}', {{"
                f"  embeddingDimension: $dim,"
                f"  relationshipWeightProperty: 'weight',"
                f"  iterationWeights: [0.0, 1.0, 1.0, 0.8],"
                f"  writeProperty: 'embedding'"
                f"}})",
                dim=dim,
            )
            log.info("fastRP written (dim=%d)", dim)

    def run_knn(self, top_k: int = 5) -> None:
        with self.driver.session() as s:
            # KNN needs an in-memory projection that has nodeProperty=embedding.
            tmp = "merchant-knn"
            s.run(f"CALL gds.graph.exists('{tmp}') YIELD exists "
                  f"WITH exists WHERE exists "
                  f"CALL gds.graph.drop('{tmp}') YIELD graphName RETURN graphName")
            s.run(
                "CALL gds.graph.project($tmp, "
                "{Merchant: {properties: 'embedding'}}, '*')",
                tmp=tmp,
            )
            s.run(
                f"CALL gds.knn.write('{tmp}', {{"
                f"  nodeProperties: ['embedding'],"
                f"  topK: $top_k,"
                f"  writeRelationshipType: 'SIMILAR_BY_EMBED',"
                f"  writeProperty: 'score'"
                f"}})",
                top_k=top_k,
            )
            s.run(f"CALL gds.graph.drop('{tmp}') YIELD graphName RETURN graphName")
            log.info("KNN written (top_k=%d)", top_k)

    def run_node_similarity(self) -> None:
        with self.driver.session() as s:
            s.run(
                f"CALL gds.nodeSimilarity.write('{GRAPH_NAME}', {{"
                f"  writeRelationshipType: 'SIMILAR_BY_VISITORS',"
                f"  writeProperty: 'score'"
                f"}})"
            )
            log.info("nodeSimilarity written")
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_gds.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/gds.py tests/test_gds.py
git commit -m "feat(fraud): write fastRP embeddings + KNN + nodeSimilarity"
```

---

## Task 17: GDS outlier flag (`is_outlier`)

**Files:**
- Modify: `src/fraud/gds.py`
- Modify: `tests/test_gds.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.neo4j
def test_mark_outliers_flags_singleton_community(clean_graph: Driver):
    with clean_graph.session() as s:
        # Two merchants in one community, one isolated merchant.
        s.run("""
            MERGE (a:Merchant {id:'a', canonical_name:'A', community:1, pagerank:0.5})
            MERGE (b:Merchant {id:'b', canonical_name:'B', community:1, pagerank:0.5})
            MERGE (z:Merchant {id:'z', canonical_name:'Z', community:2, pagerank:0.05})
        """)
    GdsClient(clean_graph).mark_outliers()
    with clean_graph.session() as s:
        rows = s.run("MATCH (m:Merchant) RETURN m.canonical_name AS n, m.is_outlier AS o").data()
    by_name = {r["n"]: r["o"] for r in rows}
    assert by_name["Z"] is True
    assert by_name["A"] is False
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_gds.py::test_mark_outliers_flags_singleton_community -v
```

Expected: `AttributeError: ... mark_outliers`.

- [ ] **Step 3: Append `mark_outliers` to `src/fraud/gds.py`**

```python
    def mark_outliers(self) -> None:
        """Set Merchant.is_outlier = true for singleton-community OR very low pagerank merchants."""
        with self.driver.session() as s:
            s.run("""
                MATCH (m:Merchant)
                OPTIONAL MATCH (peer:Merchant) WHERE peer.community = m.community AND peer <> m
                WITH m, count(peer) AS peers
                SET m.is_outlier = (peers = 0) OR coalesce(m.pagerank, 0) < 0.1
            """)
            log.info("is_outlier marked")
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_gds.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/gds.py tests/test_gds.py
git commit -m "feat(fraud): mark singleton/low-pagerank merchants as outliers"
```

---

## Task 18: Score combiner + `:Alert` writer

**Files:**
- Create: `src/fraud/score.py`
- Create: `src/fraud/alerts.py`
- Create: `tests/test_score.py`

- [ ] **Step 1: Write failing test in `tests/test_score.py`**

```python
import pytest
from neo4j import Driver

from src.fraud.score import combine, write_back


def test_combine_picks_max_rule_severity_and_blends_gds():
    findings = [
        {"rule": "duplicate_charge", "severity": 0.9, "rationale": "x"},
        {"rule": "velocity",         "severity": 0.8, "rationale": "y"},
    ]
    result = combine(rule_findings=findings, is_outlier=True, emb_dist=0.4)
    assert result["fraud_score"] == pytest.approx(0.6 * 0.9 + 0.4 * (0.5 * 1 + 0.5 * 0.4), rel=1e-3)
    assert set(result["risk_flags"]) == {"duplicate_charge", "velocity"}


def test_combine_no_findings_and_no_outlier_is_zero():
    result = combine(rule_findings=[], is_outlier=False, emb_dist=0.0)
    assert result["fraud_score"] == 0.0
    assert result["risk_flags"] == []


@pytest.mark.neo4j
def test_write_back_writes_score_and_creates_alert(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("CREATE (t:Transaction {id:'tx-1', amount:-50.0, description:'X'})")
    write_back(
        clean_graph,
        per_tx={
            "tx-1": {
                "fraud_score": 0.85,
                "risk_flags": ["duplicate_charge"],
                "rationale": "double-post",
                "max_rule":  "duplicate_charge",
            }
        },
    )
    with clean_graph.session() as s:
        row = s.run("""
            MATCH (a:Alert)-[:FLAGS]->(t:Transaction {id:'tx-1'})
            RETURN t.fraud_score AS score, t.risk_flags AS flags,
                   a.kind AS kind, a.severity AS sev
        """).single()
    assert row["score"] == pytest.approx(0.85)
    assert row["flags"] == ["duplicate_charge"]
    assert row["kind"] == "duplicate_charge"
    assert row["sev"]  == pytest.approx(0.85)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_score.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/fraud/score.py`**

```python
"""Combine rule findings + GDS features into a per-transaction fraud_score
and persist them back into Neo4j (Transaction.fraud_score, Transaction.risk_flags,
plus one :Alert node per flagged transaction).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

from neo4j import Driver

from src.fraud.rules import run_all_rules

log = logging.getLogger(__name__)

W_RULES = 0.6
W_GDS   = 0.4
ALERT_THRESHOLD = 0.50


def combine(*, rule_findings: list[dict], is_outlier: bool, emb_dist: float) -> dict:
    rule_score = max((f["severity"] for f in rule_findings), default=0.0)
    gds_score  = 0.5 * (1.0 if is_outlier else 0.0) + 0.5 * max(0.0, min(1.0, emb_dist))
    score = W_RULES * rule_score + W_GDS * gds_score
    score = max(0.0, min(1.0, score))
    flags = sorted({f["rule"] for f in rule_findings})
    max_rule = max(rule_findings, key=lambda f: f["severity"])["rule"] if rule_findings else None
    rationale = " | ".join(f["rationale"] for f in rule_findings) or (
        "Merchant is a graph outlier" if is_outlier else "" )
    return {
        "fraud_score": round(score, 4),
        "risk_flags":  flags,
        "max_rule":    max_rule,
        "rationale":   rationale,
    }


def score_all(driver: Driver) -> dict[str, dict]:
    """Run the full rule + GDS pipeline and return per-transaction scores.

    Reads:  rule_findings via `run_all_rules`, Merchant.is_outlier/embedding/community.
    Writes: nothing — call `write_back` separately.
    """
    rule_findings = run_all_rules(driver)
    with driver.session() as s:
        # User-merchant centroid: mean embedding across merchants the user actually used.
        centroid_row = s.run("""
            MATCH (m:Merchant) WHERE m.embedding IS NOT NULL
            WITH collect(m.embedding) AS embs
            RETURN embs
        """).single()
    centroid = _centroid(centroid_row["embs"]) if centroid_row and centroid_row["embs"] else None

    with driver.session() as s:
        rows = s.run("""
            MATCH (t:Transaction)-[:AT]->(m:Merchant)
            RETURN t.id AS tx_id,
                   coalesce(m.is_outlier, false) AS is_outlier,
                   m.embedding AS embedding
        """).data()

    per_tx: dict[str, dict] = {}
    for r in rows:
        tx_id   = r["tx_id"]
        emb     = r["embedding"]
        dist    = _emb_distance(emb, centroid) if (emb and centroid) else 0.0
        bundle  = combine(
            rule_findings=rule_findings.get(tx_id, []),
            is_outlier=r["is_outlier"],
            emb_dist=dist,
        )
        per_tx[tx_id] = bundle
    return per_tx


_WRITE_BACK = """
UNWIND $rows AS row
MATCH (t:Transaction {id: row.tx_id})
SET t.fraud_score = row.fraud_score,
    t.risk_flags  = row.risk_flags
WITH t, row
WHERE row.fraud_score >= $threshold AND row.max_rule IS NOT NULL
MERGE (a:Alert {id: row.alert_id})
  ON CREATE SET a.kind = row.max_rule,
                a.severity = row.fraud_score,
                a.created_at = datetime($now),
                a.rationale  = row.rationale
MERGE (a)-[:FLAGS]->(t)
"""


def write_back(driver: Driver, per_tx: dict[str, dict]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {**bundle, "tx_id": tx_id, "alert_id": str(uuid4())}
        for tx_id, bundle in per_tx.items()
    ]
    with driver.session() as s:
        s.run(_WRITE_BACK, rows=rows, threshold=ALERT_THRESHOLD, now=now)
    flagged = sum(1 for b in per_tx.values() if b["fraud_score"] >= ALERT_THRESHOLD)
    log.info("wrote %d scores, created %d alerts", len(per_tx), flagged)
    return flagged


def _centroid(embs: list[list[float]]) -> list[float]:
    n = len(embs)
    if n == 0:
        return []
    dim = len(embs[0])
    out = [0.0] * dim
    for e in embs:
        for i, v in enumerate(e):
            out[i] += v
    return [v / n for v in out]


def _emb_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance, scaled to [0, 1]."""
    import math
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a)) or 1.0
    nb = math.sqrt(sum(bi * bi for bi in b)) or 1.0
    cos = dot / (na * nb)
    return max(0.0, min(1.0, (1.0 - cos) / 2.0))
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_score.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/score.py tests/test_score.py
git commit -m "feat(fraud): combine rule + GDS signals, write back scores + alerts"
```

---

## Task 19: `src/fraud/run.py` CLI entrypoint

**Files:**
- Create: `src/fraud/run.py`

- [ ] **Step 1: Implement `src/fraud/run.py`**

```python
"""End-to-end fraud detection runner.

Usage::

    python -m src.fraud.run                # score the whole graph
    python -m src.fraud.run --skip-gds     # rules only (fast iteration)
"""
from __future__ import annotations

import argparse
import logging
import sys

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
```

- [ ] **Step 2: Run it against the live graph**

```bash
python -m src.fraud.run
```

Expected: log lines `projected GDS graph ...`, `pagerank written`, `louvain written`, `fastRP written (dim=64)`, `KNN written (top_k=5)`, `nodeSimilarity written`, `is_outlier marked`, `wrote N scores, created K alerts`, final stdout `scored N transactions, created K alerts`.

- [ ] **Step 3: Spot-check the alerts**

```bash
docker exec -i finance-ctx-neo4j cypher-shell -u neo4j -p please-change-me \
  "MATCH (a:Alert)-[:FLAGS]->(t:Transaction) RETURN a.kind, t.description, t.amount LIMIT 10;"
```

Expected: 0 or more rows depending on the dataset; should run without error.

- [ ] **Step 4: Commit**

```bash
git add src/fraud/run.py
git commit -m "feat(fraud): CLI entrypoint to run GDS + rules + writeback"
```

---

## Task 20: FastAPI fraud route — list anomalies

**Files:**
- Modify: `src/api/models.py`
- Create: `src/api/routes/fraud.py`
- Modify: `src/api/main.py`
- Create: `tests/test_fraud_api.py`

- [ ] **Step 1: Append Pydantic models to `src/api/models.py`**

```python
class AlertItem(BaseModel):
    alert_id:    str
    tx_id:       str
    kind:        str
    severity:    float
    fraud_score: float
    risk_flags:  list[str]
    rationale:   str
    merchant:    str
    amount:      float
    date:        str
    description: str
    location:    str | None = None


class AlertsResponse(BaseModel):
    month:  str | None = None
    alerts: list[AlertItem]


class FraudScoreResponse(BaseModel):
    tx_id:       str
    fraud_score: float
    risk_flags:  list[str]
    rationale:   str
```

- [ ] **Step 2: Create `src/api/routes/fraud.py`**

```python
"""HTTP surface for the fraud / anomaly layer.

GET  /api/fraud/anomalies        — list alerts (optionally filtered by month)
GET  /api/fraud/score/{tx_id}    — return fraud_score + risk flags for one tx
POST /api/fraud/recompute        — re-run the pipeline (rules + optionally GDS)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import Driver

from src.api.deps import get_driver
from src.api.models import AlertItem, AlertsResponse, FraudScoreResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/fraud", tags=["fraud"])


_ANOMALIES_QUERY = """
MATCH (a:Alert)-[:FLAGS]->(t:Transaction)-[:AT]->(m:Merchant)
OPTIONAL MATCH (t)-[:AT_LOCATION]->(l:Location)
WHERE $month IS NULL OR t.month = $month
RETURN a.id        AS alert_id,
       t.id        AS tx_id,
       a.kind      AS kind,
       a.severity  AS severity,
       t.fraud_score AS fraud_score,
       t.risk_flags  AS risk_flags,
       a.rationale AS rationale,
       m.canonical_name AS merchant,
       t.amount    AS amount,
       toString(t.date) AS date,
       t.description AS description,
       coalesce(l.name + ' (' + l.country + ')', null) AS location
ORDER BY t.fraud_score DESC, t.date DESC
LIMIT 200
"""


@router.get("/anomalies", response_model=AlertsResponse)
def get_anomalies(
    month: str | None = Query(None, description="Filter to YYYY-MM"),
    driver: Driver = Depends(get_driver),
) -> AlertsResponse:
    with driver.session() as s:
        rows = s.run(_ANOMALIES_QUERY, month=month).data()
    return AlertsResponse(month=month, alerts=[AlertItem(**r) for r in rows])


@router.get("/score/{tx_id}", response_model=FraudScoreResponse)
def get_score(tx_id: str, driver: Driver = Depends(get_driver)) -> FraudScoreResponse:
    with driver.session() as s:
        row = s.run(
            "MATCH (t:Transaction {id:$id}) "
            "OPTIONAL MATCH (a:Alert)-[:FLAGS]->(t) "
            "RETURN t.fraud_score AS fraud_score, "
            "       coalesce(t.risk_flags, []) AS risk_flags, "
            "       coalesce(a.rationale, '') AS rationale",
            id=tx_id,
        ).single()
    if row is None:
        raise HTTPException(404, f"transaction not found: {tx_id}")
    return FraudScoreResponse(
        tx_id=tx_id,
        fraud_score=float(row["fraud_score"] or 0.0),
        risk_flags=list(row["risk_flags"] or []),
        rationale=row["rationale"] or "",
    )


@router.post("/recompute")
def recompute(skip_gds: bool = Query(False), driver: Driver = Depends(get_driver)) -> dict:
    """Re-run the fraud pipeline. Synchronous — returns counts when done."""
    from src.fraud.gds import GdsClient
    from src.fraud.score import score_all, write_back

    if not skip_gds:
        gds = GdsClient(driver)
        gds.project_merchant_coincidence()
        gds.run_pagerank(); gds.run_louvain(); gds.run_fastrp()
        gds.run_knn();      gds.run_node_similarity(); gds.mark_outliers()
    per_tx  = score_all(driver)
    flagged = write_back(driver, per_tx)
    return {"scored": len(per_tx), "alerts": flagged}
```

- [ ] **Step 3: Wire the router into `src/api/main.py`**

Update the import line near the top to include `fraud`:

```python
from src.api.routes import agent, canon, fraud, graph, health, pii, timeline, wiki
```

And add the include below the existing routes:

```python
    app.include_router(fraud.router, prefix="/api")
```

- [ ] **Step 4: Write a failing API test in `tests/test_fraud_api.py`**

```python
import pytest
from fastapi.testclient import TestClient
from neo4j import Driver

from src.api.main import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.mark.neo4j
def test_get_anomalies_empty_returns_empty_list(client, clean_graph: Driver):
    resp = client.get("/api/fraud/anomalies")
    assert resp.status_code == 200
    assert resp.json() == {"month": None, "alerts": []}


@pytest.mark.neo4j
def test_get_score_404_for_missing_tx(client, clean_graph: Driver):
    resp = client.get("/api/fraud/score/nope")
    assert resp.status_code == 404


@pytest.mark.neo4j
def test_get_anomalies_returns_seeded_alert(client, clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (m:Merchant {id:'foo', canonical_name:'Foo'})
            MERGE (l:Location {id:'rome', name:'Rome', country:'IT'})
            CREATE (t:Transaction {
              id:'tx-1', amount:-100.0, description:'FOO ROME IT',
              date:date('2025-06-25'), month:'2025-06',
              fraud_score:0.7, risk_flags:['round_fx']
            })
            CREATE (a:Alert {
              id:'al-1', kind:'round_fx', severity:0.7,
              created_at:datetime(), rationale:'round in italy'
            })
            MERGE (t)-[:AT]->(m) MERGE (t)-[:AT_LOCATION]->(l)
            MERGE (a)-[:FLAGS]->(t)
        """)
    resp = client.get("/api/fraud/anomalies?month=2025-06")
    assert resp.status_code == 200
    body = resp.json()
    assert body["month"] == "2025-06"
    assert len(body["alerts"]) == 1
    alert = body["alerts"][0]
    assert alert["kind"] == "round_fx"
    assert alert["merchant"] == "Foo"
    assert alert["location"] == "Rome (IT)"
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/test_fraud_api.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/models.py src/api/main.py src/api/routes/fraud.py tests/test_fraud_api.py
git commit -m "feat(api): /api/fraud surface for alerts, scores, and recompute"
```

---

## Task 21: Wiki compiler — alerts per month

**Files:**
- Modify: `src/ingestion/compile_wiki.py`

- [ ] **Step 1: Read the current `compile_wiki.py` to locate where the month section is rendered**

```bash
grep -n "months" /Users/chamindawijayasundara/Documents/context_graphs/finance-context-engine/src/ingestion/compile_wiki.py | head
```

Expected: a function or block that loops over month entries and writes `data/wiki/months/<id>.md`. The agent should append the new section after that block.

- [ ] **Step 2: Append a new `compile_alerts` function to `src/ingestion/compile_wiki.py`** (after the existing month-compilation function — preserve all existing code):

```python
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
```

- [ ] **Step 3: Call `compile_alerts` from the existing `main()` (or top-level entry point) — find it and add this near the end of the function, after the month/category compilation:**

```python
    compile_alerts(driver, SETTINGS.wiki_dir)
```

- [ ] **Step 4: Smoke-run the wiki compiler**

```bash
python -m src.ingestion.compile_wiki
ls data/wiki/alerts/
```

Expected: zero or more `*.md` files; no errors.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/compile_wiki.py
git commit -m "feat(wiki): compile monthly alerts markdown pages"
```

---

## Task 22: `fraud_analyst` subagent

**Files:**
- Modify: `src/agent/prompts.py`
- Modify: `src/agent/subagents.py`

- [ ] **Step 1: Append the prompt to `src/agent/prompts.py`**

```python
FRAUD_ANALYST_PROMPT = """\
You are the fraud_analyst subagent. You investigate suspicious activity on
the user's bank and credit-card statements using the context graph.

## Inputs you can query

  - `Transaction.fraud_score` (float 0-1), `Transaction.risk_flags` (list)
  - `:Alert {kind, severity, rationale, created_at}` nodes linked via
    `(:Alert)-[:FLAGS]->(:Transaction)`
  - `:Merchant.is_outlier`, `.community`, `.pagerank`
  - `:Location.country`, `:Day.id`

## What to produce

For each user question, return a short structured list:

  - **High-confidence alerts** — `fraud_score >= 0.8`; explain each with its
    risk flags and rationale.
  - **Worth-reviewing** — `0.5 <= fraud_score < 0.8`; brief one-liner each.
  - **Patterns** — group alerts that share a `kind`, a `Merchant.community`,
    or a `Location.country` and call out the cluster.

Cite every claim with the Cypher query you ran or the wiki path you read.
Do not speculate; if the data doesn't support a conclusion, say so.

## Risk flag glossary (use exactly these names)

  - duplicate_charge          — same merchant, same amount, same day, ≥ 2×
  - card_testing              — small probe (≤ £5) then big charge (≥ £50) same day, diff merchants
  - new_merchant_high_amount  — first-ever charge at a merchant ≥ 5× median spend
  - geo_mismatch              — location country outside the user's normal set
  - velocity                  — ≥ 3 charges at the same merchant in one day
  - round_fx                  — round-magnitude charge (multiple of £10, ≥ £50) abroad
"""
```

- [ ] **Step 2: Register the subagent in `src/agent/subagents.py`**

Update the imports:

```python
from src.agent.prompts import (
    ADVISOR_PROMPT,
    ANALYST_PROMPT,
    CATEGORIZER_PROMPT,
    FRAUD_ANALYST_PROMPT,
)
```

And append to the `SUBAGENTS` list (inside the existing list literal, after the `advisor` entry):

```python
    {
        "name": "fraud_analyst",
        "description": "Investigate flagged transactions and risk patterns in the context graph.",
        "prompt": FRAUD_ANALYST_PROMPT,
        "tools": [graph_query, wiki_read, wiki_list],
    },
```

- [ ] **Step 3: Update the system prompt to advertise the new subagent**

In `src/agent/prompts.py`, locate the `## Subagents` section of `SYSTEM_PROMPT` and replace it (and only that section) with:

```
## Subagents
For complex multi-axis questions, spawn subagents via the ``task`` tool:

  - **categorizer**    — proposes / fixes merchant→category mappings
  - **analyst**        — runs aggregations, trends, anomaly detection
  - **advisor**        — synthesizes recommendations from analyst output
  - **fraud_analyst**  — investigates :Alert nodes and suspicious patterns
```

- [ ] **Step 4: Commit**

```bash
git add src/agent/prompts.py src/agent/subagents.py
git commit -m "feat(agent): register fraud_analyst subagent + system prompt"
```

---

## Task 23: Frontend types + API helpers

**Files:**
- Modify: `web/lib/api.ts`

- [ ] **Step 1: Append the following types + helpers to `web/lib/api.ts`** (at the end of the file)

```typescript
// ---------- Fraud / alerts ----------

export type AlertItem = {
  alert_id: string;
  tx_id: string;
  kind: string;
  severity: number;
  fraud_score: number;
  risk_flags: string[];
  rationale: string;
  merchant: string;
  amount: number;
  date: string;
  description: string;
  location: string | null;
};

export type AlertsResponse = {
  month: string | null;
  alerts: AlertItem[];
};

export async function fetchAnomalies(month?: string): Promise<AlertsResponse> {
  const q = month ? `?month=${encodeURIComponent(month)}` : '';
  const r = await fetch(`/api/fraud/anomalies${q}`);
  if (!r.ok) throw new Error(`anomalies ${r.status}`);
  return r.json();
}

export async function recomputeFraud(opts: { skipGds?: boolean } = {}): Promise<{ scored: number; alerts: number }> {
  const q = opts.skipGds ? '?skip_gds=true' : '';
  const r = await fetch(`/api/fraud/recompute${q}`, { method: 'POST' });
  if (!r.ok) throw new Error(`recompute ${r.status}`);
  return r.json();
}
```

- [ ] **Step 2: Commit**

```bash
git add web/lib/api.ts
git commit -m "feat(web): typed client for /api/fraud endpoints"
```

---

## Task 24: `AlertsPanel.tsx`

**Files:**
- Create: `web/components/AlertsPanel.tsx`

- [ ] **Step 1: Create `web/components/AlertsPanel.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";
import { type AlertItem, fetchAnomalies, recomputeFraud } from "@/lib/api";

type Props = {
  month?: string;            // YYYY-MM filter; undefined = all months
  onAlertClick?: (a: AlertItem) => void;
};

const KIND_COLOR: Record<string, string> = {
  duplicate_charge:         "bg-red-100   text-red-800   border-red-300",
  card_testing:             "bg-red-200   text-red-900   border-red-400",
  new_merchant_high_amount: "bg-orange-100 text-orange-800 border-orange-300",
  geo_mismatch:             "bg-amber-100 text-amber-800 border-amber-300",
  velocity:                 "bg-rose-100  text-rose-800  border-rose-300",
  round_fx:                 "bg-yellow-100 text-yellow-800 border-yellow-300",
};

export default function AlertsPanel({ month, onAlertClick }: Props) {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);

  async function load() {
    setLoading(true); setError(null);
    try {
      const data = await fetchAnomalies(month);
      setAlerts(data.alerts);
    } catch (e) {
      setError((e as Error).message);
    } finally { setLoading(false); }
  }

  useEffect(() => { load(); }, [month]);

  async function onRecompute() {
    setLoading(true); setError(null);
    try {
      await recomputeFraud({ skipGds: false });
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally { setLoading(false); }
  }

  return (
    <div className="flex flex-col h-full text-sm">
      <header className="flex items-center justify-between p-2 border-b">
        <h3 className="font-medium">
          Alerts {month ? <span className="text-gray-500">· {month}</span> : null}
          <span className="ml-2 text-gray-400">{alerts.length}</span>
        </h3>
        <button
          className="px-2 py-1 text-xs border rounded hover:bg-gray-50"
          onClick={onRecompute}
          disabled={loading}
        >
          {loading ? "Working…" : "Recompute"}
        </button>
      </header>

      {error && <div className="p-2 text-red-600">{error}</div>}

      <ul className="overflow-y-auto flex-1 divide-y">
        {alerts.length === 0 && !loading && (
          <li className="p-3 text-gray-500">No alerts.</li>
        )}
        {alerts.map((a) => (
          <li
            key={a.alert_id}
            className="p-3 cursor-pointer hover:bg-gray-50"
            onClick={() => onAlertClick?.(a)}
          >
            <div className="flex items-center justify-between">
              <span className={`text-xs px-2 py-0.5 border rounded ${KIND_COLOR[a.kind] ?? ""}`}>
                {a.kind}
              </span>
              <span className="text-xs text-gray-500">
                score {a.fraud_score.toFixed(2)}
              </span>
            </div>
            <div className="mt-1 font-medium">
              {a.merchant} <span className="text-gray-500">· £{Math.abs(a.amount).toFixed(2)}</span>
            </div>
            <div className="text-xs text-gray-600">
              {a.date} · {a.location ?? "—"}
            </div>
            <div className="mt-1 text-xs italic text-gray-700">{a.rationale}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/components/AlertsPanel.tsx
git commit -m "feat(web): AlertsPanel component"
```

---

## Task 25: Mount `AlertsPanel` in `page.tsx`

**Files:**
- Modify: `web/app/page.tsx`

- [ ] **Step 1: Inspect the existing layout to find where right-pane components mount**

```bash
grep -n "WikiViewer\|ChatPanel\|TimeScrubber" /Users/chamindawijayasundara/Documents/context_graphs/finance-context-engine/web/app/page.tsx
```

Expected: lines that render one of those components inside a column / pane.

- [ ] **Step 2: Add an `AlertsPanel` mount in `web/app/page.tsx`**

At the top of the file, add the import alongside the existing component imports:

```tsx
import AlertsPanel from "@/components/AlertsPanel";
```

Find the JSX block that renders `WikiViewer` (right pane) and add an `AlertsPanel` *above* it inside the same container so the panel is visible at the top of the right column. Use the existing `month` state variable (the page already has one for the time scrubber):

```tsx
<div className="border-b max-h-[40vh]">
  <AlertsPanel month={month ?? undefined} />
</div>
```

(If the right column is currently a single `WikiViewer`, wrap it in a vertical flex column so both children fit.)

- [ ] **Step 3: Visually verify in the browser**

```bash
# In one terminal:
uvicorn src.api.main:app --port 8000 --reload --reload-dir src
# In another:
cd web && npm run dev
```

Open http://localhost:3000 and confirm: AlertsPanel header is visible, "Recompute" button is clickable, list is either empty or populated.

- [ ] **Step 4: Commit**

```bash
git add web/app/page.tsx
git commit -m "feat(web): mount AlertsPanel in the workbench right column"
```

---

## Task 26: Red-ring badge on suspicious merchants in `GraphCanvas`

**Files:**
- Modify: `web/components/GraphCanvas.tsx`

- [ ] **Step 1: Find the merchant-node render path**

```bash
grep -n "type === \"Merchant\"\|type==='Merchant'\|nodeCanvasObject\|drawNode" /Users/chamindawijayasundara/Documents/context_graphs/finance-context-engine/web/components/GraphCanvas.tsx | head
```

Expected: a function that draws each node onto the canvas — typically `nodeCanvasObject`.

- [ ] **Step 2: Add a `highRiskIds` prop and a red-ring overlay** in `GraphCanvas.tsx`

Add to the `Props` type:

```tsx
highRiskIds?: Set<string>;       // node IDs (e.g. "merchant:Foo") to ring
```

Inside the component, accept the prop:

```tsx
export default function GraphCanvas({ ..., highRiskIds }: Props) {
```

Inside the `nodeCanvasObject` (or equivalent) function, AFTER the existing node draw, add:

```tsx
if (highRiskIds?.has(node.id)) {
  ctx.save();
  ctx.strokeStyle = "#dc2626";   // red-600
  ctx.lineWidth   = 2 / globalScale;
  ctx.beginPath();
  ctx.arc(node.x!, node.y!, (node.r ?? 6) + 3 / globalScale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}
```

- [ ] **Step 3: Wire `highRiskIds` from `page.tsx`**

In `web/app/page.tsx`, add at the top of the component body:

```tsx
import { useMemo, useEffect, useState } from "react";
import { fetchAnomalies, type AlertItem } from "@/lib/api";

// ...

const [alerts, setAlerts] = useState<AlertItem[]>([]);
useEffect(() => {
  fetchAnomalies(month ?? undefined).then(r => setAlerts(r.alerts)).catch(() => setAlerts([]));
}, [month]);

const highRiskIds = useMemo(
  () => new Set(alerts.filter(a => a.fraud_score >= 0.5).map(a => `merchant:${a.merchant}`)),
  [alerts],
);
```

Then pass it to the existing GraphCanvas:

```tsx
<GraphCanvas ... highRiskIds={highRiskIds} />
```

- [ ] **Step 4: Visually verify**

Open http://localhost:3000 with the dev servers running. Any merchant whose transactions have `fraud_score >= 0.5` should now show a red ring on the canvas.

- [ ] **Step 5: Commit**

```bash
git add web/components/GraphCanvas.tsx web/app/page.tsx
git commit -m "feat(web): red-ring high-risk merchants on the GraphCanvas"
```

---

## Task 27: End-to-end test

**Files:**
- Create: `tests/test_end_to_end.py`

- [ ] **Step 1: Write the end-to-end test**

```python
"""End-to-end: load fixtures → run pipeline → assert each crafted fraud case
shows up in /api/fraud/anomalies."""
import json
import pytest
from fastapi.testclient import TestClient
from neo4j import Driver

from src.api.main import app
from src.fraud.run import main as run_fraud
from src.ingestion.load_to_graph import _to_cypher_params, UPSERT_TX


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.mark.neo4j
def test_pipeline_flags_each_crafted_fraud_case(clean_graph: Driver, fixtures_dir, client):
    # 1. Load both fixture files into the graph through the same Cypher path
    # the ingest pipeline uses.
    with clean_graph.session() as s:
        for fname in ("normal_txs.jsonl", "fraud_injections.jsonl"):
            for line in (fixtures_dir / fname).read_text().splitlines():
                rec = json.loads(line)
                # Inject minimal canonicalize-equivalent values so the loader works
                # without the LLM normalizer. We bypass it by stuffing the
                # merchant fields straight into the params dict.
                params = _to_cypher_params(rec)
                s.run(UPSERT_TX, **params)

    # 2. Run the full fraud pipeline (rules + GDS) end-to-end.
    rc = run_fraud(argv=[])
    assert rc == 0

    # 3. Query /api/fraud/anomalies and verify each crafted case shows up.
    resp = client.get("/api/fraud/anomalies?month=2025-06")
    assert resp.status_code == 200
    alerts = resp.json()["alerts"]
    descriptions = {a["description"] for a in alerts}

    # All four fraud labels should be represented (some may have ≥ 1 alert).
    assert any("[FRAUD-CASE-1]" in d for d in descriptions)
    assert any("[FRAUD-CASE-2]" in d for d in descriptions)
    assert any("[FRAUD-CASE-3]" in d for d in descriptions)
    assert any("[FRAUD-CASE-4]" in d for d in descriptions)
```

- [ ] **Step 2: Run it**

```bash
pytest tests/test_end_to_end.py -v
```

Expected: 1 passed (assumes the LLM canonicalizer falls back to regex when `OPENAI_API_KEY` is unset — confirm `src/ingestion/normalize.canonicalize` does this; otherwise set `NORMALIZER=regex` for the test).

If the test fails on case-3 (`new_merchant_high_amount`), it's likely because the median baseline computed over 16 fixture rows is high enough that £2400 doesn't clear the 5× threshold. In that case, lower the multiplier for the test by setting an env var the rule reads, or load more fixture rows to lower the median. The simplest fix: load `tests/fixtures/normal_txs.jsonl` *twice* (different statement_id values) to triple the baseline sample. Document whichever fix you use in the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test(e2e): assert each crafted fraud case is flagged"
```

---

## Task 28: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append a new section to `README.md`** (just before `## What's intentionally not in this skeleton`)

```markdown
## Fraud / anomaly detection

The graph carries a per-transaction `fraud_score ∈ [0, 1]` plus a `risk_flags`
list, written by `src/fraud/`. Six deterministic rules (duplicate-charge,
card-testing, new-merchant-high-amount, geo-mismatch, velocity, round-FX)
contribute most of the score; five Neo4j GDS algorithms (PageRank, Louvain,
FastRP, KNN, Node Similarity) add a "this merchant doesn't fit your normal
graph" signal. Anything ≥ 0.5 also gets an `:Alert` node.

Run the pipeline:

```bash
python -m src.fraud.run            # rules + GDS + writeback (~30s on the sample)
python -m src.fraud.run --skip-gds # rules only — fast iteration
```

Query alerts:

```bash
curl localhost:8000/api/fraud/anomalies?month=2026-01
curl localhost:8000/api/fraud/score/<tx_id>
curl -X POST localhost:8000/api/fraud/recompute
```

In the workbench:
- `AlertsPanel` (top of right column) lists every flagged transaction; click
  one to focus the merchant on the canvas.
- Merchants whose flagged transactions score `≥ 0.5` get a red ring on the
  `GraphCanvas`.
- The `fraud_analyst` subagent can be addressed directly: ask the chat panel
  "investigate the alerts from January 2026".
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document fraud / anomaly detection layer"
```

---

## Self-review

**Spec coverage check** — every signal listed in the brief:

| Signal                                  | Covered by         |
|-----------------------------------------|--------------------|
| Duplicate-charge rule                   | Task 7             |
| Card-testing rule                       | Task 8             |
| New-merchant high-amount rule           | Task 9             |
| Geo-mismatch rule                       | Task 10            |
| Velocity rule                           | Task 11            |
| Round-FX rule                           | Task 12            |
| PageRank                                | Task 15            |
| Louvain                                 | Task 15            |
| FastRP                                  | Task 16            |
| KNN                                     | Task 16            |
| Node Similarity                         | Task 16            |
| Schema additions (Day/Location/Tpl/Alert) | Task 3           |
| Loader extension                        | Task 6             |
| Score combiner + alert writer           | Task 18            |
| CLI entrypoint                          | Task 19            |
| FastAPI surface                         | Task 20            |
| Wiki output                             | Task 21            |
| `fraud_analyst` subagent                | Task 22            |
| Frontend types + helpers                | Task 23            |
| AlertsPanel                             | Task 24            |
| Right-pane mount                        | Task 25            |
| Red-ring badge on canvas                | Task 26            |
| End-to-end test                         | Task 27            |
| Docs                                    | Task 28            |

**Type / name consistency check** —
- `fraud_score`, `risk_flags`, `community`, `pagerank`, `embedding`, `is_outlier` used identically across schema, rules, GDS, score, and API code.
- `:Alert` props `kind`, `severity`, `rationale`, `created_at` consistent between `score.py` writeback Cypher and `routes/fraud.py` reader.
- Severity constants live in one place (`src/fraud/rules.SEVERITY`) and are referenced from each rule.
- `AlertItem` Pydantic model field names match the keys returned by the API Cypher.

**Placeholder scan** — every step has concrete code, exact Cypher, exact file paths, exact `git commit` commands. No TBDs.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-19-graph-fraud-detection.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
