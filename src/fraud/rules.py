"""Deterministic fraud-rule detectors.

Each rule is a pure function `(driver) -> list[dict]`. The dict shape is:

    {"tx_id": str, "rule": str, "severity": float, "rationale": str}
"""
from __future__ import annotations

from collections import defaultdict

from neo4j import Driver

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


_CARD_TESTING_QUERY = """
MATCH (small:Transaction)-[:ON_DAY]->(d:Day)<-[:ON_DAY]-(big:Transaction),
      (small)-[:AT]->(ms:Merchant), (big)-[:AT]->(mb:Merchant)
WHERE small.amount < 0 AND big.amount < 0
  AND abs(small.amount) <= 5
  AND abs(big.amount)   >= 50
  AND ms.canonical_name <> mb.canonical_name
  AND small.id <> big.id
RETURN small.id AS small_id, big.id AS big_id,
       ms.canonical_name AS small_merchant, mb.canonical_name AS big_merchant,
       d.id AS day, small.amount AS small_amount, big.amount AS big_amount
"""


def card_testing(driver: Driver) -> list[dict]:
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


_NEW_MERCHANT_QUERY = """
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


_GEO_MISMATCH_QUERY = """
MATCH (t:Transaction)-[:AT_LOCATION]->(l:Location)
WITH l.country AS country, count(t) AS n
WHERE n >= 5
WITH collect(country) AS home_countries
MATCH (t:Transaction)-[:AT_LOCATION]->(l:Location)
WHERE NOT l.country IN home_countries AND t.amount < 0
RETURN t.id AS tx_id, l.country AS country, l.name AS place, t.amount AS amount
"""


def geo_mismatch(driver: Driver) -> list[dict]:
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


_VELOCITY_QUERY = """
MATCH (t:Transaction)-[:AT]->(m:Merchant), (t)-[:ON_DAY]->(d:Day)
WHERE t.amount < 0
WITH m, d, collect(t.id) AS tx_ids, count(*) AS n
WHERE n >= $threshold
UNWIND tx_ids AS tx_id
RETURN tx_id, m.canonical_name AS merchant, d.id AS day, n AS charges
"""


def velocity(driver: Driver, threshold: int = 3) -> list[dict]:
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


_ROUND_FX_QUERY = """
MATCH (t:Transaction)-[:AT_LOCATION]->(l:Location)
WHERE l.country <> 'GB'
  AND t.amount < 0
  AND abs(t.amount) % 10 = 0
  AND abs(t.amount) >= 50
RETURN t.id AS tx_id, l.country AS country, t.amount AS amount, l.name AS place
"""


def round_fx(driver: Driver) -> list[dict]:
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


def run_all_rules(driver: Driver) -> dict[str, list[dict]]:
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
