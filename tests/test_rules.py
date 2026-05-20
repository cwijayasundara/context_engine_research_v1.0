import pytest
from neo4j import Driver

from src.fraud.rules import (
    card_testing,
    duplicate_charge,
    geo_mismatch,
    new_merchant_high_amount,
    round_fx,
    run_all_rules,
    velocity,
)


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


@pytest.mark.neo4j
def test_new_merchant_high_amount_flagged(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            UNWIND range(1, 10) AS i
            MERGE (m:Merchant {id:'known' + i, canonical_name:'Known' + i})
            CREATE (t:Transaction {id:'norm-' + i, amount:-15.0, date:date('2025-06-01') + duration({days:i})})
            MERGE (t)-[:AT]->(m)
        """)
        s.run("""
            MERGE (lux:Merchant {id:'lux', canonical_name:'Lux'})
            CREATE (t:Transaction {id:'big', amount:-2400.0, date:date('2025-06-25')})
            MERGE (t)-[:AT]->(lux)
        """)
    flagged = new_merchant_high_amount(clean_graph, multiplier=5.0)
    assert any(f["tx_id"] == "big" for f in flagged)
    assert not any(f["tx_id"].startswith("norm-") for f in flagged)


@pytest.mark.neo4j
def test_geo_mismatch_flags_unusual_country(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (gb:Location {id:'watford', country:'GB'})
            MERGE (th:Location {id:'bangkok-th', country:'TH'})
            MERGE (m:Merchant {id:'foo', canonical_name:'Foo'})
            WITH gb, th, m
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


@pytest.mark.neo4j
def test_velocity_flags_three_charges_same_merchant_same_day(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            MERGE (m:Merchant {id:'vel', canonical_name:'VeloShop'})
            MERGE (d:Day {id:'2025-06-26'})
            WITH m, d
            UNWIND range(1,4) AS i
            CREATE (t:Transaction {id:'vt-' + i, amount:-25.0, date:date('2025-06-26')})
            MERGE (t)-[:AT]->(m) MERGE (t)-[:ON_DAY]->(d)
        """)
    flagged = velocity(clean_graph, threshold=3)
    assert len({f["tx_id"] for f in flagged}) == 4


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
    assert {f["rule"] for f in grouped["dup-a"]} == {"duplicate_charge"}
    assert {f["rule"] for f in grouped["dup-b"]} == {"duplicate_charge"}
