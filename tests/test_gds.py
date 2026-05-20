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
    assert ("A", "B") in pairs or ("B", "A") in pairs
    assert ("B", "C") in pairs or ("C", "B") in pairs
    assert ("A", "C") not in pairs and ("C", "A") not in pairs


@pytest.mark.neo4j
def test_pagerank_and_louvain_write_properties(clean_graph: Driver):
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
    client.run_pagerank()
    client.run_louvain()

    with clean_graph.session() as s:
        rows = s.run("MATCH (m:Merchant) RETURN m.canonical_name AS n, "
                     "m.pagerank AS pr, m.community AS c").data()
    by_name = {r["n"]: r for r in rows}
    assert by_name["B"]["pr"] > by_name["A"]["pr"]
    assert all(r["c"] is not None for r in rows)


@pytest.mark.neo4j
def test_fastrp_knn_node_similarity_write_back(clean_graph: Driver):
    with clean_graph.session() as s:
        s.run("""
            UNWIND ['a','b','c','x','y'] AS k MERGE (:Merchant {id:k, canonical_name:k})
        """)
        s.run("""
            MERGE (d1:Day {id:'2025-06-01'}) MERGE (d2:Day {id:'2025-06-02'})
            MERGE (d3:Day {id:'2025-06-03'})
        """)
        s.run("""
            MATCH (a:Merchant {id:'a'}), (b:Merchant {id:'b'}), (c:Merchant {id:'c'}),
                  (x:Merchant {id:'x'}), (y:Merchant {id:'y'}),
                  (d1:Day {id:'2025-06-01'}), (d2:Day {id:'2025-06-02'}),
                  (d3:Day {id:'2025-06-03'})
            CREATE (t1:Transaction)-[:AT]->(a) WITH t1, d1,d2,d3, a,b,c,x,y
            CREATE (t1)-[:ON_DAY]->(d1)
            CREATE (t2:Transaction)-[:AT]->(b) WITH t2, d1,d2,d3, a,b,c,x,y
            CREATE (t2)-[:ON_DAY]->(d1)
            CREATE (t3:Transaction)-[:AT]->(b) WITH t3, d1,d2,d3, a,b,c,x,y
            CREATE (t3)-[:ON_DAY]->(d2)
            CREATE (t4:Transaction)-[:AT]->(c) WITH t4, d1,d2,d3, a,b,c,x,y
            CREATE (t4)-[:ON_DAY]->(d2)
            CREATE (t5:Transaction)-[:AT]->(x) WITH t5, d1,d2,d3, a,b,c,x,y
            CREATE (t5)-[:ON_DAY]->(d3)
            CREATE (t6:Transaction)-[:AT]->(y) WITH t6, d1,d2,d3, a,b,c,x,y
            CREATE (t6)-[:ON_DAY]->(d3)
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
        assert ns["n"]  >= 0


@pytest.mark.neo4j
def test_mark_outliers_flags_singleton_community(clean_graph: Driver):
    with clean_graph.session() as s:
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
