"""Neo4j GDS pipeline for the fraud layer.

Builds a `merchant-coincidence` graph (Merchants linked when they appear on
the same calendar day), then runs PageRank, Louvain, FastRP, KNN, and Node
Similarity over it. All writes happen back onto the persistent Neo4j store.
"""
from __future__ import annotations

import logging

from neo4j import Driver

log = logging.getLogger(__name__)

GRAPH_NAME = "merchant-coincidence"


_PROJECT_COINCIDENCE = """
MATCH (m1:Merchant)<-[:AT]-(:Transaction)-[:ON_DAY]->(d:Day)<-[:ON_DAY]-(:Transaction)-[:AT]->(m2:Merchant)
WHERE id(m1) < id(m2)
WITH m1, m2, count(DISTINCT d) AS w
MERGE (m1)-[r:CO_OCCURRED]->(m2)
  SET r.weight = w
"""


class GdsClient:
    def __init__(self, driver: Driver):
        self.driver = driver

    def project_merchant_coincidence(self) -> None:
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

    def mark_outliers(self) -> None:
        with self.driver.session() as s:
            s.run("""
                MATCH (m:Merchant)
                OPTIONAL MATCH (peer:Merchant) WHERE peer.community = m.community AND peer <> m
                WITH m, count(peer) AS peers
                SET m.is_outlier = (peers = 0) OR coalesce(m.pagerank, 0) < 0.1
            """)
            log.info("is_outlier marked")
