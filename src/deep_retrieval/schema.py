"""Schema context and Cypher examples for Deep Agents retrieval."""
from __future__ import annotations

from src.ontology.model import Ontology, load_ontology


def build_schema_context(ontology: Ontology | None = None) -> str:
    """Return compact schema text for Cypher generation prompts."""
    ontology = ontology or load_ontology()
    lines = [
        "Use only these labels, properties, and relationship directions.",
        "Do not invent labels, relationship types, or properties.",
        "Parameterize user-provided values; do not inline quoted values in WHERE.",
        "Labels:",
    ]
    for entity in ontology.entities:
        public_props = [
            p.name for p in entity.properties
            if not p.pii and not p.vector and p.name not in {"embedding", "desc_embedding"}
        ]
        lines.append(f"- ({entity.name}) properties: {', '.join(public_props) or '(none)'}")
    lines.append("Relationships:")
    for rel in ontology.relationships:
        lines.append(f"- ({rel.source})-[:{rel.type}]->({rel.target})")
    return "\n".join(lines)


def finance_cypher_examples() -> list[dict]:
    """Few-shot examples grounded in the finance ontology."""
    return [
        {
            "name": "Merchant spend by year",
            "question": "How much did I spend at Costco in 2025?",
            "cypher": (
                "MATCH (t:Transaction)-[:AT]->(m:Merchant) "
                "WHERE m.canonical_name = $merchant AND t.year = $year AND t.amount < 0 "
                "RETURN m.canonical_name AS merchant, sum(-t.amount) AS spend, count(t) AS visits"
            ),
            "params": {"merchant": "Costco", "year": 2025},
        },
        {
            "name": "Top categories by month",
            "question": "What were my top spending categories in 2026-04?",
            "cypher": (
                "MATCH (t:Transaction)-[:AT]->(m:Merchant)-[:IN_CATEGORY]->(c:Category) "
                "WHERE t.month = $month AND t.amount < 0 "
                "RETURN c.name AS category, sum(-t.amount) AS spend, count(t) AS transactions "
                "ORDER BY spend DESC LIMIT 10"
            ),
            "params": {"month": "2026-04"},
        },
        {
            "name": "Fraud alerts by severity",
            "question": "Show high severity fraud alerts for May 2026",
            "cypher": (
                "MATCH (a:Alert)-[:FLAGS]->(t:Transaction)-[:AT]->(m:Merchant) "
                "WHERE t.month = $month AND a.severity >= $min_severity "
                "RETURN a.id AS alert_id, a.kind AS kind, a.severity AS severity, "
                "t.id AS tx_id, m.canonical_name AS merchant, t.amount AS amount "
                "ORDER BY severity DESC LIMIT 20"
            ),
            "params": {"month": "2026-05", "min_severity": 0.5},
        },
    ]
