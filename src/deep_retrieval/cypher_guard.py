"""Conservative validation for LLM-generated read-only Cypher."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.ontology.model import Ontology


class CypherValidationError(ValueError):
    """Raised when generated Cypher violates retrieval safety rules."""


@dataclass(frozen=True)
class CypherValidationResult:
    cypher: str
    limit: int | None
    labels: set[str]
    relationships: set[str]
    properties: set[tuple[str, str]]


_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV)\b|"
    r"\bCALL\s+apoc\.(?:cypher\.run|create|merge|periodic|load|refactor)",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r"\([A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*)")
_REL_RE = re.compile(
    r"\((?P<left_var>[A-Za-z_]\w*)\s*:\s*(?P<left_label>[A-Za-z_]\w*)[^)]*\)"
    r"\s*(?P<left_arrow><-|-)\s*\[:\s*(?P<rel>[A-Za-z_]\w*)\s*\]\s*"
    r"(?P<right_arrow>->|-)\s*"
    r"\((?P<right_var>[A-Za-z_]\w*)\s*:\s*(?P<right_label>[A-Za-z_]\w*)[^)]*\)"
)
_PROP_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
_ALIAS_RE = re.compile(r"\(([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)")
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
_AGG_RE = re.compile(r"\b(count|sum|avg|min|max|collect)\s*\(", re.IGNORECASE)
_WHERE_LITERAL_RE = re.compile(
    r"\bWHERE\b(?:(?!\bRETURN\b).)*=\s*(['\"])[^'\"]+\1",
    re.IGNORECASE | re.DOTALL,
)


class CypherGuard:
    """Validate generated Cypher against ontology and read-only constraints."""

    def __init__(
        self,
        *,
        labels: set[str],
        properties: dict[str, set[str]],
        relationships: set[tuple[str, str, str]],
        max_limit: int = 100,
    ) -> None:
        self.labels = labels
        self.properties = properties
        self.relationships = relationships
        self.relationship_types = {rtype for _, rtype, _ in relationships}
        self.max_limit = max_limit

    @classmethod
    def from_ontology(cls, ontology: Ontology) -> "CypherGuard":
        labels = {entity.name for entity in ontology.entities}
        properties = {
            entity.name: {prop.name for prop in entity.properties}
            for entity in ontology.entities
        }
        relationships = {
            (rel.source, rel.type, rel.target)
            for rel in ontology.relationships
        }
        return cls(labels=labels, properties=properties, relationships=relationships)

    def validate(
        self,
        cypher: str,
        *,
        params: dict | None,
    ) -> CypherValidationResult:
        normalized = " ".join(cypher.strip().split())
        if not normalized:
            raise CypherValidationError("empty Cypher query")
        if _WRITE_RE.search(normalized):
            raise CypherValidationError("write operation is not allowed")
        if _WHERE_LITERAL_RE.search(normalized):
            raise CypherValidationError("user values in WHERE must use parameters")

        aliases = dict(_ALIAS_RE.findall(normalized))
        labels = set(_LABEL_RE.findall(normalized))
        unknown_labels = labels - self.labels
        if unknown_labels:
            raise CypherValidationError(f"unknown label: {sorted(unknown_labels)[0]}")

        relationships: set[str] = set()
        for match in _REL_RE.finditer(normalized):
            left = match.group("left_label")
            right = match.group("right_label")
            rel = match.group("rel")
            relationships.add(rel)
            if rel not in self.relationship_types:
                raise CypherValidationError(f"unknown relationship: {rel}")
            actual = (
                (left, rel, right)
                if match.group("left_arrow") == "-" and match.group("right_arrow") == "->"
                else (right, rel, left)
            )
            if actual not in self.relationships:
                raise CypherValidationError(
                    f"wrong direction for relationship {rel}: {actual[0]}->{actual[2]}"
                )

        properties: set[tuple[str, str]] = set()
        for var, prop in _PROP_RE.findall(normalized):
            label = aliases.get(var)
            if not label:
                continue
            if prop not in self.properties.get(label, set()):
                raise CypherValidationError(f"unknown property: {label}.{prop}")
            properties.add((label, prop))

        limit = self._extract_limit(normalized)
        if limit is None and self._returns_rows(normalized):
            raise CypherValidationError("row-returning Cypher queries must include LIMIT")
        if limit is not None and limit > self.max_limit:
            raise CypherValidationError(f"LIMIT must be <= {self.max_limit}")

        return CypherValidationResult(
            cypher=normalized,
            limit=limit,
            labels=labels,
            relationships=relationships,
            properties=properties,
        )

    @staticmethod
    def _extract_limit(cypher: str) -> int | None:
        match = _LIMIT_RE.search(cypher)
        return int(match.group(1)) if match else None

    @staticmethod
    def _returns_rows(cypher: str) -> bool:
        if " RETURN " not in f" {cypher.upper()} ":
            return False
        return not bool(_AGG_RE.search(cypher))
