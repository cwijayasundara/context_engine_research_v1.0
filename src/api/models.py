"""Pydantic response models for the JSON surface."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str
    type: str                # "Merchant" | "Category" | "Month"
    category: str | None = None
    total_spend: float | None = None
    visits: int | None = None
    income: float | None = None
    expense: float | None = None


class GraphLink(BaseModel):
    source: str
    target: str
    type: str                # "IN_CATEGORY" | "ACTIVE_IN"
    weight: float | None = None
    visits: int | None = None


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphLink]
    range: list[str] = Field(default_factory=list,
                             description="The month-IDs this graph covers.")


# ---------- Context-graph contract ---------------------------------------
# A typed view of "the slice of the graph the UI should show right now."
# The contract is intentionally light: nodes and rels carry only what the
# canvas needs to render and select; everything else (full property dump,
# rationale, fraud flags) belongs in detail panels.


class GraphViewNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphViewRel(BaseModel):
    id: str
    source: str
    target: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphUpdate(BaseModel):
    """Single message that drives the canvas state machine.

    mode='replace' wipes the current view; mode='merge' adds without
    removing what's there. focus_ids is what the canvas should highlight
    (and zoom to) after applying the update.
    """
    nodes: list[GraphViewNode] = Field(default_factory=list)
    relationships: list[GraphViewRel] = Field(default_factory=list)
    focus_ids: list[str] = Field(default_factory=list)
    mode: Literal["replace", "merge"] = "replace"


class SchemaNode(BaseModel):
    id: str
    label: str
    type: str
    count: int | None = None
    description: str | None = None


class SchemaRel(BaseModel):
    id: str
    source: str
    target: str
    type: str
    description: str | None = None


class SchemaResponse(BaseModel):
    nodes: list[SchemaNode]
    relationships: list[SchemaRel]


class WikiPage(BaseModel):
    type: str                # "merchant" | "category" | "month" | "annual" | "index"
    name: str
    path: str                # relative path within the vault
    frontmatter: dict
    markdown: str
    outbound_links: list[str] = Field(default_factory=list)


class WikiTreeNode(BaseModel):
    section: str             # "merchants" | "categories" | "months" | "annual"
    pages: list[str]


class WikiTreeResponse(BaseModel):
    sections: list[WikiTreeNode]


class TimelinePoint(BaseModel):
    month: str
    income: float
    expense: float
    net: float
    transactions: int


class TimelineResponse(BaseModel):
    points: list[TimelinePoint]


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    wiki_root_exists: bool
    transaction_count: int


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
