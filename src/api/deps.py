"""Shared API dependencies: Neo4j driver singleton, wiki root path."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from neo4j import Driver, GraphDatabase

from src.config import SETTINGS


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    """Process-wide Neo4j driver. Closed by the FastAPI lifespan handler."""
    return GraphDatabase.driver(
        SETTINGS.neo4j_uri,
        auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password),
    )


def get_wiki_root() -> Path:
    return SETTINGS.wiki_dir
