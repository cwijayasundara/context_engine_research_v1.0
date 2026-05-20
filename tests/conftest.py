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

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "please-change-me")

# IMPORTANT: tests must NOT run against the dev `neo4j` database — the
# clean_graph fixture wipes everything and previously destroyed populated
# dev data when CI/local tests ran. The docker-compose image is Neo4j
# Enterprise, which allows multiple databases, so we use a dedicated
# `test_finance` DB. Override via NEO4J_TEST_DATABASE if needed.
TEST_DATABASE = os.getenv("NEO4J_TEST_DATABASE", "test-finance")


def _driver_or_skip() -> Driver | None:
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        drv.verify_connectivity()
        return drv
    except Exception:
        return None


def _ensure_test_db(drv: Driver) -> bool:
    """Create the isolated test database on first use. Returns False if the
    server is Community Edition and can't host multiple DBs — caller should
    skip neo4j-marked tests rather than fall back to wiping `neo4j`."""
    try:
        with drv.session(database="system") as s:
            # Backtick-quote — Neo4j DB names with hyphens require it.
            s.run(f"CREATE DATABASE `{TEST_DATABASE}` IF NOT EXISTS WAIT")
        return True
    except Exception as exc:
        import logging
        logging.getLogger(__name__).info("ensure_test_db skipped: %s", exc)
        return False


@pytest.fixture(scope="session")
def neo4j_driver() -> Driver:
    drv = _driver_or_skip()
    if drv is None:
        pytest.skip("Neo4j not reachable on {} — start docker compose first".format(NEO4J_URI))
    if not _ensure_test_db(drv):
        pytest.skip(
            f"Cannot create isolated test database '{TEST_DATABASE}' "
            "(Community Edition?). Refusing to wipe the dev `neo4j` DB."
        )
    yield drv
    drv.close()


class _TestDbDriver:
    """Driver proxy whose ``session()`` always targets the test database.

    Necessary because the test bodies call ``clean_graph.session()`` with
    no ``database=`` kwarg; without this proxy the session would fall
    through to the server-default `neo4j` DB and wipe / mutate dev data.
    """

    def __init__(self, inner: Driver) -> None:
        self._inner = inner

    def session(self, **kwargs):
        kwargs.setdefault("database", TEST_DATABASE)
        return self._inner.session(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture()
def clean_graph(neo4j_driver: Driver):
    """Wipe the isolated test database before each test."""
    with neo4j_driver.session(database=TEST_DATABASE) as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield _TestDbDriver(neo4j_driver)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
