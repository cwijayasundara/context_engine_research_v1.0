"""FastAPI app factory.

Boot::

    uvicorn src.api.main:app --reload --port 8000

Phase 1 surface (read-only):
  * GET /api/health
  * GET /api/graph
  * GET /api/wiki/tree
  * GET /api/wiki/home
  * GET /api/wiki/page
  * GET /api/wiki/search
  * GET /api/timeline
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.deps import get_driver
from src.api.routes import agent, canon, fraud, graph, health, pii, timeline, wiki


def _configure_logging() -> None:
    """Make our agent-runner traces visible alongside uvicorn's access log."""
    level = os.getenv("FCE_LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s · %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quiet the noisier libraries — they're not useful for app debugging.
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    # Bump our own surface — these are the lines you actually want to read.
    for mod in ("src.api", "src.api.agent_runner", "src.api.routes.agent",
                "src.api.decisions", "src.api.sessions"):
        logging.getLogger(mod).setLevel(logging.DEBUG if level == "DEBUG" else logging.INFO)


_configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the Neo4j driver on boot, close it on shutdown."""
    driver = get_driver()
    try:
        yield
    finally:
        driver.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="finance-context-engine API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        # Locked to the local Next dev server. Update when deploying.
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(health.router,   prefix="/api")
    app.include_router(graph.router,    prefix="/api")
    app.include_router(wiki.router,     prefix="/api")
    app.include_router(timeline.router, prefix="/api")
    app.include_router(agent.router,    prefix="/api")
    app.include_router(pii.router,      prefix="/api")
    app.include_router(canon.router,    prefix="/api")
    app.include_router(fraud.router, prefix="/api")
    return app


app = create_app()
