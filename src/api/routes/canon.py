"""Live editor for the LLM canonicalizer's decisions.

Reads the on-disk cache (the JSON file ``llm_normalize`` writes) and the
in-memory ``ALIASES`` / ``CATEGORY_LOCK`` dicts. Edits made via this
surface mutate the runtime dicts immediately so the agent picks up
overrides on the next call; persisting them back to ``llm_normalize.py``
is a separate "save to source" action.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.ingestion import llm_normalize

router = APIRouter(prefix="/canon", tags=["canon"])


@router.get("/cache")
def list_cache() -> dict:
    """Every cached classification + the active alias / category-lock tables."""
    # Force-load the file cache so cold starts return something sensible.
    llm_normalize._load_cache()  # noqa: SLF001 — module-level helper, fine to poke
    entries = [
        {
            "raw": raw,
            "canonical_name": v["canonical_name"],
            "category": v["category"],
            "kind": v.get("kind", "expense"),
            # What the user-facing dispatcher will *actually* return for this row.
            "effective": _resolve(v),
        }
        for raw, v in sorted(llm_normalize._cache.items())  # noqa: SLF001
    ]
    return {
        "cache": entries,
        "aliases":        dict(llm_normalize.ALIASES),
        "category_lock":  dict(llm_normalize.CATEGORY_LOCK),
        "cache_path":     str(llm_normalize.CACHE_PATH),
    }


def _resolve(v: dict) -> dict:
    """Run the alias + category-lock passes the way the live agent does."""
    name = llm_normalize._apply_alias(v["canonical_name"])      # noqa: SLF001
    category, kind = llm_normalize._apply_category_lock(        # noqa: SLF001
        name, v["category"], v.get("kind", "expense"),
    )
    return {"canonical_name": name, "category": category, "kind": kind}


class AliasReq(BaseModel):
    variant: str = Field(..., min_length=1, description="LLM-emitted name to fold")
    canonical: str = Field(..., min_length=1, description="Project's canonical spelling")


@router.post("/aliases")
def add_alias(req: AliasReq) -> dict:
    llm_normalize.ALIASES[req.variant.strip().lower()] = req.canonical.strip()
    return {"ok": True, "aliases": dict(llm_normalize.ALIASES)}


class CategoryLockReq(BaseModel):
    canonical: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)


@router.post("/category-lock")
def lock_category(req: CategoryLockReq) -> dict:
    if req.category not in llm_normalize.ALLOWED_CATEGORIES:
        raise HTTPException(400, f"category must be one of {llm_normalize.ALLOWED_CATEGORIES}")
    llm_normalize.CATEGORY_LOCK[req.canonical.strip()] = req.category.strip()
    return {"ok": True, "category_lock": dict(llm_normalize.CATEGORY_LOCK)}


@router.delete("/cache/{raw}")
def evict(raw: str) -> dict:
    """Drop one cached row so the next canonicalize() re-asks the LLM."""
    if raw not in llm_normalize._cache:                         # noqa: SLF001
        raise HTTPException(404, f"no cache entry for {raw!r}")
    del llm_normalize._cache[raw]                               # noqa: SLF001
    Path(llm_normalize.CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(llm_normalize.CACHE_PATH).write_text(
        json.dumps(llm_normalize._cache, indent=2, sort_keys=True),  # noqa: SLF001
    )
    return {"ok": True}


@router.get("/categories")
def categories() -> dict:
    return {"categories": list(llm_normalize.ALLOWED_CATEGORIES)}
