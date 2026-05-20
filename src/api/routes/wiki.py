"""Serve the Obsidian-style wiki vault as JSON.

Reads ``data/wiki/`` directly off disk. Each page is parsed with
``python-frontmatter`` so the YAML metadata round-trips into the API
response untouched, and ``[[wikilinks]]`` are extracted as outbound edges
so the UI can build its own navigation index without re-parsing markdown.
"""
from __future__ import annotations

import re
from pathlib import Path

import frontmatter
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_wiki_root
from src.api.models import WikiPage, WikiTreeNode, WikiTreeResponse

router = APIRouter(prefix="/wiki", tags=["wiki"])

_SECTIONS = ("merchants", "categories", "months", "annual")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


# ---------- Tree ----------------------------------------------------------

@router.get("/tree", response_model=WikiTreeResponse)
def get_tree(root: Path = Depends(get_wiki_root)) -> WikiTreeResponse:
    sections: list[WikiTreeNode] = []
    for section in _SECTIONS:
        d = root / section
        if not d.is_dir():
            continue
        pages = sorted(p.stem for p in d.glob("*.md"))
        if pages:
            sections.append(WikiTreeNode(section=section, pages=pages))
    return WikiTreeResponse(sections=sections)


# ---------- Search --------------------------------------------------------

@router.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Substring match (case-insensitive)"),
    root: Path = Depends(get_wiki_root),
) -> dict:
    needle = q.lower()
    hits: list[dict] = []
    for md_path in root.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        if needle not in text.lower():
            continue
        rel = md_path.relative_to(root).as_posix()
        # First non-frontmatter line is usually the H1 — good enough preview.
        lines = [ln for ln in text.splitlines()
                 if ln.strip() and not ln.startswith("---")]
        hits.append({
            "path": rel,
            "section": md_path.parent.name if md_path.parent != root else "root",
            "name": md_path.stem,
            "preview": (lines[0] if lines else "").lstrip("# ").strip()[:200],
        })
    return {"results": hits[:100], "total": len(hits)}


# ---------- Page ----------------------------------------------------------

@router.get("/page", response_model=WikiPage)
def get_page(
    section: str = Query(..., pattern="^(merchants|categories|months|annual)$"),
    name: str = Query(..., min_length=1),
    root: Path = Depends(get_wiki_root),
) -> WikiPage:
    return _read_page(root, section, name)


@router.get("/home", response_model=WikiPage)
def get_home(root: Path = Depends(get_wiki_root)) -> WikiPage:
    home = root / "Home.md"
    if not home.exists():
        raise HTTPException(404, "Home.md not generated — run compile_wiki first")
    post = frontmatter.load(home)
    return WikiPage(
        type=str(post.metadata.get("type", "index")),
        name="Home",
        path="Home.md",
        frontmatter=post.metadata,
        markdown=post.content,
        outbound_links=_extract_wikilinks(post.content),
    )


def _read_page(root: Path, section: str, name: str) -> WikiPage:
    # ``section`` is regex-validated upstream, but defend in depth.
    if section not in _SECTIONS:
        raise HTTPException(400, f"unknown section {section!r}")
    # Strip any path separators in name; the vault is a flat dir per section.
    safe_name = Path(name).name
    md_path = root / section / f"{safe_name}.md"
    if not md_path.exists():
        raise HTTPException(404, f"{section}/{safe_name} not found")
    post = frontmatter.load(md_path)
    return WikiPage(
        type=str(post.metadata.get("type", section.rstrip("s"))),
        name=safe_name,
        path=md_path.relative_to(root).as_posix(),
        frontmatter=post.metadata,
        markdown=post.content,
        outbound_links=_extract_wikilinks(post.content),
    )


def _extract_wikilinks(markdown: str) -> list[str]:
    """Return the distinct ``[[target]]`` tokens found in the body."""
    return sorted({m.group(1).strip() for m in _WIKILINK_RE.finditer(markdown)})
