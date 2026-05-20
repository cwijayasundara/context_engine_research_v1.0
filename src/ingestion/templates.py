"""Hash a raw transaction description down to a stable 'template' fingerprint.

Inspired by the IEEE-CIS Identity columns where many transactions share the
same template string. Two descriptions that differ only in trailing IDs /
amounts hash to the same template, so we can group them in the graph and
later run community detection over the resulting bipartite Transaction ↔
Template graph.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    id: str       # first 8 chars of sha256(template)
    template: str


# Mask: any run of 3+ alphanumerics that contains a digit → "<ID>".
# Standalone digit groups → "<NUM>".
_ID_RE  = re.compile(r"\b(?=\w*\d)\w{3,}\b")
_NUM_RE = re.compile(r"\b\d+\b")


def fingerprint(description: str) -> Template:
    template = _ID_RE.sub("<ID>", description.upper())
    template = _NUM_RE.sub("<NUM>", template)
    template = re.sub(r"\s+", " ", template).strip()
    h = hashlib.sha256(template.encode()).hexdigest()[:8]
    return Template(id=h, template=template)
