"""Ontology compiler CLI.

Reads ``data/ontology/finance.yaml`` and produces ``src/ontology/schema.cypher``.
The YAML is the source of truth — never hand-edit the cypher file. The
generator is deterministic so ``--check`` is suitable for CI.

::

    python -m src.ontology.compile           # print cypher to stdout
    python -m src.ontology.compile --write   # overwrite schema.cypher
    python -m src.ontology.compile --check   # exit 1 if regen would change anything
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.ontology.model import load_ontology, to_cypher

log = logging.getLogger(__name__)

_DEFAULT_YAML = Path(__file__).resolve().parents[2] / "data" / "ontology" / "finance.yaml"
_DEFAULT_CYPHER = Path(__file__).resolve().parent / "schema.cypher"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml",   default=str(_DEFAULT_YAML),
                        help="ontology YAML path (default: data/ontology/finance.yaml)")
    parser.add_argument("--cypher", default=str(_DEFAULT_CYPHER),
                        help="cypher output path (default: src/ontology/schema.cypher)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true",
                       help="overwrite the cypher file")
    group.add_argument("--check", action="store_true",
                       help="exit 1 if regenerating would change the cypher")
    args = parser.parse_args(argv)

    ontology = load_ontology(args.yaml)
    generated = to_cypher(ontology)

    target = Path(args.cypher)
    if args.write:
        target.write_text(generated, encoding="utf-8")
        log.info("wrote %s (%d entities, %d relationships)",
                 target, len(ontology.entities), len(ontology.relationships))
        return 0
    if args.check:
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if existing != generated:
            log.error("schema.cypher is out of date — run `python -m src.ontology.compile --write`")
            return 1
        log.info("schema.cypher up to date")
        return 0

    sys.stdout.write(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
