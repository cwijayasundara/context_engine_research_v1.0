"""Standalone CLI utility to mask PII in markdown statements.

Walks a directory of `.md` statements (credit-card and/or savings-account)
and writes a masked copy of every file to a parallel output tree, using
the same local PIIFilter (Microsoft Presidio + finance-specific regex)
that powers the in-process agent middleware.

Everything stays on the local machine — Presidio is invoked in-process
and no statement text leaves the host.

Example:

    # mask everything under data/statements/ into data/statements_masked/
    python -m src.pii.mask_statements \\
        --input  data/statements \\
        --output data/statements_masked \\
        --vault-out data/statements_masked/.vault.json

    # mask just the credit-card statements
    python -m src.pii.mask_statements \\
        --input  data/statements/crdit_stmt \\
        --output data/statements_masked/crdit_stmt

The `--vault-out` JSON contains the token → original-value map and is at
least as sensitive as the original statements — store it accordingly, or
omit the flag to discard the mapping when the process exits.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .filter import FilterConfig, PIIFilter
from .vault import PIIVault

# Defaults for the standalone masking CLI. The agent middleware uses its
# own FilterConfig and is not affected.
#   * DATE_TIME — dates aren't sensitive in this dataset, leave them in place.
#   * PERSON    — replace every detected name with one fixed dummy so the
#                 masked output reads naturally instead of having `<PERSON_NN>`.
DEFAULT_DISABLED_ENTITIES: frozenset[str] = frozenset({"DATE_TIME"})
DEFAULT_REPLACEMENTS: dict[str, str] = {"PERSON": "Tony Stark"}

# Strings that Presidio's PERSON NER tags at 0.85 confidence despite being
# obvious PDF-extraction artifacts in this dataset's column headers.
DEFAULT_NOISE_ALLOWLIST: frozenset[str] = frozenset({
    "Bal nce", "Bal", "Descript", "Descript on",
    "Money I", "Money Out", "Date", "Type",
})

log = logging.getLogger(__name__)


def mask_file(filter_: PIIFilter, src: Path, dst: Path) -> int:
    """Mask ``src`` into ``dst``. Returns the number of *new* vault
    entries minted by this file (i.e. excludes values that were already
    seen earlier in the run)."""
    text = src.read_text(encoding="utf-8")
    before = filter_.vault.size()
    masked = filter_.tokenize(text)
    after = filter_.vault.size()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(masked.text, encoding="utf-8")
    return after - before


def mask_directory(
    input_dir: Path,
    output_dir: Path,
    vault: PIIVault | None = None,
    config: FilterConfig | None = None,
    pattern: str = "*.md",
) -> tuple[int, int]:
    """Mask every file matching ``pattern`` under ``input_dir`` into
    ``output_dir``, preserving the relative directory structure.

    A single shared vault is used across all files, so the same person /
    account / card always maps to the same token across the whole run.

    Returns ``(files_processed, total_unique_pii_values)``.
    """
    vault = vault or PIIVault()
    filter_ = PIIFilter(vault=vault, config=config)
    n_files = 0
    for src in sorted(input_dir.rglob(pattern)):
        if not src.is_file():
            continue
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        minted = mask_file(filter_, src, dst)
        n_files += 1
        log.info("masked %s  (+%d new PII values)", rel, minted)
    return n_files, vault.size()


def dump_vault(vault: PIIVault, path: Path) -> None:
    """Persist the token → value map to ``path`` as JSON.

    The output keys are sorted for reproducible diffs. Treat this file
    as containing the raw PII it indexes — it does.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {tok: val for tok, val in sorted(vault.items())}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mask_statements",
        description="Mask PII in bank / credit-card statements using Presidio + regex.",
    )
    p.add_argument(
        "--input", type=Path, required=True,
        help="Directory containing markdown statements (searched recursively).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Directory to write masked copies into. Layout is preserved.",
    )
    p.add_argument(
        "--vault-out", type=Path, default=None,
        help="Optional path to write the JSON token → value map for re-hydration.",
    )
    p.add_argument(
        "--salt", default="",
        help="Vault salt — affects internal fingerprint hashing only.",
    )
    p.add_argument(
        "--pattern", default="*.md",
        help="Glob for input files (default: *.md).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.input.is_dir():
        log.error("--input must be an existing directory: %s", args.input)
        return 2

    vault = PIIVault(salt=args.salt)
    # 0.7 threshold suppresses Presidio's lower-confidence PERSON matches
    # on corrupted column headers ("Bal nce", "Descript on") that show up
    # in some PDF-extracted statements.
    config = FilterConfig(
        presidio_threshold=0.7,
        disabled_entities=DEFAULT_DISABLED_ENTITIES,
        replacements=DEFAULT_REPLACEMENTS,
        merchant_allowlist=DEFAULT_NOISE_ALLOWLIST,
    )
    n_files, n_unique = mask_directory(
        args.input, args.output, vault=vault, config=config, pattern=args.pattern,
    )
    log.info("done: %d files masked, %d unique PII values vaulted", n_files, n_unique)

    if args.vault_out:
        dump_vault(vault, args.vault_out)
        log.info("wrote vault → %s", args.vault_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
