"""CLI: generate dummy Halifax statements with gpt-5.4-mini.

Examples
--------
Generate the full default window (01/01/2025 → 31/05/2026):

    python examples/generate_sample_data.py

Generate a smaller window into a custom directory:

    python examples/generate_sample_data.py \\
        --start 2025-01-01 --end 2025-03-31 \\
        --out data/statements_generated_smoke

If ``OPENAI_API_KEY`` is unset (or the call fails), the script falls back to
a deterministic offline generator so it still produces files.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Allow running as `python examples/...` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_gen import generate_all  # noqa: E402


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_parse_date, default=date(2025, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2026, 5, 31))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "statements_generated",
        help="Output directory (savings_stmt/ and crdit_stmt/ are created here).",
    )
    parser.add_argument("--seed", type=int, default=1588)
    args = parser.parse_args()

    written = generate_all(args.start, args.end, args.out, seed=args.seed)
    print(f"Savings statements written: {len(written['savings'])}")
    print(f"Credit statements written:  {len(written['credit'])}")
    print(f"Output directory: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
