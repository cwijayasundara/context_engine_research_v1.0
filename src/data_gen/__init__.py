"""Synthetic statement generator for the finance-context-engine.

Produces realistic-looking Halifax savings + credit card statements as
Markdown, using gpt-5.4-mini to vary the discretionary spend each month.
"""

from .generator import generate_all

__all__ = ["generate_all"]
