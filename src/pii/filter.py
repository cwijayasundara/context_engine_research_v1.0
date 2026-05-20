"""PII detection + redaction.

Combines:
  1. Microsoft Presidio (NER-based, catches PERSON, EMAIL, PHONE, ADDRESS, ...).
  2. A bank-/finance-specific regex layer (account numbers, IBAN, sort code,
     last-4 patterns, routing numbers, common ID formats).

Both layers feed the same `PIIVault` so the LLM only ever sees opaque tokens.

The filter is **deterministic within a session** — the same input string
tokenizes to the same output. That keeps the agent's reasoning consistent.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from .vault import PIIVault, TokenizedText, get_default_vault

log = logging.getLogger(__name__)


# Map Presidio entity types → our short vault-type names.
PRESIDIO_TYPE_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "LOCATION": "ADDRESS",
    "US_SSN": "SSN",
    "US_BANK_NUMBER": "ACCT",
    "US_DRIVER_LICENSE": "DL",
    "IBAN_CODE": "IBAN",
    "CREDIT_CARD": "CARD",
    "IP_ADDRESS": "IP",
    "URL": "URL",
}

# Regex layer — patterns Presidio doesn't reliably catch on bank/CC data.
# Full-match patterns: every character of the regex match becomes a token.
BANK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Last-4 of a card or account, often written as ****1234 or ending in 1234
    ("ACCT", re.compile(r"\*{2,4}\d{4}\b")),
    ("ACCT", re.compile(r"\bending\s+in\s+\d{4}\b", re.IGNORECASE)),
    # UK sort code:   12-34-56  — MUST run before phone patterns.
    ("SORTCODE", re.compile(r"\b\d{2}-\d{2}-\d{2}\b")),
    # US routing number (9 digits, often after "routing" keyword)
    ("ROUTING", re.compile(r"\brouting[:\s]*\d{9}\b", re.IGNORECASE)),
    # Plaid-style ids
    ("ACCT", re.compile(r"\bplaid_account_id[:=]\s*\S+\b", re.IGNORECASE)),

    # Card numbers — partially masked (e.g. "5286 83** **** 3344") or full 16-digit.
    ("CARD", re.compile(r"\b\d{4}[\s-]?\d{2}\*{2}[\s-]?\*{4}[\s-]?\d{4}\b")),
    ("CARD", re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b")),

    # UK postcode — outward (1-2 letters + digit + optional letter/digit)
    # + inward (digit + 2 letters), with an optional space between halves.
    ("POSTCODE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")),

    # UK landline / mobile, with optional space or dash separators.
    # Anchored on the leading 0 to avoid eating account numbers / IDs.
    ("PHONE", re.compile(r"\b0\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b")),
    # International, +44 prefix.
    ("PHONE", re.compile(r"\+44[\s-]?\d{1,4}[\s-]?\d{3}[\s-]?\d{3,4}\b")),
    # US-style xxx-xxx-xxxx (appears in transaction descriptions).
    ("PHONE", re.compile(r"\b\d{3}-\d{3}-\d{4}\b")),

    # UK vehicle registration prefixed with DVLA (statement reference style).
    ("VRN", re.compile(r"\bDVLA[-\s]?[A-Z]{2}\d{2}\s?[A-Z]{3}\b", re.IGNORECASE)),

    # Person-name patterns Presidio's NER misses on all-caps text:
    #   "C WIJAYASUNDARA"   — single initial + all-caps surname (>=4 chars).
    #   "HETTI A G PERERA"  — caps first name + 1-4 single-letter initials + caps last name.
    # Word boundaries on each component prevent eating into adjacent words
    # (e.g. won't match "TFL TRAVEL CH" or "WATFORD BOROUGH COUNCIL").
    ("PERSON", re.compile(r"\b[A-Z]\b\s+[A-Z]{4,}\b")),
    ("PERSON", re.compile(r"\b[A-Z]{3,}\b(?:\s+[A-Z]\b){1,4}\s+[A-Z]{3,}\b")),

    # Standalone 6–12 digit number on its own line (e.g. Halifax statements
    # render the Account Number on a separate line under the label).
    ("ACCT", re.compile(r"(?m)^[ \t]*\d{6,12}[ \t]*$")),
]

# Group-capture patterns: only the named capture group is replaced; the
# surrounding label (e.g. "Account no:") is preserved so the document
# remains readable after masking.
BANK_GROUP_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    # "Bank Account no: 00000000" / "Account Number: 12345678"
    (
        "ACCT",
        re.compile(r"(?i)\b(?:bank\s+)?account\s+(?:no|number)[:\s]+(\d{6,12})\b"),
        1,
    ),
    # Title + name (covers headers/addresses where Presidio's NER misses
    # the all-caps surname). Stops at a digit, line-end, or non-letter so
    # the trailing address doesn't get swallowed.
    (
        "PERSON",
        re.compile(
            r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+"
            r"([A-Za-z][A-Za-z\s\.]{2,60}[A-Za-z])"
            r"(?=\s+\d|\s*$|\s*\n|\s+[^A-Za-z\s\.])",
            re.IGNORECASE,
        ),
        1,
    ),
]

# Default Presidio score threshold — above this we treat the span as PII.
DEFAULT_THRESHOLD = 0.4


@dataclass
class FilterConfig:
    languages: tuple[str, ...] = ("en",)
    presidio_threshold: float = DEFAULT_THRESHOLD
    # Don't tokenize merchant names — we want the LLM to be able to reason
    # "you spent $X at Costco". Add merchant strings here at load time so the
    # filter doesn't accidentally redact them.
    merchant_allowlist: frozenset[str] = frozenset()
    # Presidio entity types to skip entirely (left untouched in the output).
    # Use the *mapped* short name where applicable (e.g. ADDRESS not LOCATION).
    disabled_entities: frozenset[str] = frozenset()
    # Fixed-literal substitutions for specific entity types. When an entity's
    # type appears here, the matched span is replaced with the literal string
    # instead of being assigned a `<TYPE_NN>` vault token.
    replacements: dict[str, str] = field(default_factory=dict)


class PIIFilter:
    """Detects PII spans and replaces them with stable vault tokens."""

    def __init__(
        self,
        vault: PIIVault | None = None,
        config: FilterConfig | None = None,
    ) -> None:
        self._vault = vault or get_default_vault()
        self._config = config or FilterConfig()
        self._analyzer = AnalyzerEngine()

    # -- public API ---------------------------------------------------

    @property
    def vault(self) -> PIIVault:
        """The vault this filter writes to. Callers may read tokens or
        dump it for re-hydration."""
        return self._vault

    def tokenize(self, text: str) -> TokenizedText:
        """Replace any detected PII with vault tokens. Returns the
        tokenized text and a token→original map covering only this call."""
        if not text:
            return TokenizedText(text="")
        out = text

        # Round 1a — full-match regex layer (cheap, deterministic).
        for entity_type, pattern in BANK_PATTERNS:
            if entity_type in self._config.disabled_entities:
                continue
            replacement_literal = self._config.replacements.get(entity_type)
            if replacement_literal is not None:
                out = pattern.sub(replacement_literal, out)
            else:
                out = pattern.sub(
                    lambda m, et=entity_type: self._vault.tokenize_value(et, m.group(0)),
                    out,
                )

        # Round 1b — group-capture regex (keep labels, mask only the value).
        for entity_type, pattern, group_idx in BANK_GROUP_PATTERNS:
            if entity_type in self._config.disabled_entities:
                continue
            out = self._tokenize_group(out, pattern, entity_type, group_idx)

        # Round 2 — Presidio NER for the broader categories.
        results = self._analyzer.analyze(
            text=out,
            language=self._config.languages[0],
            score_threshold=self._config.presidio_threshold,
        )
        out = self._apply_presidio(out, results)

        # Build a local view of what was tokenized this call.
        local_tokens = {
            tok: val for tok, val in self._scan_tokens(out).items()
        }
        return TokenizedText(text=out, tokens=local_tokens)

    def detokenize(self, text: str) -> str:
        return self._vault.detokenize(text)

    # -- group-capture helper ----------------------------------------

    def _tokenize_group(
        self,
        text: str,
        pattern: re.Pattern[str],
        entity_type: str,
        group: int,
    ) -> str:
        """Replace only ``group`` of every match in ``text`` with either
        a fixed-literal replacement (from config) or a freshly minted
        vault token. The rest of the match (typically a label) is kept
        intact."""
        replacement_literal = self._config.replacements.get(entity_type)
        def replace(m: re.Match[str]) -> str:
            full = m.group(0)
            target = m.group(group)
            if not target:
                return full
            if replacement_literal is not None:
                new_value = replacement_literal
            else:
                new_value = self._vault.tokenize_value(entity_type, target)
            rel_start = m.start(group) - m.start()
            rel_end = m.end(group) - m.start()
            return full[:rel_start] + new_value + full[rel_end:]
        return pattern.sub(replace, text)

    # -- internals ----------------------------------------------------

    def _apply_presidio(self, text: str, results: Iterable[RecognizerResult]) -> str:
        # Presidio commonly emits overlapping/nested PERSON spans
        # (e.g. "CHAMINDA", "CHAMINDA K", "CHAMINDA K WIJAYASUNDARA").
        # Keep only the maximal ones so replacement doesn't leave half of
        # a compound name behind.
        results = self._maximal_spans(list(results))
        # Sort by end-position desc so substring replacement doesn't shift
        # earlier indexes.
        spans = sorted(results, key=lambda r: r.end, reverse=True)
        allowlist_lower = {m.lower() for m in self._config.merchant_allowlist}
        for r in spans:
            entity_type = PRESIDIO_TYPE_MAP.get(r.entity_type, r.entity_type.upper())
            if entity_type in self._config.disabled_entities:
                continue
            original = text[r.start:r.end]
            if original.strip().lower() in allowlist_lower:
                continue
            replacement = self._config.replacements.get(entity_type)
            if replacement is None:
                replacement = self._vault.tokenize_value(entity_type, original)
            text = text[:r.start] + replacement + text[r.end:]
        return text

    @staticmethod
    def _maximal_spans(spans: list[RecognizerResult]) -> list[RecognizerResult]:
        """Drop any span that is strictly contained inside another. Two
        spans with identical (start, end) are both kept; one will be a
        no-op once the text has been rewritten."""
        keep: list[RecognizerResult] = []
        for r in spans:
            contained = any(
                (other.start <= r.start and r.end <= other.end
                 and (other.start, other.end) != (r.start, r.end))
                for other in spans
            )
            if not contained:
                keep.append(r)
        return keep

    @staticmethod
    def _scan_tokens(text: str) -> dict[str, str]:
        """Find every `<TYPE_NN>` token present in ``text``."""
        out: dict[str, str] = {}
        for m in re.finditer(r"<[A-Z]+_\d{2,}>", text):
            out[m.group(0)] = m.group(0)   # value not known here; vault holds truth
        return out
