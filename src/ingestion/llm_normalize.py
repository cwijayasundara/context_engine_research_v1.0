"""LLM-cached merchant canonicalization.

Drop-in replacement for the regex table in ``normalize.py``: given a raw
transaction description, ``canonicalize()`` returns the same
``MerchantInfo`` tuple but the decision is sourced from a JSON cache on
disk. On cache miss we ask ``gpt-5.4-mini`` to classify, write the answer
back, and never call the LLM again for that exact description.

A ``prime_cache()`` helper batches misses so loading 1,500 transactions
costs ~3 LLM calls (one per 25 unique descriptions) instead of one per
row. When ``OPENAI_API_KEY`` is unset (or the call fails) we fall back to
the regex implementation so offline/CI runs keep working.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

# The MerchantInfo dataclass + regex fallback live in the original module.
from .normalize import (
    MerchantInfo,
    canonicalize as _regex_canonicalize,
    _INCOME_CATEGORIES,
    _TRANSFER_CATEGORIES,
)

load_dotenv()

log = logging.getLogger(__name__)

# Canonical category vocabulary. The LLM is forced to pick from this list so
# wiki rollups stay stable and the ``System`` filter in compile_wiki keeps
# working. Keep this in sync with the categories used by the regex rules.
ALLOWED_CATEGORIES: list[str] = [
    "Groceries", "Dining", "Coffee", "Transport", "Fuel",
    "Household", "Health", "Education", "Clothing", "Shopping",
    "Subscriptions", "Cloud", "Mobile", "Utilities",
    "Insurance", "Pension", "Mortgage", "Finance", "Banking", "Credit Card",
    "Council Tax", "Charity", "Savings",
    "Income", "Transfers",
    "System",          # statement-housekeeping, filtered out of wiki
    "Uncategorized",
]

CACHE_PATH = Path(os.getenv("CANON_CACHE_PATH", "data/.canonicalize_cache.json"))
BATCH_SIZE = 25
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").removeprefix("openai:")

# Stable canonical names — the LLM is right about *what* a merchant is but
# wobbles on *how to spell it* ("Sainsburys" vs "Sainsbury's", "Github" vs
# "GitHub"). This dict folds those variants into one wiki page. Keys are
# matched case-insensitively against the LLM-produced canonical_name. Add
# a new entry whenever a fresh variant shows up; the table stays small
# because it's stylistic-only, not semantic.
ALIASES: dict[str, str] = {
    "sainsburys":                "Sainsbury's",
    "github":                    "GitHub",
    "disney plus":               "Disney+",
    "costco":                    "Costco Wholesale",
    "shell croxley":             "Shell",
    "creation consumer fin":     "Creation Consumer Finance",
    "e.on next ltd":             "E.ON Next",
    "ee limited":                "EE",
    "google youtube premium":    "YouTube Premium",
    "home insurance":            "Home Insurance LBIS",
    "nationwide bs mortgage":    "Nationwide Mortgage",
    "paypal payin":              "PayPal Transfer",
    "swale heating ltd":         "Swale Heating",
    "tfl":                       "Transport for London",
    "vodafone ltd":              "Vodafone",
    "vodafone ltd device":       "Vodafone Device",
    "wwf":                       "WWF UK",
    # Presidio redacts "SCOTTISH" as an NRP token, so the LLM often
    # emits the survivor without the prefix.
    "widows life":               "Scottish Widows Life",
    "widows pension":            "Scottish Widows Pension",
    "paypal":                    "PayPal Transfer",
}


def _apply_alias(name: str) -> str:
    """Fold a known stylistic variant onto the canonical spelling."""
    stripped = name.strip()
    return ALIASES.get(stripped.lower(), stripped)


# Category lock — for merchants whose categorisation the LLM wobbles on
# across cold runs. ALIASES stabilises the *name*; this stabilises the
# *category*. Keys are the post-alias canonical name; values must be one
# of ``ALLOWED_CATEGORIES``. Only add entries when you've actually seen
# drift in practice — over-locking erases the LLM's main advantage.
CATEGORY_LOCK: dict[str, str] = {
    # Saw all three of Utilities/Subscriptions/Cloud across runs.
    "Virgin Media":              "Utilities",
    # M&S sells everything; clothing dominates so "Shopping" reads truer
    # than "Groceries".
    "Marks & Spencer":           "Shopping",
    # Discount homewares chain → Household reads truer than Shopping.
    "B&M":                       "Household",
    # Meal-kit subscription, lives in the weekly grocery line item.
    "HelloFresh":                "Groceries",
    # Coffee chain inside Watford station — not a sit-down restaurant.
    "Jamaica Blue":              "Coffee",
    # Motorway services; mostly fuel/transport context.
    "Welcome Break":             "Transport",
    # Sub-merchant of PayPal's payment rail rather than a real merchant.
    "PayPal Transfer":           "Transfers",
    # SaaS / streaming buckets that occasionally drift between
    # "Subscriptions" and "Cloud" depending on framing.
    "Apple":                     "Subscriptions",
    "Disney+":                   "Subscriptions",
    "Spotify":                   "Subscriptions",
    "Netflix":                   "Subscriptions",
    "YouTube Premium":           "Subscriptions",
    "OpenAI":                    "Subscriptions",
    "GitHub":                    "Cloud",
    "AWS":                       "Cloud",
}


def _kind_for(category: str) -> str:
    """Re-derive ``kind`` consistently from the (possibly locked) category."""
    if category in _INCOME_CATEGORIES:
        return "income"
    if category in _TRANSFER_CATEGORIES:
        return "transfer"
    return "expense"


def _apply_category_lock(name: str, category: str, kind: str) -> tuple[str, str]:
    """Override the LLM's category when we've locked one for this name."""
    locked = CATEGORY_LOCK.get(name)
    if locked is None or locked == category:
        return category, kind
    return locked, _kind_for(locked)

# Thread-safe in-memory cache mirror (the JSON file is the source of truth).
_lock = threading.Lock()
_cache: dict[str, dict] = {}
_cache_loaded = False


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_cache() -> None:
    global _cache, _cache_loaded
    with _lock:
        if _cache_loaded:
            return
        if CACHE_PATH.exists():
            try:
                _cache = json.loads(CACHE_PATH.read_text())
            except json.JSONDecodeError:
                log.warning("canon cache %s unreadable; starting fresh", CACHE_PATH)
                _cache = {}
        _cache_loaded = True


def _save_cache() -> None:
    with _lock:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_cache, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You normalize messy UK bank-transaction descriptions into a
clean canonical merchant name and one category from this fixed list:
{categories}

Rules:
* Strip "(DD)", "(SO)", "(DEB)", "(BP)", "(TFR)", "(FPO)", "(BGC)" trailing
  type tags, store numbers, "POS DEBIT"/"PAYPAL *"/"TST*" prefixes, trailing
  two-letter state/country codes ("CA", "IRL"), and ``<TOKEN_NN>`` PII
  brackets — those are noise.
* "Tesco" / "Tescos Stores 3372" / "TESCO 6753" → "Tesco" / "Groceries".
* "BALANCE FROM PREVIOUS STATEMENT" → "Statement Carry-over" / "System".
* "DIRECT DEBIT PAYMENT - THANK YOU" → "Card Payment Received" / "System".
* If only ``<URL_NN>`` survives → "Redacted Online Merchant" / "Uncategorized".
* If only ``<PERSON_NN>`` survives → "Redacted Local Merchant" / "Uncategorized".
* Amazon order-code remnants like "*R684K9ZJ4" alone → "Amazon" / "Shopping".
* "CORK IRL" alone is Apple's Irish billing entity → "Apple" / "Subscriptions".
* "SCOTTISH WIDOWS" or just "WIDOWS PENSION/LIFE" → "Scottish Widows Pension"
  or "Scottish Widows Life" / "Pension" or "Insurance".
* For salaries / "BGC" credits → "<EMPLOYER> Salary" / "Income".
* Save-the-Change rounding → "Save the Change" / "Savings".

Output STRICT JSON of the shape:
{{"results": [
  {{"canonical_name": "...", "category": "<one of the list>", "kind": "expense"|"income"|"transfer"}},
  ...
]}}
Preserve the order of the input list. Return one result per input description.
"""


def _llm_classify(descriptions: list[str]) -> list[MerchantInfo]:
    """Send a batch to gpt-5.4-mini. Falls back to regex on any failure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return [_regex_canonicalize(d) for d in descriptions]

    try:
        from openai import OpenAI
    except ImportError:
        return [_regex_canonicalize(d) for d in descriptions]

    base_url = os.getenv("OPENAI_BASE_URL") or None
    client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    user_msg = (
        "Classify each description and return JSON. Inputs (one per line):\n"
        + "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": SYSTEM_PROMPT.format(categories=ALLOWED_CATEGORIES)},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        payload = json.loads(resp.choices[0].message.content)
        items = payload.get("results", [])
        out: list[MerchantInfo] = []
        for i, desc in enumerate(descriptions):
            item = items[i] if i < len(items) else None
            out.append(_coerce(item, desc))
        return out
    except Exception as exc:  # pragma: no cover — fall back rather than crash
        log.warning("LLM canonicalize failed (%s); falling back to regex", exc)
        return [_regex_canonicalize(d) for d in descriptions]


def _coerce(item: dict | None, original: str) -> MerchantInfo:
    """Validate the LLM response. If invalid, fall back to regex for this row."""
    if not isinstance(item, dict):
        return _regex_canonicalize(original)
    name = str(item.get("canonical_name", "")).strip()
    category = str(item.get("category", "")).strip()
    if category not in ALLOWED_CATEGORIES or not name:
        return _regex_canonicalize(original)
    kind = str(item.get("kind", "")).strip().lower()
    if kind not in {"expense", "income", "transfer"}:
        kind = _kind_for(category)
    # Fold stylistic variants onto canonical spellings, then lock the
    # category for any merchant we've previously seen drift on.
    name = _apply_alias(name)
    category, kind = _apply_category_lock(name, category, kind)
    return MerchantInfo(canonical_name=name, category=category, kind=kind)


# ---------------------------------------------------------------------------
# Public API (matches normalize.canonicalize)
# ---------------------------------------------------------------------------

def prime_cache(descriptions: Iterable[str]) -> dict[str, float]:
    """Pre-warm the cache for every unique description in one go.

    Returns a small stats dict: ``{"hits": int, "misses": int, "seconds": float}``.
    Call this before bulk-loading so each row gets a hot cache lookup.
    """
    _load_cache()
    seen: set[str] = set()
    misses: list[str] = []
    for d in descriptions:
        raw = (d or "").strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        if raw not in _cache:
            misses.append(raw)

    started = time.time()
    for i in range(0, len(misses), BATCH_SIZE):
        batch = misses[i:i + BATCH_SIZE]
        results = _llm_classify(batch)
        for raw, info in zip(batch, results):
            _cache[raw] = {
                "canonical_name": info.canonical_name,
                "category": info.category,
                "kind": info.kind,
            }
    if misses:
        _save_cache()
    return {
        "hits": len(seen) - len(misses),
        "misses": len(misses),
        "seconds": round(time.time() - started, 2),
    }


def canonicalize(description: str) -> MerchantInfo:
    """Cache-first canonicalization. On miss, calls the LLM (and caches)."""
    _load_cache()
    raw = (description or "").strip()
    if not raw:
        return _regex_canonicalize(raw)

    cached = _cache.get(raw)
    if cached:
        # Re-apply aliases and category lock at read time so cache entries
        # written before a rule was added still resolve to today's rule set.
        name = _apply_alias(cached["canonical_name"])
        category, kind = _apply_category_lock(
            name, cached["category"], cached["kind"],
        )
        return MerchantInfo(canonical_name=name, category=category, kind=kind)

    # Single-row miss — call the LLM, cache, return.
    info = _llm_classify([raw])[0]
    _cache[raw] = {
        "canonical_name": info.canonical_name,
        "category": info.category,
        "kind": info.kind,
    }
    _save_cache()
    return info
