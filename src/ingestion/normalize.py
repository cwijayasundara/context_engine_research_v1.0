"""Merchant canonicalization.

Raw transaction descriptions are messy:
    "COSTCO WHOLESALE #1234 SEATTLE WA"
    "COSTCO.COM"
    "POS DEBIT COSTCO 0034"

We want all three to collapse to a single `Merchant` node with canonical name
`Costco Wholesale`. This module keeps a simple rule table plus a fallback
strategy: strip prefixes, collapse store numbers, drop the `(TYPE)` suffix
the parser leaves behind, title-case.

In production you'd back this with an LLM-categorization subagent that
proposes canonical_name + category for unknown descriptions and writes
decisions back to the graph as a `CategorizationDecision` (an event-clock
trace).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Hand-curated rules. Anything not matched here falls through to heuristics.
# Order matters: the first matching rule wins. Keep more-specific patterns
# above more-general ones (e.g. "JUST EAT" must beat "PAYPAL").
RULES: list[tuple[re.Pattern[str], str, str]] = [
    # ----- Groceries & household ---------------------------------------
    (re.compile(r"\bCOSTCO\b",                 re.I), "Costco Wholesale",      "Groceries"),
    (re.compile(r"\bTESCOS?\b",                re.I), "Tesco",                  "Groceries"),
    (re.compile(r"\bSAINSBURY",                re.I), "Sainsbury's",            "Groceries"),
    (re.compile(r"\bMORRISONS?\b",             re.I), "Morrisons",              "Groceries"),
    (re.compile(r"\bMARKS\s*&?\s*SPENCER\b",   re.I), "Marks & Spencer",        "Groceries"),
    (re.compile(r"\bWHOLE\s*FOODS",            re.I), "Whole Foods",            "Groceries"),
    (re.compile(r"\bB\s*&\s*M\b",              re.I), "B&M",                    "Household"),
    (re.compile(r"\bB\s*&\s*Q\b",              re.I), "B&Q",                    "Household"),
    (re.compile(r"\bBOOTS\b",                  re.I), "Boots",                  "Health"),
    (re.compile(r"\bALLI\s+BHAVAN",            re.I), "Alli Bhavan",            "Dining"),
    (re.compile(r"\bJAMAICA\s+BLUE",           re.I), "Jamaica Blue",           "Dining"),

    # ----- Transport & travel ------------------------------------------
    (re.compile(r"\bUBER\b",                   re.I), "Uber",                   "Transport"),
    (re.compile(r"\bLYFT\b",                   re.I), "Lyft",                   "Transport"),
    (re.compile(r"\bTFL\b",                    re.I), "Transport for London",   "Transport"),
    (re.compile(r"\bWELCOME\s*BREAK",          re.I), "Welcome Break",          "Transport"),
    (re.compile(r"\bSHELL\b",                  re.I), "Shell",                  "Fuel"),

    # ----- Food delivery -----------------------------------------------
    (re.compile(r"JUST\s*EAT|JUSTEAT",         re.I), "Just Eat",               "Dining"),
    (re.compile(r"UBER\s*EATS",                re.I), "Uber Eats",              "Dining"),
    (re.compile(r"\bHELLO\s*FRESH",            re.I), "HelloFresh",             "Dining"),

    # ----- Streaming, subs, dev tools ----------------------------------
    (re.compile(r"\bNETFLIX",                  re.I), "Netflix",                "Subscriptions"),
    (re.compile(r"\bSPOTIFY",                  re.I), "Spotify",                "Subscriptions"),
    (re.compile(r"DISNEY\s*PLUS|DISNEY\+",     re.I), "Disney+",                "Subscriptions"),
    (re.compile(r"YOUTUBE\s*PREMIUM",          re.I), "YouTube Premium",        "Subscriptions"),
    # APPLE patterns: also catch the orphaned "CORK IRL" fragment that
    # Apple's Irish billing entity leaves when Presidio masks "APPLE".
    (re.compile(r"APPLE\.COM|\bAPPLE\b|\bCORK\s+IRL\b", re.I), "Apple",          "Subscriptions"),
    (re.compile(r"\bGOOGLE\b",                 re.I), "Google",                 "Subscriptions"),
    (re.compile(r"\bOPENAI\b|CHATGPT",         re.I), "OpenAI",                 "Subscriptions"),
    (re.compile(r"\bAWS\b|AMAZON\s*WEB",       re.I), "AWS",                    "Cloud"),
    (re.compile(r"GITHUB",                     re.I), "GitHub",                 "Cloud"),
    # AMZN often appears glued ("AMZNMktplace") so don't require a trailing
    # word boundary. R<digits>... is the Amazon order-code prefix that's
    # left behind when the AMAZON/AMZN portion gets stripped by PII filters.
    (re.compile(r"\bAMAZON|\bAMZN|\bR\d{2,}[A-Z0-9]+", re.I), "Amazon",         "Shopping"),

    # ----- Retail clothing ---------------------------------------------
    (re.compile(r"\bUNIQLO\b",                 re.I), "Uniqlo",                 "Clothing"),
    (re.compile(r"FOOT\s*LOCKER",              re.I), "Foot Locker",            "Clothing"),
    (re.compile(r"VISION\s*EXPRESS",           re.I), "Vision Express",         "Health"),

    # ----- Education ----------------------------------------------------
    (re.compile(r"TUTORFUL",                   re.I), "Tutorful",               "Education"),
    (re.compile(r"PARENTPAY",                  re.I), "ParentPay",              "Education"),

    # ----- Utilities & council -----------------------------------------
    # Match either the full "WATFORD BOROUGH …" or the fragment Presidio
    # leaves after redacting "WATFORD" as a LOCATION.
    (re.compile(r"WATFORD\s+BOROUGH|BOROUGH\s+COUNCIL", re.I), "Watford Borough Council","Council Tax"),
    (re.compile(r"\bE\.?ON\b",                 re.I), "E.ON Next",              "Utilities"),
    (re.compile(r"AFFINITY\s*WATER",           re.I), "Affinity Water",         "Utilities"),
    (re.compile(r"VIRGIN\s*MEDIA",             re.I), "Virgin Media",           "Utilities"),
    (re.compile(r"TV\s*LICENCE",               re.I), "TV Licence",             "Utilities"),
    (re.compile(r"SWALE\s*HEATING",            re.I), "Swale Heating",          "Utilities"),

    # ----- Telecoms -----------------------------------------------------
    (re.compile(r"VODAFONE",                   re.I), "Vodafone",               "Mobile"),
    (re.compile(r"\bEE\s*(LTD|LIMITED|MOBILE)", re.I), "EE",                    "Mobile"),

    # ----- Insurance ----------------------------------------------------
    (re.compile(r"ADMIRAL.*MOTOR",             re.I), "Admiral Motor Insurance","Insurance"),
    (re.compile(r"ADMIRAL.*HOME",              re.I), "Admiral Home Insurance", "Insurance"),
    (re.compile(r"\bADMIRAL\b",                re.I), "Admiral",                "Insurance"),
    (re.compile(r"HOME\s*INSURANCE",           re.I), "Home Insurance LBIS",    "Insurance"),
    # "SCOTTISH" often gets redacted as NRP, leaving just "WIDOWS LIFE/PENSION".
    (re.compile(r"WIDOWS.*LIFE",               re.I), "Scottish Widows Life",   "Insurance"),
    (re.compile(r"WIDOWS.*PENSION|WIDOWS\b",   re.I), "Scottish Widows Pension","Pension"),
    (re.compile(r"CREATION\s*CONSUMER",        re.I), "Creation Consumer Finance","Finance"),

    # ----- Mortgage, savings, charity ----------------------------------
    (re.compile(r"NATIONWIDE.*MORTGAGE|MORTGAGE", re.I), "Nationwide Mortgage", "Mortgage"),
    (re.compile(r"HALIFAX.*CREDIT\s*CARD|HALIFAX\s*CC", re.I), "Halifax Credit Card","Credit Card"),
    (re.compile(r"\bHALIFAX\b",                re.I), "Halifax",                "Banking"),
    (re.compile(r"PEPPER\s*STARK",             re.I), "Pepper Stark Savings",   "Savings"),
    (re.compile(r"SAVETHECHANGE",              re.I), "Save the Change",        "Savings"),
    (re.compile(r"\bWWF\b",                    re.I), "WWF UK",                 "Charity"),

    # ----- Income -------------------------------------------------------
    (re.compile(r"STARK\s*INDUSTRIES",         re.I), "Stark Industries Salary","Income"),
    (re.compile(r"\bDIRECT DEPOSIT",           re.I), "Direct Deposit",         "Income"),
    (re.compile(r"INTEREST\s*\(GROSS",         re.I), "Interest",               "Income"),
    (re.compile(r"\bSALARY\b",                 re.I), "Salary",                 "Income"),

    # ----- Coffee (kept from defaults) ---------------------------------
    (re.compile(r"\bSTARBUCKS",                re.I), "Starbucks",              "Coffee"),

    # ----- Statement housekeeping (filtered out of wiki) ---------------
    (re.compile(r"BALANCE\s+FROM\s+PREVIOUS",  re.I), "Statement Carry-over",   "System"),
    (re.compile(r"DIRECT\s+DEBIT\s+PAYMENT\s*-\s*THANK\s+YOU", re.I), "Card Payment Received", "System"),

    # ----- Fully-redacted catch-alls (must be LAST) --------------------
    # When the PII filter erased everything meaningful, group the survivors
    # so they don't fragment the wiki. Order matters: specific tokens first,
    # bare-empty last.
    (re.compile(r"<URL_\d+>", re.I),                   "Redacted Online Merchant","Uncategorized"),
    (re.compile(r"<PERSON_\d+>", re.I),                "Redacted Local Merchant", "Uncategorized"),
    (re.compile(r"^\s*$", re.I),                       "Redacted Merchant",       "Uncategorized"),

    # ----- Payment rails (low-priority — match last so brand rules win) -
    (re.compile(r"PAYPAL\s*\*?PYPL",           re.I), "PayPal Transfer",        "Transfers"),
]


@dataclass(frozen=True)
class MerchantInfo:
    canonical_name: str
    category: str
    kind: str = "expense"   # "income" | "expense" | "transfer"


_INCOME_CATEGORIES = {"Income"}
_TRANSFER_CATEGORIES = {"Transfers", "Savings"}


# Common stripping patterns used in the fallback path -----------------------
_PREFIX = re.compile(r"^(POS\s+DEBIT|TST\*|SQ\s+\*|PAYPAL\s+\*)\s+", re.I)
_STORE_NUM = re.compile(r"#\s*\d+\b")
_TRAILING_STATE = re.compile(r"\s+[A-Z]{2}\s*$")            # "...  CA"
# The parser appends a "(DD)" / "(SO)" / "(DEB)" / etc. type tag to the
# description so the type survives the markdown→JSONL roundtrip. We don't
# want that in the canonical merchant name.
_TYPE_SUFFIX = re.compile(
    r"\s*\((?:DD|SO|DEB|BP|TFR|FPO|FPI|BGC|CHG|CHQ|PAY|MPI|MPO|DEP|FEE|CPT|COR)\)\s*$",
    re.I,
)
# Bracketed PII tokens the filter inserts (e.g. "TUTORFUL <ADDRESS_02>") —
# strip so they don't fragment merchant pages.
_PII_TOKEN = re.compile(r"\s*<[A-Z]+_\d+>\s*")
_MULTISPACE = re.compile(r"\s+")


def _strip_for_canonical(text: str) -> str:
    cleaned = _TYPE_SUFFIX.sub("", text)
    cleaned = _PII_TOKEN.sub(" ", cleaned)
    cleaned = _PREFIX.sub("", cleaned)
    cleaned = _STORE_NUM.sub("", cleaned)
    cleaned = _TRAILING_STATE.sub("", cleaned)
    return _MULTISPACE.sub(" ", cleaned).strip()


def _regex_canonicalize(description: str) -> MerchantInfo:
    """Rule-based canonicalization. Always returns *something*."""
    raw = description.strip()
    # The pre-stripped form (no PII tokens, no type suffix) is what the
    # main brand rules match against. We run rules against BOTH so the
    # final catch-alls (which look for `<PERSON_NN>` / `<URL_NN>` tokens)
    # can still see the original markers before the stripper eats them.
    text = _strip_for_canonical(raw)
    for pattern, canonical, category in RULES:
        if pattern.search(text) or pattern.search(raw):
            if category in _INCOME_CATEGORIES:
                kind = "income"
            elif category in _TRANSFER_CATEGORIES:
                kind = "transfer"
            else:
                kind = "expense"
            return MerchantInfo(canonical, category, kind)

    # Fallback: title-case whatever's left after the strip pass.
    cleaned = text.title() or "Unknown"
    return MerchantInfo(
        canonical_name=cleaned,
        category="Uncategorized",
        kind="expense",
    )


def canonicalize(description: str) -> MerchantInfo:
    """Dispatcher.

    Honours the ``NORMALIZER`` env var:
      * ``regex`` (default) — fast, deterministic, ~50 hand-written rules.
      * ``llm``             — cached gpt-5.4-mini classification with a
                              regex fallback when the API key is missing.

    Callers don't need to know which backend is in use; the
    ``MerchantInfo`` shape is identical either way.
    """
    backend = os.getenv("NORMALIZER", "regex").lower()
    if backend == "llm":
        # Import lazily so the regex path stays zero-dep.
        from .llm_normalize import canonicalize as _llm
        return _llm(description)
    return _regex_canonicalize(description)
