"""Extract a (name, country) Location from a transaction description tail.

Statements don't ship structured merchant addresses, but most descriptions end
in a town / state / country token (e.g. ``SAINSBURY'S S/MKT WATFORD``,
``GITHUB, INC. SAN FRANCISCO CA``, ``APPLE.COM/BILL CORK IRL``). This module
recovers a coarse Location signal from those suffixes — used by the
geo-mismatch rule and Louvain projection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    country: str  # ISO 3166-1 alpha-2


# Explicit country tokens we recognise at the very end of the description.
_COUNTRY_SUFFIX = {
    "FR": "FR", "DE": "DE", "ES": "ES", "IT": "IT", "NL": "NL",
    "IRL": "IE", "IE": "IE", "TH": "TH", "JP": "JP", "CN": "CN",
    "LUX": "LU", "LU": "LU", "BR": "BR", "AU": "AU",
}

# US state codes — when one is the last token we treat it as US.
_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}

# Common ALL-CAPS prefix tokens that participate in multi-word US city names
# (e.g. "SAN FRANCISCO", "NEW YORK", "LOS ANGELES", "ST LOUIS"). When we see
# one of these immediately before the city token, we include it in the name.
_CITY_PREFIXES = {
    "SAN", "NEW", "LOS", "LAS", "ST", "ST.", "SANTA", "FORT", "FT",
    "FT.", "PORT", "WEST", "EAST", "NORTH", "SOUTH", "LAKE", "MOUNT",
    "MT", "MT.",
}

# Strip common merchant-tail noise so the *location* tokens are isolated.
# Includes corporate suffixes and common statement/bookkeeping words that
# would otherwise be mistaken for a town name by the GB fallback.
_NOISE_TOKENS = {
    "INC", "CORP", "LTD", "PLC", "LLC", "LIMITED", "COMPANY",
    "CO", "INC.", "CO.", "BILL", "BILL.", "EU", "EU.",
    "STATEMENT", "BALANCE", "PREVIOUS", "FROM", "PAYMENT",
    "THANK", "YOU",
}

_TOKEN_RE = re.compile(r"[A-Z][A-Z\.\-']+")


def _strip_noise(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.rstrip(".") not in _NOISE_TOKENS]


def parse_location(description: str) -> Location | None:
    """Return a `Location` for the trailing geo tokens, or `None`.

    Strategy:
      1. Take the last 1–3 ALL-CAPS tokens.
      2. If the very last token is an explicit country code → use it.
      3. Else if the last token is a 2-letter US state → country = US.
      4. Else if the second-to-last token looks like a UK town → country = GB.
      5. Otherwise return None.
    """
    tokens = _TOKEN_RE.findall(description.upper())
    tokens = _strip_noise(tokens)
    if not tokens:
        return None

    last = tokens[-1].rstrip(".")
    # Case 1: explicit country
    if last in _COUNTRY_SUFFIX:
        country = _COUNTRY_SUFFIX[last]
        prefix = tokens[-2] if len(tokens) >= 2 else None
        if prefix and prefix not in _COUNTRY_SUFFIX:
            name = f"{prefix.title()} {last}"
            sid = f"{prefix.lower()}-{last.lower()}"
        else:
            name = last
            sid = last.lower()
        return Location(id=sid, name=name, country=country)

    # Case 2: US state code
    if last in _US_STATES and len(tokens) >= 2:
        city = tokens[-2]
        # Handle multi-word cities like "SAN FRANCISCO", "NEW YORK".
        if (
            len(tokens) >= 3
            and tokens[-3].rstrip(".") in _CITY_PREFIXES
        ):
            prefix = tokens[-3].rstrip(".")
            city_full = f"{prefix} {city}"
            sid = f"{prefix.lower()}-{city.lower()}-{last.lower()}"
            name = f"{city_full.title()} {last}"
        else:
            sid = f"{city.lower()}-{last.lower()}"
            name = f"{city.title()} {last}"
        return Location(id=sid, name=name, country="US")

    # Case 3: UK-style trailing ALL-CAPS token, defaulting to GB.
    if last.isalpha() and 3 <= len(last) <= 16:
        return Location(id=last.lower(), name=last.title(), country="GB")

    return None
