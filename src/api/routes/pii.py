"""PII surface — preview the persona that lives behind the vault tokens.

The vault itself is a hash-keyed dict in ``src/pii/vault.py``; for the UI's
"what does the LLM see" toggle we only need a stable preview that pairs
each persona field with its tokenized form. The mapping is the same one
the parser uses at ingest time, so this stays a thin read.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.config import SETTINGS
from src.pii.vault import get_default_vault

router = APIRouter(prefix="/pii", tags=["pii"])


# Persona fields the UI needs to mask in "LLM view". This matches the
# synthetic persona in src/data_gen/generator.py; in a real deployment
# you'd source these from a profile record.
_PERSONA_FIELDS: list[tuple[str, str]] = [
    ("PERSON",   "Mr Tony Stark"),
    ("ADDRESS",  "14 Avengers Close"),
    ("ADDRESS",  "Watford"),
    ("ADDRESS",  "Hertfordshire"),
    ("POSTCODE", "WD99 9ZZ"),
    ("SORTCODE", "11-22-33"),
    ("ACCT",     "12345678"),
    ("CARD",     "5286 83** **** 1588"),
    ("PHONE",    "01632 960000"),
    ("PHONE",    "01632 960123"),
    ("NRP",      "STARK INDUSTRIES"),
]


@router.get("/preview")
def preview() -> dict:
    """Return ``{real → token}`` pairs the UI can swap into rendered text.

    Tokenization is deterministic — same input + same salt always yields
    the same token — so calling this once on boot is enough.
    """
    vault = get_default_vault(salt=SETTINGS.pii_token_salt)
    pairs: list[dict] = []
    seen: set[str] = set()
    for kind, value in _PERSONA_FIELDS:
        if value in seen:
            continue
        seen.add(value)
        token = vault.tokenize_value(kind, value)
        pairs.append({"real": value, "token": token, "kind": kind})
    return {"pairs": pairs}
