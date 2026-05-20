"""In-memory token ↔ value vault.

The vault is the only place the *real* PII values exist after the
filter has run. It is process-local and never serialized — restart the
process and you lose the mapping. That is intentional: we don't want
PII durable anywhere other than the original source files.

Tokens are deterministic *within a session* — the same `John Smith`
maps to the same `<PERSON_03>` for the life of the process so the
agent's reasoning stays internally consistent, but tokens differ
across sessions so they can't be correlated across runs.
"""
from __future__ import annotations

import hashlib
import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TokenizedText:
    """The output of `PIIFilter.tokenize(...)`.

    `text` contains only opaque tokens like `<ACCT_01>`. The vault knows
    how to swap them back via `detokenize`.
    """

    text: str
    tokens: dict[str, str] = field(default_factory=dict)  # token → original


class PIIVault:
    """Thread-safe, process-local PII vault.

    Token format: ``<TYPE_NN>`` where ``TYPE`` is e.g. ``PERSON``, ``ACCT``,
    ``EMAIL``, ``ADDRESS`` and ``NN`` is a session-local counter. The
    counter resets per type.
    """

    def __init__(self, salt: str = "") -> None:
        self._salt = salt
        self._lock = threading.RLock()
        self._by_value: dict[tuple[str, str], str] = {}     # (type, value) → token
        self._by_token: dict[str, str] = {}                  # token → value
        self._counters: dict[str, int] = defaultdict(int)    # type → next index

    # -- mutators -----------------------------------------------------

    def tokenize_value(self, entity_type: str, value: str) -> str:
        """Return the stable token for ``value`` of ``entity_type``.

        If the value has already been seen this session, returns the
        existing token. Otherwise mints a new one.
        """
        value = value.strip()
        if not value:
            return value
        key = (entity_type, self._fingerprint(value))
        with self._lock:
            if key in self._by_value:
                return self._by_value[key]
            self._counters[entity_type] += 1
            token = f"<{entity_type}_{self._counters[entity_type]:02d}>"
            self._by_value[key] = token
            self._by_token[token] = value
            return token

    # -- readers ------------------------------------------------------

    def detokenize(self, text: str) -> str:
        """Restore real values in ``text``. Any token without a vault
        entry is left as-is (never invent values)."""
        if not text:
            return text
        # Sort by length desc so e.g. <ACCT_10> wins over <ACCT_1>.
        with self._lock:
            for token in sorted(self._by_token, key=len, reverse=True):
                if token in text:
                    text = text.replace(token, self._by_token[token])
        return text

    def lookup(self, token: str) -> str | None:
        with self._lock:
            return self._by_token.get(token)

    def size(self) -> int:
        return len(self._by_token)

    def items(self) -> list[tuple[str, str]]:
        """Snapshot of every ``(token, original_value)`` pair currently
        in the vault. Returns a list (not a view) so callers can iterate
        without holding the lock."""
        with self._lock:
            return list(self._by_token.items())

    # -- helpers ------------------------------------------------------

    def _fingerprint(self, value: str) -> str:
        """Hash so the in-memory map key isn't the raw PII string."""
        h = hashlib.blake2b(digest_size=16)
        h.update(self._salt.encode())
        h.update(b"\x00")
        h.update(value.encode())
        return h.hexdigest()


# Module-level default vault. Tests can construct their own.
_default_vault: PIIVault | None = None


def get_default_vault(salt: str = "") -> PIIVault:
    global _default_vault
    if _default_vault is None:
        _default_vault = PIIVault(salt=salt)
    return _default_vault
