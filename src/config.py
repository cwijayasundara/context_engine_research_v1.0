"""Process-wide settings loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    # Overrides the default api.openai.com base. Point this at any
    # OpenAI-compatible endpoint — e.g. Google's Gemini OpenAI-compat layer:
    #   OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    #   OPENAI_API_KEY=<your gemini api key>
    #   OPENAI_MODEL=gemini-3.5-flash
    openai_base_url: str | None
    model: str

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    statements_dir: Path
    wiki_dir: Path
    ontology_path: Path

    pii_token_salt: str
    langsmith_api_key: str | None
    langsmith_project: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=_required("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("OPENAI_MODEL", "openai:gpt-5.4-mini"),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=_required("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            statements_dir=ROOT / os.getenv("STATEMENTS_DIR", "data/statements"),
            wiki_dir=ROOT / os.getenv("WIKI_DIR", "data/wiki"),
            ontology_path=ROOT / os.getenv("ONTOLOGY_PATH", "data/ontology/finance.yaml"),
            pii_token_salt=os.getenv("PII_TOKEN_SALT", "dev-only-do-not-use-in-prod"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY") or None,
            langsmith_project=os.getenv("LANGSMITH_PROJECT") or None,
        )


def _required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


SETTINGS = Settings.from_env() if os.getenv("FINCTX_LAZY") != "1" else None  # type: ignore[assignment]
