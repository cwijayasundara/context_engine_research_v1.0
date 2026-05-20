"""Harness profile helpers for Gemini Flash Deep Agents runs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiFlashHarnessProfile:
    """Local description of the intended Deep Agents harness profile."""

    model: str = "google_genai:gemini-3.5-flash"
    prefer_structured_tools: bool = True
    use_ptc: bool = True
    max_tool_result_chars: int = 4000
    final_answer_style: str = "concise-grounded"


def register_gemini_flash_profile() -> GeminiFlashHarnessProfile:
    """Register a Gemini profile when Deep Agents exposes profile hooks.

    Deep Agents 0.6 profile APIs may not be installed in local test runs. This
    function is intentionally best-effort and returns the local profile object
    either way.
    """
    profile = GeminiFlashHarnessProfile()
    try:
        from deepagents.profiles import HarnessProfile, register_profile  # type: ignore
    except Exception:
        return profile

    try:
        register_profile(
            profile.model,
            HarnessProfile(
                model=profile.model,
                prefer_structured_tools=profile.prefer_structured_tools,
                use_ptc=profile.use_ptc,
                max_tool_result_chars=profile.max_tool_result_chars,
            ),
        )
    except Exception:
        # Profile registration is an optimization; builder fallback still uses
        # explicit model/tools/middleware settings.
        return profile
    return profile
