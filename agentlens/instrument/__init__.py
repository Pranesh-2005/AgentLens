"""Auto-instrumentation registry.

``apply_all()`` walks every patcher and applies the ones whose SDK is installed.
Each patcher is import-time safe and returns True only when it actually patched
something, so ``monitor.start()`` can report what it hooked. ``revert_all()``
undoes every patch (used by ``monitor.stop()``).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import anthropic, gemini, groq, litellm, openai
from ._patch import revert_all

# id -> patcher module exposing apply() -> bool
PATCHERS = {
    "openai": openai,
    "anthropic": anthropic,
    "gemini": gemini,
    "groq": groq,
    "litellm": litellm,
}


def apply_all(only: Optional[List[str]] = None) -> Dict[str, bool]:
    """Apply every (or a selected subset of) LLM SDK patcher. Returns a map of
    id -> whether it hooked anything."""
    targets = only or list(PATCHERS)
    result: Dict[str, bool] = {}
    for name in targets:
        mod = PATCHERS.get(name)
        if mod is None:
            continue
        try:
            result[name] = mod.apply()
        except Exception:
            result[name] = False
    return result


__all__ = ["apply_all", "revert_all", "PATCHERS"]
