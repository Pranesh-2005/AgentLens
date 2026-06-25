"""Diagnostics for the framework adapters and auto-patchers.

Adapters hook into framework internals (callback signatures, method names) that
move between versions. When they drift the failure is silent — a span just stops
being captured. These helpers turn that silence into a visible
``AgentLensWarning`` so version drift is noticed instead of producing empty
dashboards.
"""
from __future__ import annotations

import warnings
from typing import Iterable


class AgentLensWarning(UserWarning):
    """Emitted when an adapter/patcher can't hook something it expected to."""


def warn(message: str) -> None:
    warnings.warn(f"[agentlens] {message}", AgentLensWarning, stacklevel=3)


def require_methods(obj: object, methods: Iterable[str], what: str) -> None:
    """Warn if ``obj`` is missing every one of ``methods`` (so the wrapper would
    silently capture nothing). ``methods`` is treated as "at least one must
    exist"."""
    present = [m for m in methods if callable(getattr(obj, m, None))]
    if not present:
        warn(
            f"{what}: {type(obj).__name__} has none of {list(methods)} — "
            f"that will not be captured (framework version drift?)"
        )
