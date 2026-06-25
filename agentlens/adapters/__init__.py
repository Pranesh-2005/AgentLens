"""Framework adapters.

Auto-patching (``monitor.start()``) covers the raw LLM SDKs. Agent frameworks
expose their own callback/hook systems that carry far richer structure (agent
boundaries, tool calls, decisions), so each framework gets a dedicated adapter
here. Every adapter is import-time safe: importing this package never requires
any framework to be installed.

Common, framework-agnostic helpers live in ``common`` (``wrap_tool``,
``traced_tool``) and work everywhere.
"""
from __future__ import annotations

from .common import traced_tool, wrap_tool

__all__ = ["traced_tool", "wrap_tool"]
