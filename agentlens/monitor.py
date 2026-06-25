"""The one-line entry point.

    from agentlens import monitor
    monitor.start()

``start()`` initializes the local store and auto-patches whatever LLM SDKs are
installed. Manual ``@trace`` / ``log_tool`` / ``log_memory`` and the framework
adapters keep working alongside it. ``stop()`` removes the patches and flushes.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import client as _client
from . import instrument

_started = False


def start(project: str = "default", tracking_uri: Optional[str] = None,
          db_path: Optional[str] = None, frameworks: Optional[List[str]] = None) -> Dict[str, bool]:
    """Initialize AgentLens and auto-patch installed LLM SDKs.

    Parameters mirror ``agentlens.init`` plus ``frameworks`` to restrict which
    LLM SDK patchers run (defaults to all). Returns the patch report
    (``{"openai": True, "anthropic": False, ...}``). Idempotent."""
    global _started
    _client.init(project=project, tracking_uri=tracking_uri, db_path=db_path)
    report = instrument.apply_all(only=frameworks)
    _started = True
    return report


def stop() -> None:
    """Remove all auto-patches and flush buffered spans."""
    global _started
    instrument.revert_all()
    _client.flush()
    _started = False


def is_started() -> bool:
    return _started
