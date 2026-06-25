"""PydanticAI adapter.

Wraps a PydanticAI ``Agent`` so each run logs an agent span (with token usage
when the result exposes ``usage()``). Tools registered on the agent can be
wrapped with the framework-agnostic ``wrap_tool``.

Example::

    from agentlens.adapters.pydanticai import instrument_agent
    agent = instrument_agent(agent)
"""
from __future__ import annotations

import functools
import time
from typing import Any

from ..events import SpanKind, jsonable
from ..tracing import _log
from .common import wrap_tool  # re-exported for convenience

instrument_tool = wrap_tool


def _log_run(agent: Any, result: Any, t0: float, status: str = "ok") -> None:
    try:
        usage = None
        getu = getattr(result, "usage", None)
        if callable(getu):
            try:
                usage = getu()
            except Exception:
                usage = None
        output = getattr(result, "output", None) or getattr(result, "data", None) or result
        # _log nests under the active trace (contextvar), so wrapping the run in
        # agentlens.trace_session(...) keeps the agent span(s) in a single run.
        _log(
            SpanKind.AGENT.value,
            getattr(agent, "name", None) or "pydanticai.agent",
            {
                "agent_name": getattr(agent, "name", None),
                "model": jsonable(getattr(agent, "model", None)),
                "output": jsonable(output),
                "input_tokens": getattr(usage, "request_tokens", None),
                "output_tokens": getattr(usage, "response_tokens", None),
            },
            status,
            (time.time() - t0) * 1000.0,
        )
    except Exception:
        pass


def instrument_agent(agent: Any) -> Any:
    """Wrap ``run`` / ``run_sync`` so each run logs an agent span. Idempotent."""
    for attr in ("run_sync", "run"):
        original = getattr(agent, attr, None)
        if not callable(original) or getattr(original, "__agentlens__", False):
            continue
        if attr == "run":  # async
            @functools.wraps(original)
            async def awrapper(*a, __orig=original, **k):
                t0 = time.time()
                try:
                    res = await __orig(*a, **k)
                except Exception:
                    _log_run(agent, None, t0, "error")
                    raise
                _log_run(agent, res, t0)
                return res
            awrapper.__agentlens__ = True
            try:
                agent.run = awrapper
            except Exception:
                pass
        else:
            @functools.wraps(original)
            def wrapper(*a, __orig=original, **k):
                t0 = time.time()
                try:
                    res = __orig(*a, **k)
                except Exception:
                    _log_run(agent, None, t0, "error")
                    raise
                _log_run(agent, res, t0)
                return res
            wrapper.__agentlens__ = True
            try:
                agent.run_sync = wrapper
            except Exception:
                pass
    return agent
