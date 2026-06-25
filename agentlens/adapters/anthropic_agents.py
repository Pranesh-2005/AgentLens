"""Anthropic Agent SDK (``claude-agent-sdk``) adapter.

The Claude Agent SDK runs tools you define and streams messages. The most
reliable, version-stable instrumentation point is the tools themselves: wrap
each tool callable so its invocation logs a tool span (args, result, errors).
Raw Claude API calls the SDK makes are additionally captured by
``monitor.start()`` auto-patching the underlying ``anthropic`` client.

Example::

    from agentlens.adapters.anthropic_agents import traced_tool

    @traced_tool("get_weather")
    async def get_weather(args): ...
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional

from .common import traced_tool, wrap_tool

instrument_tool = wrap_tool


def traced_async_tool(name: Optional[str] = None):
    """Async variant of :func:`traced_tool` for ``async def`` tool handlers."""
    def deco(fn: Callable) -> Callable:
        tool_name = name or getattr(fn, "__name__", "tool")

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            from ..tracing import log_tool

            t0 = time.time()
            try:
                result = await fn(*args, **kwargs)
            except Exception as e:
                log_tool(tool_name, args={"args": args, "kwargs": kwargs}, error=e,
                         duration_ms=(time.time() - t0) * 1000.0)
                raise
            log_tool(tool_name, args={"args": args, "kwargs": kwargs}, result=result,
                     duration_ms=(time.time() - t0) * 1000.0)
            return result

        return wrapper

    return deco


__all__ = ["traced_tool", "traced_async_tool", "wrap_tool", "instrument_tool"]
