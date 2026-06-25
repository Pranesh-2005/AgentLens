"""Framework-agnostic instrumentation helpers.

These work with any agent framework (or none): wrap a plain callable as a tool
so each invocation logs a ``tool`` span — including its arguments, result,
duration, and any exception (which automatically feeds the Failure Explorer).
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional


def wrap_tool(fn: Callable, name: Optional[str] = None) -> Callable:
    """Return a wrapped callable that logs a ``tool`` span per call. Re-raises
    the original exception after recording it, so behavior is unchanged."""
    tool_name = name or getattr(fn, "__name__", None) or "tool"

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from ..tracing import log_tool

        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            log_tool(tool_name, args={"args": args, "kwargs": kwargs}, error=e,
                     duration_ms=(time.time() - t0) * 1000.0)
            raise
        log_tool(tool_name, args={"args": args, "kwargs": kwargs}, result=result,
                 duration_ms=(time.time() - t0) * 1000.0)
        return result

    return wrapper


def traced_tool(name: Optional[str] = None):
    """Decorator form of :func:`wrap_tool`.

        @traced_tool()
        def web_search(q): ...
    """
    def deco(fn: Callable) -> Callable:
        return wrap_tool(fn, name)

    return deco
