"""FastMCP adapter.

FastMCP keeps registered tools in a tool manager; this adapter wraps each tool's
underlying function so calls log tool spans. Call ``instrument_server(mcp)``
after your tools are registered.

Example::

    from fastmcp import FastMCP
    from agentlens.adapters.fastmcp import instrument_server

    mcp = FastMCP("demo")

    @mcp.tool()
    def add(a: int, b: int) -> int: return a + b

    instrument_server(mcp)
"""
from __future__ import annotations

from typing import Any

from .anthropic_agents import traced_async_tool
from .common import traced_tool, wrap_tool

instrument_tool = wrap_tool


def instrument_server(mcp: Any) -> Any:
    """Wrap each registered FastMCP tool's function with a tool-span logger."""
    manager = getattr(mcp, "_tool_manager", None) or getattr(mcp, "tool_manager", None)
    tools = None
    if manager is not None:
        tools = getattr(manager, "_tools", None) or getattr(manager, "tools", None)
    tools = tools if tools is not None else getattr(mcp, "_tools", None)

    items = []
    if isinstance(tools, dict):
        items = list(tools.items())
    elif isinstance(tools, (list, tuple)):
        items = [(getattr(t, "name", i), t) for i, t in enumerate(tools)]

    for key, tool in items:
        for attr in ("fn", "func", "handler", "callback"):
            fn = getattr(tool, attr, None)
            if callable(fn) and not getattr(fn, "__agentlens_patched__", False):
                name = getattr(tool, "name", None) or str(key)
                try:
                    setattr(tool, attr, wrap_tool(fn, name))
                except Exception:
                    pass
                break
    return mcp


__all__ = ["traced_tool", "traced_async_tool", "wrap_tool", "instrument_tool", "instrument_server"]
