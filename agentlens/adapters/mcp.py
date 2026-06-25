"""MCP (Model Context Protocol) SDK adapter.

Wraps the tool handlers registered on a low-level ``mcp.server.Server`` (or any
object exposing a tool registry) so each tool dispatch logs a tool span. Use
``traced_tool`` to decorate individual MCP tool functions you control.

Example::

    from agentlens.adapters.mcp import traced_tool

    @server.call_tool()
    @traced_tool()
    async def handle(name, arguments): ...
"""
from __future__ import annotations

from typing import Any

from .anthropic_agents import traced_async_tool
from .common import traced_tool, wrap_tool

instrument_tool = wrap_tool


def instrument_server(server: Any) -> Any:
    """Best-effort: wrap any callables found in a server's tool registry so their
    invocation logs a tool span. Tolerant of differing MCP server internals."""
    for attr in ("_tool_handlers", "_tools", "tools"):
        registry = getattr(server, attr, None)
        if isinstance(registry, dict) and registry:
            for key, entry in list(registry.items()):
                fn = getattr(entry, "fn", None) or getattr(entry, "handler", None) or entry
                if callable(fn):
                    name = getattr(entry, "name", None) or str(key)
                    wrapped = wrap_tool(fn, name)
                    if hasattr(entry, "fn"):
                        try:
                            entry.fn = wrapped
                        except Exception:
                            pass
                    else:
                        registry[key] = wrapped
            break
    return server


__all__ = ["traced_tool", "traced_async_tool", "wrap_tool", "instrument_tool", "instrument_server"]
