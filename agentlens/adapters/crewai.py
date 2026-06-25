"""CrewAI adapter.

CrewAI agents/crews accept ``step_callback`` and ``task_callback`` hooks, and
their tools are plain callables. This adapter provides:

  * ``step_callback`` / ``task_callback`` — pass to ``Agent(...)`` / ``Crew(...)``
    to log agent-step and task spans,
  * ``instrument_tools(tools)`` — wrap a list of tools so each invocation logs a
    tool span (uses the framework-agnostic ``wrap_tool``).

Example::

    from agentlens.adapters.crewai import step_callback, instrument_tools
    agent = Agent(..., tools=instrument_tools(tools), step_callback=step_callback)
"""
from __future__ import annotations

from typing import Any, List

from ..events import SpanKind, jsonable
from ..tracing import _log
from .common import wrap_tool


def step_callback(step: Any) -> None:
    """Log a CrewAI agent step as an agent span (best-effort field extraction).

    Uses the contextvar-aware logger so the span nests under whatever trace is
    active — wrap the crew kickoff in ``agentlens.trace_workflow(...)`` to keep a
    crew's steps in one run instead of scattering them into top-level rows."""
    try:
        _log(SpanKind.AGENT.value, "crewai.step",
             {"agent_name": getattr(step, "agent", None) and str(getattr(step, "agent")),
              "output": jsonable(getattr(step, "output", None) or step)})
    except Exception:
        pass


def task_callback(task_output: Any) -> None:
    """Log a CrewAI task result as a workflow span (nested under the active trace)."""
    try:
        _log(SpanKind.WORKFLOW.value, "crewai.task",
             {"output": jsonable(getattr(task_output, "raw", None) or task_output)})
    except Exception:
        pass


def instrument_tools(tools: List[Any]) -> List[Any]:
    """Wrap each tool's underlying callable so calls log tool spans. Works for
    CrewAI ``BaseTool`` instances (wraps ``_run``/``run``) and plain functions."""
    out = []
    for t in tools or []:
        for attr in ("_run", "run", "func"):
            fn = getattr(t, attr, None)
            if callable(fn):
                name = getattr(t, "name", None) or getattr(fn, "__name__", "tool")
                try:
                    setattr(t, attr, wrap_tool(fn, name))
                except Exception:
                    pass
                break
        else:
            if callable(t):
                t = wrap_tool(t)
        out.append(t)
    return out
