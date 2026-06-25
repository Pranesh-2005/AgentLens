"""Strands Agents adapter.

Strands agents accept a ``callback_handler`` that receives streaming events. This
adapter supplies one that logs LLM/tool/agent activity, plus the framework-
agnostic ``wrap_tool`` for Strands tools.

Example::

    from agentlens.adapters.strands import callback_handler
    agent = Agent(..., callback_handler=callback_handler)
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..tracing import log_tool
from .common import wrap_tool

instrument_tool = wrap_tool

# Strands streams ``current_tool_use`` on *every* token delta while it builds the
# tool call, so the same call arrives dozens of times with *partial* input — the
# first sighting has ``input=""`` and the result hasn't run yet. We therefore
# buffer each tool use by ``toolUseId``, keep the most-complete input seen, and
# emit the span only once its result message arrives (or at run end as a
# fallback), so the Tool Explorer gets real input *and* output instead of blanks.
_pending: Dict[str, Dict[str, Any]] = {}
_logged: set[str] = set()


def _tool_results(msg: Any) -> List[Dict[str, Any]]:
    """Pull ``toolResult`` blocks out of a completed Strands message dict."""
    out: List[Dict[str, Any]] = []
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and isinstance(c.get("toolResult"), dict):
                out.append(c["toolResult"])
    return out


def callback_handler(**kwargs: Any) -> None:
    """Strands callback. Buffers streaming tool input, then logs one tool span per
    tool use — with real input + output — nested under whatever trace is active
    (wrap the agent call in ``agentlens.trace_agent(...)`` so the spans land in a
    single run instead of separate top-level rows)."""
    try:
        # 1. accumulate the streamed (partial) tool call; keep the longest input.
        tu = kwargs.get("current_tool_use")
        if isinstance(tu, dict) and tu.get("name"):
            tid = tu.get("toolUseId") or tu.get("name")
            cur = _pending.setdefault(tid, {"name": tu["name"], "input": None})
            inp = tu.get("input")
            if isinstance(inp, str):
                if cur["input"] is None or len(inp) > len(cur["input"] or ""):
                    cur["input"] = inp
            elif inp not in (None, "") and cur["input"] is None:
                cur["input"] = inp

        # 2. when the result message lands, emit the span (input + output) once.
        for block in _tool_results(kwargs.get("message")):
            tid = block.get("toolUseId")
            if not tid or tid in _logged:
                continue
            _logged.add(tid)
            pend = _pending.pop(tid, {})
            log_tool(pend.get("name") or str(tid), args=pend.get("input"),
                     result=block.get("content"))

        # 3. fallback: at run end, flush any tool use that never produced a result
        # message so its input isn't lost.
        if kwargs.get("result") is not None or kwargs.get("complete"):
            for tid, pend in list(_pending.items()):
                if tid in _logged:
                    _pending.pop(tid, None)
                    continue
                _logged.add(tid)
                _pending.pop(tid, None)
                log_tool(pend.get("name") or str(tid), args=pend.get("input"))
    except Exception:
        pass


def reset() -> None:
    """Clear the per-process tool-use buffers (call between independent runs)."""
    _seen_tool_uses.clear()
    _pending.clear()
    _logged.clear()


# Back-compat alias: older callers referenced the dedup set directly.
_seen_tool_uses = _logged
