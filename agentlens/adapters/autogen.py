"""AutoGen (AG2 / pyautogen) adapter.

AutoGen conversable agents support registered reply/hook functions. This adapter
instruments an agent so each generated reply logs an agent span, and provides a
tool wrapper for function-calling tools.

Example::

    from agentlens.adapters.autogen import instrument_agent
    instrument_agent(assistant)
"""
from __future__ import annotations

from typing import Any

from .. import client as _client
from ..events import Span, SpanKind, jsonable
from .common import wrap_tool

instrument_tool = wrap_tool


def instrument_agent(agent: Any) -> Any:
    """Hook an AutoGen agent so its replies log agent spans. Uses
    ``register_hook('process_message_before_send')`` when available, else wraps
    ``generate_reply``."""
    name = getattr(agent, "name", None) or type(agent).__name__

    def _log(content: Any) -> None:
        try:
            ev = Span(project=_client.get_project(), kind=SpanKind.AGENT.value, name=name,
                      attributes={"agent_name": name, "output": jsonable(content)})
            _client.get_client().log_event(ev.finish().model_dump())
        except Exception:
            pass

    if hasattr(agent, "register_hook"):
        def hook(sender=None, message=None, recipient=None, silent=None, **kw):
            _log(message)
            return message
        try:
            agent.register_hook("process_message_before_send", hook)
            return agent
        except Exception:
            pass

    # fallback: wrap generate_reply
    original = getattr(agent, "generate_reply", None)
    if callable(original):
        import functools

        @functools.wraps(original)
        def wrapper(*a, **k):
            reply = original(*a, **k)
            _log(reply)
            return reply
        try:
            agent.generate_reply = wrapper
        except Exception:
            pass
    return agent
