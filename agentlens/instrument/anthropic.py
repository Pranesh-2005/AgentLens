"""Auto-patch the Anthropic Python SDK so every ``messages.create`` emits an
``llm`` span with tokens + cost. Patches both sync and async clients."""
from __future__ import annotations

from ._patch import instrument_llm


def _parse(resp) -> dict:
    usage = getattr(resp, "usage", None)
    text = None
    try:
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    except Exception:
        pass
    return {
        "model": getattr(resp, "model", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "response": text,
    }


def apply() -> bool:
    try:
        from anthropic.resources import messages as m
    except ImportError:
        return False
    done = False
    done |= instrument_llm(m.Messages, "create", _parse, "anthropic")
    if hasattr(m, "AsyncMessages"):
        done |= instrument_llm(m.AsyncMessages, "create", _parse, "anthropic")
    return done
