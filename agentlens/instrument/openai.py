"""Auto-patch the OpenAI Python SDK (>=1.0) so every chat completion emits an
``llm`` span with tokens + cost. Patches both sync and async clients."""
from __future__ import annotations

from ._patch import instrument_llm


def _parse(resp) -> dict:
    usage = getattr(resp, "usage", None)
    text = None
    try:
        text = resp.choices[0].message.content
    except Exception:
        pass
    return {
        "model": getattr(resp, "model", None),
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "response": text,
    }


def apply() -> bool:
    try:
        from openai.resources.chat import completions as c
    except ImportError:
        return False
    done = False
    done |= instrument_llm(c.Completions, "create", _parse, "openai")
    if hasattr(c, "AsyncCompletions"):
        done |= instrument_llm(c.AsyncCompletions, "create", _parse, "openai")
    return done
