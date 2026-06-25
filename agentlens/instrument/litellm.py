"""Auto-patch LiteLLM's module-level ``completion`` / ``acompletion`` so every
call through LiteLLM (any backend it proxies) emits an ``llm`` span."""
from __future__ import annotations

from ._patch import instrument_llm


def _parse(resp) -> dict:
    # LiteLLM returns an OpenAI-compatible ModelResponse
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
        import litellm
    except ImportError:
        return False
    done = False
    done |= instrument_llm(litellm, "completion", _parse, "litellm")
    if hasattr(litellm, "acompletion"):
        done |= instrument_llm(litellm, "acompletion", _parse, "litellm")
    return done
