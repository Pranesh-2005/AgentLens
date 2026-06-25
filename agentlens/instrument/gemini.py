"""Auto-patch Google Gemini (``google.generativeai``) so every
``GenerativeModel.generate_content`` emits an ``llm`` span."""
from __future__ import annotations

from ._patch import instrument_llm


def _parse(resp) -> dict:
    meta = getattr(resp, "usage_metadata", None)
    text = None
    try:
        text = resp.text
    except Exception:
        pass
    return {
        "model": None,  # not on the response; falls back to call kwargs
        "input_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
        "response": text,
    }


def apply() -> bool:
    try:
        import google.generativeai as genai
    except ImportError:
        return False
    done = False
    if hasattr(genai, "GenerativeModel"):
        done |= instrument_llm(genai.GenerativeModel, "generate_content", _parse, "gemini")
    return done
