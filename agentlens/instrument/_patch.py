"""Safe monkeypatch helpers shared by the LLM auto-patchers.

Guarantees:
  * idempotent — patching twice is a no-op (marker attribute on the wrapper),
  * restorable — every patch records how to undo itself for ``monitor.stop()``,
  * non-intrusive — a failure while *emitting* a span never propagates into the
    instrumented call (observability must not crash the host app).
"""
from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, List, Tuple

_MARK = "__agentlens_patched__"
_RESTORE: List[Tuple[Any, str, Any]] = []


def already_patched(fn: Any) -> bool:
    return getattr(fn, _MARK, False)


def patch(owner: Any, attr: str, make_wrapper: Callable[[Callable], Callable]) -> bool:
    """Replace ``owner.attr`` with ``make_wrapper(original)``. Returns True if a
    patch was applied, False if it was already patched or the attr is missing."""
    original = getattr(owner, attr, None)
    if original is None or already_patched(original):
        return False
    wrapper = make_wrapper(original)
    functools.update_wrapper(wrapper, original)
    setattr(wrapper, _MARK, True)
    setattr(owner, attr, wrapper)
    _RESTORE.append((owner, attr, original))
    return True


def revert_all() -> None:
    while _RESTORE:
        owner, attr, original = _RESTORE.pop()
        try:
            setattr(owner, attr, original)
        except Exception:
            pass


def instrument_llm(owner: Any, attr: str, parse: Callable[[Any], dict], provider: str) -> bool:
    """Patch a (possibly async) LLM ``create``-style method so each call emits an
    ``llm`` span. ``parse(response)`` must return a dict with optional keys
    ``model, input_tokens, output_tokens, response``."""
    def make_wrapper(original):
        is_async = inspect.iscoroutinefunction(original)
        if is_async:
            @functools.wraps(original)
            async def awrapper(*args, **kwargs):
                t0 = time.time()
                try:
                    resp = await original(*args, **kwargs)
                except Exception as e:
                    _emit_error(provider, kwargs, e, t0)
                    raise
                _emit(provider, kwargs, resp, parse, t0)
                return resp
            return awrapper

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                resp = original(*args, **kwargs)
            except Exception as e:
                _emit_error(provider, kwargs, e, t0)
                raise
            _emit(provider, kwargs, resp, parse, t0)
            return resp
        return wrapper

    return patch(owner, attr, make_wrapper)


def _emit(provider, kwargs, resp, parse, t0) -> None:
    try:
        from ..tracing import log_llm
        from ..events import estimate_tokens
        info = parse(resp) or {}
        prompt = kwargs.get("messages") or kwargs.get("prompt") or kwargs.get("contents")
        response = info.get("response")
        # Streaming responses (and some OpenAI-compatible gateways) don't report a
        # usage block, so the provider gives back no token counts. Fall back to a
        # cheap char-based estimate from the prompt/response so the cost dashboards
        # show real numbers instead of blanks; flag it so it's not mistaken for exact.
        in_tok, out_tok = info.get("input_tokens"), info.get("output_tokens")
        estimated = False
        if in_tok is None and prompt is not None:
            in_tok, estimated = estimate_tokens(prompt), True
        if out_tok is None and response:
            out_tok, estimated = estimate_tokens(response), True
        log_llm(
            model=info.get("model") or kwargs.get("model") or "unknown",
            provider=provider, prompt=prompt, response=response,
            input_tokens=in_tok, output_tokens=out_tok,
            tokens_estimated=estimated or None,
            duration_ms=(time.time() - t0) * 1000.0,
        )
    except Exception:
        pass


def _emit_error(provider, kwargs, exc, t0) -> None:
    try:
        from ..tracing import log_llm
        log_llm(
            model=kwargs.get("model") or "unknown", provider=provider,
            prompt=kwargs.get("messages") or kwargs.get("prompt") or kwargs.get("contents"),
            status="error", error=repr(exc), exception=type(exc).__name__,
            duration_ms=(time.time() - t0) * 1000.0,
        )
    except Exception:
        pass
