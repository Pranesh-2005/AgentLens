"""Tracing primitives: ``trace`` works as a context manager and a decorator,
with nesting via contextvars producing the Session -> Workflow -> Agent -> Span
tree, plus convenience loggers that emit correctly-shaped events so the
diagnostic dashboards light up without users learning the event schema."""
from __future__ import annotations

import contextvars
import functools
import time
from typing import Any, Dict, List, Optional

from . import client as _client
from .events import Span, SpanKind, jsonable, normalize_result

_current_span: "contextvars.ContextVar[Optional[Span]]" = contextvars.ContextVar(
    "agentlens_current_span", default=None
)


def current_trace_id() -> Optional[str]:
    span = _current_span.get()
    return span.trace_id if span else None


def current_session_id() -> Optional[str]:
    span = _current_span.get()
    return span.session_id if span else None


def _start_span(name: str, kind: str, attributes: Dict[str, Any]) -> Span:
    parent = _current_span.get()
    ev = Span(project=_client.get_project(), name=name, kind=kind, attributes=attributes)
    if parent is not None:
        ev.trace_id = parent.trace_id
        ev.session_id = parent.session_id or parent.span_id
        ev.parent_span_id = ev.parent_span_id or parent.span_id
    elif kind == SpanKind.SESSION.value:
        ev.session_id = ev.span_id
    return ev


class _TraceHandle:
    """Returned by ``trace(...)`` — usable as a ``with`` block or ``@`` decorator."""

    def __init__(self, name: str, kind: str, attributes: Dict[str, Any]):
        self._name = name
        self._kind = kind
        self._attributes = attributes
        self._span: Optional[Span] = None
        self._token = None

    def __enter__(self) -> Span:
        self._span = _start_span(self._name, self._kind, dict(self._attributes))
        self._token = _current_span.set(self._span)
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        span = self._span
        _current_span.reset(self._token)
        span.finish(status="error" if exc_type else "ok")
        if exc is not None:
            span.attributes.setdefault("error", repr(exc))
            span.attributes.setdefault("exception", type(exc).__name__)
        _client.get_client().log_event(span.model_dump())
        return False

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _TraceHandle(self._name or fn.__name__, self._kind, dict(self._attributes)):
                return fn(*args, **kwargs)

        return wrapper


def trace(name: Any = None, kind: Optional[str] = None, **attributes):
    """``with agentlens.trace("step"): ...`` or ``@agentlens.trace``. Generic
    span; prefer ``trace_session``/``trace_workflow``/``trace_agent`` for the
    common boundaries."""
    if callable(name):  # bare @agentlens.trace
        fn = name
        return _TraceHandle(fn.__name__, kind or SpanKind.OTHER.value, {})(fn)
    name = name or ""
    return _TraceHandle(name, kind or SpanKind.OTHER.value, attributes)


def trace_session(name: str = "session", **attributes):
    return _TraceHandle(name, SpanKind.SESSION.value, attributes)


def trace_workflow(name: str = "workflow", **attributes):
    return _TraceHandle(name, SpanKind.WORKFLOW.value, attributes)


def trace_agent(agent_name: str = "agent", role: Optional[str] = None, **attributes):
    attrs = {"agent_name": agent_name, "role": role, **attributes}
    return _TraceHandle(agent_name, SpanKind.AGENT.value, attrs)


# ---------------------------------------------------------------- loggers

def _log(kind: str, name: str, attributes: Dict[str, Any], status: str = "ok",
         duration_ms: Optional[float] = None) -> Span:
    parent = _current_span.get()
    now = time.time()
    ev = Span(
        project=_client.get_project(),
        name=name or kind,
        kind=kind,
        start_time=now if duration_ms is None else now - duration_ms / 1000.0,
        end_time=now,
        duration_ms=duration_ms or 0.0,
        status=status,
        attributes=attributes,
    )
    if parent is not None:
        ev.trace_id = parent.trace_id
        ev.session_id = parent.session_id or parent.span_id
        ev.parent_span_id = parent.span_id
    _client.get_client().log_event(ev.model_dump())
    return ev


def log_llm(model: str, provider: Optional[str] = None, prompt: Any = None,
            response: Any = None, input_tokens: Optional[int] = None,
            output_tokens: Optional[int] = None, cost: Optional[float] = None,
            duration_ms: Optional[float] = None, status: str = "ok", **extra) -> Span:
    attrs = {"model": model, "provider": provider, "prompt": jsonable(prompt),
             "response": jsonable(response), "input_tokens": input_tokens,
             "output_tokens": output_tokens, "cost": cost, **extra}
    return _log(SpanKind.LLM.value, model or "llm", attrs, status, duration_ms)


def log_tool(tool_name: str, args: Any = None, result: Any = None,
             retry_count: Optional[int] = None, error: Any = None,
             duration_ms: Optional[float] = None, **extra) -> Span:
    status = "error" if error is not None else "ok"
    j_args, j_result = jsonable(args), jsonable(result)
    # canonical "input"/"output" keys (UI reads these uniformly across adapters);
    # "args"/"result" kept as back-compat aliases.
    attrs = {"tool_name": tool_name, "input": j_args, "output": j_result,
             "args": j_args, "result": j_result,
             "retry_count": retry_count, **extra}
    if error is not None:
        attrs["error"] = repr(error) if isinstance(error, BaseException) else str(error)
        attrs["exception"] = type(error).__name__ if isinstance(error, BaseException) else None
    return _log(SpanKind.TOOL.value, tool_name, attrs, status, duration_ms)


def log_memory(op: str, memory_key: Optional[str] = None, value: Any = None,
               hit: Optional[bool] = None, duration_ms: Optional[float] = None, **extra) -> Span:
    attrs = {"op": op, "memory_key": memory_key, "value": jsonable(value), "hit": hit, **extra}
    return _log(SpanKind.MEMORY.value, f"memory.{op}", attrs, "ok", duration_ms)


def log_decision(options: Optional[List[Any]] = None, chosen: Any = None,
                 reason: Optional[str] = None, **extra) -> Span:
    attrs = {"options": jsonable(options), "chosen": jsonable(chosen), "reason": reason, **extra}
    return _log(SpanKind.DECISION.value, "decision", attrs)


def log_retrieval(query: str, results: List[Any], retriever: Optional[str] = None,
                  duration_ms: Optional[float] = None, **extra) -> Span:
    norm = [normalize_result(r) for r in results]
    for i, r in enumerate(norm):
        r.setdefault("rank", i + 1)
    attrs = {"query": query, "results": norm, "top_k": len(norm), "retriever": retriever, **extra}
    return _log(SpanKind.RETRIEVAL.value, retriever or "retrieval", attrs, "ok", duration_ms)


def log_error(error: Any, kind: str = SpanKind.OTHER.value, retry_count: Optional[int] = None,
              **extra) -> Span:
    attrs = {
        "error": repr(error) if isinstance(error, BaseException) else str(error),
        "exception": type(error).__name__ if isinstance(error, BaseException) else None,
        "retry_count": retry_count, **extra,
    }
    return _log(kind, "error", attrs, "error")
