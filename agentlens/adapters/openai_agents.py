"""OpenAI Agents SDK adapter.

The OpenAI Agents SDK emits traces/spans to registered trace processors. This
adapter registers a processor that converts agent/generation/function spans into
AgentLens spans.

Example::

    from agentlens.adapters.openai_agents import install
    install()   # call once before running your agents
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .. import client as _client
from ..events import Span, SpanKind, jsonable


class AgentLensTracingProcessor:
    """Implements the OpenAI Agents SDK ``TracingProcessor`` protocol (best-effort
    duck-typed). Each ended span is mapped to an AgentLens span."""

    def __init__(self, project: Optional[str] = None):
        self.project = project or _client.get_project()

    def on_trace_start(self, trace) -> None:  # noqa: D401
        pass

    def on_trace_end(self, trace) -> None:
        pass

    def on_span_start(self, span) -> None:
        pass

    def on_span_end(self, span) -> None:
        try:
            data = getattr(span, "span_data", None)
            tname = type(data).__name__.lower() if data is not None else ""
            if "generation" in tname or "response" in tname:
                kind = SpanKind.LLM.value
            elif "function" in tname or "tool" in tname:
                kind = SpanKind.TOOL.value
            elif "agent" in tname:
                kind = SpanKind.AGENT.value
            else:
                kind = SpanKind.OTHER.value

            attrs: dict = {"data": jsonable(data)}
            # pull model + token usage off generation/response spans so the cost
            # dashboards have real numbers instead of blanks
            if kind == SpanKind.LLM.value:
                usage = getattr(data, "usage", None) or {}
                if isinstance(usage, dict):
                    attrs["input_tokens"] = usage.get("input_tokens") or usage.get("prompt_tokens")
                    attrs["output_tokens"] = usage.get("output_tokens") or usage.get("completion_tokens")
                attrs["model"] = getattr(data, "model", None)
                # ResponseSpanData stashes the full request/result; surface them
                # so the input/output columns aren't blank for LLM spans either.
                attrs["input"] = jsonable(getattr(data, "input", None))
                attrs["output"] = jsonable(getattr(data, "response", None)
                                           or getattr(data, "output", None))
            # FunctionSpanData carries the tool's real args + return value; lift
            # them into top-level columns instead of burying inside "data".
            if kind == SpanKind.TOOL.value:
                attrs["input"] = jsonable(getattr(data, "input", None))
                attrs["output"] = jsonable(getattr(data, "output", None))

            # a readable name: the span_data's own name, else the model, else a
            # clean label derived from the kind (never the raw "GenerationSpanData").
            name = (getattr(data, "name", None) or attrs.get("model")
                    or {"llm": "llm call", "tool": "tool", "agent": "agent"}.get(kind)
                    or tname.replace("spandata", "") or "span")
            ev = Span(project=self.project, kind=kind, name=name, attributes=attrs)
            # honour the SDK's own trace/span ids so a whole agent run nests into a
            # single AgentLens run instead of one top-level row per SDK span.
            tid = getattr(span, "trace_id", None)
            sid = getattr(span, "span_id", None)
            pid = getattr(span, "parent_id", None)
            if tid:
                ev.trace_id = tid
            if sid:
                ev.span_id = sid
            if pid:
                ev.parent_span_id = pid
            _client.get_client().log_event(ev.finish().model_dump())
        except Exception:
            pass

    def shutdown(self) -> None:
        _client.flush()

    def force_flush(self) -> None:
        _client.flush()


def install(project: Optional[str] = None) -> bool:
    """Register the AgentLens trace processor with the OpenAI Agents SDK."""
    candidates = []
    try:
        import agents  # type: ignore
        candidates.append(agents)
        try:
            from agents import tracing  # type: ignore
            candidates.append(tracing)
        except ImportError:
            pass
    except ImportError:
        try:
            from openai.agents import tracing  # type: ignore
            candidates.append(tracing)
        except ImportError:
            return False
    proc = AgentLensTracingProcessor(project)
    for mod in candidates:
        for fn in ("add_trace_processor", "set_trace_processors", "register_processor"):
            adder = getattr(mod, fn, None)
            if callable(adder):
                try:
                    adder([proc] if "set" in fn else proc)
                    return True
                except Exception:
                    continue
    return False
