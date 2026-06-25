"""LlamaIndex adapter: a callback handler mapping LlamaIndex events to spans.

Usage::

    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager
    from agentlens.adapters.llamaindex import AgentLensLlamaHandler

    Settings.callback_manager = CallbackManager([AgentLensLlamaHandler()])
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .. import client as _client
from ..events import Span, SpanKind, jsonable, normalize_result

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler  # type: ignore
    from llama_index.core.callbacks.schema import CBEventType  # type: ignore
    _OK = True
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object
    CBEventType = None
    _OK = False


class AgentLensLlamaHandler(BaseCallbackHandler):
    """Maps LLM / FUNCTION_CALL / AGENT_STEP / RETRIEVE events to AgentLens spans."""

    def __init__(self, project: Optional[str] = None):
        if not _OK:
            raise ImportError("LlamaIndex is not installed. Run: pip install agentlens[llamaindex]")
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.project = project or _client.get_project()
        self._trace_id = Span().trace_id
        self._starts: Dict[str, float] = {}
        self._payloads: Dict[str, Any] = {}

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._trace_id = Span().trace_id

    def end_trace(self, trace_id: Optional[str] = None, trace_map=None) -> None:
        pass

    def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kw) -> str:
        self._starts[event_id] = time.time()
        self._payloads[event_id] = payload or {}
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kw) -> None:
        t0 = self._starts.pop(event_id, time.time())
        payload = payload or self._payloads.pop(event_id, {})
        name = str(getattr(event_type, "value", event_type))
        kind, attrs = self._map(event_type, payload)
        ev = Span(trace_id=self._trace_id, project=self.project, kind=kind,
                  name=name, start_time=t0, attributes=attrs)
        _client.get_client().log_event(ev.finish().model_dump())

    def _map(self, event_type, payload) -> tuple:
        et = str(getattr(event_type, "value", event_type)).lower()
        if "llm" in et:
            return SpanKind.LLM.value, {"model": jsonable(payload.get("serialized", {})),
                                        "response": jsonable(payload.get("response"))}
        if "retrieve" in et:
            nodes = payload.get("nodes") or []
            return SpanKind.RETRIEVAL.value, {"query": jsonable(payload.get("query_str")),
                                              "results": [normalize_result(n) for n in nodes]}
        if "function" in et or "tool" in et or "agent" in et:
            jp = jsonable(payload)
            return SpanKind.TOOL.value, {"tool_name": et, "input": jp, "args": jp}
        return SpanKind.OTHER.value, {"payload": jsonable(payload)}
