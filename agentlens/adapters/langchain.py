"""LangChain adapter: a callback handler that converts LangChain run events
into AgentLens spans.

Usage::

    from agentlens.adapters.langchain import AgentLensCallbackHandler
    chain.invoke(question, config={"callbacks": [AgentLensCallbackHandler()]})

LangGraph runs through the same LangChain callback system, so this handler works
for LangGraph graphs too (see ``adapters/langgraph.py``).

The pure mapping helpers below (dict in -> span dict out) are testable without
LangChain installed.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .. import client as _client
from ..events import Span, SpanKind, estimate_tokens, jsonable, normalize_result

try:
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object


# --------------------------------------------------------------- pure mapping

def llm_span(prompts: List[str], response_text: str, model: Optional[str],
             trace_id: str, parent_span_id: Optional[str], project: str,
             start_time: float, token_usage: Optional[Dict[str, Any]] = None,
             status: str = "ok") -> Dict[str, Any]:
    usage = token_usage or {}
    ev = Span(
        trace_id=trace_id, parent_span_id=parent_span_id, project=project,
        kind=SpanKind.LLM.value, name=model or "langchain.llm", start_time=start_time,
        attributes={"model": model, "provider": "langchain", "prompt": "\n\n".join(prompts),
                    "response": response_text, "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens")},
    )
    return ev.finish(status).model_dump()


def tool_span(tool_name: str, args: Any, result: Any, trace_id: str,
              parent_span_id: Optional[str], project: str, start_time: float,
              error: Any = None) -> Dict[str, Any]:
    j_args, j_result = jsonable(args), jsonable(result)
    attrs: Dict[str, Any] = {"tool_name": tool_name, "input": j_args, "output": j_result,
                             "args": j_args, "result": j_result}
    status = "ok"
    if error is not None:
        status = "error"
        attrs["error"] = repr(error)
        attrs["exception"] = type(error).__name__ if isinstance(error, BaseException) else None
    ev = Span(trace_id=trace_id, parent_span_id=parent_span_id, project=project,
              kind=SpanKind.TOOL.value, name=tool_name, start_time=start_time, attributes=attrs)
    return ev.finish(status).model_dump()


# ------------------------------------------------------------------- handler

class AgentLensCallbackHandler(BaseCallbackHandler):
    """Maps on_chain_* to the run/agent boundary, on_tool_* to tool spans,
    on_retriever_* to retrieval spans and on_llm_* to llm spans."""

    def __init__(self, project: Optional[str] = None):
        if BaseCallbackHandler is object:
            raise ImportError("LangChain is not installed. Run: pip install agentlens[langchain]")
        self.project = project or _client.get_project()
        self._trace_id: Optional[str] = None
        self._root_run: Optional[str] = None
        self._starts: Dict[str, float] = {}
        self._names: Dict[str, str] = {}
        self._trace_start: Optional[float] = None
        self._name: Optional[str] = None

    def _ensure_trace(self) -> str:
        if self._trace_id is None:
            self._trace_id = Span().trace_id
        return self._trace_id

    def _emit(self, ev: Dict[str, Any]) -> None:
        _client.get_client().log_event(ev)

    # -- chain = run/agent boundary --------------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        if parent_run_id is None and self._root_run is None:
            self._root_run = str(run_id)
            self._trace_id = Span().trace_id
            self._trace_start = time.time()
            self._name = (serialized or {}).get("name") if isinstance(serialized, dict) else None

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kw):
        if str(run_id) == self._root_run:
            ev = Span(trace_id=self._trace_id, project=self.project, kind=SpanKind.WORKFLOW.value,
                      name=self._name or "langchain.chain", start_time=self._trace_start or time.time(),
                      attributes={"name": self._name})
            self._emit(ev.finish().model_dump())
            self._root_run = None

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kw):
        if str(run_id) == self._root_run:
            ev = Span(trace_id=self._trace_id, project=self.project, kind=SpanKind.WORKFLOW.value,
                      name=self._name or "langchain.chain", start_time=self._trace_start or time.time(),
                      attributes={"error": repr(error), "exception": type(error).__name__})
            self._emit(ev.finish("error").model_dump())
            self._root_run = None

    # -- tools -----------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._starts[rid] = time.time()
        self._names[rid] = (serialized or {}).get("name", "tool") if isinstance(serialized, dict) else "tool"
        self._names[rid + ":in"] = input_str

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._emit(tool_span(self._names.pop(rid, "tool"), self._names.pop(rid + ":in", None),
                             output, self._ensure_trace(), None, self.project,
                             self._starts.pop(rid, time.time())))

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._emit(tool_span(self._names.pop(rid, "tool"), self._names.pop(rid + ":in", None),
                             None, self._ensure_trace(), None, self.project,
                             self._starts.pop(rid, time.time()), error=error))

    # -- retriever -------------------------------------------------------
    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._starts[rid] = time.time()
        self._names[rid] = query

    def on_retriever_end(self, documents, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        results = [normalize_result(d) for d in (documents or [])]
        ev = Span(trace_id=self._ensure_trace(), project=self.project, kind=SpanKind.RETRIEVAL.value,
                  name="langchain.retriever", start_time=self._starts.pop(rid, time.time()),
                  attributes={"query": self._names.pop(rid, ""), "results": results, "top_k": len(results)})
        self._emit(ev.finish().model_dump())

    # -- llm -------------------------------------------------------------
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._starts[rid] = time.time()
        self._names[rid] = "\n\n".join(prompts)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._starts[rid] = time.time()
        flat = []
        for batch in messages:
            for m in batch:
                flat.append(f"{getattr(m, 'type', 'msg')}: {getattr(m, 'content', m)}")
        self._names[rid] = "\n".join(flat)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        text, model, usage = "", None, None
        try:
            out = getattr(response, "llm_output", None) or {}
            model = out.get("model_name") or out.get("model")
            usage = out.get("token_usage") or out.get("usage")
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = getattr(gen, "text", "") or getattr(msg, "content", "")
            meta = getattr(msg, "response_metadata", None) or {}
            model = model or meta.get("model_name") or meta.get("model")
            usage = usage or meta.get("token_usage")
            um = getattr(msg, "usage_metadata", None)
            if not usage and um:
                usage = {"prompt_tokens": um.get("input_tokens"), "completion_tokens": um.get("output_tokens")}
        except (AttributeError, IndexError):
            pass
        self._emit(llm_span([self._names.pop(rid, "")], text, model, self._ensure_trace(),
                            None, self.project, self._starts.pop(rid, time.time()), usage))

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._emit(llm_span([self._names.pop(rid, "")], repr(error), None, self._ensure_trace(),
                            None, self.project, self._starts.pop(rid, time.time()), status="error"))
