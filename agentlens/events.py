"""Universal agent event model.

Every agent system, regardless of framework, produces the same handful of
diagnostic events: a session/workflow runs, agents act, they call tools, read
and write memory, make decisions, and invoke LLMs — and any of those can fail.
A single ``Span`` schema covers every one; kind-specific payloads live in
``attributes`` using the documented keys below. Field names follow the
OpenTelemetry GenAI semantic conventions where one exists so a future OTLP
bridge is cheap.

Span kind attribute conventions
-------------------------------
session:   name, status
workflow:  name, status, graph?  (e.g. {nodes, edges} for a known DAG)
agent:     agent_name, role, input, output, model
llm:       model, provider, prompt, response, input_tokens, output_tokens, cost
tool:      tool_name, args, result, retry_count, error
memory:    op (read|write|update|delete), memory_key, value, hit
decision:  options, chosen, reason
retrieval: query, results=[{id?, text?, score, rank?}]
"""
from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    SESSION = "session"
    WORKFLOW = "workflow"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    MEMORY = "memory"
    DECISION = "decision"
    RETRIEVAL = "retrieval"
    OTHER = "other"


KINDS = [k.value for k in SpanKind]


def new_id() -> str:
    return uuid.uuid4().hex


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) so we avoid a tokenizer dependency."""
    return max(1, len(str(text)) // 4)


class Span(BaseModel):
    event_id: str = Field(default_factory=new_id)
    trace_id: str = Field(default_factory=new_id)
    span_id: str = Field(default_factory=new_id)
    parent_span_id: Optional[str] = None
    session_id: Optional[str] = None
    project: str = "default"
    kind: str = SpanKind.OTHER.value
    name: str = ""
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "ok"  # ok | error
    attributes: Dict[str, Any] = Field(default_factory=dict)

    def finish(self, status: str = "ok") -> "Span":
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status
        return self


def jsonable(value: Any, _depth: int = 0) -> Any:
    """Coerce arbitrary tool args/results into something JSON-serializable so we
    never crash the host app while logging. Objects fall back to ``repr``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _depth > 6:
        return repr(value)[:2000]
    if isinstance(value, dict):
        return {str(k): jsonable(v, _depth + 1) for k, v in list(value.items())[:200]}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v, _depth + 1) for v in list(value)[:200]]
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(), _depth + 1)
    for attr in ("model_dump", "dict", "to_dict", "export"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return jsonable(fn(), _depth + 1)
            except Exception:
                break
    text = getattr(value, "text", None) or getattr(value, "content", None) \
        or getattr(value, "page_content", None)
    if isinstance(text, str):
        return text
    return repr(value)[:2000]


def normalize_result(item: Any) -> Dict[str, Any]:
    """Coerce a retrieval result (str | dict | doc-like) into the wire shape."""
    if isinstance(item, str):
        return {"text": item}
    if isinstance(item, BaseModel):
        item = item.model_dump()
    if isinstance(item, dict):
        out = dict(item)
        if "id" not in out and "chunk_id" in out:
            out["id"] = out.pop("chunk_id")
        return out
    text = getattr(item, "text", None) or getattr(item, "page_content", None)
    if text is not None:
        return {"text": text, "metadata": dict(getattr(item, "metadata", {}) or {})}
    return {"text": str(item)}
