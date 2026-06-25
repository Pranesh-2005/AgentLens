import time

from agentlens.events import Span, SpanKind


def _span(store, **kw):
    s = Span(project="p", **kw).finish(kw.get("status", "ok"))
    store.ingest_events([s.model_dump()])
    return s


def test_side_tables_populated(store):
    tid = "t1"
    _span(store, trace_id=tid, kind=SpanKind.TOOL.value, name="search",
          attributes={"tool_name": "search"})
    _span(store, trace_id=tid, kind=SpanKind.MEMORY.value,
          attributes={"op": "read", "memory_key": "k", "hit": True})
    _span(store, trace_id=tid, kind=SpanKind.TOOL.value, status="error",
          attributes={"tool_name": "bad", "error": "X", "exception": "ValueError"})

    ts = store.tool_stats("p")
    assert any(t["tool_name"] == "search" for t in ts["tools"])
    assert ts["failed"] and ts["failed"][0]["tool_name"] == "bad"

    ms = store.memory_stats("p")
    assert ms["reads"] == 1 and ms["hit_rate"] == 1.0

    fs = store.failure_stats("p")
    assert fs["recent"][0]["exception"] == "ValueError"


def test_overview_rates(store):
    _span(store, trace_id="ok1", kind=SpanKind.SESSION.value, status="ok")
    _span(store, trace_id="err1", kind=SpanKind.SESSION.value, status="error",
          attributes={"error": "boom", "exception": "RuntimeError"})
    o = store.overview("p")
    assert o["total_runs"] == 2
    assert o["success"] == 1 and o["failed"] == 1
    assert o["failure_rate"] == 0.5


def test_cost_backfill_and_summary(store):
    _span(store, trace_id="g1", kind=SpanKind.LLM.value,
          attributes={"model": "gpt-4o-mini", "input_tokens": 1_000_000, "output_tokens": 1_000_000})
    cs = store.cost_summary("p")
    # gpt-4o-mini = (0.15, 0.60) per 1M => 0.75
    assert round(cs["totals"]["cost"], 2) == 0.75
    assert cs["by_model"][0]["model"] == "gpt-4o-mini"