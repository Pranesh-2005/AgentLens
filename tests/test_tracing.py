import agentlens


def test_manual_loggers_and_nesting(local):
    store = local
    with agentlens.trace_session("chat"):
        with agentlens.trace_agent("planner", role="lead"):
            agentlens.log_tool("web_search", args={"q": "ai"}, result=["r1"], retry_count=1)
            agentlens.log_memory("read", "prefs", hit=True)
            agentlens.log_llm(model="gpt-4o", input_tokens=1000, output_tokens=200)

    runs = store.list_runs("test")
    assert len(runs) == 1
    detail = store.get_run(runs[0]["trace_id"])
    kinds = {s["kind"] for s in detail["spans"]}
    assert {"session", "agent", "tool", "memory", "llm"} <= kinds
    # tree: session at root, agent under it
    assert detail["tree"][0]["kind"] == "session"


def test_tool_error_feeds_failures(local):
    store = local
    with agentlens.trace_agent("worker"):
        agentlens.log_tool("flaky", error=ConnectionError("boom"), retry_count=2)
    f = store.failure_stats("test")
    assert f["recent"]
    assert f["recent"][0]["exception"] == "ConnectionError"


def test_trace_decorator_records_error(local):
    store = local

    @agentlens.trace("risky")
    def boom():
        raise ValueError("x")

    try:
        boom()
    except ValueError:
        pass
    runs = store.list_runs("test")
    assert runs[0]["status"] == "error"


def test_llm_cost_backfilled(local):
    store = local
    agentlens.log_llm(model="gpt-4o", input_tokens=1_000_000, output_tokens=0)
    cost = store.cost_summary("test")
    assert cost["totals"]["cost"] > 0  # priced from the book