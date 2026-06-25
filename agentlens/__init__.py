"""AgentLens — local-first diagnostics & observability for AI agents.

Answers one question: *"Why did my agent behave this way?"*

Zero-config quickstart::

    from agentlens import monitor
    monitor.start()                      # auto-patches installed LLM SDKs
    # ... run your agent ...

Manual instrumentation (framework-agnostic)::

    import agentlens
    agentlens.init(project="research-bot")
    with agentlens.trace_agent("planner"):
        agentlens.log_tool("web_search", args={"q": "..."}, result=[...], retry_count=1)
        agentlens.log_memory("read", "user_prefs", hit=True)
        agentlens.log_llm(model="gpt-4o", input_tokens=1200, output_tokens=400)

Then ``agentlens ui`` to explore the dashboard (default http://127.0.0.1:7180).
"""
from . import monitor
from .client import flush, get_client, init
from .events import Span, SpanKind
from .tracing import (
    current_session_id,
    current_trace_id,
    log_decision,
    log_error,
    log_llm,
    log_memory,
    log_retrieval,
    log_tool,
    trace,
    trace_agent,
    trace_session,
    trace_workflow,
)

__version__ = "0.1.0"

__all__ = [
    "init", "flush", "get_client", "monitor",
    "trace", "trace_session", "trace_workflow", "trace_agent",
    "current_trace_id", "current_session_id",
    "log_llm", "log_tool", "log_memory", "log_decision", "log_retrieval", "log_error",
    "Span", "SpanKind", "__version__",
]
