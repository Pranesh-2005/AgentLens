"""LangGraph search agent — AgentLens-instrumented.

The LangChain/LangGraph callback handler captures the graph's llm + tool +
workflow spans (LangChain reports token usage through the callback, so the llm
spans are complete). We use ``agentlens.init()`` rather than ``monitor.start()``
so the OpenAI SDK isn't also auto-patched — that would double-log every llm call.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import agentlens
from agentlens.adapters.langgraph import AgentLensCallbackHandler

import langgraph_agent as base


def main() -> None:
    query = _bootstrap.query_from_argv("Who won the most recent Formula 1 race?")
    agentlens.init(project="langgraph-agent")
    try:
        app = base.build_graph()
        final = app.invoke(
            {"messages": [("user", query)]},
            config={"callbacks": [AgentLensCallbackHandler()], "recursion_limit": 8},
        )
        print(final["messages"][-1].content)
    finally:
        agentlens.flush()


if __name__ == "__main__":
    main()
