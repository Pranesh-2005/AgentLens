"""LangChain search agent — AgentLens-instrumented.

Same agent as ``../langchain_agent.py``. The ``AgentLensCallbackHandler`` is the
instrumentation here: LangChain reports token usage through the callback, so it
produces complete llm spans (model + tokens + cost). We deliberately use
``agentlens.init()`` rather than ``monitor.start()`` — auto-patching the OpenAI
SDK on top of the callback would log every call twice (once with tokens from the
callback, once without from the patched stream), inflating the llm count.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (loads env + sys.path)

import agentlens
from agentlens.adapters.langchain import AgentLensCallbackHandler

import langchain_agent as base  # the original suite script


def main() -> None:
    query = _bootstrap.query_from_argv("What is the latest stable version of Python?")
    agentlens.init(project="langchain-agent")
    try:
        agent = base.build_agent_langgraph()
        result = agent.invoke(
            {"messages": [("user", query)]},
            config={"callbacks": [AgentLensCallbackHandler()], "recursion_limit": 8},
        )
        print(result["messages"][-1].content)
    finally:
        agentlens.flush()


if __name__ == "__main__":
    main()
