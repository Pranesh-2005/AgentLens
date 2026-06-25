"""PydanticAI search agent — AgentLens-instrumented.

``instrument_agent`` wraps the agent run (agent spans); ``monitor.start()``
auto-patch captures the Nebius llm calls.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import agentlens
from agentlens import monitor
from agentlens.adapters.pydanticai import instrument_agent

import pydanticai_agent as base


def main() -> None:
    query = _bootstrap.query_from_argv("What is the latest stable version of Python?")
    monitor.start(project="pydanticai-agent")
    try:
        with agentlens.trace_session("pydanticai-search", input=query):
            agent = instrument_agent(base.agent)
            result = agent.run_sync(query)
            print(getattr(result, "output", result))
    finally:
        monitor.stop()
        agentlens.flush()


if __name__ == "__main__":
    main()