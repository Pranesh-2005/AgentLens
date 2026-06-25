"""AutoGen (AgentChat) search agent — AgentLens-instrumented.

AutoGen's ``OpenAIChatCompletionClient`` talks to Nebius via the OpenAI SDK, so
``monitor.start()`` auto-patch captures the llm calls.
"""
from __future__ import annotations

import asyncio

import _bootstrap  # noqa: F401

import agentlens
from agentlens import monitor

import autogen_agent as base


def main() -> None:
    query = _bootstrap.query_from_argv("What is the latest stable version of Python?")
    monitor.start(project="autogen-agent")
    try:
        with agentlens.trace_session("autogen-search", input=query):
            asyncio.run(base.main(query))
    finally:
        monitor.stop()
        agentlens.flush()


if __name__ == "__main__":
    main()