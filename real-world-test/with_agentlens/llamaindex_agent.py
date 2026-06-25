"""LlamaIndex search agent — AgentLens-instrumented.

LlamaIndex 0.14 ``FunctionAgent`` bypasses CallbackManager, so structure spans
aren't emitted by the adapter; ``monitor.start()`` auto-patch still captures the
underlying OpenAI-compatible (Nebius) llm calls.
"""
from __future__ import annotations

import asyncio

import _bootstrap  # noqa: F401

import agentlens
from agentlens import monitor

import llamaindex_agent as base


def main() -> None:
    query = _bootstrap.query_from_argv("What is the latest stable version of Python?")
    monitor.start(project="llamaindex-agent")
    try:
        with agentlens.trace_session("llamaindex-search", input=query):
            print(asyncio.run(base.main(query)))
    finally:
        monitor.stop()
        agentlens.flush()


if __name__ == "__main__":
    main()