"""Strands search agent — AgentLens-instrumented.

The Strands ``callback_handler`` streams agent/tool events into AgentLens;
``monitor.start()`` auto-patch captures the Nebius llm calls.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import agentlens
from agentlens import monitor
from agentlens.adapters.strands import callback_handler, reset

import strands_agent as base


def main() -> None:
    query = _bootstrap.query_from_argv("What is the latest stable version of Python?")
    monitor.start(project="strands-agent")  # auto-patch captures the Nebius llm calls
    try:
        reset()  # clear the per-process tool-use dedup cache for a clean run
        try:
            base.agent.callback_handler = callback_handler
        except Exception:
            pass
        # wrap the whole agent call in one trace so the callback's tool spans nest
        # under a single run instead of scattering into separate top-level rows.
        with agentlens.trace_agent("strands-search", input=query):
            result = base.agent(query)
        print(result)
    finally:
        monitor.stop()
        agentlens.flush()


if __name__ == "__main__":
    main()