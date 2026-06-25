"""CrewAI search agent — AgentLens-instrumented.

The suite script uses CrewAI's native OpenAI provider (no LiteLLM) pointed at
Nebius, so ``monitor.start()`` auto-patch on the OpenAI SDK captures the llm
calls. The CrewAI step/task callbacks add agent/task structure.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import agentlens
from agentlens import monitor

import crewai_agent as base


def main() -> None:
    query = _bootstrap.query_from_argv("What is the latest stable version of Python?")
    monitor.start(project="crewai-agent")
    try:
        # base.run() builds the crew + kicks it off; attach AgentLens callbacks
        # where the suite exposes them.
        try:
            from agentlens.adapters.crewai import step_callback, task_callback
            base.researcher.step_callback = step_callback  # type: ignore[attr-defined]
        except Exception:
            pass
        # one trace around the whole crew so steps + llm calls nest in a single run
        with agentlens.trace_workflow("crewai-research", input=query):
            print(base.run(query))
    finally:
        monitor.stop()
        agentlens.flush()


if __name__ == "__main__":
    main()