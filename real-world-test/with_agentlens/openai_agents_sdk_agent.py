"""OpenAI Agents SDK search agent — AgentLens-instrumented.

Two AgentLens-specific adjustments vs the suite script:
  * the suite passes ``model="meta-llama/..."`` which the SDK parses as a
    provider prefix (``UserError``); we wrap it in an explicit
    ``OpenAIChatCompletionsModel`` bound to the Nebius client instead.
  * the suite disables tracing; the AgentLens trace processor needs it on, so we
    re-enable it and ``install()`` the processor.

The processor is the single source of truth here: it maps the SDK's own
trace/span ids, so the whole agent run nests into one AgentLens run. We use
``agentlens.init()`` (not ``monitor.start()``) so the OpenAI SDK isn't also
auto-patched — that would log every generation a second time in a separate trace.
"""
from __future__ import annotations

import asyncio

import _bootstrap  # noqa: F401

import agentlens
from agentlens.adapters.openai_agents import install

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from openai import AsyncOpenAI

import openai_agents_sdk_agent as base
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL


def main() -> None:
    query = _bootstrap.query_from_argv("What is the latest stable version of Python?")
    agentlens.init(project="openai-agents-sdk")
    try:
        set_tracing_disabled(False)
        install()
        client = AsyncOpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)
        model = OpenAIChatCompletionsModel(model=NEBIUS_MODEL, openai_client=client)
        agent = Agent(name="Search Agent", instructions=base.agent.instructions,
                      tools=base.agent.tools, model=model)
        result = asyncio.run(Runner.run(agent, query))
        print(result.final_output)
    finally:
        agentlens.flush()


if __name__ == "__main__":
    main()