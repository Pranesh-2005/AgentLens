"""
OpenAI Agents SDK search agent.

LLM:   Nebius Token Factory via a custom `AsyncOpenAI` client (base_url
       override) registered as the SDK-wide default client, plus the
       Chat Completions API shape (Nebius doesn't speak OpenAI's newer
       Responses API).
       Docs: https://openai.github.io/openai-agents-python/models/
             https://openai.github.io/openai-agents-python/config/
Tool:  Serper.dev wrapped with the `@function_tool` decorator.
       Docs: https://openai.github.io/openai-agents-python/tools/
Agent: `agents.Agent` run via `Runner`.

Install:
    pip install openai-agents requests
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402
from serper_search import serper_search  # noqa: E402

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

# Point the SDK's default OpenAI client at Nebius Token Factory.
set_default_openai_client(
    AsyncOpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)
)
# Nebius implements Chat Completions, not the newer Responses API.
set_default_openai_api("chat_completions")
# No OpenAI platform key -> disable OpenAI's hosted tracing export.
set_tracing_disabled(True)


@function_tool
def web_search(query: str) -> str:
    """Search the live web via Serper.dev (Google Search API) for current
    information, news, or facts."""
    return serper_search(query)


agent = Agent(
    name="Search Agent",
    instructions="You are a helpful research assistant. Use the web_search "
    "tool whenever you need current or factual information.",
    tools=[web_search],
    model=NEBIUS_MODEL,
)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What's the latest SpaceX launch?"
    result = Runner.run_sync(agent, q)
    print(result.final_output)
