"""
PydanticAI search agent.

LLM:   Nebius Token Factory via `OpenAIChatModel` + `OpenAIProvider(base_url=...)`
       -- PydanticAI's generic "any OpenAI-compatible API" path (it also ships
       a dedicated `NebiusProvider`, shown commented-out below, if you'd
       rather use `NEBIUS_API_KEY`-only auth and skip the explicit base_url).
       Docs: https://ai.pydantic.dev/models/openai/
Tool:  Serper.dev registered with the `@agent.tool_plain` decorator (no
       RunContext needed since the tool doesn't need agent dependencies).
       Docs: https://ai.pydantic.dev/tools/
Agent: `pydantic_ai.Agent`.

Install:
    pip install pydantic-ai-slim[openai] requests
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402
from serper_search import serper_search  # noqa: E402

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Generic OpenAI-compatible provider pointed at Nebius:
model = OpenAIChatModel(
    NEBIUS_MODEL,
    provider=OpenAIProvider(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY),
)

# Equivalent shortcut using PydanticAI's dedicated Nebius provider class:
# from pydantic_ai.providers.nebius import NebiusProvider
# model = OpenAIChatModel(NEBIUS_MODEL, provider=NebiusProvider(api_key=NEBIUS_API_KEY))

agent = Agent(
    model,
    system_prompt="You are a helpful research assistant with web search access.",
)


@agent.tool_plain
def web_search(query: str) -> str:
    """Search the live web via Serper.dev (Google Search API) for current
    information, news, or facts not already known."""
    return serper_search(query)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What's the current price of Bitcoin?"
    result = agent.run_sync(q)
    print(result.output)
