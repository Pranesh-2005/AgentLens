"""
Strands Agents (AWS) search agent.

LLM:   Nebius Token Factory via Strands' `OpenAIModel` provider, which talks
       to any OpenAI-compatible API via `client_args={"base_url", "api_key"}`.
       Nebius is an officially documented community model provider for
       Strands: https://strandsagents.com/docs/community/model-providers/nebius-token-factory/
       Install: pip install 'strands-agents[openai]'
Tool:  Serper.dev wrapped with the `@tool` decorator from `strands`.
       Docs: https://strandsagents.com/docs/user-guide/concepts/tools/python-tools/
Agent: `strands.Agent`.

Install:
    pip install 'strands-agents[openai]' requests
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402
from serper_search import serper_search  # noqa: E402

from strands import Agent, tool
from strands.models.openai import OpenAIModel


@tool
def web_search(query: str) -> str:
    """Search the live web via Serper.dev (Google Search API) for current
    information, news, or facts.

    Args:
        query: The search query string.
    """
    return serper_search(query)


model = OpenAIModel(
    client_args={
        "api_key": NEBIUS_API_KEY,
        "base_url": NEBIUS_BASE_URL,
    },
    model_id=NEBIUS_MODEL,
    params={"temperature": 0.2, "max_tokens": 2000},
)

agent = Agent(
    model=model,
    tools=[web_search],
    system_prompt="You are a helpful research assistant with web search access.",
)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What's the latest in renewable energy storage?"
    response = agent(q)
    print(response)
