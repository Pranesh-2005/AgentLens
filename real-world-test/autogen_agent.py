"""
AutoGen (autogen-agentchat / AG2-style) search agent.

LLM:   Nebius Token Factory via `autogen_ext.models.openai.OpenAIChatCompletionClient`,
       AutoGen's standard "any OpenAI-compatible endpoint" client -- pass
       `base_url` + `api_key` + an explicit `model_info` dict (required for
       non-OpenAI model names so AutoGen knows the model supports function
       calling).
       Docs: https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.openai.html
Tool:  Serper.dev as a plain async Python function -- AssistantAgent infers
       the tool schema from the function signature + docstring.
       Docs: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/tools.html
Agent: `AssistantAgent` from `autogen_agentchat.agents`.

Install:
    pip install autogen-agentchat autogen-ext[openai] requests
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402
from serper_search import serper_search  # noqa: E402

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def web_search(query: str) -> str:
    """Search the live web via Serper.dev (Google Search API) for current
    information, news, or facts."""
    return serper_search(query)


model_client = OpenAIChatCompletionClient(
    model=NEBIUS_MODEL,
    base_url=NEBIUS_BASE_URL,
    api_key=NEBIUS_API_KEY,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
)

agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    tools=[web_search],
    system_message="You are a helpful assistant. Use the web_search tool for "
    "anything requiring current or factual information.",
    reflect_on_tool_use=True,
)


async def main(query: str) -> None:
    await Console(agent.run_stream(task=query))
    await model_client.close()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What's new in the latest Linux kernel release?"
    asyncio.run(main(q))
