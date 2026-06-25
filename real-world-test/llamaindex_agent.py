"""
LlamaIndex search agent.

LLM:   Nebius Token Factory via `llama_index.llms.openai_like.OpenAILike`,
       LlamaIndex's standard adapter for any OpenAI-compatible chat
       endpoint (set `api_base` + `is_function_calling_model=True`).
       Docs: https://docs.llamaindex.ai/en/stable/api_reference/llms/openai_like/
Tool:  Serper.dev wrapped as a `FunctionTool`.
       Docs: https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/
Agent: `FunctionAgent`, LlamaIndex's current single-agent class for
       tool-calling LLMs.
       Docs: https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/

Install:
    pip install llama-index-core llama-index-llms-openai-like requests
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402
from serper_search import serper_search  # noqa: E402

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike


def web_search(query: str) -> str:
    """Search the live web via Serper.dev (Google Search API) for current
    information, news, or facts not in the model's training data."""
    return serper_search(query)


search_tool = FunctionTool.from_defaults(fn=web_search)

llm = OpenAILike(
    model=NEBIUS_MODEL,
    api_base=NEBIUS_BASE_URL,
    api_key=NEBIUS_API_KEY,
    is_chat_model=True,
    is_function_calling_model=True,  # Nebius models support OpenAI-style tool calls
    temperature=0.2,
)

agent = FunctionAgent(
    tools=[search_tool],
    llm=llm,
    system_prompt="You are a helpful research assistant with web search access.",
)


async def main(query: str) -> str:
    response = await agent.run(query)
    return str(response)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What's the latest news in AI hardware?"
    print(asyncio.run(main(q)))
