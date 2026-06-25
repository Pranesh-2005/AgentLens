"""
Anthropic Claude Agent SDK search agent.

LLM:   Nebius Token Factory -- routed through `nebius_anthropic_proxy.py`
       (in this same folder), a small hand-written FastAPI shim that speaks
       the Anthropic Messages API on one side and Nebius's OpenAI-compatible
       API on the other, using the official `openai` SDK -- NOT litellm.
       The Claude Agent SDK is pointed at this local proxy via
       `ANTHROPIC_BASE_URL`, exactly like you would point it at any other
       Anthropic-Messages-compatible gateway.
       Docs: https://docs.claude.com/en/api/agent-sdk/python
Tool:  Serper.dev wrapped as an in-process SDK MCP tool with `@tool` +
       `create_sdk_mcp_server` (the Agent SDK's native custom-tool mechanism).
       Docs: https://platform.claude.com/docs/en/agent-sdk/custom-tools
Agent: `ClaudeSDKClient` + `ClaudeAgentOptions`.

Run:
    pip install claude-agent-sdk fastapi uvicorn openai requests

    # Terminal 1 -- start the bridge proxy:
    NEBIUS_API_KEY=... uvicorn nebius_anthropic_proxy:app --port 8317

    # Terminal 2 -- run the agent:
    NEBIUS_API_KEY=... python anthropic_agent.py "your question"
"""

import asyncio
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from serper_search import serper_search  # noqa: E402

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)

PROXY_BASE_URL = "http://127.0.0.1:8317"


@tool("web_search", "Search the live web via Serper.dev (Google Search API) "
      "for current information, news, or facts.", {"query": str})
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    result = serper_search(args["query"])
    return {"content": [{"type": "text", "text": result}]}


search_server = create_sdk_mcp_server(
    name="search-tools", version="1.0.0", tools=[web_search]
)

options = ClaudeAgentOptions(
    system_prompt="You are a helpful research assistant with web search access.",
    mcp_servers={"search": search_server},
    allowed_tools=["mcp__search__web_search"],
    env={
        # Redirect the SDK's Anthropic API calls to our local Nebius bridge.
        "ANTHROPIC_BASE_URL": PROXY_BASE_URL,
        "ANTHROPIC_API_KEY": "not-needed-by-the-local-proxy",
    },
)


async def main(query: str) -> None:
    async with ClaudeSDKClient(options=options) as client:
        await client.query(query)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
        print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What's the latest news about quantum computing?"
    asyncio.run(main(q))
