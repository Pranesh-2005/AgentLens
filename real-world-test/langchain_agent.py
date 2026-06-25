"""
LangChain search agent.

LLM:   Nebius Token Factory, via langchain_openai.ChatOpenAI pointed at
       Nebius's OpenAI-compatible endpoint (base_url override).
       Docs: https://python.langchain.com/docs/integrations/chat/openai/
Tool:  Serper.dev, wrapped with the `@tool` decorator.
       Docs: https://python.langchain.com/docs/concepts/tools/
Agent: LangChain's prebuilt tool-calling ReAct agent (`create_react_agent`
       from `langgraph.prebuilt`, the path LangChain's own docs now point to
       for "agents" -- see https://python.langchain.com/docs/tutorials/agents/).
       If you want pure-LangChain (no LangGraph) you can swap in
       `create_tool_calling_agent` + `AgentExecutor` instead -- both are shown
       below.

Install:
    pip install langchain langchain-openai langgraph requests
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402
from serper_search import serper_search  # noqa: E402

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def web_search(query: str) -> str:
    """Search the live web for up-to-date information using Serper.dev
    (Google Search API). Use this whenever you need current facts, news,
    or anything that might have changed since your training data."""
    return serper_search(query)


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=NEBIUS_MODEL,
        base_url=NEBIUS_BASE_URL,
        api_key=NEBIUS_API_KEY,
        temperature=0.2,
    )


def build_agent_langgraph():
    """Recommended path: LangChain's prebuilt ReAct-style tool agent."""
    from langgraph.prebuilt import create_react_agent

    llm = build_llm()
    return create_react_agent(llm, tools=[web_search])


def build_agent_classic():
    """Classic pure-LangChain path: AgentExecutor + tool-calling agent."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful research assistant with web search."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, [web_search], prompt)
    return AgentExecutor(agent=agent, tools=[web_search], verbose=True)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is the latest stable version of Python?"

    print("--- LangGraph-based ReAct agent ---")
    agent = build_agent_langgraph()
    result = agent.invoke({"messages": [("user", query)]})
    print(result["messages"][-1].content)

    # Uncomment to try the classic AgentExecutor path instead:
    # print("\n--- Classic AgentExecutor agent ---")
    # agent2 = build_agent_classic()
    # print(agent2.invoke({"input": query})["output"])
