"""
LangGraph search agent.

LLM:   Nebius Token Factory via langchain_openai.ChatOpenAI (base_url
       override) -- LangGraph uses LangChain's chat-model interface, it
       doesn't have its own separate model layer.
       Docs: https://langchain-ai.github.io/langgraph/agents/agents/
Tool:  Serper.dev wrapped with `@tool`.
Agent: Hand-rolled LangGraph StateGraph with a model node + ToolNode, which
       is the pattern LangGraph's own docs use to explain how
       `create_react_agent` works under the hood:
       https://langchain-ai.github.io/langgraph/tutorials/introduction/

Install:
    pip install langgraph langchain-openai requests
"""

import sys
from pathlib import Path
from typing import Annotated

sys.path.append(str(Path(__file__).resolve().parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402
from serper_search import serper_search  # noqa: E402

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


@tool
def web_search(query: str) -> str:
    """Search the live web via Serper.dev (Google Search API) for current
    information, news, or facts."""
    return serper_search(query)


TOOLS = [web_search]

llm = ChatOpenAI(
    model=NEBIUS_MODEL,
    base_url=NEBIUS_BASE_URL,
    api_key=NEBIUS_API_KEY,
    temperature=0.2,
).bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Who won the most recent Formula 1 race?"
    app = build_graph()
    final_state = app.invoke({"messages": [("user", query)]})
    print(final_state["messages"][-1].content)
