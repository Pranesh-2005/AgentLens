"""LangGraph adapter.

LangGraph executes nodes through the LangChain callback system, so the
``AgentLensCallbackHandler`` from the LangChain adapter captures graph node
LLM/tool/retrieval activity directly. Pass it in the run config::

    from agentlens.adapters.langgraph import AgentLensCallbackHandler
    graph.invoke(state, config={"callbacks": [AgentLensCallbackHandler()]})

Each graph node fires chain start/end events; tool nodes fire tool events; the
parent/child span tree reconstructs the workflow path in the Run detail view.
"""
from __future__ import annotations

from .langchain import AgentLensCallbackHandler

__all__ = ["AgentLensCallbackHandler"]
