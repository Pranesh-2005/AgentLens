"""Drive the real Nebius+Serper web-search agents in this suite *through
AgentLens* and verify spans are captured.

For every framework we:
  1. ``monitor.start(project=...)`` — auto-patches the OpenAI SDK, so every
     Nebius (OpenAI-compatible) call becomes an ``llm`` span automatically.
  2. attach the framework's AgentLens adapter where one exists (richer
     tool/workflow spans), otherwise rely on the auto-patch alone.
  3. run the suite's *own* agent code (a real web search via Serper).
  4. flush, then read the captured spans back out of ./.agentlens/agentlens.db.

Frameworks not installed are SKIPPED. Run from this directory::

    python agentlens_runner.py
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# suite imports resolve from this dir; force a model that exists on Nebius and
# supports tool calling (the suite default model id returns empty content).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).resolve().parent / ".env")
os.environ["NEBIUS_MODEL"] = os.environ.get(
    "NEBIUS_MODEL_OVERRIDE", "meta-llama/Llama-3.3-70B-Instruct"
)

import agentlens
from agentlens import monitor
from agentlens.server.db import Store
from agentlens.storage import resolve

QUERY = "What is the latest stable version of Python? Use web search."
RESULTS: list[tuple[str, str, str]] = []


def _spans(project: str) -> dict[str, int]:
    store = Store(resolve(None))
    rows = store._conn.execute(
        "SELECT kind, COUNT(*) n FROM spans WHERE project=? GROUP BY kind", (project,)
    ).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def run(name: str, project: str, fn) -> None:
    print(f"\n=== {name} ===")
    if not os.getenv("NEBIUS_API_KEY"):
        RESULTS.append((name, "SKIP", "no NEBIUS_API_KEY"))
        print("  SKIP (no key)")
        return
    try:
        from agentlens import client as _c
        _c._config.client = None
        monitor.start(project=project)          # init + auto-patch openai SDK
        fn()
        monitor.stop()                           # revert patches + flush
        agentlens.flush()
        kinds = _spans(project)
        if kinds:
            RESULTS.append((name, "PASS", str(kinds)))
            print(f"  PASS captured: {kinds}")
        else:
            RESULTS.append((name, "FAIL", "no spans captured"))
            print("  FAIL no spans")
    except ImportError as e:
        RESULTS.append((name, "SKIP", f"not installed: {e}"))
        print(f"  SKIP not installed: {e}")
    except Exception as e:
        RESULTS.append((name, "FAIL", repr(e)))
        print(f"  FAIL {e!r}")
        traceback.print_exc()


# ---------------------------------------------------------------- LangChain
def t_langchain():
    import langchain_agent as m
    from agentlens.adapters.langchain import AgentLensCallbackHandler
    agent = m.build_agent_langgraph()
    agent.invoke({"messages": [("user", QUERY)]},
                 config={"callbacks": [AgentLensCallbackHandler()], "recursion_limit": 8})


# ---------------------------------------------------------------- LangGraph
def t_langgraph():
    import langgraph_agent as m
    from agentlens.adapters.langgraph import AgentLensCallbackHandler
    app = m.build_graph()
    app.invoke({"messages": [("user", QUERY)]},
               config={"callbacks": [AgentLensCallbackHandler()], "recursion_limit": 8})


# ---------------------------------------------------------------- LlamaIndex
def t_llamaindex():
    import asyncio
    import llamaindex_agent as m
    # FunctionAgent (0.14) bypasses CallbackManager; auto-patch on openai SDK
    # still captures the llm call.
    asyncio.run(m.main(QUERY))


# ---------------------------------------------------------------- PydanticAI
def t_pydanticai():
    import pydanticai_agent as m
    from agentlens.adapters.pydanticai import instrument_agent
    instrument_agent(m.agent).run_sync(QUERY)


# ---------------------------------------------------------------- Strands
def t_strands():
    import strands_agent as m
    from agentlens.adapters.strands import callback_handler
    try:
        m.agent.callback_handler = callback_handler
    except Exception:
        pass
    m.agent(QUERY)


# ---------------------------------------------------------------- CrewAI
def t_crewai():
    import crewai_agent as m
    m.run(QUERY)             # builds crew + kickoff (native OpenAI provider)


# ---------------------------------------------------------------- OpenAI Agents SDK
def t_openai_agents():
    import asyncio
    import openai_agents_sdk_agent as m
    from agentlens.adapters.openai_agents import install
    from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
    from openai import AsyncOpenAI
    from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL
    # The SDK parses "meta-llama/..." as a provider prefix; wrap the model in an
    # explicit Chat-Completions model bound to the Nebius client instead.
    set_tracing_disabled(False)          # module disabled it; adapter needs it on
    install()
    client = AsyncOpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)
    model = OpenAIChatCompletionsModel(model=NEBIUS_MODEL, openai_client=client)
    agent = Agent(name="Search Agent", instructions=m.agent.instructions,
                  tools=m.agent.tools, model=model)
    asyncio.run(Runner.run(agent, QUERY))


# ---------------------------------------------------------------- AutoGen
def t_autogen():
    import asyncio
    import autogen_agent as m
    asyncio.run(m.main(QUERY))


if __name__ == "__main__":
    run("langchain", "rw-langchain", t_langchain)
    run("langgraph", "rw-langgraph", t_langgraph)
    run("llamaindex", "rw-llamaindex", t_llamaindex)
    run("pydanticai", "rw-pydanticai", t_pydanticai)
    run("strands", "rw-strands", t_strands)
    run("crewai", "rw-crewai", t_crewai)
    run("openai-agents", "rw-openai-agents", t_openai_agents)
    run("autogen", "rw-autogen", t_autogen)

    print("\n" + "=" * 60 + "\nSUMMARY")
    for name, status, detail in RESULTS:
        print(f"  {status:5} {name:16} {detail[:80]}")
    npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"\n{npass}/{len(RESULTS)} frameworks captured real spans via AgentLens.")
