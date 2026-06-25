"""Real-world, multi-framework integration test for AgentLens.

Builds a *real* calculator/chat agent in each installed framework, runs it
against a real LLM (Groq), and then verifies AgentLens actually captured spans
for that framework by reading them back out of the SQLite store.

Each framework writes into its own project so results are isolated. Frameworks
that aren't installed are SKIPPED. Run::

    python examples/test_frameworks.py

Requires a GROQ_API_KEY (loaded from F:/rag obs/.env if present).
"""
from __future__ import annotations

import os
import traceback

# ---- load GROQ key from the sibling project's .env if not already set --------
_ENV = r"F:/rag obs/.env"
if os.path.exists(_ENV) and not os.getenv("GROQ_API_KEY"):
    for line in open(_ENV):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

GROQ_MODEL = "llama-3.3-70b-versatile"
QUESTION = "What is 23 * 17 + 5? Use the calculator tool, then state the number."

import agentlens
from agentlens.server.db import Store
from agentlens.storage import resolve

RESULTS: list[tuple[str, str, str]] = []  # (framework, status, detail)


def calc(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))


def _fresh_client(project: str):
    """Point AgentLens at a fresh project (shared local DB)."""
    from agentlens import client as _c
    _c._config.client = None
    agentlens.init(project=project)


def _spans(project: str):
    store = Store(resolve(None))
    rows = store._conn.execute(
        "SELECT kind, COUNT(*) n FROM spans WHERE project=? GROUP BY kind", (project,)
    ).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def run(name: str, project: str, fn):
    print(f"\n=== {name} ===")
    if not os.getenv("GROQ_API_KEY"):
        RESULTS.append((name, "SKIP", "no GROQ_API_KEY"))
        print("  SKIP (no key)")
        return
    try:
        _fresh_client(project)
        fn()
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


# --------------------------------------------------------------- 1. raw Groq SDK (auto-patch)
def t_groq():
    from agentlens import monitor
    import groq
    monitor.start(project="fw-groq-autopatch")  # re-points + patches
    client = groq.Groq()
    client.chat.completions.create(
        model=GROQ_MODEL, max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in 3 words."}],
    )
    monitor.stop()


# --------------------------------------------------------------- 2. LiteLLM (auto-patch)
def t_litellm():
    from agentlens import monitor
    import litellm
    monitor.start(project="fw-litellm", frameworks=["litellm"])
    litellm.completion(
        model=f"groq/{GROQ_MODEL}", max_tokens=64,
        messages=[{"role": "user", "content": "Reply with one word: ping"}],
    )
    monitor.stop()


# --------------------------------------------------------------- 3. LangChain (callback handler)
def t_langchain():
    from langchain_groq import ChatGroq
    from agentlens.adapters.langchain import AgentLensCallbackHandler

    llm = ChatGroq(model=GROQ_MODEL, max_tokens=64)
    llm.invoke("What is the capital of France? One word.",
               config={"callbacks": [AgentLensCallbackHandler()]})


# --------------------------------------------------------------- 4. LangGraph (ReAct + tool)
def t_langgraph():
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent
    from agentlens.adapters.langgraph import AgentLensCallbackHandler

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression like '2*3+1'."""
        return calc(expression)

    agent = create_react_agent(ChatGroq(model=GROQ_MODEL), [calculator])
    agent.invoke({"messages": [("user", QUESTION)]},
                 config={"callbacks": [AgentLensCallbackHandler()]})


# --------------------------------------------------------------- 5. LlamaIndex (ReAct + tool)
def t_llamaindex():
    # LlamaIndex 0.14 moved agents to a new instrumentation system that bypasses
    # CallbackManager; the AgentLens handler hooks the (still-supported) callback
    # manager, so we exercise a real LLM completion through it.
    from llama_index.core.callbacks import CallbackManager
    from llama_index.llms.groq import Groq as LIGroq
    from agentlens.adapters.llamaindex import AgentLensLlamaHandler

    cm = CallbackManager([AgentLensLlamaHandler()])
    llm = LIGroq(model=GROQ_MODEL, callback_manager=cm, max_tokens=64)
    llm.complete("What is 23*17+5? Reply with only the number.")


# --------------------------------------------------------------- 6. PydanticAI (tool agent)
def t_pydanticai():
    from pydantic_ai import Agent
    from agentlens.adapters.pydanticai import instrument_agent

    agent = Agent(f"groq:{GROQ_MODEL}", system_prompt="Use tools to compute. Be terse.")

    @agent.tool_plain
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return calc(expression)

    agent = instrument_agent(agent)
    agent.run_sync(QUESTION)


# --------------------------------------------------------------- 7. FastMCP (tool server)
def t_fastmcp():
    from fastmcp import FastMCP
    from agentlens.adapters.fastmcp import instrument_server

    mcp = FastMCP("calc-server")

    @mcp.tool()
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return calc(expression)

    instrument_server(mcp)
    # invoke the (now-wrapped) underlying tool function directly
    mgr = getattr(mcp, "_tool_manager", None) or getattr(mcp, "tool_manager", None)
    tools = getattr(mgr, "_tools", None) or getattr(mgr, "tools", None) or {}
    entry = list(tools.values())[0]
    fn = getattr(entry, "fn", None) or getattr(entry, "func", None)
    fn(expression="23*17+5")


# --------------------------------------------------------------- 8. CrewAI (tool agent)
def t_crewai():
    # Route through CrewAI's *native* OpenAI provider (Groq is OpenAI-compatible).
    # The litellm fallback path (model="groq/...") leaks CrewAI's internal
    # `cache_breakpoint` message flag straight into the request, which Groq
    # rejects; the native provider strips it via base_llm._format_messages.
    from crewai import Agent, Crew, LLM, Task
    from agentlens.adapters.crewai import step_callback, task_callback

    llm = LLM(model=GROQ_MODEL, provider="openai",
              base_url="https://api.groq.com/openai/v1",
              api_key=os.environ["GROQ_API_KEY"])
    agent = Agent(role="mathematician", goal="Compute arithmetic exactly",
                  backstory="You are a precise calculator.", llm=llm,
                  step_callback=step_callback, verbose=False)
    task = Task(description="What is 23*17+5? Give only the number.",
                expected_output="a single number", agent=agent)
    Crew(agents=[agent], tasks=[task], task_callback=task_callback).kickoff()


# --------------------------------------------------------------- 9. OpenAI Agents SDK
def t_openai_agents():
    # Simple chat agent (no tool): Groq rejects this SDK's tool-call format, so
    # we verify the AgentLens trace processor captures a plain agent run.
    import asyncio

    from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled
    from openai import AsyncOpenAI
    from agentlens.adapters.openai_agents import install

    if not install():
        raise RuntimeError("could not register AgentLens trace processor")
    set_tracing_disabled(False)
    client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1",
                         api_key=os.environ["GROQ_API_KEY"])
    set_default_openai_client(client, use_for_tracing=False)

    agent = Agent(name="chat", instructions="Answer briefly with the number.", model=GROQ_MODEL)
    asyncio.run(Runner.run(agent, "What is 23*17+5?"))


# --------------------------------------------------------------- 10. Strands Agents
def t_strands():
    from strands import Agent, tool
    from strands.models.litellm import LiteLLMModel
    from agentlens.adapters.strands import callback_handler

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return calc(expression)

    model = LiteLLMModel(model_id=f"groq/{GROQ_MODEL}")
    agent = Agent(model=model, tools=[calculator], callback_handler=callback_handler)
    agent(QUESTION)


if __name__ == "__main__":
    run("groq-autopatch", "fw-groq-autopatch", t_groq)
    run("litellm-autopatch", "fw-litellm", t_litellm)
    run("langchain", "fw-langchain", t_langchain)
    run("langgraph", "fw-langgraph", t_langgraph)
    run("llamaindex", "fw-llamaindex", t_llamaindex)
    run("pydanticai", "fw-pydanticai", t_pydanticai)
    run("fastmcp", "fw-fastmcp", t_fastmcp)
    run("crewai", "fw-crewai", t_crewai)
    run("openai-agents", "fw-openai-agents", t_openai_agents)
    run("strands", "fw-strands", t_strands)

    print("\n" + "=" * 60 + "\nSUMMARY")
    for name, status, detail in RESULTS:
        print(f"  {status:5} {name:18} {detail[:80]}")
    npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"\n{npass}/{len(RESULTS)} frameworks captured real spans.")
