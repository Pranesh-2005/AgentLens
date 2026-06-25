# AgentLens

**Local-first diagnostics & observability for AI agents.** One line to integrate,
a single SQLite file, a built-in dashboard. AgentLens answers the question other
tools don't:

> **Why did my agent behave this way?**

Not a prompt tracker. Not a token dashboard. A diagnostics platform for *agent
behavior* — tool usage, memory access, workflow paths, decisions, failures,
retries, latency, and cost.

```bash
pip install pyagentlens               # core: store + dashboard + manual API + auto-patch
pip install "pyagentlens[langchain]"  # framework adapters as extras (langgraph, crewai, …)
pip install "pyagentlens[all]"        # every adapter
```

> Package name on PyPI is `pyagentlens`; the import name is `agentlens`.

```python
from agentlens import monitor
monitor.start()          # auto-patches installed LLM SDKs — that's it
```

> **Full docs:** see [`Guide.md`](./Guide.md) — in-depth reference for humans *and* AI
> coding agents (when-to-use-what tables, every API signature, per-framework
> snippets, troubleshooting, and an agent cheat sheet).

```bash
agentlens ui             # dashboard at http://127.0.0.1:7180
```

No server to deploy. No cloud. No external services. Data lives in a hidden
`./.agentlens/agentlens.db` (SQLite, WAL).

---

## Three ways to instrument (mix freely)

### 1. Zero-config auto-patch
`monitor.start()` detects and patches whatever is installed — **OpenAI,
Anthropic, Google Gemini, Groq, LiteLLM** — so every LLM call becomes an `llm`
span with tokens and cost, with no code changes.

### 2. Manual (framework-agnostic, always available)
```python
import agentlens
agentlens.init(project="research-bot")

with agentlens.trace_session("chat"):
    with agentlens.trace_agent("planner", role="lead"):
        agentlens.log_decision(options=["search", "answer"], chosen="search", reason="needs fresh info")
        agentlens.log_memory("read", "user_prefs", hit=True)
        agentlens.log_tool("web_search", args={"q": "ai news"}, result=[...], retry_count=1)
        agentlens.log_llm(model="gpt-4o", input_tokens=1200, output_tokens=400)
```
A tool/LLM logged with an `error=` (or a `trace_*` block that raises) is
automatically recorded as a failure.

### 3. Framework adapters
| Framework | Import | Maturity |
|---|---|---|
| LangChain | `agentlens.adapters.langchain.AgentLensCallbackHandler` | full |
| LangGraph | `agentlens.adapters.langgraph.AgentLensCallbackHandler` | full (LangChain callbacks) |
| LlamaIndex | `agentlens.adapters.llamaindex.AgentLensLlamaHandler` | full |
| CrewAI | `agentlens.adapters.crewai` (`step_callback`, `instrument_tools`) | best-effort |
| AutoGen | `agentlens.adapters.autogen.instrument_agent` | best-effort |
| PydanticAI | `agentlens.adapters.pydanticai.instrument_agent` | best-effort |
| OpenAI Agents SDK | `agentlens.adapters.openai_agents.install` | best-effort |
| Strands Agents | `agentlens.adapters.strands.callback_handler` | best-effort |
| Anthropic Agent SDK | `agentlens.adapters.anthropic_agents.traced_async_tool` | tool-level |
| MCP | `agentlens.adapters.mcp.instrument_server` / `traced_tool` | tool-level |
| FastMCP | `agentlens.adapters.fastmcp.instrument_server` | tool-level |

Plus a universal `agentlens.adapters.wrap_tool(fn)` / `@traced_tool()` that works
with any framework or none.

---

## The dashboard

- **Overview** — runs, success/failure rate, cost, p50/p95 latency.
- **Runs → Run detail** — the **workflow execution graph** (span tree: session →
  workflow → agent → tool/memory/llm/decision), click any node to inspect.
- **Agent Explorer** — runs, failures, latency, and attributed cost per agent.
- **Tool Explorer** — most used / slowest / most failed tools, retries.
- **Memory Explorer** — reads/writes/updates/deletes and read hit-rate.
- **Failure Explorer** — exceptions, messages, retries, jump to the failing run.
- **Cost Explorer** — token usage, cost by day, by model, most expensive agents.

---

## Try it now (no keys needed)

```bash
pip install -e .
python examples/demo_agent.py
agentlens ui
```

## Design

Local-first, zero-config, framework-agnostic, async-safe. Observability code
never crashes your app (emit failures are swallowed). One `Span` model covers
every event; kind-specific data lives in `attributes`. Cost is computed from an
editable price book (`agentlens/server/pricing.py`).

CLI: `agentlens ui [--port 7180] [--host] [--backend-store-uri PATH]`,
`agentlens providers`, `agentlens version`. Override the port with
`AGENTLENS_PORT`.

**Single process** writes straight to local SQLite. For **multiple
processes/workers** sharing one dashboard, run `agentlens ui` once and point each
worker at it: `agentlens.init(project=..., tracking_uri="http://127.0.0.1:7180")`.

See [`Guide.md`](./Guide.md) for the complete reference.
