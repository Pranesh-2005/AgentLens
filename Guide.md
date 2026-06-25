# AgentLens — The Complete Guide

> Local-first diagnostics & observability for AI agents.
> Answers one question: **"Why did my agent behave this way?"**

This guide is written for **two audiences**:

- **Humans** — read top to bottom, or jump to the section you need.
- **AI coding agents** — start at [§0 Decision Card](#0-decision-card-read-this-first). Every "when to use what" choice is a lookup table, not prose. Snippets are copy-paste-ready.

---

## Table of contents

0. [Decision card (read this first)](#0-decision-card-read-this-first)
1. [What AgentLens is (and isn't)](#1-what-agentlens-is-and-isnt)
2. [Install](#2-install)
3. [The three instrumentation layers — when to use which](#3-the-three-instrumentation-layers--when-to-use-which)
4. [Layer 1 — Auto-patch (`monitor.start`)](#4-layer-1--auto-patch-monitorstart)
5. [Layer 2 — Manual API (`trace_*` / `log_*`)](#5-layer-2--manual-api-trace_--log_)
6. [Layer 3 — Framework adapters](#6-layer-3--framework-adapters)
7. [Span model & `SpanKind` reference](#7-span-model--spankind-reference)
8. [The dashboard — page by page](#8-the-dashboard--page-by-page)
9. [CLI reference](#9-cli-reference)
10. [Configuration & environment variables](#10-configuration--environment-variables)
11. [Local vs HTTP client — when to use which](#11-local-vs-http-client--when-to-use-which)
12. [Cost & token tracking](#12-cost--token-tracking)
13. [Recipes](#13-recipes)
14. [Safety guarantees](#14-safety-guarantees)
15. [Troubleshooting](#15-troubleshooting)
16. [AI-agent cheat sheet](#16-ai-agent-cheat-sheet)

---

## 0. Decision card (read this first)

**Goal → action** lookup. Pick the row that matches the situation.

| Situation | Do this |
|---|---|
| Just want LLM calls/tokens/cost captured, zero code changes | `from agentlens import monitor; monitor.start()` |
| Using a supported agent framework (LangChain, CrewAI, …) | Add that framework's [adapter](#6-layer-3--framework-adapters) |
| Custom agent loop / no framework | [Manual API](#5-layer-2--manual-api-trace_--log_): `trace_agent` + `log_tool`/`log_memory`/`log_llm` |
| Want the richest traces (workflow graph + agents + tools + cost) | Combine: `monitor.start()` **and** the adapter **and** manual spans where the framework is blind |
| Production service, many processes/workers | [HTTP client](#11-local-vs-http-client--when-to-use-which) → one shared server |
| One script, notebook, local dev | Default [local client](#11-local-vs-http-client--when-to-use-which) (direct SQLite, no server) |
| Need to see what happened | `agentlens ui` → http://127.0.0.1:7180 |

**Three rules that never change:**

1. AgentLens **never raises into your code.** If observability fails, your agent keeps running. Safe to wrap anything.
2. **No keys, no cloud, no network** (unless you opt into the HTTP client). Everything lands in `./.agentlens/agentlens.db`.
3. Layers **stack.** Auto-patch + adapter + manual can all run at once; spans merge into one trace tree.

---

## 1. What AgentLens is (and isn't)

**Is:** a local, zero-config tool that records *spans* of agent execution — LLM calls, tool calls, memory ops, decisions, retrievals, agent runs, workflows — into a single SQLite file, then serves a dashboard to explore them. Built to debug *agent behavior*: why it picked a tool, where it looped, what a step cost, which call failed.

**Isn't:** a hosted APM, a prompt-eval framework, a vector DB, or a tracing standard. It does not require OpenTelemetry, an account, or an API key. (It can replay LLMs for cost backfill, but that is optional.)

**Mental model:** like `mlflow ui`, but the unit of observation is an **agent span tree** instead of an ML run.

```
session
└─ workflow            (the run)
   └─ agent            (a reasoning unit)
      ├─ llm           (a model call: model, tokens, cost)
      ├─ tool          (a tool call: args, result, retries, error)
      ├─ memory        (read/write/update/delete, hit/miss)
      └─ decision      (options, chosen, reason)
```

---

## 2. Install

```bash
pip install pyagentlens               # core (SQLite + dashboard + manual API + auto-patch)

# framework adapters as extras (only what you use):
pip install "pyagentlens[langchain]"
pip install "pyagentlens[langgraph]"
pip install "pyagentlens[llamaindex]"
pip install "pyagentlens[crewai]"
pip install "pyagentlens[autogen]"
pip install "pyagentlens[pydanticai]"
pip install "pyagentlens[openai-agents]"
pip install "pyagentlens[mcp]"
pip install "pyagentlens[fastmcp]"
pip install "pyagentlens[all]"        # every adapter
pip install "pyagentlens[dev]"        # + pytest
```

> Package name on PyPI is `pyagentlens`; the import name stays `agentlens` (`from agentlens import monitor`).

Auto-patch needs **no extra** — it patches whatever LLM SDK is already installed (`openai`, `anthropic`, `google-generativeai`, `groq`, `litellm`).

Requires Python 3.10+.

---

## 3. The three instrumentation layers — when to use which

| Layer | What it captures | Code cost | Use when |
|---|---|---|---|
| **1. Auto-patch** `monitor.start()` | `llm` spans (model, tokens, cost) for installed SDKs | 1 line | You call LLM SDKs directly and want them tracked for free. |
| **2. Manual API** `trace_*` / `log_*` | Anything you wrap: agents, tools, memory, decisions, workflows | per-span | Custom loops, or to enrich what frameworks don't expose. |
| **3. Adapters** | Framework-native agent/tool/llm/workflow spans | 1 import + 1 hook | You use a supported framework and want its structure for free. |

They are **additive**. Best traces = all three:

```python
from agentlens import monitor
from agentlens.adapters.langchain import AgentLensCallbackHandler
import agentlens

monitor.start(project="my-bot")                          # layer 1: raw LLM SDK calls
cb = AgentLensCallbackHandler()                           # layer 3: framework structure
with agentlens.trace_workflow("nightly-research"):        # layer 2: your own boundary
    agent.invoke(state, config={"callbacks": [cb]})
```

---

## 4. Layer 1 — Auto-patch (`monitor.start`)

The headline one-liner. Detects installed LLM SDKs, monkeypatches their completion methods (sync **and** async), and emits an `llm` span — with tokens and backfilled cost — for every call.

```python
from agentlens import monitor
monitor.start()                       # project="default", local store
# ... any openai / anthropic / gemini / groq / litellm call now produces an llm span ...
monitor.stop()                        # remove patches + flush (optional; auto-flush on exit)
```

**Signature**

```python
monitor.start(
    project: str = "default",
    tracking_uri: str | None = None,   # set → send to a running server instead of local SQLite
    db_path: str | None = None,        # override the SQLite path
    frameworks: list[str] | None = None,  # restrict which SDK patchers run; None = all
) -> dict[str, bool]                   # patch report, e.g. {"openai": True, "groq": True, "anthropic": False}
```

**Covered SDKs:** `openai`, `anthropic`, `gemini`, `groq`, `litellm`.

**When to use:** always, as a baseline — it's free LLM/cost visibility. Combine with adapters/manual for structure.

**When `frameworks=` helps:** narrow patching in a process that imports many SDKs but you only want one tracked: `monitor.start(frameworks=["litellm"])`.

`monitor.stop()` restores the original methods (idempotent) and flushes. You rarely need it — patches are harmless and a flush runs at exit — but it's there for tests and clean teardown.

---

## 5. Layer 2 — Manual API (`trace_*` / `log_*`)

Framework-agnostic. Use for custom agents, or to record what a framework hides (memory hits, decisions, retries).

### 5.1 Initialize (once)

```python
import agentlens
agentlens.init(project="research-bot")    # local ./.agentlens/agentlens.db
```

`init(project="default", tracking_uri=None, db_path=None)`. Omit `tracking_uri` for local SQLite; set it (e.g. `"http://127.0.0.1:7180"`) to stream to a server.

> `monitor.start()` calls `init()` for you. Only call `init()` directly when you use the manual/adapter layers **without** auto-patch.

### 5.2 Span boundaries (context manager **or** decorator)

```python
with agentlens.trace_session("user-123-chat"):
    with agentlens.trace_workflow("answer-question"):
        with agentlens.trace_agent("planner", role="orchestrator"):
            ...   # log_* calls here nest under "planner"

@agentlens.trace_agent("researcher")      # also works as a decorator
def research(q): ...
```

| Function | Span kind | Use for |
|---|---|---|
| `trace_session(name="session", **attrs)` | session | A whole conversation / user session (outermost). |
| `trace_workflow(name="workflow", **attrs)` | workflow | One run of a pipeline/graph → becomes the **workflow graph**. |
| `trace_agent(agent_name="agent", role=None, **attrs)` | agent | One reasoning unit / sub-agent. |
| `trace(name=None, kind=None, **attrs)` | other (or given `kind`) | Generic span; prefer the typed ones above. |

Nesting is automatic via contextvars — children inherit `trace_id`/`session_id` from whatever boundary is open. On exception, the span is marked `status="error"` and the exception type/message recorded (then re-raised — `trace_*` does **not** swallow your errors, only observability errors are swallowed).

### 5.3 Event loggers (leaf spans)

```python
agentlens.log_llm(model="gpt-4o", provider="openai",
                  prompt=msgs, response=text,
                  input_tokens=1200, output_tokens=400,
                  cost=None,                 # None → backfilled from the price book
                  duration_ms=812.4)

agentlens.log_tool("web_search", args={"q": "agent obs"}, result=[...],
                   retry_count=1, error=None, duration_ms=120.0)

agentlens.log_memory("read", "user_prefs", value={"lang": "en"}, hit=True)   # op: read|write|update|delete

agentlens.log_decision(options=["search", "answer"], chosen="search",
                       reason="not enough context yet")

agentlens.log_retrieval("query text", results=[...], retriever="pgvector", duration_ms=33.0)

agentlens.log_error(exc, kind="tool", retry_count=2)
```

**Auto-status:** `log_tool(..., error=...)` and `log_error(...)` set `status="error"` automatically → they show up in the **Failure Explorer** with no extra call.

| Logger | Span kind | Key attributes |
|---|---|---|
| `log_llm` | llm | model, provider, prompt, response, input_tokens, output_tokens, cost |
| `log_tool` | tool | tool_name, args, result, retry_count, error, exception |
| `log_memory` | memory | op, memory_key, value, hit |
| `log_decision` | decision | options, chosen, reason |
| `log_retrieval` | retrieval | query, results, top_k, retriever |
| `log_error` | (given kind) | error, exception, retry_count |

### 5.4 Helpers

```python
agentlens.current_trace_id()      # correlate AgentLens spans with your own logs
agentlens.current_session_id()
agentlens.flush()                 # force-flush buffered spans (HTTP client only; local is sync)
```

---

## 6. Layer 3 — Framework adapters

Each adapter is **import-time safe** (won't crash if the framework isn't installed) and pure-mapping helpers are testable without the framework. Pick your framework:

### LangChain — `AgentLensCallbackHandler`
```python
from agentlens.adapters.langchain import AgentLensCallbackHandler
llm.invoke("...", config={"callbacks": [AgentLensCallbackHandler()]})
```
Captures chain/tool/retriever/llm/chat-model events → agent/tool/llm/retrieval spans.

### LangGraph — same handler
```python
from agentlens.adapters.langgraph import AgentLensCallbackHandler
agent = create_react_agent(llm, tools)
agent.invoke({"messages": [("user", q)]},
             config={"callbacks": [AgentLensCallbackHandler()]})
```
Produces the **workflow graph** (workflow → llm → tool → llm …).

### LlamaIndex — `AgentLensLlamaHandler`
```python
from llama_index.core.callbacks import CallbackManager
from agentlens.adapters.llamaindex import AgentLensLlamaHandler
cm = CallbackManager([AgentLensLlamaHandler()])
llm = SomeLLM(callback_manager=cm)
```
> **Note:** LlamaIndex ≥0.14 routed *agents* through a new instrumentation system that bypasses `CallbackManager`. The handler hooks the (still-supported) callback manager — attach `cm` to the LLM/index to capture llm/retrieval spans.

### CrewAI — `step_callback` / `task_callback` / `instrument_tools`
```python
from agentlens.adapters.crewai import step_callback, task_callback, instrument_tools
agent = Agent(..., tools=instrument_tools(tools), step_callback=step_callback)
Crew(agents=[agent], tasks=[task], task_callback=task_callback).kickoff()
```

### AutoGen — `instrument_agent`
```python
from agentlens.adapters.autogen import instrument_agent
agent = instrument_agent(agent)
```

### PydanticAI — `instrument_agent`
```python
from agentlens.adapters.pydanticai import instrument_agent
agent = instrument_agent(agent)
agent.run_sync("...")
```

### OpenAI Agents SDK — `install()`
```python
from agentlens.adapters.openai_agents import install
install()          # register the AgentLens trace processor once, before running agents
```
Returns `True` on success, `False` if the SDK isn't importable.

### Strands — `callback_handler`
```python
from agentlens.adapters.strands import callback_handler
agent = Agent(model=model, tools=tools, callback_handler=callback_handler)
```

### Anthropic Agent SDK — `traced_async_tool`
```python
from agentlens.adapters.anthropic_agents import traced_async_tool

@traced_async_tool("calculator")
async def calc(expression: str) -> str: ...
```

### MCP / FastMCP — `instrument_server`
```python
from fastmcp import FastMCP
from agentlens.adapters.fastmcp import instrument_server   # or agentlens.adapters.mcp
mcp = FastMCP("calc-server")
@mcp.tool()
def calculator(expression: str) -> str: ...
instrument_server(mcp)        # wraps every registered tool → tool spans (args/result/errors)
```

### Any framework — generic tool wrappers (`adapters.common`)
```python
from agentlens.adapters.common import wrap_tool, traced_tool

search = wrap_tool(search_fn, name="web_search")   # wrap an existing callable

@traced_tool("calculator")                          # or decorate
def calc(expr): ...
```

**Verified coverage** (real agents against a live LLM): all of the above capture real spans. See [§15 troubleshooting](#15-troubleshooting) for the per-framework gotchas discovered during that testing.

---

## 7. Span model & `SpanKind` reference

Everything is one `Span`. You rarely construct it by hand (the loggers do), but here's the shape:

```python
Span(
    event_id, trace_id, span_id, parent_span_id, session_id,
    project, kind, name,
    start_time, end_time, duration_ms,
    status,                 # "ok" | "error"
    attributes={...},       # kind-specific (see table below)
)
```

`SpanKind` values and their conventional `attributes`:

| `SpanKind` | Meaning | Conventional attributes |
|---|---|---|
| `session` | A conversation / user session | name, status |
| `workflow` | One run of a pipeline / graph | name, status |
| `agent` | A reasoning unit / sub-agent | agent_name, role, input, output, model |
| `llm` | A model call | model, provider, prompt, response, input_tokens, output_tokens, cost |
| `tool` | A tool/function call | tool_name, args, result, retry_count, error |
| `memory` | A memory operation | op (read/write/update/delete), memory_key, value, hit |
| `decision` | A branch/choice | options, chosen, reason |
| `retrieval` | A retrieval step | query, results, top_k, retriever |
| `other` | Anything else | free-form |

At ingest, spans fan out into denormalized side-tables (`tool_calls`, `memory_ops`, `failures`) that power the explorer pages, and `llm` spans get **cost backfilled** from the price book when `cost` is missing.

---

## 8. The dashboard — page by page

Launch: `agentlens ui` → http://127.0.0.1:7180. Use the project selector (top) to scope every page.

| Page | Route | Shows | Use it to… |
|---|---|---|---|
| **Overview** | `/` | total runs, success/failure rate, p50/p95 latency, total cost, tool-call count | Get the health snapshot at a glance. |
| **Runs** | `/runs` | list of runs (trace_id, name, status, cost, counts, duration) | Find a specific run to inspect. |
| **Run detail** | `/runs/{id}` | the **workflow execution graph** — span tree with per-node kind, status, duration, cost | Answer "what happened, step by step, and where did it go wrong?" |
| **Agents** | `/agents` | per-agent runs, success rate, cost | See which agent is expensive / failing. |
| **Tools** | `/tools` | most-used / slowest / failed tools, retries | Debug tool usage and flakiness. |
| **Memory** | `/memory` | reads/writes/updates/deletes, hit rate | Audit memory access and cache effectiveness. |
| **Failures** | `/failures` | exceptions, timeouts, rate limits, retries | Triage everything that errored. |
| **Cost** | `/cost` | tokens & cost by model, by day, by agent | Track spend and find the expensive paths. |

The Run-detail tree is the centerpiece — e.g. a real ReAct run renders as:

```
workflow  langchain.chain            851.1ms ok
  llm     llama-3.3-70b-versatile    317.6ms ok
  tool    calculator                   1.2ms ok
  llm     llama-3.3-70b-versatile    306.7ms ok
```

---

## 9. CLI reference

```bash
agentlens ui                          # start the dashboard (default 127.0.0.1:7180)
agentlens ui --host 0.0.0.0 --port 9000
agentlens ui --backend-store-uri /path/to/agentlens.db
agentlens providers                   # list LLM providers available for optional replay/cost
agentlens version
```

`agentlens ui` reads `AGENTLENS_HOST` / `AGENTLENS_PORT` / `AGENTLENS_STORE` as defaults.

---

## 10. Configuration & environment variables

| Variable | Default | Effect |
|---|---|---|
| `AGENTLENS_PORT` | `7180` | Dashboard port (also `--port`). |
| `AGENTLENS_HOST` | `127.0.0.1` | Dashboard host (also `--host`). |
| `AGENTLENS_STORE` | *(hidden)* | SQLite path (also `--backend-store-uri` / `db_path=`). |
| `AGENTLENS_OPENAI_BASE_URL` | — | Override base URL for the optional LLM replay layer (OpenAI-compatible endpoints, e.g. Groq). |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `GROQ_API_KEY` | — | Only used by the optional replay/`/generate` feature; **not** needed for tracing. |

**Storage location:** hidden `./.agentlens/agentlens.db` (per project directory), `PRAGMA journal_mode=WAL`. The dot-folder survives accidental cleanup better than a plain logs dir and stays out of `git status`.

---

## 11. Local vs HTTP client — when to use which

| | **LocalClient** (default) | **HttpClient** |
|---|---|---|
| Trigger | `init()` / `monitor.start()` with no `tracking_uri` | pass `tracking_uri="http://host:7180"` |
| Path | writes straight to SQLite | batches spans on a background thread → `POST /api/events` |
| Best for | single script, notebook, local dev, tests | long-running services, multiple processes/workers feeding one store |
| Flush | synchronous (nothing to flush) | auto every ~1s + on exit; `agentlens.flush()` to force |

**Rule:** one process → local. Many processes that must share one dashboard → run `agentlens ui` once, point everyone at it with `tracking_uri`.

```python
# worker processes
agentlens.init(project="prod-bot", tracking_uri="http://127.0.0.1:7180")
# one server
#   agentlens ui
```

---

## 12. Cost & token tracking

- `log_llm(...)` / auto-patch record `input_tokens`, `output_tokens`. If `cost` is omitted, it's **backfilled at ingest** from a built-in, editable price book (includes current Claude/GPT/Groq models).
- Token estimation falls back to a heuristic when an SDK doesn't report usage.
- The **Cost Explorer** aggregates by model, by day, and by agent.
- `agentlens providers` shows which providers are configured for the optional replay feature.

To track a provider/model the price book doesn't know, either pass `cost=` explicitly to `log_llm`, or add the model to `agentlens/server/pricing.py`.

---

## 13. Recipes

### 13.1 Minimal custom agent, fully traced
```python
import agentlens
agentlens.init(project="demo")

with agentlens.trace_workflow("answer"):
    with agentlens.trace_agent("planner"):
        agentlens.log_decision(options=["search", "answer"], chosen="search",
                               reason="missing facts")
        agentlens.log_memory("read", "user_prefs", hit=True)
        try:
            result = search_tool(q)
            agentlens.log_tool("search", args={"q": q}, result=result)
        except Exception as e:
            agentlens.log_tool("search", args={"q": q}, error=e, retry_count=2)
        agentlens.log_llm(model="gpt-4o", input_tokens=900, output_tokens=210)
```

### 13.2 Zero-config + framework, the lazy best-practice
```python
from agentlens import monitor
from agentlens.adapters.langgraph import AgentLensCallbackHandler

monitor.start(project="research")
agent.invoke(state, config={"callbacks": [AgentLensCallbackHandler()]})
```

### 13.3 Catch a tool failure in the dashboard
```python
try:
    out = flaky_tool(x)
    agentlens.log_tool("flaky_tool", args={"x": x}, result=out)
except Exception as e:
    agentlens.log_tool("flaky_tool", args={"x": x}, error=e)   # → Failure Explorer
    raise
```

### 13.4 Correlate AgentLens with your own logs
```python
logger.info("starting step", extra={"trace_id": agentlens.current_trace_id()})
```

---

## 14. Safety guarantees

- **Observability never breaks your agent.** Every emit path swallows its own errors. A bad span, a full disk, a downed server → your code runs on.
- **`trace_*` re-raises your exceptions** (it records `status="error"` first). Only *observability* errors are swallowed, never *your* errors.
- **Idempotent patching.** `monitor.start()` twice is safe; `monitor.stop()` restores originals.
- **Import-time safe adapters.** Importing an adapter for a framework you don't have installed won't crash.
- **Offline & private.** No network, no keys, no telemetry unless you opt into the HTTP client or the replay feature.

---

## 15. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| No spans in dashboard | Did you `init()` / `monitor.start()`? Right **project** selected in the UI? Right `db_path`? |
| LLM calls not captured by auto-patch | SDK imported *after* `monitor.start()` in some setups — start first, or use the adapter/manual layer. |
| **FastMCP** import errors / empty module | fastmcp 3.x can install as a broken empty namespace package on some environments. Pin **`fastmcp==2.14.7`**. |
| **CrewAI + Groq** → `400 cache_breakpoint unsupported` | CrewAI's litellm path (`model="groq/..."`) leaks its internal `cache_breakpoint` message flag. Use the native provider, which strips it: `LLM(model=MODEL, provider="openai", base_url="https://api.groq.com/openai/v1", api_key=KEY)`. |
| **LlamaIndex ≥0.14** agent spans missing | Agents bypass `CallbackManager`. Attach `AgentLensLlamaHandler` via `CallbackManager` to the LLM/index to capture llm/retrieval spans. |
| **OpenAI Agents SDK + Groq** tool calls fail | Groq rejects that SDK's tool-call wire format. Use a plain chat agent, or an OpenAI/Anthropic backend, for tool agents. |
| Cost shows 0 | Model not in price book — pass `cost=` to `log_llm` or add it to `pricing.py`. |
| Port already in use | `agentlens ui --port 9000` or set `AGENTLENS_PORT`. |
| Errors not in Failure Explorer | Pass the exception to `log_tool(error=...)` / `log_error(...)`, or let `trace_*` see it (don't catch-and-swallow before the span closes). |

---

## 16. AI-agent cheat sheet

Copy-paste decision rules for an autonomous coding agent integrating AgentLens.

```text
IF user calls LLM SDKs directly (openai/anthropic/gemini/groq/litellm):
    from agentlens import monitor; monitor.start(project="<name>")

IF user uses a framework:
    langchain/langgraph -> AgentLensCallbackHandler in config={"callbacks":[...]}
    llamaindex          -> AgentLensLlamaHandler via CallbackManager on the LLM
    crewai              -> step_callback/task_callback + instrument_tools(tools)
    autogen             -> instrument_agent(agent)
    pydanticai          -> instrument_agent(agent)
    openai-agents       -> install()  # once, before running
    strands             -> Agent(callback_handler=callback_handler)
    anthropic-agents    -> @traced_async_tool(name)
    mcp/fastmcp         -> instrument_server(server)
    none/custom         -> trace_agent + log_tool/log_memory/log_llm/log_decision

IF custom tool not in a framework:
    from agentlens.adapters.common import traced_tool  # decorator
    # or wrap_tool(fn, name)

ALWAYS safe: every log_/trace_ swallows its own errors; trace_* still re-raises user errors.
INIT once: monitor.start() implies init(); else call agentlens.init(project=...).
MULTI-PROCESS: init(..., tracking_uri="http://host:7180"); run one `agentlens ui`.
VIEW: agentlens ui  ->  http://127.0.0.1:7180
STORE: ./.agentlens/agentlens.db (WAL, hidden, offline).
PIN: fastmcp==2.14.7.
```

---

*AgentLens — see why your agent did what it did. Local. Zero-config. Yours.*
