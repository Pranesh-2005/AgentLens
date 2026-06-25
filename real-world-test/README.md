# Nebius + Serper Search Agents, in 9 Agent Frameworks

The same "search agent" -- one LLM-backed agent with one web-search tool --
implemented natively in each of:

| Framework | File | LLM wiring | Tool wiring |
|---|---|---|---|
| LangChain | `langchain_agent.py` | `ChatOpenAI(base_url=...)` | `@tool` |
| LangGraph | `langgraph_agent.py` | `ChatOpenAI(base_url=...).bind_tools()` | `@tool` + `ToolNode` |
| LlamaIndex | `llamaindex_agent.py` | `OpenAILike(api_base=...)` | `FunctionTool` |
| CrewAI | `crewai_agent.py` | `LLM(model="openai/...", base_url=...)` **(native, no LiteLLM)** | `@tool` (crewai.tools) |
| AutoGen | `autogen_agent.py` | `OpenAIChatCompletionClient(base_url=...)` | plain async function |
| PydanticAI | `pydanticai_agent.py` | `OpenAIChatModel` + `OpenAIProvider(base_url=...)` | `@agent.tool_plain` |
| OpenAI Agents SDK | `openai_agents_sdk_agent.py` | custom `AsyncOpenAI(base_url=...)` via `set_default_openai_client` | `@function_tool` |
| Strands Agents | `strands_agent.py` | `OpenAIModel(client_args={base_url,...})` | `@tool` (strands) |
| Anthropic Claude Agent SDK | `anthropic_agent_sdk/` | local bridge proxy (`nebius_anthropic_proxy.py`) translating Anthropic Messages API <-> Nebius's OpenAI API | `@tool` + `create_sdk_mcp_server` |

**LLM:** [Nebius Token Factory](https://tokenfactory.nebius.com) — an
OpenAI-compatible inference API (`https://api.tokenfactory.nebius.com/v1/`).
Every framework except the Claude Agent SDK talks to it directly via that
framework's own "custom OpenAI-compatible endpoint" mechanism.

**Search tool:** [Serper.dev](https://serper.dev) — a Google Search API.
The actual HTTP call lives in one place, `common/serper_search.py`
(plain `requests`, ~40 lines), and every framework wraps that same function
with its own native tool-registration decorator/class.

**No LiteLLM, anywhere** — including for CrewAI (which now ships a
LiteLLM-free native provider path, see `crewai_agent.py`'s docstring) and for
the Claude Agent SDK, which normally requires a LiteLLM proxy to talk to
non-Anthropic models. Instead, `anthropic_agent_sdk/nebius_anthropic_proxy.py`
is a small hand-rolled FastAPI shim that translates Anthropic's Messages API
into Nebius's OpenAI-compatible API using the official `openai` SDK.

## Setup

```bash
cp .env.example .env   # fill in NEBIUS_API_KEY, SERPER_API_KEY, etc.
pip install -r requirements.txt   # or just the section(s) you need
```

Then export the vars (or use `python-dotenv` / `direnv`) and run any script:

```bash
python langchain_agent.py "What's the latest news about Mars rovers?"
python crewai_agent.py "Who won the last F1 race?"
...
```

For the Claude Agent SDK example, start the bridge proxy first:

```bash
cd anthropic_agent_sdk
uvicorn nebius_anthropic_proxy:app --port 8317 &
python anthropic_agent.py "your question"
```

## Notes

- `common/config.py` centralizes the Nebius `base_url` / `api_key` /
  `model` so every script picks them up consistently.
- Pick any Nebius model id that supports tool/function calling -- see
  `https://docs.tokenfactory.nebius.com/ai-models-inference/overview`.
- These scripts are written directly from each framework's current official
  docs (linked in each file's docstring) but have not been live-tested
  end-to-end against real Nebius/Serper accounts in this environment (no
  network egress to those domains here) -- double check error messages
  against the linked docs if something doesn't match your installed
  package version.
