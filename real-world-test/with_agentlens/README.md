# AgentLens-instrumented variants

Each file here is a standalone, runnable copy of the matching script in the
parent directory, with **AgentLens wired in directly** — no external runner.

Every variant does the same two things:

1. `monitor.start(project=...)` — inits the local store and auto-patches the
   OpenAI SDK, so every Nebius (OpenAI-compatible) call becomes an `llm` span.
2. attaches that framework's AgentLens adapter where one exists (LangChain /
   LangGraph callback handler, PydanticAI / Strands hooks, OpenAI-Agents trace
   processor) for richer agent / tool / workflow spans.

`_bootstrap.py` loads `../.env` (Nebius + Serper keys), pins a tool-calling
Nebius model, and puts the suite root on `sys.path`.

## Run

```bash
cd real-world-test
python with_agentlens/langchain_agent.py     "What is the latest Python release?"
python with_agentlens/langgraph_agent.py     "Who won the last F1 race?"
python with_agentlens/llamaindex_agent.py
python with_agentlens/pydanticai_agent.py
python with_agentlens/strands_agent.py
python with_agentlens/crewai_agent.py
python with_agentlens/openai_agents_sdk_agent.py
python with_agentlens/autogen_agent.py
```

Then explore the captured spans:

```bash
agentlens ui        # DB: real-world-test/.agentlens/agentlens.db
```

Each script writes to its own project (`langchain-agent`, `langgraph-agent`, …)
so you can scope the dashboard per framework.

## Notes (AgentLens-specific fixes baked in)

- **Model:** the suite default model returns empty content; these pin
  `meta-llama/Llama-3.3-70B-Instruct` (override with `NEBIUS_MODEL`).
- **OpenAI Agents SDK:** wraps the model in `OpenAIChatCompletionsModel` (the
  SDK otherwise reads `meta-llama/` as a provider prefix → `UserError`) and
  re-enables tracing so the AgentLens processor receives spans.
- **LlamaIndex 0.14:** `FunctionAgent` bypasses CallbackManager — llm spans come
  from the auto-patch.