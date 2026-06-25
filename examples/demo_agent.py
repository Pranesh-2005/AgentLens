"""End-to-end AgentLens demo — no API keys, no frameworks required.

Simulates a research agent: planner decides, calls a (sometimes failing) search
tool with retries, reads memory, calls an LLM, and answers. Reproduces the
shapes from the AgentScope spec's Few-Shot examples.

Run::

    python examples/demo_agent.py
    agentlens ui            # then open http://127.0.0.1:7180
"""
import random
import time

import agentlens

agentlens.init(project="research-agent")

QUERIES = ["Find latest AI news", "Latest stock price of NVDA", "Summarize today's headlines"]


def flaky_search(q: str):
    """Tool that fails the first time ~half the runs, then succeeds (with retry)."""
    retries = 0
    while True:
        time.sleep(random.uniform(0.02, 0.12))
        if retries < 1 and random.random() < 0.5:
            retries += 1
            continue  # simulate a transient ConnectionError + retry
        return [{"title": f"Result for {q}", "score": 0.9}], retries


def run_once(query: str):
    with agentlens.trace_session("research"):
        with agentlens.trace_agent("planner", role="orchestrator"):
            agentlens.log_decision(options=["search", "answer_directly"], chosen="search",
                                   reason="needs fresh info")
            # memory read
            hit = random.random() < 0.7
            agentlens.log_memory("read", "user_preferences", hit=hit)

            # tool call with retry; occasionally fully fails
            t0 = time.time()
            try:
                if random.random() < 0.15:
                    raise ConnectionError("search backend unreachable")
                results, retries = flaky_search(query)
                agentlens.log_tool("web_search", args={"query": query}, result=results,
                                   retry_count=retries, duration_ms=(time.time() - t0) * 1000)
            except ConnectionError as e:
                agentlens.log_tool("web_search", args={"query": query}, error=e, retry_count=2,
                                   duration_ms=(time.time() - t0) * 1000)
                results = []

            # memory write
            agentlens.log_memory("write", "last_query", value=query)

            # llm call (cost auto-priced from the model)
            agentlens.log_llm(model=random.choice(["gpt-4o", "claude-opus-4-8", "gpt-4o-mini"]),
                              prompt=f"Answer using: {results}",
                              response="Here is what I found ...",
                              input_tokens=random.randint(800, 1400),
                              output_tokens=random.randint(200, 500),
                              duration_ms=random.uniform(600, 1900))


if __name__ == "__main__":
    for i in range(25):
        run_once(random.choice(QUERIES))
    agentlens.flush()
    print("Logged 25 runs. Now run:  agentlens ui   ->  http://127.0.0.1:7180")