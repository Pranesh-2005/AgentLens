"""
CrewAI search agent.

LLM:   Nebius Token Factory, via CrewAI's NATIVE OpenAI-compatible provider
       (model id prefixed "openai/" + a custom base_url). This deliberately
       avoids LiteLLM -- CrewAI's litellm-free path, documented at:
       https://docs.crewai.com/en/learn/litellm-removal-guide
       Install the native extra (NOT the litellm extra):
           pip install 'crewai[openai]'
Tool:  Serper.dev wrapped with the `@tool` decorator from `crewai.tools`.
       Docs: https://docs.crewai.com/en/concepts/tools
Agent: A single CrewAI `Agent` + `Task` run through a `Crew`.
       Docs: https://docs.crewai.com/en/concepts/agents

Install:
    pip install 'crewai[openai]' requests
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402
from serper_search import serper_search  # noqa: E402

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool


@tool("Web Search")
def web_search(query: str) -> str:
    """Search the live web via Serper.dev (Google Search API) for current
    information, news, or facts the model may not already know."""
    return serper_search(query)


# Native (non-LiteLLM) provider: pass `provider="openai"` explicitly +
# `base_url`/`api_key`. CrewAI then talks to Nebius's OpenAI-compatible
# endpoint directly via the `openai` Python SDK extra, never touching
# litellm.
#
# Note: CrewAI's `LLM(model="openai/<model>")` shorthand only routes
# natively if "<model>" matches one of CrewAI's known OpenAI model-name
# constants -- arbitrary third-party model ids (like Nebius's, which are
# themselves "org/model" style, e.g. "meta-llama/Meta-Llama-3.1-70B-Instruct")
# fail that check and silently fall back to requiring LiteLLM. Passing
# `provider="openai"` explicitly skips that whitelist check and forces the
# native OpenAI-SDK path unconditionally -- confirmed against crewai 1.14.7.
llm = LLM(
    model=NEBIUS_MODEL,
    provider="openai",
    base_url=NEBIUS_BASE_URL,
    api_key=NEBIUS_API_KEY,
    temperature=0.2,
)

researcher = Agent(
    role="Web Research Specialist",
    goal="Find accurate, up-to-date answers to the user's question using web search.",
    backstory="A meticulous researcher who always verifies facts with a live search.",
    tools=[web_search],
    llm=llm,
    verbose=True,
)


def run(query: str) -> str:
    task = Task(
        description=f"Research and answer this question thoroughly: {query}",
        expected_output="A clear, well-sourced answer with key facts.",
        agent=researcher,
    )
    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)
    return str(crew.kickoff())


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the latest developments in fusion energy?"
    print(run(q))
