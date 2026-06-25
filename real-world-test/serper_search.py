"""
Tiny, dependency-light wrapper around the Serper.dev Google Search API.

Every framework in this repo imports `serper_search()` from this module and
wraps it as a "tool" using that framework's own native tool-registration
mechanism (no shared abstraction is forced on the frameworks themselves).

Docs: https://serper.dev/playground  (POST https://google.serper.dev/search)
"""

from __future__ import annotations

import os
import requests

SERPER_URL = "https://google.serper.dev/search"


def serper_search(query: str, num_results: int = 5) -> str:
    """Search the web with Serper.dev (Google Search API) and return a
    compact, LLM-friendly digest of the top organic results.

    Args:
        query: The search query.
        num_results: Max number of organic results to include (default 5).

    Returns:
        A newline-separated string of "Title — Snippet (URL)" lines, or an
        error message if the request failed.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return "ERROR: SERPER_API_KEY environment variable is not set."

    try:
        resp = requests.post(
            SERPER_URL,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": num_results},
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"ERROR: Serper request failed: {exc}"

    data = resp.json()
    lines = []

    # Optional "answer box" / knowledge graph snippet, if Google has one.
    if "answerBox" in data:
        ab = data["answerBox"]
        snippet = ab.get("answer") or ab.get("snippet")
        if snippet:
            lines.append(f"Answer box: {snippet}")

    for item in data.get("organic", [])[:num_results]:
        title = item.get("title", "Untitled")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        lines.append(f"- {title}: {snippet} ({link})")

    if not lines:
        return f"No results found for query: {query!r}"

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual smoke test: `SERPER_API_KEY=... python serper_search.py "your query"`
    import sys

    q = " ".join(sys.argv[1:]) or "latest news about Claude AI"
    print(serper_search(q))
