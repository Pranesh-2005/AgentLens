"""Shared bootstrap for the AgentLens-instrumented variants.

Each ``with_agentlens/<framework>_agent.py`` is a standalone, runnable copy of
the matching suite script with AgentLens wired in directly. They all import this
module first to:

  * load ``real-world-test/.env`` (Nebius + Serper keys),
  * pin a Nebius model that exists and supports tool calling,
  * put the suite root on ``sys.path`` so ``config`` / ``serper_search`` and the
    original ``*_agent`` modules import cleanly.

Run any of them like the originals, e.g.::

    python with_agentlens/langchain_agent.py "What is the latest Python release?"

then view the captured spans with ``agentlens ui`` (DB: ./.agentlens/).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUITE_ROOT))

# Windows consoles default to a legacy codepage (cp1252) that can't encode
# characters the models emit (e.g. U+202F narrow no-break space), which crashes
# a plain print() of the answer. Force UTF-8 on stdout/stderr where supported.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv(SUITE_ROOT / ".env")

# The suite default model returns empty content; pin a tool-calling model.
os.environ.setdefault("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")


def query_from_argv(default: str) -> str:
    return " ".join(sys.argv[1:]) or default
