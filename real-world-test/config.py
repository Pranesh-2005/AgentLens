"""
Shared Nebius Token Factory connection settings.

Nebius Token Factory exposes an OpenAI-compatible Chat Completions API:
  https://docs.tokenfactory.nebius.com/quickstart

    base_url = "https://api.tokenfactory.nebius.com/v1/"
    api_key  = <NEBIUS_API_KEY>

Pick any chat/instruct model id that supports tool/function calling.
See: https://docs.tokenfactory.nebius.com/ai-models-inference/overview
"""

import os
from dotenv import load_dotenv

load_dotenv()

NEBIUS_BASE_URL = os.environ.get(
    "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
)
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")

# Any Nebius Token Factory model id that supports function/tool calling.
# e.g. "meta-llama/Meta-Llama-3.1-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct",
# "deepseek-ai/DeepSeek-V3-0324" ...
NEBIUS_MODEL = os.environ.get(
    "NEBIUS_MODEL", "nvidia/Nemotron-3-Nano-Omni"
)
