"""
nebius_anthropic_proxy.py
==========================

The Claude Agent SDK (and the Claude Code CLI it runs on top of) only ever
speaks Anthropic's native Messages API (`POST /v1/messages`) -- pointing
`ANTHROPIC_BASE_URL` at a plain OpenAI-compatible host (like Nebius Token
Factory) does NOT work, because the request/response shapes are different.

Every public guide that bridges "Claude Agent SDK + a non-Anthropic model"
does so by putting a LiteLLM proxy in front of it. Since this project
explicitly avoids LiteLLM, this file is a small, dependency-light, *hand
rolled* translation layer instead:

    Claude Agent SDK  --(Anthropic /v1/messages)-->  this proxy
                                                          |
                                                          v  (translate request)
                                                  Nebius Token Factory
                                                  (OpenAI /chat/completions,
                                                   via the official `openai`
                                                   Python SDK -- NOT litellm)
                                                          |
                                                          v  (translate response)
    Claude Agent SDK  <--(Anthropic /v1/messages)--  this proxy

It supports: system prompts, multi-turn history, tool definitions, tool_use /
tool_result blocks (so the SDK MCP `@tool` functions keep working), and both
streaming and non-streaming responses (streaming is "chunked" -- the full
Nebius completion is fetched first, then re-emitted as Anthropic SSE events,
since Nebius's OpenAI-compatible stream deltas don't need to be forwarded
token-by-token for this to work correctly).

This is a minimal compatibility shim for demos/dev use, not a
production-grade gateway (no retries, batching, prompt caching, etc).

Run it:
    pip install fastapi uvicorn openai
    NEBIUS_API_KEY=... uvicorn nebius_anthropic_proxy:app --port 8317

Then point the Claude Agent SDK at it (see anthropic_agent.py in this folder).
"""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL  # noqa: E402

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI

app = FastAPI(title="Nebius -> Anthropic Messages API bridge (no litellm)")
nebius_client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)


# --------------------------------------------------------------------------
# Anthropic request -> OpenAI/Nebius request
# --------------------------------------------------------------------------
def anthropic_to_openai_messages(body: dict) -> list[dict]:
    messages: list[dict] = []

    system = body.get("system")
    if system:
        if isinstance(system, list):
            system = "\n".join(
                b.get("text", "") for b in system if isinstance(b, dict)
            )
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif btype == "tool_result":
                # Anthropic puts tool results in a "user" message; OpenAI
                # wants a dedicated "tool" role message instead.
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = "\n".join(
                        c.get("text", "") for c in result_content if isinstance(c, dict)
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": result_content,
                    }
                )

        if text_parts or tool_calls:
            oa_msg: dict[str, Any] = {"role": role, "content": "\n".join(text_parts)}
            if tool_calls:
                oa_msg["tool_calls"] = tool_calls
            messages.append(oa_msg)

    return messages


def anthropic_to_openai_tools(body: dict) -> list[dict] | None:
    tools = body.get("tools")
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


# --------------------------------------------------------------------------
# OpenAI/Nebius response -> Anthropic response
# --------------------------------------------------------------------------
FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def openai_to_anthropic_response(completion, model: str) -> dict:
    choice = completion.choices[0]
    message = choice.message
    content: list[dict] = []

    if message.content:
        content.append({"type": "text", "text": message.content})

    for tc in message.tool_calls or []:
        try:
            tool_input = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            tool_input = {}
        content.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": tool_input,
            }
        )

    usage = completion.usage
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": FINISH_REASON_MAP.get(choice.finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        },
    }


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_anthropic_events(anthropic_msg: dict):
    """Re-emit an already-complete Anthropic message as an SSE stream, since
    we fetch the full completion from Nebius before translating (Nebius's
    raw token deltas are OpenAI-shaped and not worth re-streaming token by
    token through a hand-rolled translator for this demo)."""
    msg_id = anthropic_msg["id"]
    yield sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {**anthropic_msg, "content": [], "stop_reason": None},
        },
    )

    for idx, block in enumerate(anthropic_msg["content"]):
        yield sse_event(
            "content_block_start",
            {"type": "content_block_start", "index": idx, "content_block": block},
        )
        if block["type"] == "text":
            yield sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
            )
        yield sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": idx}
        )

    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": anthropic_msg["stop_reason"],
                "stop_sequence": None,
            },
            "usage": anthropic_msg["usage"],
        },
    )
    yield sse_event("message_stop", {"type": "message_stop"})


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()

    oa_messages = anthropic_to_openai_messages(body)
    oa_tools = anthropic_to_openai_tools(body)

    kwargs: dict[str, Any] = {
        "model": NEBIUS_MODEL,
        "messages": oa_messages,
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 1.0),
    }
    if oa_tools:
        kwargs["tools"] = oa_tools

    completion = nebius_client.chat.completions.create(**kwargs)
    anthropic_msg = openai_to_anthropic_response(completion, body.get("model", NEBIUS_MODEL))

    if body.get("stream"):
        return StreamingResponse(
            stream_anthropic_events(anthropic_msg), media_type="text/event-stream"
        )
    return JSONResponse(anthropic_msg)


@app.get("/v1/models")
async def list_models():
    # Lets Claude Code's gateway-model-discovery populate its /model picker.
    return {"data": [{"id": NEBIUS_MODEL, "display_name": NEBIUS_MODEL}]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8317)
