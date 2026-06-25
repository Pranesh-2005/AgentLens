"""Pure-mapping tests — exercise adapter helpers without the frameworks installed."""
from agentlens.adapters.langchain import llm_span, tool_span
from agentlens.adapters.common import wrap_tool


def test_langchain_llm_span_shape():
    ev = llm_span(["hi"], "answer", "gpt-4o", "trace1", None, "p", 0.0,
                  token_usage={"prompt_tokens": 10, "completion_tokens": 5})
    assert ev["kind"] == "llm"
    assert ev["attributes"]["input_tokens"] == 10
    assert ev["attributes"]["response"] == "answer"


def test_langchain_tool_span_error():
    ev = tool_span("search", {"q": "x"}, None, "trace1", None, "p", 0.0, error=ValueError("nope"))
    assert ev["status"] == "error"
    assert ev["attributes"]["exception"] == "ValueError"


def test_wrap_tool_logs_and_reraises(local):
    store = local

    @wrap_tool
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    calls = store.tool_stats("test")["tools"]
    assert calls and calls[0]["calls"] == 1