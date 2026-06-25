"""Auto-patch tests using a fake OpenAI-shaped client (no real SDK needed)."""
from agentlens.instrument._patch import instrument_llm, revert_all


class _Usage:
    prompt_tokens = 11
    completion_tokens = 7


class _Msg:
    content = "hello"


class _Choice:
    message = _Msg()


class _Resp:
    model = "gpt-4o"
    usage = _Usage()
    choices = [_Choice()]


class FakeCompletions:
    def create(self, **kwargs):
        return _Resp()


def _parse(resp):
    return {"model": resp.model, "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens, "response": resp.choices[0].message.content}


def test_instrument_llm_emits_span(local):
    store = local
    patched = instrument_llm(FakeCompletions, "create", _parse, "openai")
    assert patched is True
    try:
        FakeCompletions().create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        calls = store.list_llm_calls("test")
        assert len(calls) == 1
        assert calls[0]["model"] == "gpt-4o"
        assert calls[0]["input_tokens"] == 11
        assert calls[0]["cost"] is not None  # backfilled
    finally:
        revert_all()


def test_patch_is_idempotent():
    assert instrument_llm(FakeCompletions, "create", _parse, "openai") is True
    assert instrument_llm(FakeCompletions, "create", _parse, "openai") is False
    revert_all()