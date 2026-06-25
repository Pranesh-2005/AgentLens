from agentlens.events import Span, SpanKind, estimate_tokens, jsonable, normalize_result


def test_span_finish_sets_duration_and_status():
    s = Span(kind=SpanKind.TOOL.value, name="t")
    s.finish("error")
    assert s.status == "error"
    assert s.duration_ms is not None and s.duration_ms >= 0


def test_jsonable_handles_objects():
    class Foo:
        def __init__(self): self.x = 1
    assert jsonable({"a": [1, 2]}) == {"a": [1, 2]}
    assert isinstance(jsonable(Foo()), str)  # repr fallback
    assert jsonable("hi") == "hi"
    assert jsonable(None) is None


def test_jsonable_uses_model_dump():
    from pydantic import BaseModel

    class M(BaseModel):
        a: int = 5
    assert jsonable(M()) == {"a": 5}


def test_normalize_result_str_and_dict():
    assert normalize_result("hello") == {"text": "hello"}
    assert normalize_result({"chunk_id": "x", "text": "t"}) == {"id": "x", "text": "t"}


def test_estimate_tokens():
    assert estimate_tokens("a" * 8) == 2
    assert estimate_tokens("") == 1