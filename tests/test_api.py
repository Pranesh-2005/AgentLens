from fastapi.testclient import TestClient

from agentlens.events import Span, SpanKind
from agentlens.server.app import create_app


def _client(tmp_path):
    app = create_app(str(tmp_path / "agentlens.db"))
    return TestClient(app)


def test_ingest_and_query(tmp_path):
    c = _client(tmp_path)
    span = Span(project="p", trace_id="t1", kind=SpanKind.LLM.value,
                attributes={"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50}).finish()
    r = c.post("/api/events", json={"events": [span.model_dump()]})
    assert r.status_code == 200 and r.json()["ingested"] == 1

    assert c.get("/api/projects").json()[0]["name"] == "p"
    ov = c.get("/api/overview?project=p").json()
    assert ov["total_runs"] == 1
    runs = c.get("/api/runs?project=p").json()
    assert runs[0]["trace_id"] == "t1"
    detail = c.get("/api/runs/t1").json()
    assert detail["run"]["trace_id"] == "t1"
    costs = c.get("/api/costs?project=p").json()
    assert costs["totals"]["generations"] == 1


def test_pages_render(tmp_path):
    c = _client(tmp_path)
    for path in ["/", "/runs", "/agents", "/tools", "/memory", "/failures", "/cost"]:
        assert c.get(path).status_code == 200