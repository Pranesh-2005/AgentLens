"""REST API for span ingestion and dashboard queries."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..events import Span, SpanKind
from . import pricing


class EventBatch(BaseModel):
    events: List[Dict[str, Any]]


class GenerateRequest(BaseModel):
    provider: str = "anthropic"
    model: Optional[str] = None
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    project: str = "default"
    max_tokens: int = 1024


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    def store(request: Request):
        return request.app.state.store

    @router.post("/events")
    def ingest(batch: EventBatch, request: Request):
        return {"ingested": store(request).ingest_events(batch.events)}

    @router.get("/projects")
    def projects(request: Request):
        return store(request).list_projects()

    @router.get("/overview")
    def overview(request: Request, project: Optional[str] = None):
        return store(request).overview(project)

    @router.get("/runs")
    def runs(request: Request, project: Optional[str] = None, limit: int = 200):
        return store(request).list_runs(project, limit)

    @router.get("/runs/{trace_id}")
    def run_detail(trace_id: str, request: Request):
        r = store(request).get_run(trace_id)
        if r is None:
            raise HTTPException(404, "run not found")
        return r

    @router.get("/tools")
    def tools(request: Request, project: str):
        return store(request).tool_stats(project)

    @router.get("/memory")
    def memory(request: Request, project: str):
        return store(request).memory_stats(project)

    @router.get("/failures")
    def failures(request: Request, project: str, limit: int = 200):
        return store(request).failure_stats(project, limit)

    @router.get("/agents")
    def agents(request: Request, project: str):
        return store(request).agent_stats(project)

    @router.get("/costs")
    def costs(request: Request, project: str):
        return store(request).cost_summary(project)

    @router.get("/llm-calls")
    def llm_calls(request: Request, project: str, limit: int = 200):
        return store(request).list_llm_calls(project, limit)

    @router.get("/providers")
    def providers():
        from . import llm
        return llm.available_providers()

    @router.post("/generate")
    def generate(req: GenerateRequest, request: Request):
        """Optional replay: run an LLM over an ad-hoc prompt, log it as an llm
        span, and return the answer with computed cost."""
        from . import llm

        if not req.prompt:
            raise HTTPException(400, "prompt is required")
        t0 = time.time()
        try:
            result = llm.generate(req.provider, req.model or "", req.prompt,
                                  req.system_prompt, req.max_tokens)
        except llm.ProviderError as e:
            raise HTTPException(502, str(e))
        duration_ms = (time.time() - t0) * 1000.0
        cost = pricing.estimate_cost(result["model"], result.get("input_tokens"),
                                     result.get("output_tokens"))
        ev = Span(
            project=req.project, kind=SpanKind.LLM.value, name="llm (replay)",
            start_time=t0, end_time=t0 + duration_ms / 1000.0, duration_ms=duration_ms,
            attributes={"model": result["model"], "provider": result["provider"],
                        "prompt": req.prompt, "system_prompt": req.system_prompt,
                        "response": result["text"], "input_tokens": result.get("input_tokens"),
                        "output_tokens": result.get("output_tokens"), "cost": cost, "replayed": True},
        )
        store(request).ingest_events([ev.model_dump()])
        return {"trace_id": ev.trace_id, "model": result["model"], "provider": result["provider"],
                "response": result["text"], "cost": cost, "duration_ms": duration_ms,
                "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens")}

    return router
