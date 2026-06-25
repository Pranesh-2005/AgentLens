"""FastAPI app factory: serves both the REST API and the dashboard."""
from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import build_router
from .db import Store

_HERE = os.path.dirname(os.path.abspath(__file__))


def create_app(db_path: str | None = None) -> FastAPI:
    from ..storage import resolve

    app = FastAPI(title="AgentLens", docs_url="/api/docs")
    app.state.store = Store(resolve(db_path))

    app.include_router(build_router())
    app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")
    templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))

    def page(request: Request, name: str, **ctx):
        ctx["page"] = name
        return templates.TemplateResponse(request, f"{name}.html", ctx)

    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request, project: str = ""):
        return page(request, "overview", project=project)

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request, project: str = ""):
        return page(request, "runs", project=project)

    @app.get("/runs/{trace_id}", response_class=HTMLResponse)
    def run_detail(request: Request, trace_id: str):
        return page(request, "run_detail", trace_id=trace_id)

    @app.get("/agents", response_class=HTMLResponse)
    def agents(request: Request, project: str = ""):
        return page(request, "agents", project=project)

    @app.get("/tools", response_class=HTMLResponse)
    def tools(request: Request, project: str = ""):
        return page(request, "tools", project=project)

    @app.get("/memory", response_class=HTMLResponse)
    def memory(request: Request, project: str = ""):
        return page(request, "memory", project=project)

    @app.get("/failures", response_class=HTMLResponse)
    def failures(request: Request, project: str = ""):
        return page(request, "failures", project=project)

    @app.get("/cost", response_class=HTMLResponse)
    def cost(request: Request, project: str = ""):
        return page(request, "cost", project=project)

    @app.get("/docs", response_class=HTMLResponse)
    def docs():
        from .docs import guide_html
        return HTMLResponse(guide_html())

    return app
