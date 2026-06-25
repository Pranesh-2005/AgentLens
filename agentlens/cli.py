"""``agentlens`` command line interface (``agentlens ui``, like ``mlflow ui``)."""
from __future__ import annotations

import argparse
import os

# Dedicated default port for the AgentLens dashboard (overridable via --port or
# the AGENTLENS_PORT env var). 7180 is uncommon and kept distinct from common
# dev servers (8000/8080) and MLflow (5000) so it runs alongside them cleanly.
DEFAULT_PORT = int(os.getenv("AGENTLENS_PORT", "7180"))
DEFAULT_HOST = os.getenv("AGENTLENS_HOST", "127.0.0.1")
DEFAULT_STORE = os.getenv("AGENTLENS_STORE")  # None -> hidden ./.agentlens/agentlens.db


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentlens",
                                     description="AgentLens — diagnostics & observability for AI agents")
    sub = parser.add_subparsers(dest="command")

    ui = sub.add_parser("ui", help="start the AgentLens dashboard")
    ui.add_argument("--host", default=DEFAULT_HOST)
    ui.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"dashboard port (default: {DEFAULT_PORT}, or $AGENTLENS_PORT)")
    ui.add_argument("--backend-store-uri", default=DEFAULT_STORE,
                    help="path to the SQLite database (default: hidden ./.agentlens/agentlens.db)")

    sub.add_parser("providers", help="list LLM providers available for replay")
    sub.add_parser("version", help="print version")

    args = parser.parse_args(argv)
    if args.command == "version":
        from . import __version__
        print(f"agentlens {__version__}")
        return 0
    if args.command == "providers":
        from .server import llm
        for p in llm.available_providers():
            mark = "✓" if p["ready"] else "·"
            print(f"  {mark} {p['id']:<12} {p['label']:<22} {'ready' if p['ready'] else p['hint']}")
        return 0
    if args.command == "ui":
        import uvicorn

        from .server.app import create_app
        from .storage import resolve

        store = resolve(args.backend_store_uri)
        app = create_app(store)
        print(f"AgentLens UI: http://{args.host}:{args.port}  (store: {store})")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
