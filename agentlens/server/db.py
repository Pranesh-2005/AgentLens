"""SQLite storage for AgentLens.

Local-first by design: defaults to a hidden ``./.agentlens/agentlens.db``. Uses
stdlib sqlite3 in WAL mode with a process-wide lock; fine for the local
single-user tool this is. One ``spans`` table holds every event; denormalized
side-tables (tool_calls / memory_ops / failures) are populated at ingest time so
the Tool / Memory / Failure explorers are cheap to query.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from . import pricing

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    name TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    trace_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    name TEXT,
    session_id TEXT,
    input TEXT,
    start_time REAL,
    end_time REAL,
    status TEXT DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
CREATE TABLE IF NOT EXISTS spans (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    span_id TEXT,
    parent_span_id TEXT,
    session_id TEXT,
    project TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT,
    start_time REAL,
    end_time REAL,
    duration_ms REAL,
    status TEXT DEFAULT 'ok',
    attributes TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_project_kind ON spans(project, kind);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    status TEXT,
    duration_ms REAL,
    retry_count INTEGER,
    error TEXT,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_tc_project ON tool_calls(project, tool_name);
CREATE TABLE IF NOT EXISTS memory_ops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    op TEXT NOT NULL,
    memory_key TEXT,
    hit INTEGER,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_mo_project ON memory_ops(project, op);
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT,
    exception TEXT,
    message TEXT,
    retry_count INTEGER,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_fail_project ON failures(project);
"""


def _loads(s: Optional[str]) -> Any:
    try:
        return json.loads(s) if s else {}
    except (TypeError, ValueError):
        return {}


class Store:
    def __init__(self, path: str = "agentlens.db"):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.Error:
                pass
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------- ingest

    def ingest_events(self, events: List[Dict[str, Any]]) -> int:
        with self._lock:
            cur = self._conn.cursor()
            for ev in events:
                self._ingest_one(cur, ev)
            self._conn.commit()
        return len(events)

    def _ingest_one(self, cur: sqlite3.Cursor, ev: Dict[str, Any]) -> None:
        project = ev.get("project") or "default"
        trace_id = ev.get("trace_id") or ev.get("event_id")
        kind = ev.get("kind", "other")
        attrs = ev.get("attributes") or {}

        # backfill LLM cost from the price book when not supplied
        if kind == "llm" and attrs.get("cost") in (None, 0):
            est = pricing.estimate_cost(attrs.get("model"), attrs.get("input_tokens"),
                                        attrs.get("output_tokens"))
            if est is not None:
                attrs = {**attrs, "cost": est, "cost_estimated": True}

        cur.execute("INSERT OR IGNORE INTO projects(name, created_at) VALUES (?, ?)",
                    (project, time.time()))

        # upsert run row: extend bounds, capture an input/name from any event
        start = ev.get("start_time") or time.time()
        end = ev.get("end_time") or start
        run_input = attrs.get("input") or attrs.get("query") or attrs.get("prompt")
        name = ev.get("name") or kind
        session_id = ev.get("session_id")
        row = cur.execute("SELECT trace_id, name, input, start_time, end_time, status"
                          " FROM runs WHERE trace_id=?", (trace_id,)).fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO runs(trace_id, project, name, session_id, input, start_time, end_time, status)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (trace_id, project, name, session_id, _short(run_input), start, end, ev.get("status", "ok")),
            )
        else:
            new_start = min(row["start_time"] or start, start)
            new_end = max(row["end_time"] or end, end)
            new_input = row["input"] or _short(run_input)
            # a run is the top-level boundary; title it after a structural span
            # (session/workflow/agent) rather than a leaf llm/tool span's name.
            new_name = name if kind in ("session", "workflow", "agent") else row["name"]
            status = "error" if (row["status"] == "error" or ev.get("status") == "error") else "ok"
            cur.execute("UPDATE runs SET start_time=?, end_time=?, input=?, name=?, status=? WHERE trace_id=?",
                        (new_start, new_end, new_input, new_name, status, trace_id))

        cur.execute(
            "INSERT OR REPLACE INTO spans(event_id, trace_id, span_id, parent_span_id, session_id, project,"
            " kind, name, start_time, end_time, duration_ms, status, attributes)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ev.get("event_id"), trace_id, ev.get("span_id"), ev.get("parent_span_id"),
                session_id, project, kind, ev.get("name", ""),
                ev.get("start_time"), ev.get("end_time"), ev.get("duration_ms"),
                ev.get("status", "ok"), json.dumps(attrs, default=str),
            ),
        )
        self._extract(cur, project, trace_id, ev, kind, attrs)

    def _extract(self, cur, project, trace_id, ev, kind, attrs) -> None:
        """Fan a span into the denormalized analytics side-tables."""
        eid, ts, status = ev.get("event_id"), ev.get("start_time") or time.time(), ev.get("status", "ok")
        if kind == "tool":
            cur.execute(
                "INSERT INTO tool_calls(event_id, trace_id, project, tool_name, status, duration_ms,"
                " retry_count, error, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (eid, trace_id, project, attrs.get("tool_name") or ev.get("name") or "tool",
                 status, ev.get("duration_ms"), attrs.get("retry_count"), attrs.get("error"), ts),
            )
        elif kind == "memory":
            hit = attrs.get("hit")
            cur.execute(
                "INSERT INTO memory_ops(event_id, trace_id, project, op, memory_key, hit, ts)"
                " VALUES (?,?,?,?,?,?,?)",
                (eid, trace_id, project, attrs.get("op") or "read", attrs.get("memory_key"),
                 None if hit is None else int(bool(hit)), ts),
            )
        if status == "error" or attrs.get("error") or attrs.get("exception"):
            cur.execute(
                "INSERT INTO failures(event_id, trace_id, project, kind, name, exception, message,"
                " retry_count, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (eid, trace_id, project, kind, ev.get("name"),
                 attrs.get("exception"), attrs.get("error"), attrs.get("retry_count"), ts),
            )

    # ------------------------------------------------------------- queries

    def list_projects(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.name,
                       COUNT(r.trace_id) AS runs,
                       AVG((r.end_time - r.start_time) * 1000.0) AS avg_latency_ms,
                       MAX(r.end_time) AS last_activity
                FROM projects p LEFT JOIN runs r ON r.project = p.name
                GROUP BY p.name ORDER BY last_activity DESC
                """
            ).fetchall()
            out = []
            for r in rows:
                cost = self._conn.execute(
                    "SELECT SUM(json_extract(attributes,'$.cost')) AS c FROM spans"
                    " WHERE project=? AND kind='llm'", (r["name"],)).fetchone()
                out.append({"name": r["name"], "runs": r["runs"],
                            "avg_latency_ms": r["avg_latency_ms"],
                            "total_cost": cost["c"] or 0, "last_activity": r["last_activity"]})
            return out

    def overview(self, project: Optional[str] = None) -> Dict[str, Any]:
        where, args = ("WHERE project = ?", [project]) if project else ("", [])
        cost_where = "WHERE kind='llm'" + (" AND project=?" if project else "")
        with self._lock:
            runs = self._conn.execute(f"SELECT status, start_time, end_time FROM runs {where}", args).fetchall()
            cost = self._conn.execute(
                f"SELECT SUM(json_extract(attributes,'$.cost')) AS c FROM spans {cost_where}", args).fetchone()
            tools = self._conn.execute(f"SELECT COUNT(*) AS c FROM tool_calls {where}", args).fetchone()
            fails = self._conn.execute(f"SELECT COUNT(*) AS c FROM failures {where}", args).fetchone()
        total = len(runs)
        ok = sum(1 for r in runs if r["status"] == "ok")
        lat = sorted((r["end_time"] - r["start_time"]) * 1000.0
                     for r in runs if r["end_time"] and r["start_time"])
        return {
            "total_runs": total,
            "success": ok, "failed": total - ok,
            "success_rate": (ok / total) if total else None,
            "failure_rate": ((total - ok) / total) if total else None,
            "p50_latency_ms": _pct(lat, 0.50), "p95_latency_ms": _pct(lat, 0.95),
            "total_cost": cost["c"] or 0,
            "tool_calls": tools["c"], "failures": fails["c"],
        }

    def list_runs(self, project: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        where, args = ("WHERE r.project = ?", [project]) if project else ("", [])
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT r.*,
                  (SELECT SUM(json_extract(s.attributes,'$.cost')) FROM spans s
                     WHERE s.trace_id=r.trace_id AND s.kind='llm') AS cost,
                  (SELECT COUNT(*) FROM spans s WHERE s.trace_id=r.trace_id AND s.kind='agent') AS agents,
                  (SELECT COUNT(*) FROM spans s WHERE s.trace_id=r.trace_id AND s.kind='llm') AS llm_calls,
                  (SELECT COUNT(*) FROM tool_calls t WHERE t.trace_id=r.trace_id) AS tool_calls,
                  (SELECT COUNT(*) FROM failures f WHERE f.trace_id=r.trace_id) AS failures
                FROM runs r {where}
                ORDER BY r.start_time DESC LIMIT ?
                """,
                args + [limit],
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["duration_ms"] = ((r["end_time"] - r["start_time"]) * 1000.0
                                if r["end_time"] and r["start_time"] else None)
            out.append(d)
        return out

    def get_run(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._conn.execute("SELECT * FROM runs WHERE trace_id=?", (trace_id,)).fetchone()
            if run is None:
                return None
            spans = self._conn.execute(
                "SELECT * FROM spans WHERE trace_id=? ORDER BY start_time", (trace_id,)).fetchall()
        evs = []
        for s in spans:
            d = dict(s)
            d["attributes"] = _loads(s["attributes"])
            evs.append(d)
        run = dict(run)
        if run["end_time"] and run["start_time"]:
            run["duration_ms"] = (run["end_time"] - run["start_time"]) * 1000.0
        return {"run": run, "spans": evs, "tree": _build_tree(evs)}

    # ---------------------------------------------------- explorers

    def tool_stats(self, project: str) -> Dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tool_name,
                       COUNT(*) AS calls,
                       SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS failures,
                       AVG(duration_ms) AS avg_ms, MAX(duration_ms) AS max_ms,
                       SUM(COALESCE(retry_count,0)) AS retries
                FROM tool_calls WHERE project=?
                GROUP BY tool_name ORDER BY calls DESC
                """, (project,)).fetchall()
        tools = [dict(r) for r in rows]
        return {
            "tools": tools,
            "most_used": sorted(tools, key=lambda t: t["calls"], reverse=True)[:10],
            "slowest": sorted([t for t in tools if t["avg_ms"]], key=lambda t: t["avg_ms"], reverse=True)[:10],
            "failed": sorted([t for t in tools if t["failures"]], key=lambda t: t["failures"], reverse=True)[:10],
        }

    def memory_stats(self, project: str) -> Dict[str, Any]:
        with self._lock:
            by_op = self._conn.execute(
                "SELECT op, COUNT(*) AS n FROM memory_ops WHERE project=? GROUP BY op", (project,)).fetchall()
            hits = self._conn.execute(
                "SELECT SUM(hit) AS h, COUNT(hit) AS n FROM memory_ops"
                " WHERE project=? AND op='read' AND hit IS NOT NULL", (project,)).fetchone()
            keys = self._conn.execute(
                "SELECT memory_key, COUNT(*) AS n FROM memory_ops WHERE project=? AND memory_key IS NOT NULL"
                " GROUP BY memory_key ORDER BY n DESC LIMIT 20", (project,)).fetchall()
        counts = {r["op"]: r["n"] for r in by_op}
        return {
            "by_op": counts,
            "reads": counts.get("read", 0), "writes": counts.get("write", 0),
            "updates": counts.get("update", 0), "deletes": counts.get("delete", 0),
            "hit_rate": (hits["h"] / hits["n"]) if hits and hits["n"] else None,
            "top_keys": [dict(r) for r in keys],
        }

    def failure_stats(self, project: str, limit: int = 200) -> Dict[str, Any]:
        with self._lock:
            by_exc = self._conn.execute(
                "SELECT COALESCE(exception,'(unknown)') AS exception, COUNT(*) AS n"
                " FROM failures WHERE project=? GROUP BY exception ORDER BY n DESC", (project,)).fetchall()
            recent = self._conn.execute(
                "SELECT * FROM failures WHERE project=? ORDER BY ts DESC LIMIT ?", (project, limit)).fetchall()
        return {"by_exception": [dict(r) for r in by_exc], "recent": [dict(r) for r in recent]}

    def agent_stats(self, project: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT json_extract(attributes,'$.agent_name') AS agent_name,
                       COUNT(*) AS runs,
                       SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS failures,
                       AVG(duration_ms) AS avg_ms
                FROM spans WHERE project=? AND kind='agent'
                GROUP BY agent_name ORDER BY runs DESC
                """, (project,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["agent_name"] = d["agent_name"] or "(unnamed)"
            # cost attributable to this agent: llm spans whose parent is one of its spans
            d["cost"] = self._agent_cost(project, d["agent_name"])
            out.append(d)
        return out

    def _agent_cost(self, project: str, agent_name: str) -> float:
        row = self._conn.execute(
            """
            SELECT SUM(json_extract(l.attributes,'$.cost')) AS c
            FROM spans l JOIN spans a ON l.parent_span_id = a.span_id
            WHERE l.project=? AND l.kind='llm' AND a.kind='agent'
              AND json_extract(a.attributes,'$.agent_name')=?
            """, (project, agent_name)).fetchone()
        return row["c"] or 0.0

    def cost_summary(self, project: str) -> Dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT trace_id, name, start_time, duration_ms,
                       json_extract(attributes,'$.model')         AS model,
                       json_extract(attributes,'$.provider')      AS provider,
                       json_extract(attributes,'$.cost')          AS cost,
                       json_extract(attributes,'$.input_tokens')  AS input_tokens,
                       json_extract(attributes,'$.output_tokens') AS output_tokens
                FROM spans WHERE project=? AND kind='llm' ORDER BY start_time
                """, (project,)).fetchall()
        gens = [dict(r) for r in rows]
        total_cost = sum((g["cost"] or 0) for g in gens)
        total_in = sum((g["input_tokens"] or 0) for g in gens)
        total_out = sum((g["output_tokens"] or 0) for g in gens)
        by_model: Dict[str, Dict[str, Any]] = {}
        by_day: Dict[str, Dict[str, Any]] = {}
        for g in gens:
            model = g["model"] or "unknown"
            m = by_model.setdefault(model, {"model": model, "calls": 0, "cost": 0.0,
                                            "input_tokens": 0, "output_tokens": 0})
            m["calls"] += 1
            m["cost"] += g["cost"] or 0
            m["input_tokens"] += g["input_tokens"] or 0
            m["output_tokens"] += g["output_tokens"] or 0
            day = time.strftime("%Y-%m-%d", time.localtime(g["start_time"] or 0))
            d = by_day.setdefault(day, {"day": day, "calls": 0, "cost": 0.0,
                                        "input_tokens": 0, "output_tokens": 0})
            d["calls"] += 1
            d["cost"] += g["cost"] or 0
            d["input_tokens"] += g["input_tokens"] or 0
            d["output_tokens"] += g["output_tokens"] or 0
        models = sorted(by_model.values(), key=lambda x: x["cost"], reverse=True)
        days = sorted(by_day.values(), key=lambda x: x["day"])
        return {
            "totals": {"generations": len(gens), "cost": total_cost,
                       "input_tokens": total_in, "output_tokens": total_out,
                       "total_tokens": total_in + total_out, "models": len(by_model)},
            "by_model": models, "by_day": days,
            "by_agent": sorted(self.agent_stats(project), key=lambda a: a["cost"], reverse=True)[:10],
        }

    def list_llm_calls(self, project: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_id, trace_id, name, start_time, duration_ms, status,
                       json_extract(attributes,'$.model')         AS model,
                       json_extract(attributes,'$.provider')      AS provider,
                       json_extract(attributes,'$.cost')          AS cost,
                       json_extract(attributes,'$.input_tokens')  AS input_tokens,
                       json_extract(attributes,'$.output_tokens') AS output_tokens,
                       attributes
                FROM spans WHERE project=? AND kind='llm' ORDER BY start_time DESC LIMIT ?
                """, (project, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            attrs = _loads(d.pop("attributes"))
            d["response"] = attrs.get("response")
            d["prompt"] = attrs.get("prompt")
            out.append(d)
        return out


def _short(v: Any, n: int = 300) -> Optional[str]:
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s[:n]


def _pct(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def _build_tree(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assemble parent/child span tree for the workflow graph view."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for s in spans:
        node = dict(s)
        node["children"] = []
        by_id[s["span_id"]] = node
    roots: List[Dict[str, Any]] = []
    for s in spans:
        node = by_id[s["span_id"]]
        parent = by_id.get(s.get("parent_span_id"))
        if parent is not None and parent is not node:
            parent["children"].append(node)
        else:
            roots.append(node)
    return roots
