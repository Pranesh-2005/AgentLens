"use strict";
// AgentLens dashboard — vanilla JS, no frameworks. One file, page dispatch.

const B = document.body;
const PAGE = B.dataset.page || "overview";
const TRACE_ID = B.dataset.traceId || "";
const app = document.getElementById("app");

// ---- helpers ----
const $ = (sel, el = document) => el.querySelector(sel);
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, c => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = (m) => m == null ? "—" : (m < 1000 ? `${m.toFixed(0)}ms` : `${(m / 1000).toFixed(2)}s`);
const fmtCost = (c) => c == null ? "—" : `$${Number(c).toFixed(4)}`;
const fmtPct = (p) => p == null ? "—" : `${(p * 100).toFixed(0)}%`;
const fmtNum = (n) => n == null ? "—" : Number(n).toLocaleString();
const ago = (t) => { if (!t) return "—"; const s = Date.now() / 1000 - t; if (s < 60) return `${s | 0}s ago`;
  if (s < 3600) return `${s / 60 | 0}m ago`; if (s < 86400) return `${s / 3600 | 0}h ago`; return `${s / 86400 | 0}d ago`; };
async function api(path) { const r = await fetch("/api" + path); if (!r.ok) throw new Error(r.status); return r.json(); }
function setApp(html) { app.innerHTML = html; }
function project() {
  // URL wins, then this page's server-rendered value, then the last project the
  // user picked (persisted) — so switching tabs never silently jumps to projects[0].
  let p = new URLSearchParams(location.search).get("project") || B.dataset.project || "";
  if (!p) { try { p = localStorage.getItem("al_project") || ""; } catch (e) {} }
  return p;
}

// ---- project selector + nav ----
async function initChrome() {
  const sel = $("#projectSelect");
  document.querySelectorAll("[data-nav]").forEach(a => {
    if (a.dataset.nav === PAGE) a.classList.add("active");
  });
  let projects = [];
  try { projects = await api("/projects"); } catch (e) {}
  const names = projects.map(p => p.name);
  let cur = project();
  if (cur && !names.includes(cur)) cur = "";   // remembered project no longer exists
  sel.innerHTML = projects.map(p =>
    `<option value="${esc(p.name)}" ${p.name === cur ? "selected" : ""}>${esc(p.name)} (${p.runs})</option>`).join("")
    || `<option value="">no data</option>`;
  // sync URL + nav links + storage to the resolved project (falls back to the
  // first project only when nothing is remembered) — keeps every tab consistent.
  if (cur) navProject(cur, false);
  else if (projects.length) navProject(projects[0].name, false);
  sel.onchange = () => navProject(sel.value, true);
}
function navProject(name, reload) {
  B.dataset.project = name;
  try { localStorage.setItem("al_project", name); } catch (e) {}
  const u = new URL(location);
  u.searchParams.set("project", name);
  history.replaceState({}, "", u);
  document.querySelectorAll("[data-nav]").forEach(a => {
    const base = a.getAttribute("href").split("?")[0];
    a.href = `${base}?project=${encodeURIComponent(name)}`;
  });
  if (reload) render();
}

// ---- pages ----
const kindBadge = (k) => `<span class="badge kind-${esc(k)}">${esc(k)}</span>`;
const statusBadge = (s) => `<span class="badge ${s === "error" ? "error" : "ok"}">${esc(s || "ok")}</span>`;

async function renderOverview() {
  const p = project();
  const o = await api(`/overview?project=${encodeURIComponent(p)}`);
  setApp(`
    <h1>Overview</h1><div class="muted small">${esc(p) || "all projects"}</div>
    <div class="cards">
      <div class="card accent"><div class="label">Total Runs</div><div class="big">${fmtNum(o.total_runs)}</div></div>
      <div class="card good"><div class="label">Success Rate</div><div class="big">${fmtPct(o.success_rate)}</div></div>
      <div class="card bad"><div class="label">Failure Rate</div><div class="big">${fmtPct(o.failure_rate)}</div></div>
      <div class="card"><div class="label">Total Cost</div><div class="big">${fmtCost(o.total_cost)}</div></div>
      <div class="card"><div class="label">Latency p50 / p95</div><div class="big">${fmtMs(o.p50_latency_ms)}</div>
        <div class="muted small">p95 ${fmtMs(o.p95_latency_ms)}</div></div>
      <div class="card"><div class="label">Tool Calls</div><div class="big">${fmtNum(o.tool_calls)}</div></div>
      <div class="card"><div class="label">Failures</div><div class="big">${fmtNum(o.failures)}</div></div>
    </div>
    <h2>Recent runs</h2><div id="recent"><div class="empty">…</div></div>`);
  const runs = await api(`/runs?project=${encodeURIComponent(p)}&limit=10`);
  $("#recent").replaceWith(runsTable(runs));
}

function runsTable(runs) {
  if (!runs.length) return el(`<div class="empty">No runs yet. Instrument your agent and run it.</div>`);
  const rows = runs.map(r => `
    <tr class="rowlink" data-href="/runs/${esc(r.trace_id)}?project=${encodeURIComponent(r.project)}">
      <td>${statusBadge(r.status)}</td>
      <td>${esc(r.name || "run")}<div class="muted small mono preview">${esc((r.input || "").replace(/\s+/g, " ").trim().slice(0, 80))}</div></td>
      <td class="num">${fmtNum(r.agents)}</td>
      <td class="num">${fmtNum(r.tool_calls)}</td>
      <td class="num">${fmtNum(r.llm_calls)}</td>
      <td class="num">${r.failures ? `<span class="badge error">${r.failures}</span>` : "0"}</td>
      <td class="num">${fmtMs(r.duration_ms)}</td>
      <td class="num">${fmtCost(r.cost)}</td>
      <td class="num muted">${ago(r.start_time)}</td>
    </tr>`).join("");
  const t = el(`<table class="data"><thead><tr>
    <th>Status</th><th>Run</th><th class="num">Agents</th><th class="num">Tools</th><th class="num">LLM</th><th class="num">Fails</th>
    <th class="num">Duration</th><th class="num">Cost</th><th class="num">When</th></tr></thead><tbody>${rows}</tbody></table>`);
  t.querySelectorAll(".rowlink").forEach(tr => tr.onclick = () => location.href = tr.dataset.href);
  return t;
}

async function renderRuns() {
  const p = project();
  setApp(`<h1>Runs</h1><div id="runs"><div class="empty">…</div></div>`);
  const runs = await api(`/runs?project=${encodeURIComponent(p)}&limit=200`);
  $("#runs").replaceWith(runsTable(runs));
}

async function renderRunDetail() {
  const d = await api(`/runs/${encodeURIComponent(TRACE_ID)}`);
  const r = d.run;
  setApp(`
    <a href="/runs?project=${encodeURIComponent(r.project)}" class="small">← runs</a>
    <h1>${esc(r.name || "run")} ${statusBadge(r.status)}</h1>
    <div class="kv" style="margin-top:10px">
      <dt>trace</dt><dd>${esc(r.trace_id)}</dd>
      <dt>duration</dt><dd>${fmtMs(r.duration_ms)}</dd>
      <dt>input</dt><dd>${esc(r.input || "—")}</dd>
    </div>
    <h2>Workflow path</h2>
    <div class="cols">
      <div class="tree" id="tree"></div>
      <div class="detail-panel" id="spanDetail"><div class="muted small">Select a node to inspect.</div></div>
    </div>`);
  const tree = $("#tree");
  const spansById = {};
  d.spans.forEach(s => spansById[s.span_id] = s);
  function node(n, root) {
    const err = n.status === "error";
    const wrap = el(`<div class="tnode ${root ? "root" : ""}"></div>`);
    const row = el(`<div class="trow ${err ? "err" : ""}">
      ${kindBadge(n.kind)}<span class="tname">${esc(n.name || n.kind)}</span>
      <span class="tdur">${fmtMs(n.duration_ms)}</span></div>`);
    row.onclick = (e) => { e.stopPropagation(); showSpan(n); document.querySelectorAll(".trow").forEach(x => x.classList.remove("selected")); row.classList.add("selected"); };
    wrap.appendChild(row);
    (n.children || []).forEach(c => wrap.appendChild(node(c, false)));
    return wrap;
  }
  if (!d.tree.length) tree.innerHTML = `<div class="empty">No spans.</div>`;
  d.tree.forEach(n => tree.appendChild(node(n, true)));
  function showSpan(s) {
    const a = s.attributes || {};
    $("#spanDetail").innerHTML = `
      <div>${kindBadge(s.kind)} <strong>${esc(s.name)}</strong> ${statusBadge(s.status)}</div>
      <div class="muted small" style="margin:6px 0">${fmtMs(s.duration_ms)}${a.cost != null ? " · " + fmtCost(a.cost) : ""}</div>
      <pre class="block">${esc(JSON.stringify(a, null, 2))}</pre>`;
  }
}

async function renderAgents() {
  const p = project();
  const ags = await api(`/agents?project=${encodeURIComponent(p)}`);
  const rows = ags.map(a => `<tr>
    <td>${esc(a.agent_name)}</td><td class="num">${fmtNum(a.runs)}</td>
    <td class="num">${a.failures ? `<span class="badge error">${a.failures}</span>` : "0"}</td>
    <td class="num">${fmtMs(a.avg_ms)}</td><td class="num">${fmtCost(a.cost)}</td></tr>`).join("");
  setApp(`<h1>Agent Explorer</h1>` + (ags.length ?
    `<table class="data"><thead><tr><th>Agent</th><th class="num">Runs</th><th class="num">Failures</th><th class="num">Avg latency</th><th class="num">Cost</th></tr></thead><tbody>${rows}</tbody></table>`
    : `<div class="empty">No agent spans. Wrap agents with <code>trace_agent</code> or an adapter.</div>`));
}

function barChart(items, label, value, cls) {
  const max = Math.max(1, ...items.map(value));
  return `<div class="bars">${items.map(i => `
    <div class="bar-row"><span>${esc(label(i))}</span>
      <div class="bar-track"><div class="bar-fill ${cls || ""}" style="width:${(value(i) / max * 100).toFixed(1)}%"></div></div>
      <span class="num">${fmtNum(value(i))}</span></div>`).join("")}</div>`;
}

async function renderTools() {
  const p = project();
  const t = await api(`/tools?project=${encodeURIComponent(p)}`);
  if (!t.tools.length) return setApp(`<h1>Tool Explorer</h1><div class="empty">No tool calls logged.</div>`);
  const rows = t.tools.map(x => `<tr>
    <td>${esc(x.tool_name)}</td><td class="num">${fmtNum(x.calls)}</td>
    <td class="num">${x.failures ? `<span class="badge error">${x.failures}</span>` : "0"}</td>
    <td class="num">${fmtNum(x.retries)}</td><td class="num">${fmtMs(x.avg_ms)}</td><td class="num">${fmtMs(x.max_ms)}</td></tr>`).join("");
  setApp(`<h1>Tool Explorer</h1>
    <div class="cols">
      <div class="panel"><h3>Most used</h3>${barChart(t.most_used, x => x.tool_name, x => x.calls)}</div>
      <div class="panel"><h3>Slowest (avg ms)</h3>${barChart(t.slowest, x => x.tool_name, x => Math.round(x.avg_ms || 0), "amber")}</div>
    </div>
    <h2>All tools</h2>
    <table class="data"><thead><tr><th>Tool</th><th class="num">Calls</th><th class="num">Failures</th><th class="num">Retries</th><th class="num">Avg</th><th class="num">Max</th></tr></thead>
    <tbody>${rows}</tbody></table>`);
}

async function renderMemory() {
  const p = project();
  const m = await api(`/memory?project=${encodeURIComponent(p)}`);
  const keys = (m.top_keys || []).map(k => `<tr><td>${esc(k.memory_key)}</td><td class="num">${fmtNum(k.n)}</td></tr>`).join("");
  setApp(`<h1>Memory Explorer</h1>
    <div class="cards">
      <div class="card"><div class="label">Reads</div><div class="big">${fmtNum(m.reads)}</div></div>
      <div class="card"><div class="label">Writes</div><div class="big">${fmtNum(m.writes)}</div></div>
      <div class="card"><div class="label">Updates</div><div class="big">${fmtNum(m.updates)}</div></div>
      <div class="card"><div class="label">Deletes</div><div class="big">${fmtNum(m.deletes)}</div></div>
      <div class="card accent"><div class="label">Read hit rate</div><div class="big">${fmtPct(m.hit_rate)}</div></div>
    </div>
    <h2>Top memory keys</h2>` + (keys ?
      `<table class="data"><thead><tr><th>Key</th><th class="num">Accesses</th></tr></thead><tbody>${keys}</tbody></table>`
      : `<div class="empty">No memory operations logged.</div>`));
}

async function renderFailures() {
  const p = project();
  const f = await api(`/failures?project=${encodeURIComponent(p)}`);
  if (!f.recent.length) return setApp(`<h1>Failure Explorer</h1><div class="empty">No failures recorded. 🎉</div>`);
  const byExc = f.by_exception.map(e => `<tr><td>${esc(e.exception)}</td><td class="num">${fmtNum(e.n)}</td></tr>`).join("");
  const recent = f.recent.map(r => `<tr class="rowlink" data-href="/runs/${esc(r.trace_id)}?project=${encodeURIComponent(p)}">
    <td>${kindBadge(r.kind)}</td><td>${esc(r.name || "")}</td>
    <td class="mono">${esc(r.exception || "")}</td>
    <td class="mono preview">${esc((r.message || "").replace(/\s+/g, " ").trim().slice(0, 100))}</td>
    <td class="num">${r.retry_count != null ? r.retry_count : "—"}</td>
    <td class="num muted">${ago(r.ts)}</td></tr>`).join("");
  setApp(`<h1>Failure Explorer</h1>
    <div class="panel"><h3>By exception</h3>
      <table class="data"><thead><tr><th>Exception</th><th class="num">Count</th></tr></thead><tbody>${byExc}</tbody></table></div>
    <h2>Recent failures</h2>
    <table class="data"><thead><tr><th>Kind</th><th>Where</th><th>Exception</th><th>Message</th><th class="num">Retries</th><th class="num">When</th></tr></thead>
    <tbody>${recent}</tbody></table>`);
  app.querySelectorAll(".rowlink").forEach(tr => tr.onclick = () => location.href = tr.dataset.href);
}

async function renderCost() {
  const p = project();
  const c = await api(`/costs?project=${encodeURIComponent(p)}`);
  const t = c.totals;
  const models = c.by_model.map(m => `<tr><td>${esc(m.model)}</td><td class="num">${fmtNum(m.calls)}</td>
    <td class="num">${fmtNum(m.input_tokens)}</td><td class="num">${fmtNum(m.output_tokens)}</td><td class="num">${fmtCost(m.cost)}</td></tr>`).join("");
  const days = c.by_day;
  setApp(`<h1>Cost Explorer</h1>
    <div class="cards">
      <div class="card accent"><div class="label">Total Cost</div><div class="big">${fmtCost(t.cost)}</div></div>
      <div class="card"><div class="label">LLM Calls</div><div class="big">${fmtNum(t.generations)}</div></div>
      <div class="card"><div class="label">Total Tokens</div><div class="big">${fmtNum(t.total_tokens)}</div></div>
      <div class="card"><div class="label">Models</div><div class="big">${fmtNum(t.models)}</div></div>
    </div>
    <div class="cols">
      <div class="panel"><h3>Cost by day</h3>${days.length ? barChart(days, d => d.day, d => +d.cost.toFixed(4), "green") : '<div class="muted small">no data</div>'}</div>
      <div class="panel"><h3>Most expensive agents</h3>${(c.by_agent || []).length ? barChart(c.by_agent, a => a.agent_name, a => +(a.cost || 0).toFixed(4), "amber") : '<div class="muted small">no agent-attributed cost</div>'}</div>
    </div>
    <h2>By model</h2>` + (models ?
      `<table class="data"><thead><tr><th>Model</th><th class="num">Calls</th><th class="num">In tokens</th><th class="num">Out tokens</th><th class="num">Cost</th></tr></thead><tbody>${models}</tbody></table>`
      : `<div class="empty">No LLM calls logged.</div>`));
}

const ROUTES = {
  overview: renderOverview, runs: renderRuns, run_detail: renderRunDetail,
  agents: renderAgents, tools: renderTools, memory: renderMemory,
  failures: renderFailures, cost: renderCost,
};

async function render() {
  try { await (ROUTES[PAGE] || renderOverview)(); }
  catch (e) { setApp(`<div class="empty">Error loading view: ${esc(e.message || e)}</div>`); }
}

(async function main() {
  await initChrome();
  await render();
})();