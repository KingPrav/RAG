"""The dashboard page: static HTML + a little vanilla JS that draws SVG.
`__DATA__` is replaced with the JSON from dashboard.py. Palette and chart
chrome follow a validated categorical palette (route = fixed color slot)."""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FastAPI Docs Assistant · cost & latency</title>
<style>
.viz-root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --good: #006300;
  --cache: #2a78d6; --fast: #eb6834; --standard: #1baf7a; --escalate: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --good: #0ca30c;
    --cache: #3987e5; --fast: #d95926; --standard: #199e70; --escalate: #c98500;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); }
.viz-root { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink);
  background: var(--page); max-width: 1180px; margin: 0 auto; padding: 28px 24px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 15px; margin: 0 0 2px; font-weight: 600; }
.sub, .note { color: var(--ink-2); font-size: 13px; }
.note { margin: 2px 0 12px; }
.banner { background: #fff4d6; color: #5c4300; border: 1px solid #f0d58a; padding: 8px 12px; border-radius: 8px;
  margin: 12px 0 0; font-size: 13px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0; }
.kpi, .card { background: var(--surface-1); border: 1px solid var(--ring); border-radius: 12px; padding: 14px 16px; }
.kpi .label { color: var(--ink-2); font-size: 12px; }
.kpi .value { font-size: 28px; font-weight: 650; letter-spacing: -0.01em; margin-top: 2px; font-variant-numeric: tabular-nums; }
.kpi .delta { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
.kpi .delta.good { color: var(--good); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 16px; margin-bottom: 16px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--ink-2); margin: 6px 0 4px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
svg { display: block; width: 100%; overflow: visible; }
svg text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg text.lbl { fill: var(--ink-2); font-size: 12px; }
svg text.val { fill: var(--ink); font-size: 12px; font-weight: 600; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.hit { fill: transparent; cursor: default; }
#tip { position: fixed; pointer-events: none; background: var(--surface-1); color: var(--ink); border: 1px solid var(--ring);
  box-shadow: 0 4px 16px rgba(0,0,0,.12); border-radius: 8px; padding: 8px 10px; font-size: 12px; opacity: 0;
  transition: opacity .08s; max-width: 280px; z-index: 10; }
#tip b { font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pill { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; color: #fff; }
.pill.escalate { color: #0b0b0b; }
.empty { color: var(--muted); padding: 30px 0; text-align: center; }
</style>
</head>
<body>
<div class="viz-root" id="root">
  <h1>FastAPI Docs Assistant: cost &amp; latency</h1>
  <div class="sub" id="period"></div>
  <div id="banner"></div>
  <div class="kpis" id="kpis"></div>

  <div class="grid2">
    <div class="card"><h2>Requests per day, by route</h2>
      <div class="note">Cache share should grow as the cache warms up.</div>
      <div class="legend" id="legend-daily"></div><div id="daily"></div></div>
    <div class="card"><h2>Latency by route</h2>
      <div class="note">Line from p50 to p95, end to end (embedding, cache, routing, retrieval, generation).</div>
      <div id="latency"></div></div>
  </div>
  <div class="grid2">
    <div class="card"><h2>Cost per 1,000 questions, by route</h2>
      <div class="note">Escalation is the expensive path, so it should stay rare.</div>
      <div id="cost"></div></div>
    <div class="card"><h2>Why questions took the standard path</h2>
      <div class="note">The router conditions that blocked the fast path most often.</div>
      <div id="reasons"></div></div>
  </div>
  <div class="grid2">
    <div class="card"><h2>Cache misses: similarity to the nearest cached question</h2>
      <div class="note" id="hist-note"></div><div id="hist"></div></div>
    <div class="card"><h2>Routing eval: pass rate vs cost</h2>
      <div class="note" id="frontier-note"></div><div id="frontier"></div></div>
  </div>
  <div class="card"><h2>Recent requests</h2><div class="note">The 25 most recent questions.</div>
    <div id="recent"></div></div>
</div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const ROUTES = ["cache", "fast", "standard", "escalate"];
const NAMES = {cache: "Cache hit", fast: "Fast", standard: "Standard", escalate: "Escalate"};
const css = v => getComputedStyle(document.getElementById("root")).getPropertyValue(v).trim();
const color = r => `var(--${r})`;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const fmtMs = ms => ms == null ? "–" : ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
const fmtUsd = v => v == null ? "–" : v >= 10 ? "$" + v.toFixed(0) : v >= 1 ? "$" + v.toFixed(2) : "$" + v.toFixed(3);
const fmtPct = v => v == null ? "–" : Math.round(v * 100) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function svg(container, h) {
  const w = container.clientWidth || 500;
  const s = el("svg", {viewBox: `0 0 ${w} ${h}`, height: h});
  container.appendChild(s);
  return [s, w];
}
const tip = $("tip");
function hover(target, html) {
  target.addEventListener("mousemove", e => {
    tip.innerHTML = html; tip.style.opacity = 1;
    const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px";
  });
  target.addEventListener("mouseleave", () => tip.style.opacity = 0);
}
function empty(id, msg) { $(id).innerHTML = `<div class="empty">${msg}</div>`; }
function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(v || 1))); return Math.ceil((v || 1) / p) * p; }

// ---------------------------------------------------------------- header + KPIs
function header() {
  const d = DATA.summary;
  if (DATA.demo) $("banner").innerHTML = '<div class="banner"><b>Demo data.</b> Synthetic traffic generated by '
    + '<code>dashboard --demo</code> to preview the layout. Run <code>python -m docs_assistant.dashboard</code> for your real traces.</div>';
  if (!d.n) { $("period").textContent = "No requests yet. Ask some questions with python -m docs_assistant.ask."; return false; }
  $("period").textContent = `${d.n.toLocaleString()} requests · ${d.first.slice(0, 10)} to ${d.last.slice(0, 10)}`
    + (d.legacy_requests ? ` · ${d.legacy_requests} from before routing (counted as standard)` : "")
    + ` · generated ${DATA.generated_at.slice(0, 16).replace("T", " ")} UTC`;
  const k = [
    ["Cost per 1k questions", fmtUsd(d.usd_per_1k), d.savings != null ? `${fmtPct(d.savings)} below standard-only (${fmtUsd(d.baseline_usd_per_1k)}, est.)` : "", d.savings > 0],
    ["p95 latency", fmtMs(d.p95_ms), d.std_p95_ms ? `standard path alone: ${fmtMs(d.std_p95_ms)}` : "", d.p95_ms < d.std_p95_ms],
    ["p50 latency", fmtMs(d.p50_ms), "", false],
    ["Cache hit rate", fmtPct(d.cache_hit_rate), `threshold ${d.cache_threshold}`, false],
    ["Routed to fast path", fmtPct(d.routes.fast.share), `escalated: ${fmtPct(d.routes.escalate.share)}`, false],
  ];
  $("kpis").innerHTML = k.map(([l, v, delta, good]) =>
    `<div class="kpi"><div class="label">${l}</div><div class="value">${v}</div><div class="delta ${good ? "good" : ""}">${delta}</div></div>`).join("");
  return true;
}

// ---------------------------------------------------------------- stacked daily bars
function daily() {
  const days = DATA.summary.daily, box = $("daily");
  $("legend-daily").innerHTML = ROUTES.map(r => `<span><i style="background:${color(r)}"></i>${NAMES[r]}</span>`).join("");
  const [s, w] = svg(box, 240), m = {l: 36, r: 8, t: 8, b: 24}, ih = 240 - m.t - m.b, iw = w - m.l - m.r;
  const max = niceMax(Math.max(...days.map(d => ROUTES.reduce((a, r) => a + d[r], 0))));
  for (let i = 0; i <= 4; i++) {
    const y = m.t + ih - ih * i / 4;
    el("line", {x1: m.l, x2: w - m.r, y1: y, y2: y, class: i ? "gridline" : "axis"}, s);
    el("text", {x: m.l - 6, y: y + 4, "text-anchor": "end"}, s).textContent = Math.round(max * i / 4);
  }
  const bw = iw / days.length, barW = Math.max(3, Math.min(28, bw - 4));
  days.forEach((d, i) => {
    let y = m.t + ih; const x = m.l + i * bw + (bw - barW) / 2, total = ROUTES.reduce((a, r) => a + d[r], 0);
    ROUTES.forEach((r, j) => {
      const h = ih * d[r] / max; if (h <= 0) return;
      const gap = j ? 2 : 0;                                   // 2px surface gap between segments
      const top = ROUTES.slice(j + 1).every(rr => !d[rr]);
      el("rect", {x, y: y - h + (top ? 0 : 0), width: barW, height: Math.max(0, h - gap), rx: top ? 3 : 0, fill: color(r)}, s);
      y -= h;
    });
    const hit = el("rect", {x: m.l + i * bw, y: m.t, width: bw, height: ih, class: "hit"}, s);
    hover(hit, `<b>${d.day}</b> · ${total} requests<br>` + ROUTES.map(r => `${NAMES[r]}: ${d[r]} (${fmtPct(d[r] / (total || 1))})`).join("<br>"));
    const every = Math.max(1, Math.ceil(days.length * 46 / iw));        // ~46px per date label
    if (i % every === 0)
      el("text", {x: m.l + i * bw + bw / 2, y: 240 - 6, "text-anchor": "middle"}, s).textContent = d.day.slice(5);
  });
}

// ---------------------------------------------------------------- latency range per route
function latency() {
  const R = DATA.summary.routes, rows = ROUTES.filter(r => R[r].n), box = $("latency");
  if (!rows.length) return empty("latency", "No data.");
  const rowH = 46, [s, w] = svg(box, rows.length * rowH + 26), m = {l: 84, r: 70};
  const max = niceMax(Math.max(...rows.map(r => R[r].p95_ms || 0)) / 1000) * 1000;
  const x = v => m.l + (w - m.l - m.r) * v / max;
  for (let i = 0; i <= 4; i++) {
    const v = max * i / 4;
    el("line", {x1: x(v), x2: x(v), y1: 4, y2: rows.length * rowH, class: i ? "gridline" : "axis"}, s);
    el("text", {x: x(v), y: rows.length * rowH + 16, "text-anchor": "middle"}, s).textContent = fmtMs(v);
  }
  rows.forEach((r, i) => {
    const cy = i * rowH + rowH / 2, a = R[r];
    el("text", {x: m.l - 10, y: cy + 4, "text-anchor": "end", class: "lbl"}, s).textContent = NAMES[r];
    el("line", {x1: x(a.p50_ms), x2: x(a.p95_ms), y1: cy, y2: cy, stroke: color(r), "stroke-width": 2, "stroke-linecap": "round"}, s);
    el("circle", {cx: x(a.p50_ms), cy, r: 5, fill: color(r), stroke: css("--surface-1"), "stroke-width": 2}, s);
    el("circle", {cx: x(a.p95_ms), cy, r: 5, fill: css("--surface-1"), stroke: color(r), "stroke-width": 2}, s);
    el("text", {x: x(a.p95_ms) + 10, y: cy + 4, class: "val"}, s).textContent = fmtMs(a.p95_ms);
    hover(el("rect", {x: 0, y: cy - rowH / 2, width: w, height: rowH, class: "hit"}, s),
      `<b>${NAMES[r]}</b> · ${a.n} requests<br>p50 ${fmtMs(a.p50_ms)} (filled dot)<br>p95 ${fmtMs(a.p95_ms)} (open dot)`);
  });
}

// ---------------------------------------------------------------- horizontal bars
function hbars(id, items, fmt, tipFn) {
  const box = $(id); if (!items.length) return empty(id, "No data yet.");
  const longest = Math.max(...items.map(i => String(i.label).length));
  const rowH = 34, [s, w] = svg(box, items.length * rowH + 4), m = {l: Math.min(260, 24 + longest * 6.6), r: 70};
  const max = Math.max(...items.map(i => i.value)) || 1, x = v => (w - m.l - m.r) * v / max;
  el("line", {x1: m.l, x2: m.l, y1: 0, y2: items.length * rowH, class: "axis"}, s);
  items.forEach((it, i) => {
    const y = i * rowH + 8, bw = Math.max(2, x(it.value));
    el("text", {x: m.l - 10, y: y + 13, "text-anchor": "end", class: "lbl"}, s).textContent = it.label;
    el("path", {d: `M${m.l},${y} h${bw - 4} a4,4 0 0 1 4,4 v10 a4,4 0 0 1 -4,4 h-${bw - 4} z`, fill: it.color}, s);
    el("text", {x: m.l + bw + 8, y: y + 13, class: "val"}, s).textContent = fmt(it.value);
    hover(el("rect", {x: 0, y: y - 6, width: w, height: rowH, class: "hit"}, s), tipFn(it));
  });
}
function cost() {
  const R = DATA.summary.routes;
  hbars("cost", ROUTES.filter(r => R[r].n).map(r => ({label: NAMES[r], value: R[r].usd_per_1k, color: color(r), n: R[r].n})),
    fmtUsd, it => `<b>${it.label}</b><br>${fmtUsd(it.value)} per 1,000 questions<br>${it.n} requests`);
}
function reasons() {
  const rs = DATA.summary.standard_reasons;
  hbars("reasons", rs.map(([label, n]) => ({label, value: n, color: color("standard")})),
    v => String(v), it => `<b>${esc(it.label)}</b><br>${it.value} standard-path requests`);
}

// ---------------------------------------------------------------- similarity histogram
function hist() {
  const d = DATA.summary, bins = d.miss_similarity_hist, box = $("hist");
  const total = bins.reduce((a, b) => a + b.n, 0);
  $("hist-note").textContent = total
    ? `${total} misses had a cached neighbour. ${d.near_misses} fell within 0.03 below the ${d.cache_threshold} threshold: `
      + `if that pile is large, re-run calibrate-cache before loosening the threshold.`
    : "No cache misses with a cached neighbour yet.";
  if (!total) return;
  const [s, w] = svg(box, 220), m = {l: 32, r: 8, t: 8, b: 24}, ih = 220 - m.t - m.b, iw = w - m.l - m.r;
  const max = niceMax(Math.max(...bins.map(b => b.n))), bw = iw / bins.length;
  const x = v => m.l + iw * (v - bins[0].lo) / (bins[bins.length - 1].hi - bins[0].lo);
  for (let i = 0; i <= 3; i++) {
    const y = m.t + ih - ih * i / 3;
    el("line", {x1: m.l, x2: w - m.r, y1: y, y2: y, class: i ? "gridline" : "axis"}, s);
    el("text", {x: m.l - 6, y: y + 4, "text-anchor": "end"}, s).textContent = Math.round(max * i / 3);
  }
  bins.forEach((b, i) => {
    const h = ih * b.n / max, bx = m.l + i * bw + 1;
    if (h > 0) el("path", {d: `M${bx},${m.t + ih} v-${Math.max(0, h - 3)} a3,3 0 0 1 3,-3 h${bw - 8} a3,3 0 0 1 3,3 v${Math.max(0, h - 3)} z`,
      fill: b.lo >= d.cache_threshold - 0.03 ? color("cache") : css("--axis")}, s);
    hover(el("rect", {x: bx, y: m.t, width: bw, height: ih, class: "hit"}, s), `similarity ${b.lo.toFixed(2)}–${b.hi.toFixed(2)}<br><b>${b.n}</b> misses`);
    if (i % 5 === 0) el("text", {x: bx, y: 220 - 6, "text-anchor": "middle"}, s).textContent = b.lo.toFixed(2);
  });
  const tx = x(d.cache_threshold);
  el("line", {x1: tx, x2: tx, y1: m.t, y2: m.t + ih, stroke: css("--ink"), "stroke-width": 1.5, "stroke-dasharray": "4 3"}, s);
  el("text", {x: tx - 6, y: m.t + 12, "text-anchor": "end", class: "lbl"}, s).textContent = `threshold ${d.cache_threshold}`;
}

// ---------------------------------------------------------------- routing frontier (eval)
function frontier() {
  const sim = DATA.eval.simulation, box = $("frontier");
  if (!sim) { $("frontier-note").textContent = ""; return empty("frontier", "Run cost_eval paths + simulate to see the eval trade-off here."); }
  $("frontier-note").textContent = `From ${sim.run}: each dot is the cheapest router policy reaching that pass rate. `
    + `Labelled points are the reference systems.`;
  const pts = sim.frontier.map(f => ({x: f.usd_per_1k, y: f.pass_rate, f}));
  const named = Object.entries(sim.named).map(([n, r]) => ({x: r.usd_per_1k, y: r.pass_rate, n, r}));
  const all = pts.concat(named), H = 260, [s, w] = svg(box, H), m = {l: 40, r: 16, t: 16, b: 28}, ih = H - m.t - m.b, iw = w - m.l - m.r;
  // Log x: escalation costs ~20x the small model, which would squash everything else on a linear axis.
  const xs = all.map(p => Math.max(p.x, 0.01)), lo = Math.pow(10, Math.floor(Math.log10(Math.min(...xs)))),
        hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...xs) * 1.05)));
  const ymin = Math.max(0, Math.floor(Math.min(...all.map(p => p.y)) * 20) / 20 - 0.05);
  const X = v => m.l + iw * (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  const Y = v => m.t + ih - ih * (v - ymin) / (1 - ymin);
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (1 - ymin) * i / 4, y = Y(v);
    el("line", {x1: m.l, x2: w - m.r, y1: y, y2: y, class: i ? "gridline" : "axis"}, s);
    el("text", {x: m.l - 6, y: y + 4, "text-anchor": "end"}, s).textContent = fmtPct(v);
  }
  for (let d = lo; d <= hi * 1.001; d *= 10) for (const t of [1, 3]) {
    const v = d * t; if (v > hi * 1.001) continue;
    el("line", {x1: X(v), x2: X(v), y1: m.t, y2: m.t + ih, class: "gridline"}, s);
    el("text", {x: X(v), y: H - 8, "text-anchor": "middle"}, s).textContent = fmtUsd(v);
  }
  el("text", {x: w - m.r, y: m.t + ih - 6, "text-anchor": "end"}, s).textContent = "$ per 1k questions (log) →";
  const line = pts.slice().sort((a, b) => a.x - b.x);
  el("polyline", {points: line.map(p => `${X(p.x)},${Y(p.y)}`).join(" "), fill: "none", stroke: css("--axis"), "stroke-width": 2}, s);
  pts.forEach(p => {
    el("circle", {cx: X(p.x), cy: Y(p.y), r: 4, fill: css("--muted"), stroke: css("--surface-1"), "stroke-width": 2}, s);
    hover(el("circle", {cx: X(p.x), cy: Y(p.y), r: 10, class: "hit"}, s),
      `<b>${p.f.passes}/${p.f.n} passed</b> · ${fmtUsd(p.x)} per 1k<br>p95 ${fmtMs(p.f.p95_ms)}<br>fast ${fmtPct(p.f.route_mix.fast)} · escalate ${fmtPct(p.f.route_mix.escalate)}`);
  });
  const nc = {"always:standard": "standard", "always:fast": "fast", "always:escalate": "escalate", "selected policy": "cache", "fixed policy": "cache"};
  const shown = named.filter(p => nc[p.n] || p.n === "oracle").sort((a, b) => a.y - b.y || a.x - b.x);
  const placed = [];                                        // greedy label placement: try above, below, right
  shown.forEach(p => {
    const cx = X(p.x), cy = Y(p.y), fill = p.n === "oracle" ? css("--ink") : color(nc[p.n]);
    el("circle", {cx, cy, r: 6, fill, stroke: css("--surface-1"), "stroke-width": 2}, s);
    const text = p.n.replace("always:", "always "), tw = text.length * 6.6;
    const options = [[cx + 9, cy - 8], [cx + 9, cy + 16], [cx - 9 - tw, cy - 8], [cx - 9 - tw, cy + 16]];
    const fits = ([x, y]) => x >= m.l && x + tw <= w && y > 10 && y < H - 30 &&
      placed.every(b => x + tw < b.x || x > b.x + b.w || y < b.y - 13 || y - 13 > b.y);
    const [lx, ly] = options.find(fits) || options[0];
    placed.push({x: lx, y: ly, w: tw});
    el("text", {x: lx, y: ly, class: "lbl"}, s).textContent = text;
    const d = p.r.vs_standard;
    hover(el("circle", {cx, cy, r: 12, class: "hit"}, s),
      `<b>${esc(p.n)}</b><br>pass ${fmtPct(p.y)} · ${fmtUsd(p.x)} per 1k · p95 ${fmtMs(p.r.p95_ms)}`
      + (d ? `<br>vs standard: ${(d.diff * 100).toFixed(1)} pts [${(d.ci[0] * 100).toFixed(0)}, ${(d.ci[1] * 100).toFixed(0)}]` : ""));
  });
}

// ---------------------------------------------------------------- table
function recent() {
  const rows = DATA.summary.recent;
  $("recent").innerHTML = `<table><thead><tr><th>time (UTC)</th><th>route</th><th class="num">latency</th>`
    + `<th class="num">cost</th><th class="num">nearest cached</th><th>question</th></tr></thead><tbody>`
    + rows.map(r => `<tr><td>${r.ts.slice(5, 16).replace("T", " ")}</td>`
      + `<td><span class="pill ${r.route}" style="background:${color(r.route)}">${NAMES[r.route]}</span></td>`
      + `<td class="num">${fmtMs(r.total_ms)}</td><td class="num">$${r.cost_usd.toFixed(5)}</td>`
      + `<td class="num">${r.best_similarity == null ? "–" : r.best_similarity.toFixed(3)}</td><td>${esc(r.question)}</td></tr>`).join("")
    + `</tbody></table>`;
}

if (header()) { daily(); latency(); cost(); reasons(); hist(); recent(); }
frontier();
</script>
</body>
</html>
"""
