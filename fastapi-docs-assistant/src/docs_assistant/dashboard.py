"""Phase 4e: cost & latency dashboard over the serving traces.

Reads data/logs/traces.jsonl (schema v4 requests from serve.Assistant), plus
the latest routing simulation and cache calibration if they exist, and writes
one self-contained HTML file (no server, no CDN): reports/dashboard.html.

Run:  python -m docs_assistant.dashboard                 # your real traces
      python -m docs_assistant.dashboard --since 7d
      python -m docs_assistant.dashboard --demo          # synthetic data, clearly labelled

Everything numeric is computed here in Python (and unit-tested); the page
only draws it.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config
from .tracing import TRACES_PATH, load_traces

ROUTES = ("cache", "fast", "standard", "escalate")
OUT_PATH = config.ROOT / "reports" / "dashboard.html"


def pct(values: list[float], p: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    return v[min(len(v) - 1, round(p * (len(v) - 1)))] if v else None


def served(traces: list[dict]) -> list[dict]:
    """Flatten v4 traces into one row per request. Older traces (no serving
    block) were answered by the standard pipeline without a cache."""
    rows = []
    for t in traces:
        s = t.get("serving")
        g = t.get("generation") or {}
        if s:
            cache = s.get("cache") or {}
            rows.append({"ts": t["timestamp"], "question": t["question"], "route": s["route"],
                         "total_ms": s["timings_ms"].get("total"), "cost_usd": s["cost_usd"],
                         "timings": s["timings_ms"], "model": g.get("model"),
                         "best_similarity": cache.get("similarity") if s["route"] == "cache"
                         else cache.get("best_similarity"),
                         "reasons": (s.get("decision") or {}).get("reasons", []), "legacy": False})
        else:
            r = t.get("retrieval") or {}
            rows.append({"ts": t["timestamp"], "question": t["question"], "route": "standard",
                         "total_ms": (r.get("latency_ms") or 0) + (g.get("latency_ms") or 0),
                         "cost_usd": g.get("cost_usd", 0.0), "timings": {}, "model": g.get("model"),
                         "best_similarity": None, "reasons": [], "legacy": True})
    return rows


def summarize(rows: list[dict], threshold: float = config.CACHE_THRESHOLD) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    by_route = defaultdict(list)
    for r in rows:
        by_route[r["route"]].append(r)
    total_cost = sum(r["cost_usd"] for r in rows)

    # Counterfactual: what the same traffic would have cost on the standard
    # path alone (the Phase 1 system), priced at the observed standard mean.
    std = [r["cost_usd"] for r in by_route.get("standard", [])]
    std_mean = sum(std) / len(std) if std else None
    baseline_cost = std_mean * n if std_mean else None

    routes = {}
    for name in ROUTES:
        rs = by_route.get(name, [])
        ms = [r["total_ms"] for r in rs]
        routes[name] = {"n": len(rs), "share": len(rs) / n,
                        "usd_per_1k": 1000 * sum(r["cost_usd"] for r in rs) / len(rs) if rs else None,
                        "p50_ms": pct(ms, 0.5), "p95_ms": pct(ms, 0.95)}

    days = defaultdict(Counter)
    for r in rows:
        days[r["ts"][:10]][r["route"]] += 1
    misses = [r["best_similarity"] for r in rows if r["route"] != "cache" and r["best_similarity"] is not None]
    edges = [round(0.70 + 0.01 * i, 2) for i in range(31)]          # 0.70 .. 1.00
    hist = [{"lo": lo, "hi": hi, "n": sum(lo <= s < hi for s in misses)} for lo, hi in zip(edges, edges[1:])]
    reasons = Counter(x for r in rows if r["route"] == "standard" for x in r["reasons"]
                      if not x.startswith(("weak top hit", "no clear", "hits spread", "long (")))
    reasons.update(x.split(" (")[0] for r in rows if r["route"] == "standard" for x in r["reasons"]
                   if x.startswith(("weak top hit", "no clear", "hits spread", "long (")))

    all_ms = [r["total_ms"] for r in rows]
    return {
        "n": n, "first": min(r["ts"] for r in rows), "last": max(r["ts"] for r in rows),
        "usd_per_1k": 1000 * total_cost / n,
        "baseline_usd_per_1k": 1000 * baseline_cost / n if baseline_cost else None,
        "savings": 1 - total_cost / baseline_cost if baseline_cost else None,
        "p50_ms": pct(all_ms, 0.5), "p95_ms": pct(all_ms, 0.95),
        "std_p95_ms": routes["standard"]["p95_ms"],
        "cache_hit_rate": routes["cache"]["share"],
        "routes": routes,
        "daily": [{"day": d, **{k: days[d].get(k, 0) for k in ROUTES}} for d in sorted(days)],
        "miss_similarity_hist": hist, "cache_threshold": threshold,
        "near_misses": sum(threshold - 0.03 <= s < threshold for s in misses),
        "standard_reasons": reasons.most_common(6),
        "legacy_requests": sum(r["legacy"] for r in rows),
        "recent": [{k: r[k] for k in ("ts", "route", "total_ms", "cost_usd", "question", "best_similarity")}
                   for r in sorted(rows, key=lambda r: r["ts"], reverse=True)[:25]],
    }


def latest(pattern: str) -> Path | None:
    found = sorted(config.EVAL_RUNS_DIR.glob(pattern))
    return found[-1] if found else None


def load_eval_context() -> dict:
    """The eval side of the story: the routing frontier and the cache threshold table."""
    out = {"simulation": None, "calibration": None}
    sim_dir = next((d for d in reversed(sorted(config.EVAL_RUNS_DIR.glob("*_paths-*")))
                    if (d / "simulation.json").exists()), None)
    if sim_dir:
        sim = json.loads((sim_dir / "simulation.json").read_text())
        out["simulation"] = {"run": sim_dir.name, "named": sim["named"], "frontier": sim["frontier"]}
    cal = latest("*_cache-calibration-*")
    if cal and (cal / "summary.json").exists():
        c = json.loads((cal / "summary.json").read_text())
        out["calibration"] = {"run": cal.name, "recommended": c["recommended"], "thresholds": c["thresholds"]}
    return out


# ---------------------------------------------------------------- demo data

DEMO_QUESTIONS = [
    "How do I add CORS?", "How do I declare a request body?", "How do I return a custom status code?",
    "How do I use dependencies with yield?", "What is the difference between Depends and Security?",
    "How do I upload a file?", "How do I set a cookie and how do I read it back?", "How do I run background tasks?",
    "How do I add OAuth2 password flow with JWT?", "What is the default port of fastapi dev?",
]


def demo_traces(days: int = 14, seed: int = 7) -> list[dict]:
    """Synthetic v4 traces with plausible shapes: the cache warms up over the
    period, ~40% of misses go fast, escalation is rare. Labelled as demo."""
    rng = random.Random(seed)
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
    out = []
    for day in range(days):
        hit_p = 0.05 + 0.30 * day / max(1, days - 1)
        for _ in range(rng.randint(60, 140)):
            ts = start + timedelta(days=day, seconds=rng.randint(8 * 3600, 22 * 3600))
            q = rng.choice(DEMO_QUESTIONS)
            if rng.random() < hit_p:
                route, ms, cost, sim = "cache", rng.gauss(160, 40), 0.0000005, round(rng.uniform(0.95, 1.0), 4)
                reasons = []
            else:
                route = rng.choices(["fast", "standard", "escalate"], [0.45, 0.53, 0.02])[0]
                ms = {"fast": rng.lognormvariate(8.0, 0.25), "standard": rng.lognormvariate(8.9, 0.28),
                      "escalate": rng.lognormvariate(9.25, 0.25)}[route]
                cost = {"fast": rng.gauss(0.00042, 0.00006), "standard": rng.gauss(0.00110, 0.00015),
                        "escalate": rng.gauss(0.0210, 0.003)}[route]
                sim = round(min(0.949, rng.betavariate(18, 4) * 0.35 + 0.62), 4)     # a miss is below the threshold
                reasons = (rng.sample(["weak top hit (0.55 < 0.62)", "no clear winner (margin 0.012)",
                                       "hits spread out (40% on top page)", "asks for a specific value/version",
                                       "multi-part"], rng.randint(1, 2)) if route == "standard"
                           else ["confident single-page match"] if route == "fast" else ["comparison"])
            cache = ({"hit": True, "exact": sim == 1.0, "similarity": sim, "source_question": q} if route == "cache"
                     else {"hit": False, "best_similarity": sim, "stored": True})
            out.append({"timestamp": ts.isoformat(timespec="milliseconds"), "schema_version": 4, "question": q,
                        "generation": {"model": "demo"},
                        "serving": {"route": route, "decision": {"reasons": reasons}, "cache": cache,
                                    "timings_ms": {"total": round(max(60, ms), 1)}, "cost_usd": max(0.0, cost)}})
    return out


def demo_eval_context() -> dict:
    def pt(passes, usd, p95, fast, esc, diff=None):
        r = {"passes": passes, "n": 41, "pass_rate": passes / 41, "usd_per_1k": usd, "p95_ms": p95,
             "route_mix": {"fast": fast, "standard": 1 - fast - esc, "escalate": esc}}
        if diff is not None:
            r["vs_standard"] = {"diff": diff, "ci": [diff - 0.07, diff + 0.07]}
        return r
    return {"simulation": {"run": "demo_paths-dev", "named": {
                "always:fast": pt(36, 0.42, 4300, 1, 0, -0.098), "always:standard": pt(40, 1.10, 12300, 0, 0),
                "always:escalate": pt(40, 21.0, 15100, 0, 1, 0.0), "oracle": pt(41, 0.71, 11800, .55, .02, 0.024),
                "selected policy": pt(40, 0.83, 11900, .40, .02, 0.0)},
            "frontier": [pt(41, 1.25, 12400, .30, .05), pt(40, 0.83, 11900, .40, .02), pt(39, 0.66, 10800, .55, 0),
                         pt(38, 0.55, 8200, .70, 0), pt(36, 0.42, 4300, 1, 0)]},
            "calibration": None}


# ---------------------------------------------------------------- render

def render(summary: dict, eval_context: dict, demo: bool = False) -> str:
    from .dashboard_template import TEMPLATE
    data = {"summary": summary, "eval": eval_context, "demo": demo,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    # "</" inside JSON would end the <script> block early (a question could contain it).
    return TEMPLATE.replace("__DATA__", json.dumps(data, default=str).replace("</", "<\\/"))


def parse_since(value: str | None) -> str | None:
    if not value:
        return None
    unit = {"h": "hours", "d": "days"}[value[-1]]
    return (datetime.now(timezone.utc) - timedelta(**{unit: float(value[:-1])})).isoformat()


def main():
    parser = argparse.ArgumentParser(description="Build the cost & latency dashboard (reports/dashboard.html).")
    parser.add_argument("--traces", type=Path, default=TRACES_PATH)
    parser.add_argument("--since", help="only requests newer than e.g. 24h or 7d")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--demo", action="store_true", help="synthetic data, to preview the layout")
    args = parser.parse_args()

    if args.demo:
        traces, eval_context = demo_traces(), demo_eval_context()
    else:
        traces, eval_context = load_traces(args.traces), load_eval_context()
    since = parse_since(args.since)
    rows = [r for r in served(traces) if not since or r["ts"] >= since]
    summary = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(summary, eval_context, demo=args.demo))
    if summary["n"]:
        print(f"{summary['n']} requests · ${summary['usd_per_1k']:.2f} per 1k · p95 {summary['p95_ms'] / 1000:.1f} s · "
              f"cache hits {summary['cache_hit_rate']:.0%}")
    print(f"Wrote {args.out}" + ("  (DEMO DATA)" if args.demo else ""))


if __name__ == "__main__":
    main()
