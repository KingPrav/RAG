"""Dashboard numbers (computed in Python, so they're tested here) and the page render."""
import json

import pytest

from docs_assistant import dashboard


def v4(route, cost, ms, ts="2026-11-10T12:00:00.000+00:00", sim=None, reasons=()):
    cache = {"hit": True, "similarity": sim, "source_question": "q"} if route == "cache" else {"best_similarity": sim}
    return {"timestamp": ts, "question": f"{route} q", "generation": {"model": "m"},
            "serving": {"route": route, "decision": {"reasons": list(reasons)}, "cache": cache,
                        "timings_ms": {"total": ms}, "cost_usd": cost}}


LEGACY = {"timestamp": "2026-10-01T09:00:00.000+00:00", "question": "old", "retrieval": {"latency_ms": 3000},
          "generation": {"model": "m", "latency_ms": 4000, "cost_usd": 0.001}}

TRACES = [v4("cache", 0.0, 150, sim=0.97), v4("fast", 0.0004, 3000, sim=0.91, reasons=["confident single-page match"]),
          v4("standard", 0.0010, 8000, sim=0.93, reasons=["weak top hit (0.55 < 0.62)", "multi-part"]),
          v4("standard", 0.0012, 9000, ts="2026-11-11T12:00:00.000+00:00", sim=0.80,
             reasons=["weak top hit (0.58 < 0.62)"])]


def test_legacy_traces_count_as_standard():
    rows = dashboard.served([LEGACY])
    assert rows[0]["route"] == "standard" and rows[0]["total_ms"] == 7000 and rows[0]["legacy"]


def test_summary_costs_savings_and_routes():
    s = dashboard.summarize(dashboard.served(TRACES), threshold=0.95)
    assert s["n"] == 4 and s["cache_hit_rate"] == 0.25
    assert s["usd_per_1k"] == pytest.approx(1000 * 0.0026 / 4)
    assert s["baseline_usd_per_1k"] == pytest.approx(1.1)                  # standard mean x all traffic
    assert s["savings"] == pytest.approx(1 - 0.0026 / 0.0044)
    assert s["routes"]["fast"]["share"] == 0.25 and s["routes"]["escalate"]["n"] == 0
    assert [d["day"] for d in s["daily"]] == ["2026-11-10", "2026-11-11"]


def test_near_misses_and_histogram_use_only_misses():
    s = dashboard.summarize(dashboard.served(TRACES), threshold=0.95)
    assert s["near_misses"] == 1                                           # 0.93 is within 0.03 below 0.95
    assert sum(b["n"] for b in s["miss_similarity_hist"]) == 3             # the 0.97 cache hit is excluded


def test_standard_reasons_are_grouped_without_numbers():
    s = dashboard.summarize(dashboard.served(TRACES))
    assert dict(s["standard_reasons"]) == {"weak top hit": 2, "multi-part": 1}


def test_empty_traces():
    assert dashboard.summarize([]) == {"n": 0}


def test_render_embeds_data_safely():
    s = dashboard.summarize(dashboard.served([v4("fast", 0.0004, 3000)] + [
        dict(v4("fast", 0.0004, 3000), question="what about </script><script>alert(1)</script>?")]))
    html = dashboard.render(s, {"simulation": None, "calibration": None})
    assert "__DATA__" not in html and "</script><script>alert" not in html
    payload = html.split("const DATA = ", 1)[1].split(";\nconst ROUTES", 1)[0]
    assert json.loads(payload.replace("<\\/", "</"))["summary"]["n"] == 2


def test_demo_data_is_labelled_and_consistent():
    rows = dashboard.served(dashboard.demo_traces(days=3))
    assert all(r["best_similarity"] < 0.95 for r in rows if r["route"] != "cache")
    assert '"demo": true' in dashboard.render(dashboard.summarize(rows), dashboard.demo_eval_context(), demo=True)
