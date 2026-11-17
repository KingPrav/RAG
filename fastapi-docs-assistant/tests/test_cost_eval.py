"""cost_eval: cache calibration math, policy replay, selection, and the judge guard. Offline."""
import time

import pytest

from docs_assistant import config, cost_eval
from docs_assistant.router import RouterPolicy

# ------------------------------------------------------------ cache calibration

EMB = {
    "add cors": [1, 0, 0, 0], "enable cors": [0.97, 0.243, 0, 0], "cors setup": [0.96, 0.28, 0, 0],
    "remove cors": [0.93, 0, 0.367, 0],                              # hard negative, 0.93 to "add cors"
    "query params": [0, 1, 0, 0], "declare query params": [0.2, 0.979, 0, 0], "query args": [0, 0.97, 0, 0.243],
    "path params": [0, 0.9, 0, 0.436],                               # hard negative, 0.90 to "query params"
}


def embed_many(texts):
    return [EMB[t] for t in texts]


PAIRS = [
    {"id": "a", "split": "dev", "question": "add cors", "paraphrases": ["enable cors", "cors setup"],
     "hard_negative": "remove cors"},
    {"id": "b", "split": "dev", "question": "query params", "paraphrases": ["declare query params", "query args"],
     "hard_negative": "path params"},
]


def at(report, t):
    return next(r for r in report["thresholds"] if r["threshold"] == t)


def test_calibration_counts_hits_and_false_hits():
    r = cost_eval.calibrate_cache(PAIRS, embed_many, "dev")
    assert r["n_paraphrases"] == 4 and r["n_hard_negatives"] == 2
    low = at(r, 0.85)
    assert low["paraphrase_hit_rate"] == 1.0 and low["false_hits"] == 2      # both hard negatives slip in
    mid = at(r, 0.94)
    assert mid["hard_negative_hits"] == 0 and mid["paraphrase_hit_rate"] == 1.0
    assert r["recommended"] == 0.94                                            # lowest with zero false hits
    assert r["closest_hard_negatives"][0]["hard_negative"] == "remove cors"


def test_paraphrase_matching_the_wrong_source_is_a_false_hit():
    pairs = [dict(PAIRS[0], paraphrases=["query args", "enable cors"])]      # "query args" isn't a CORS question
    pairs.append(PAIRS[1])
    r = cost_eval.calibrate_cache(pairs, embed_many, "dev")
    assert at(r, 0.96)["paraphrase_wrong_source"] == 1


def test_rejected_pairs_are_skipped():
    pairs = [PAIRS[0], dict(PAIRS[1], review={"status": "rejected"})]
    assert cost_eval.calibrate_cache(pairs, embed_many, "dev")["n_cached"] == 1


# ------------------------------------------------------------ routing simulation

def path(passed, cost, ms):
    return {"passed": passed, "cost_usd": cost, "total_ms": ms, "error": None}


def item(i, top1, fast_ok, std_ok=True, esc_ok=True, comparison=False):
    return {"id": f"q{i}", "features": {"n_words": 6, "comparison": comparison, "specific_value": False,
                                        "multi_part": False, "top1": top1, "margin": 0.1,
                                        "top_page_share": 1.0, "n_pages": 1},
            "paths": {"fast": path(fast_ok, 0.0004, 3000), "standard": path(std_ok, 0.0010, 7700),
                      "escalate": path(esc_ok, 0.0200, 9000)}}


# Confident questions pass on fast; the weak-hit ones only pass on standard.
ROWS = [item(i, 0.75, True) for i in range(6)] + [item(i, 0.55, False) for i in range(6, 10)]


def test_always_policies_and_oracle():
    assert cost_eval.evaluate_policy(ROWS, "always:fast")["passes"] == 6
    assert cost_eval.evaluate_policy(ROWS, "always:standard")["passes"] == 10
    oracle = cost_eval.evaluate_policy(ROWS, "oracle")
    assert oracle["passes"] == 10 and oracle["route_mix"]["fast"] == 0.6


def test_policy_replay_matches_router():
    r = cost_eval.evaluate_policy(ROWS, RouterPolicy(fast_min_top1=0.7))
    assert r["passes"] == 10 and r["route_mix"]["fast"] == 0.6
    assert r["usd_per_question"] == pytest.approx((6 * 0.0004 + 4 * 0.0010) / 10)
    assert r["p95_ms"] == 7700


def test_simulate_selects_cheapest_policy_without_regressions():
    started = time.perf_counter()
    sim = cost_eval.simulate(ROWS)
    assert time.perf_counter() - started < 20
    sel = sim["selected"]
    assert sel["passes"] == 10 and sel["route_mix"]["fast"] == 0.6           # matches the oracle here
    assert 0.55 < sel["policy"]["fast_min_top1"] <= 0.75
    assert sel["usd_per_question"] < sim["named"]["always:standard"]["usd_per_question"]
    assert sim["frontier"][0]["passes"] == 10


def test_comparison_questions_escalate_in_replay():
    rows = ROWS + [item(10, 0.75, False, std_ok=False, comparison=True)]
    r = cost_eval.evaluate_policy(rows, RouterPolicy(fast_min_top1=0.7))
    assert r["passes"] == 11 and r["route_mix"]["escalate"] == pytest.approx(1 / 11)


def test_judge_independence_guard(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "openai:big")
    monkeypatch.setattr(config, "ESCALATE_MODEL", "big")
    with pytest.raises(SystemExit, match="grade its own"):
        cost_eval.check_judge_independence()
    monkeypatch.setattr(config, "ESCALATE_MODEL", "other")
    cost_eval.check_judge_independence()


def test_render_reports_do_not_crash():
    assert "Recommended" in cost_eval.render_cache_report(cost_eval.calibrate_cache(PAIRS, embed_many, "dev"))
    assert "Selected policy" in cost_eval.render_simulation(cost_eval.simulate(ROWS), "paths-dev")


def test_router_policy_load(tmp_path):
    p = tmp_path / "policy.json"
    assert RouterPolicy.load(p) == RouterPolicy()                              # missing file -> defaults
    p.write_text('{"policy": {"fast_min_top1": 0.5, "fast_min_margin": 0.0, "fast_min_page_share": 0.0, '
                 '"fast_max_words": 999, "escalate": false, "escalate_min_pages": 3}}')
    assert RouterPolicy.load(p).fast_min_top1 == 0.5
