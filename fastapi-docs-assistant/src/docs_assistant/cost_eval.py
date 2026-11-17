"""Phase 4d: prove the cost/latency savings don't cost quality.

Two questions, measured separately:

1. Cache: at what similarity threshold does the cache stop serving answers
   to *different* questions?
       pairs           LLM drafts 2 paraphrases (same answer) and 1 hard
                       negative (minimal edit, different answer) per golden
                       question -> evals/cache_pairs_v1.jsonl (review it!)
       calibrate-cache embeds everything once, simulates a cache holding the
                       golden questions, and reports per threshold: paraphrase
                       hit rate vs false-hit rate. No LLM calls.

2. Routing: which router policy is cheapest without lowering the pass rate?
       paths           answers every golden question on ALL three paths
                       (fast / standard / escalate), judges each, and saves
                       the router features: one paid run.
       simulate        replays any number of router policies on that table
                       for free: pass rate, $/question, p50/p95 per policy,
                       against always-standard (the Phase 1 system).

Run:  python -m docs_assistant.cost_eval pairs
      python -m docs_assistant.cost_eval calibrate-cache --split dev
      python -m docs_assistant.cost_eval paths --split dev
      python -m docs_assistant.cost_eval simulate evals/runs/<ts>_paths-dev
      python -m docs_assistant.cost_eval simulate evals/runs/<ts>_paths-test --policy evals/router_policy.json
"""
import argparse
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from . import config
from .eval_metrics import bootstrap_ci, paired_bootstrap

PAIRS_PATH = config.ROOT / "evals" / "cache_pairs_v1.jsonl"
POLICY_PATH = config.ROOT / "evals" / "router_policy.json"


def new_run_dir(label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = config.EVAL_RUNS_DIR / f"{stamp}_{label}"
    d.mkdir(parents=True, exist_ok=False)
    return d


# ================================================================ 1. cache pairs

PAIRS_PROMPT = """You are building a test set for a semantic cache in front of a FastAPI documentation assistant.
A cache serves a stored answer to a new question when the two questions are similar enough.

Original question: {question}
Reference answer (for your understanding only): {answer}

Write:
1. paraphrases: two rewordings a real developer might type that have EXACTLY the same correct answer.
   Vary vocabulary and structure (casual, terse, different terms), not just word order.
2. hard_negative: one question that looks almost the same (change as few words as possible)
   but needs a DIFFERENT answer from the FastAPI docs, e.g. path -> query parameters,
   request -> response, sync -> async, set -> read, add -> remove. It must still be a sensible FastAPI question.
3. why_different: one sentence on why the hard negative's answer differs."""


class CachePairs(BaseModel):
    paraphrases: list[str] = Field(min_length=2, max_length=2)
    hard_negative: str
    why_different: str


def generate_pairs(items, llm, workers: int = config.EVAL_WORKERS) -> list[dict]:
    """One structured call per golden question. Abstain items are included:
    an out-of-scope question's paraphrase should hit just like any other."""
    chain = llm.with_structured_output(CachePairs)

    def one(item) -> dict:
        out = chain.invoke(PAIRS_PROMPT.format(question=item.question, answer=item.expected_answer))
        return {"id": item.id, "split": item.split, "question": item.question,
                "paraphrases": out.paraphrases, "hard_negative": out.hard_negative,
                "why_different": out.why_different, "review": {"status": "draft", "note": ""}}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


# ================================================================ 1b. cache calibration

THRESHOLDS = [round(t, 2) for t in np.arange(0.80, 1.0, 0.01)]


def unit_rows(vectors: list[list[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def calibrate_cache(pairs: list[dict], embed_many, split: str | None = None,
                    thresholds: list[float] = THRESHOLDS) -> dict:
    """Simulate a cache that holds every golden question (both splits: more
    look-alikes in the cache makes false hits more likely, i.e. a stricter
    test), then probe it with the paraphrases and hard negatives of `split`.

    A paraphrase is a correct hit only if its nearest cached question is its
    own source; nearest-but-wrong counts as a false hit, like any hard
    negative that clears the threshold."""
    pairs = [p for p in pairs if p.get("review", {}).get("status") != "rejected"]
    cached = unit_rows(embed_many([p["question"] for p in pairs]))
    probes = [p for p in pairs if split in (None, "all") or p["split"] == split]
    para = [(text, pairs.index(p)) for p in probes for text in p["paraphrases"]]
    negs = [p["hard_negative"] for p in probes]
    para_sims = unit_rows(embed_many([t for t, _ in para])) @ cached.T if para else np.empty((0, len(pairs)))
    neg_sims = unit_rows(embed_many(negs)) @ cached.T if negs else np.empty((0, len(pairs)))

    para_best, para_arg = para_sims.max(axis=1), para_sims.argmax(axis=1)
    para_right = para_arg == np.array([src for _, src in para])
    neg_best = neg_sims.max(axis=1)

    rows = []
    for t in thresholds:
        good = int(((para_best >= t) & para_right).sum())
        wrong = int(((para_best >= t) & ~para_right).sum())
        neg_hits = int((neg_best >= t).sum())
        hits = good + wrong + neg_hits
        rows.append({"threshold": t, "paraphrase_hit_rate": good / len(para) if para else None,
                     "paraphrase_wrong_source": wrong, "hard_negative_hits": neg_hits,
                     "false_hits": wrong + neg_hits,
                     "precision": good / hits if hits else None})

    safe = [r for r in rows if r["false_hits"] == 0]
    near_safe = [r for r in rows if r["precision"] is None or r["precision"] >= 0.99]
    worst = sorted(zip(neg_best.tolist(), negs, [p["question"] for p in probes]), reverse=True)[:5]
    return {
        "split": split, "n_cached": len(pairs), "n_paraphrases": len(para), "n_hard_negatives": len(negs),
        "paraphrase_similarity": quantiles(para_best[para_right]),
        "hard_negative_similarity": quantiles(neg_best),
        "thresholds": rows,
        "recommended": safe[0]["threshold"] if safe else None,          # lowest threshold with zero false hits
        "recommended_99": near_safe[0]["threshold"] if near_safe else None,
        "closest_hard_negatives": [{"similarity": round(s, 4), "hard_negative": n, "question": q}
                                   for s, n, q in worst],
    }


def quantiles(values) -> dict | None:
    v = np.asarray(values, dtype=float)
    if not v.size:
        return None
    return {q: round(float(np.quantile(v, p)), 4) for q, p in
            [("min", 0), ("p10", .1), ("p50", .5), ("p90", .9), ("max", 1)]}


def render_cache_report(r: dict) -> str:
    lines = [f"# Cache threshold calibration ({r['split'] or 'all'})", "",
             f"Cache holds {r['n_cached']} golden questions; probed with {r['n_paraphrases']} paraphrases "
             f"and {r['n_hard_negatives']} hard negatives.", "",
             f"- Paraphrase similarity (to own source): {r['paraphrase_similarity']}",
             f"- Hard-negative similarity (to nearest): {r['hard_negative_similarity']}",
             f"- **Recommended threshold (zero false hits): {r['recommended']}**"
             f" · at ≥99% precision: {r['recommended_99']}", "",
             "| threshold | paraphrase hit rate | false hits | precision |", "|---|---|---|---|"]
    for row in r["thresholds"]:
        hr = "–" if row["paraphrase_hit_rate"] is None else f"{row['paraphrase_hit_rate']:.0%}"
        pr = "–" if row["precision"] is None else f"{row['precision']:.1%}"
        lines.append(f"| {row['threshold']:.2f} | {hr} | {row['false_hits']} | {pr} |")
    lines += ["", "Closest hard negatives (the questions that set the threshold):", ""]
    lines += [f"- {c['similarity']:.3f}  \"{c['hard_negative']}\" vs \"{c['question']}\""
              for c in r["closest_hard_negatives"]]
    return "\n".join(lines) + "\n"


# ================================================================ 2. per-path answers

def check_judge_independence() -> None:
    """The judge must not grade answers written by its own model (self-preference bias)."""
    judge_model = config.JUDGE_MODEL.split(":", 1)[-1]
    if judge_model == config.ESCALATE_MODEL:
        raise SystemExit(f"ESCALATE_MODEL and JUDGE_MODEL are both '{judge_model}': the judge would grade its own "
                         "answers on the escalate path. Set a different ESCALATE_MODEL, or JUDGE_MODEL to another "
                         "provider (e.g. anthropic:<model>), in .env.")


def path_summary(row: dict, served) -> dict:
    """What simulate() needs from one item on one path."""
    return {"passed": bool(row.get("passed")), "error": row.get("error"),
            "answerable": row.get("answerable_predicted"), "cost_usd": round(served.cost_usd, 7) if served else 0.0,
            "total_ms": served.total_ms if served else None,
            "retrieve_ms": row.get("retrieve_ms"), "generate_ms": row.get("generate_ms"),
            "judge_cost_usd": row.get("judge_cost_usd", 0.0), "answer": row.get("answer")}


def run_paths(items, assistant, judge, *, routes=("fast", "standard", "escalate"), k: int = config.TOP_K,
              workers: int = config.EVAL_WORKERS, progress=print) -> list[dict]:
    """Every item on every route, plus the router's features for that item.
    Forced routes bypass the cache, so each path is measured cold."""
    from .evaluate import error_row, score_item
    from .router import extract_features, route

    def one(item) -> dict:
        vector = assistant.vectorstore.embeddings.embed_query(item.question)
        features = extract_features(item.question, assistant._probe(vector))
        out = {"id": item.id, "split": item.split, "category": item.category, "answer_type": item.answer_type,
               "question": item.question, "features": features.to_dict(),
               "default_route": route(features).route, "paths": {}}
        for name in routes:
            try:
                served = assistant.answer(item.question, force_route=name)
                out["paths"][name] = path_summary(score_item(item, served.result, judge, k), served)
            except Exception as e:                          # one failure must not sink the run
                out["paths"][name] = path_summary(error_row(item, e), None)
        return out

    rows = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, item): i for i, item in enumerate(items)}
        for done, future in enumerate(futures, start=1):
            rows[futures[future]] = r = future.result()
            marks = " ".join(f"{n[:4]}:{'ok' if p['passed'] else ('ERR' if p['error'] else 'x ')}"
                             for n, p in r["paths"].items())
            progress(f"[{done:>3}/{len(items)}] {marks}  {r['id']:<10} {r['question'][:60]}")
    return rows


# ================================================================ 2b. policy simulation (free)

POLICY_GRID = {
    "fast_min_top1": [0.50, 0.55, 0.60, 0.62, 0.65, 0.70],
    "fast_min_margin": [0.0, 0.01, 0.02, 0.03, 0.05],
    "fast_min_page_share": [0.0, 0.4, 0.6, 0.8],
    "fast_max_words": [12, 18, 25, 999],
    "escalate": [True, False],
}


def choose_routes(rows: list[dict], policy) -> list[str]:
    """policy: a RouterPolicy, or 'always:<route>', or 'oracle' (cheapest
    passing path per item: the ceiling no real router can beat)."""
    from .router import Features, route
    if isinstance(policy, str) and policy.startswith("always:"):
        return [policy.split(":", 1)[1]] * len(rows)
    if policy == "oracle":
        out = []
        for r in rows:
            passing = [(p["cost_usd"], n) for n, p in r["paths"].items() if p["passed"]]
            out.append(min(passing)[1] if passing else "standard")
        return out
    return [route(Features(**r["features"]), policy).route for r in rows]


def evaluate_policy(rows: list[dict], policy, baseline: list[float] | None = None,
                    resamples: int = 2000) -> dict:
    routes = choose_routes(rows, policy)
    picked = [r["paths"].get(name) or r["paths"]["standard"] for r, name in zip(rows, routes)]
    passed = [float(p["passed"]) for p in picked]
    cost = [p["cost_usd"] for p in picked]
    ms = sorted(p["total_ms"] for p in picked if p["total_ms"] is not None)
    pct = lambda q: ms[min(len(ms) - 1, round(q * (len(ms) - 1)))] if ms else None   # noqa: E731
    out = {
        "policy": policy if isinstance(policy, str) else policy.describe(),
        "pass_rate": sum(passed) / len(passed), "passes": int(sum(passed)), "n": len(passed),
        "pass_ci": bootstrap_ci(passed, resamples=resamples) if resamples else None,
        "usd_per_question": sum(cost) / len(cost), "usd_per_1k": 1000 * sum(cost) / len(cost),
        "p50_ms": pct(0.5), "p95_ms": pct(0.95),
        "route_mix": {n: routes.count(n) / len(routes) for n in ("fast", "standard", "escalate")},
        "_passed": passed,
    }
    if baseline is not None and resamples:
        out["vs_standard"] = paired_bootstrap(baseline, passed, resamples=resamples)
    return out


def simulate(rows: list[dict], grid: dict = POLICY_GRID, max_regressions: int = 0) -> dict:
    """Replay every policy in the grid. The pick is the cheapest policy whose
    pass count is within `max_regressions` items of always-standard (default:
    no worse at all), ties broken by p95 latency."""
    from .router import RouterPolicy
    standard = evaluate_policy(rows, "always:standard")
    base = standard["_passed"]
    named = {name: evaluate_policy(rows, name, base) for name in
             ("always:fast", "always:standard", "always:escalate", "oracle")}
    named["default policy"] = evaluate_policy(rows, RouterPolicy(), base)

    keys = list(grid)
    candidates = [evaluate_policy(rows, RouterPolicy(**dict(zip(keys, values))), base, resamples=0)
                  for values in itertools.product(*(grid[k] for k in keys))]
    ok = [c for c in candidates if c["passes"] >= standard["passes"] - max_regressions]
    best = min(ok, key=lambda c: (c["usd_per_question"], c["p95_ms"] or 0)) if ok else None
    if best:
        best = evaluate_policy(rows, RouterPolicy(**best["policy"]), base)     # full-precision CIs
        named["selected policy"] = best

    # Cost/quality frontier: for each pass count, the cheapest policy reaching it.
    frontier = {}
    for c in candidates:
        if c["passes"] not in frontier or c["usd_per_question"] < frontier[c["passes"]]["usd_per_question"]:
            frontier[c["passes"]] = c
    frontier = {p: evaluate_policy(rows, RouterPolicy(**c["policy"]), base) for p, c in frontier.items()}
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}   # noqa: E731
    return {"n_items": len(rows), "n_policies": len(candidates), "max_regressions": max_regressions,
            "named": {k: strip(v) for k, v in named.items()},
            "selected": strip(best) if best else None,
            "frontier": [strip(frontier[p]) for p in sorted(frontier, reverse=True)]}


def render_simulation(sim: dict, label: str) -> str:
    def row(name, r):
        ci = r.get("pass_ci")
        ci_s = f" [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else ""
        d = r.get("vs_standard")
        d_s = "–" if not d else f"{d['diff']:+.1%} [{d['ci'][0]:+.0%}, {d['ci'][1]:+.0%}]"
        mix = " / ".join(f"{r['route_mix'][n]:.0%}" for n in ("fast", "standard", "escalate"))
        return (f"| {name} | {r['pass_rate']:.1%}{ci_s} | {d_s} | ${r['usd_per_1k']:.2f} | "
                f"{(r['p50_ms'] or 0) / 1000:.1f} s | {(r['p95_ms'] or 0) / 1000:.1f} s | {mix} |")
    head = ["| policy | pass rate [95% CI] | Δ vs standard [95% CI] | $ / 1k questions | p50 | p95 "
            "| fast / standard / escalate |", "|---|---|---|---|---|---|---|"]
    lines = [f"# Routing simulation: {label}", "",
             f"{sim['n_items']} items × 3 paths, {sim['n_policies']} policies replayed. "
             f"Selection: cheapest policy with at most {sim['max_regressions']} fewer passes than always-standard.",
             "", *head, *(row(n, r) for n, r in sim["named"].items()), ""]
    if sim["selected"]:
        lines += ["Selected policy:", "", "```json", json.dumps(sim["selected"]["policy"], indent=2), "```", ""]
    else:
        lines += ["No policy in the grid matched always-standard's pass count.", ""]
    lines += ["Cost/quality frontier (cheapest policy per pass count):", "", *head,
              *(row(f"{f['passes']}/{f['n']} passes", f) for f in sim["frontier"][:8])]
    return "\n".join(lines) + "\n"


# ================================================================ CLI

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_pairs(args) -> None:
    from .evaluate import load_golden
    from .judge import get_judge_model
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists (it may hold your review). Use --force to regenerate.")
    items, _ = load_golden(args.golden)
    print(f"Drafting paraphrases + hard negatives for {len(items)} questions with {config.JUDGE_MODEL} ...")
    pairs = generate_pairs(items, get_judge_model())
    args.out.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    print(f"Wrote {args.out}. Review it: mark bad items \"review\": {{\"status\": \"rejected\"}}.")


def cmd_calibrate(args) -> None:
    from .vectorstore import get_embeddings
    if not args.pairs.exists():
        raise SystemExit(f"No {args.pairs}. Run: python -m docs_assistant.cost_eval pairs")
    report = calibrate_cache(load_jsonl(args.pairs), get_embeddings().embed_documents, args.split)
    run_dir = new_run_dir(f"cache-calibration-{args.split}")
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2))
    (run_dir / "summary.md").write_text(md := render_cache_report(report))
    print(md + f"\nSaved to {run_dir}\nSet CACHE_THRESHOLD in .env to the recommended value.")


def cmd_paths(args) -> None:
    from .evaluate import load_golden
    from .judge import build_judge, get_judge_model
    from .serve import build_assistant
    check_judge_independence()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("The test split is held out: select a policy on dev first, then add --confirm-test.")
    items, _ = load_golden(args.golden)
    items = [i for i in items if args.split == "all" or i.split == args.split][: args.limit]
    routes = tuple(args.routes.split(","))
    print(f"Answering {len(items)} items on {', '.join(routes)} · small {config.CHAT_MODEL} · "
          f"escalate {config.ESCALATE_MODEL} · judge {config.JUDGE_MODEL}\n")
    assistant = build_assistant(use_cache=False)
    started = time.perf_counter()
    rows = run_paths(items, assistant, build_judge(get_judge_model()), routes=routes, workers=args.workers)
    run_dir = new_run_dir(f"paths-{args.split}")
    cfg = {"split": args.split, "routes": routes, "chat_model": config.CHAT_MODEL,
           "escalate_model": config.ESCALATE_MODEL, "judge_model": config.JUDGE_MODEL,
           "pipeline": assistant.pipeline, "wall_seconds": round(time.perf_counter() - started, 1)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name in routes:
        ok = sum(r["paths"][name]["passed"] for r in rows)
        usd = sum(r["paths"][name]["cost_usd"] for r in rows) / len(rows)
        print(f"  {name:<9} {ok}/{len(rows)} passed · ${1000 * usd:.2f} per 1k questions")
    print(f"Saved to {run_dir}\nNext: python -m docs_assistant.cost_eval simulate {run_dir}")


def cmd_simulate(args) -> None:
    from .router import RouterPolicy
    rows = load_jsonl(args.run / "results.jsonl")
    if args.policy:                      # confirmation run: one fixed policy, no search
        policy = RouterPolicy(**json.loads(args.policy.read_text())["policy"])
        base = evaluate_policy(rows, "always:standard")["_passed"]
        named = {n: evaluate_policy(rows, n, base) for n in ("always:standard", "oracle")}
        named["fixed policy"] = evaluate_policy(rows, policy, base)
        sim = {"n_items": len(rows), "n_policies": 1, "max_regressions": None,
               "named": {k: {x: y for x, y in v.items() if not x.startswith("_")} for k, v in named.items()},
               "selected": None, "frontier": []}
    else:
        sim = simulate(rows, max_regressions=args.max_regressions)
    md = render_simulation(sim, args.run.name)
    (args.run / "simulation.json").write_text(json.dumps(sim, indent=2))
    (args.run / "simulation.md").write_text(md)
    print(md)
    if sim["selected"] and not args.policy and "dev" in args.run.name:
        POLICY_PATH.write_text(json.dumps({"policy": sim["selected"]["policy"], "selected_on": args.run.name,
                                           "pass_rate": sim["selected"]["pass_rate"],
                                           "usd_per_1k": sim["selected"]["usd_per_1k"]}, indent=2))
        print(f"Saved the selected policy to {POLICY_PATH.relative_to(config.ROOT)} (the router now uses it).")


def main():
    parser = argparse.ArgumentParser(description="Cache calibration and routing cost/quality evals.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    from .golden import GOLDEN_PATH

    p = sub.add_parser("pairs", help="draft cache test pairs (paraphrases + hard negatives)")
    p.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    p.add_argument("--out", type=Path, default=PAIRS_PATH)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_pairs)

    p = sub.add_parser("calibrate-cache", help="false-hit rate per similarity threshold (no LLM calls)")
    p.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("paths", help="answer + judge every item on every path (paid)")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    p.add_argument("--routes", default="fast,standard,escalate")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=config.EVAL_WORKERS)
    p.add_argument("--confirm-test", action="store_true")
    p.set_defaults(fn=cmd_paths)

    p = sub.add_parser("simulate", help="replay router policies on a paths run (free)")
    p.add_argument("run", type=Path)
    p.add_argument("--max-regressions", type=int, default=0)
    p.add_argument("--policy", type=Path, help="evaluate one fixed policy (e.g. evals/router_policy.json on test)")
    p.set_defaults(fn=cmd_simulate)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
