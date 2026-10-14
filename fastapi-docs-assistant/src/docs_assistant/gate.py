"""Step 9: the eval gate. Decides whether a candidate eval run is good enough
to merge, by comparing it with the committed baseline run.

    python -m docs_assistant.gate evals/runs/<candidate>          # exit 1 = blocked

Three kinds of check (policy in evals/gate.json):

1. Infrastructure: no errored items, and every baseline item was evaluated.
   Missing results are failures, never silently dropped.
2. Hard floors (absolute): e.g. zero invalid citations, every unanswerable
   trap refused, overall pass rate above a minimum.
3. Regression (relative): paired bootstrap of per-item pass/fail against the
   baseline. Blocks only when the drop is statistically significant (the 95%
   CI of the difference lies entirely below zero).

Why not a stricter "non-inferiority" margin? With ~40 dev items, ordinary
run-to-run noise (a few items flipping each way) gives a CI roughly ±12
points wide, so a tight margin would block good changes constantly. A
significant-drop rule catches real regressions (several items broken, few
fixed) without crying wolf. Tighten it as the golden set grows.
"""
import argparse
import json
import sys
from pathlib import Path

from . import config
from .compare import compare, setting
from .evaluate import load_run, summarize

POLICY_PATH = config.ROOT / "evals" / "gate.json"


def load_policy(path: Path = POLICY_PATH) -> dict:
    return json.loads(path.read_text())


def check(baseline: tuple[dict, list[dict]], candidate: tuple[dict, list[dict]], policy: dict,
          resamples: int = config.BOOTSTRAP_RESAMPLES) -> dict:
    (base_cfg, base_rows), (cand_cfg, cand_rows) = baseline, candidate
    k = cand_cfg["k"]
    s = summarize(cand_rows, k, resamples)
    floors = policy["floors"]
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, kind: str):
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "kind": kind})

    # 1. infrastructure
    add("No errored items", s["n_errors"] <= floors.get("max_errors", 0),
        f"{s['n_errors']} errored", "infrastructure")
    missing = sorted({r["id"] for r in base_rows} - {r["id"] for r in cand_rows})
    add("Every baseline item evaluated", not missing,
        f"missing: {', '.join(missing)}" if missing else f"{len(cand_rows)} items", "infrastructure")

    # 2. hard floors
    a, g = s["answers"], s["grounding"]
    inv = g["invalid_citation_rate"]["value"] or 0.0
    add("Invalid citations", inv <= floors["max_invalid_citation_rate"],
        f"{inv:.1%} (max {floors['max_invalid_citation_rate']:.0%})", "floor")
    refusal = a["correct_refusal_rate"]["value"]
    if refusal is not None:
        add("Unanswerable questions refused", refusal >= floors["min_correct_refusal_rate"],
            f"{refusal:.1%} (min {floors['min_correct_refusal_rate']:.0%})", "floor")
    passed = s["overall_pass_rate"]["value"] or 0.0
    add("Overall pass rate", passed >= floors["min_overall_pass_rate"],
        f"{passed:.1%} (min {floors['min_overall_pass_rate']:.0%})", "floor")

    # 3. regression vs baseline (paired, same items)
    result = compare(baseline, candidate, resamples=resamples)
    for metric in policy["regression"]["metrics"]:
        d = result["deltas"].get(metric)
        if d is None:
            continue
        lo, hi = d["ci"]
        add(f"No significant drop in {metric}", hi >= 0,
            f"{d['diff']:+.1%} [{lo:+.1%}, {hi:+.1%}] vs baseline (n={d['n']})", "regression")

    warnings = []
    for key in ("golden_sha256", "judge_model", "judge_prompt_version", "corpus_commit"):
        if setting(base_cfg, key) != setting(cand_cfg, key):
            warnings.append(f"`{key}` differs from the baseline, so differences mix several causes.")
    return {"passed": all(c["ok"] for c in checks), "checks": checks, "warnings": warnings,
            "fixed": result["fixed"], "broken": result["broken"], "summary": s,
            "baseline_label": base_cfg["label"], "candidate_label": cand_cfg["label"]}


def render(report: dict) -> str:
    verdict = "✅ **Eval gate passed**" if report["passed"] else "❌ **Eval gate failed**"
    lines = [f"## {verdict}", "", f"Candidate `{report['candidate_label']}` vs baseline "
             f"`{report['baseline_label']}`", "", "| Check | Result | Detail |", "|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['name']} ({c['kind']}) | {'pass' if c['ok'] else '**FAIL**'} | {c['detail']} |")
    lines += ["", f"Fixed vs baseline: {', '.join(report['fixed']) or '-'}",
              "", f"Broken vs baseline: {', '.join(report['broken']) or '-'}"]
    if report["warnings"]:
        lines += ["", *[f"> ⚠️ {w}" for w in report["warnings"]]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Gate a candidate eval run against the committed baseline.")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    baseline_dir = config.ROOT / policy["baseline_run"]
    report = check(load_run(baseline_dir), load_run(args.candidate), policy)
    print(render(report))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
