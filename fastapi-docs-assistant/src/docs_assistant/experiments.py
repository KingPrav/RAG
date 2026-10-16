"""Run a round of experiments in one command and summarize them against a baseline.

    python -m docs_assistant.experiments

Each experiment is a full dev eval (evaluate.py) with different settings; the
summary compares each one item by item with the baseline run and is written to
evals/runs/<stamp>_experiments.md.
"""
import argparse
import glob
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .compare import compare
from .evaluate import load_run, percentile, summarize

# label -> evaluate.py flags
ROUND = {
    "v3": ["--prompt", "v3"],                                             # completeness prompt, fixed refusals
    "v3-llm": ["--prompt", "v3", "--rerank", "llm"],                      # + listwise LLM reranker
    "v3-llm-norw": ["--prompt", "v3", "--rerank", "llm", "--no-rewrite"],  # reranker instead of rewrite (faster?)
}
BASELINE_GLOB = "*_rw-n1"


def run_eval(label: str, flags: list[str]) -> Path:
    cmd = [sys.executable, "-m", "docs_assistant.evaluate", "--split", "dev", "--label", label, *flags]
    print(f"\n=== {label}: {' '.join(flags)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=config.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    saved = None
    for line in proc.stdout:
        if line.startswith("[") or line.startswith("Evaluating") or "error" in line.lower():
            print(line, end="", flush=True)
        if line.startswith("Saved to "):
            saved = Path(line.removeprefix("Saved to ").strip())
    if proc.wait() != 0 or saved is None:
        raise SystemExit(f"{label} failed (exit {proc.returncode}); see output above")
    return saved


def metrics(cfg: dict, rows: list[dict]) -> dict:
    s = summarize(rows, cfg["k"], resamples=1000)
    ok = [r for r in rows if not r.get("error")]
    return {
        "pass": s["overall_pass_rate"]["value"],
        "hit@1": s["retrieval"]["hit@1"]["value"],
        "gold": s["retrieval"]["context_recall"]["value"],
        "nuggets": s["answers"]["vital_nugget_recall"]["value"],
        "refusals": s["answers"]["correct_refusal_rate"]["value"],
        "false_ref": s["answers"]["false_refusal_rate"]["value"],
        "p50": percentile([r["retrieve_ms"] + r["generate_ms"] for r in ok], .5),
        "cost": s["cost_usd"]["answers_per_query"],
        "errors": s["n_errors"],
    }


def render(base: tuple, runs: list[tuple[str, tuple]]) -> str:
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"
    lines = ["| Run | Pass | Δ pass vs baseline | hit@1 | Gold in context | Vital nuggets | Trick Qs refused | "
             "False refusals | p50 latency | $/query | Fixed / broken |", "|" + "---|" * 11]
    b = metrics(*base)
    lines.append(f"| baseline ({base[0]['label']}) | {pct(b['pass'])} | - | {pct(b['hit@1'])} | {pct(b['gold'])} | "
                 f"{pct(b['nuggets'])} | {pct(b['refusals'])} | {pct(b['false_ref'])} | {b['p50']:,.0f} ms | "
                 f"${b['cost']:.5f} | - |")
    details = []
    for label, run in runs:
        m = metrics(*run)
        c = compare(base, run, resamples=2000)
        d = c["deltas"]["pass"]
        delta = f"{d['diff']:+.1%}{' (significant)' if d['significant'] else ''}" if d else "n/a"
        lines.append(f"| {label} | {pct(m['pass'])} | {delta} | {pct(m['hit@1'])} | {pct(m['gold'])} | "
                     f"{pct(m['nuggets'])} | {pct(m['refusals'])} | {pct(m['false_ref'])} | {m['p50']:,.0f} ms | "
                     f"${m['cost']:.5f} | {len(c['fixed'])} / {len(c['broken'])} |")
        details.append(f"- **{label}** fixed: {', '.join(c['fixed']) or '-'} · broken: {', '.join(c['broken']) or '-'}"
                       + (f" · {m['errors']} errored" if m["errors"] else ""))
    return "\n".join(lines + [""] + details) + "\n"


def latest(pattern: str) -> Path | None:
    found = sorted(glob.glob(str(config.EVAL_RUNS_DIR / pattern)))
    return Path(found[-1]) if found else None


def finalize() -> None:
    """After shipping: one sealed test-split run with the shipped defaults, Ragas on
    the shipped dev run vs the previous one, and the README results table."""
    test_dir = run_eval_split("final-v3-llm", "test", ["--confirm-test"])
    ragas_runs = [p for p in (latest("*_rw-n1"), latest("*_v3-llm")) if p]
    print("\n=== Ragas: " + " vs ".join(p.name for p in ragas_runs), flush=True)
    subprocess.run([sys.executable, "-m", "docs_assistant.ragas_eval", *map(str, ragas_runs), "--yes"],
                   cwd=config.ROOT, check=True)
    table_runs = [p for p in (latest("*_baseline"), latest("*_rw-n1"), latest("*_v3-llm"),
                              latest("*_final"), test_dir) if p]
    subprocess.run([sys.executable, "-m", "docs_assistant.results", *map(str, table_runs), "--write-readme"],
                   cwd=config.ROOT, check=True)


def run_eval_split(label: str, split: str, flags: list[str]) -> Path:
    cmd = [sys.executable, "-m", "docs_assistant.evaluate", "--split", split, "--label", label, *flags]
    print(f"\n=== {label} ({split})", flush=True)
    proc = subprocess.Popen(cmd, cwd=config.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    saved = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("Saved to "):
            saved = Path(line.removeprefix("Saved to ").strip())
    if proc.wait() != 0 or saved is None:
        raise SystemExit(f"{label} failed (exit {proc.returncode})")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Run the experiment round and compare with the baseline.")
    parser.add_argument("--only", help="comma-separated subset of: " + ", ".join(ROUND))
    parser.add_argument("--finalize", action="store_true",
                        help="test-split run with shipped settings + Ragas + README results table")
    args = parser.parse_args()
    if args.finalize:
        finalize()
        return
    names = args.only.split(",") if args.only else list(ROUND)

    base_dirs = sorted(glob.glob(str(config.EVAL_RUNS_DIR / BASELINE_GLOB)))
    if not base_dirs:
        raise SystemExit(f"No baseline run matching {BASELINE_GLOB} in {config.EVAL_RUNS_DIR}")
    base = load_run(Path(base_dirs[-1]))

    runs = [(name, load_run(run_eval(name, ROUND[name]))) for name in names]
    md = f"# Experiment round vs {base[0]['label']} (dev)\n\n" + render(base, runs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = config.EVAL_RUNS_DIR / f"{stamp}_experiments.md"
    out.write_text(md)
    print("\n" + md + f"\nSaved to {out}")


if __name__ == "__main__":
    main()
