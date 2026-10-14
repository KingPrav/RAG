"""Step 9: build the README results table from eval runs, so the numbers are
generated rather than typed by hand.

    python -m docs_assistant.results evals/runs/<baseline> evals/runs/<run2> ...
    python -m docs_assistant.results <runs...> --write-readme     # replace the table in README.md

Each row after the first shows its change in pass rate against the first run
(paired bootstrap on the same items; * = 95% CI excludes zero).
"""
import argparse
import re
from pathlib import Path

from . import config
from .compare import compare, setting
from .evaluate import load_run, summarize

START, END = "<!-- results:start -->", "<!-- results:end -->"
README = config.ROOT / "README.md"


def describe(cfg: dict) -> str:
    parts = [f"k={cfg['k']}"]
    if setting(cfg, "neighbors"):
        parts.append(f"neighbors ±{setting(cfg, 'neighbors')}")
    if setting(cfg, "rewrite"):
        parts.append("query rewrite")
    return ", ".join(parts)


def pct(m: dict | None) -> str:
    if not m or m.get("value") is None:
        return "n/a"
    ci = m.get("ci95")
    return f"{m['value']:.1%}" + (f" [{ci[0]:.0%}–{ci[1]:.0%}]" if ci else "")


def table(runs: list[tuple[dict, list[dict]]], focus: str = "vocab-mismatch") -> str:
    first = runs[0]
    lines = ["| Run | Split | Retrieval | Pass rate [95% CI] | Δ vs first | Paraphrased pass | "
             "Gold in context | False refusals | $/query | p50 latency |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, (cfg, rows) in enumerate(runs):
        k = cfg["k"]
        s = summarize(rows, k, resamples=2000)
        focus_rows = [r for r in rows if r["category"] == focus]
        focus_pass = f"{sum(r['passed'] for r in focus_rows)}/{len(focus_rows)}" if focus_rows else "n/a"
        if i == 0 or cfg["split"] != first[0]["split"]:
            delta = "-"
        else:
            d = compare(first, (cfg, rows), resamples=2000)["deltas"]["pass"]
            delta = f"{d['diff']:+.1%}{'*' if d['significant'] else ''}" if d else "n/a"
        lat = s["latency_ms"]
        latency = (f"{(lat['retrieve_p50'] or 0) + (lat['generate_p50'] or 0):,.0f} ms"
                   if lat["retrieve_p50"] is not None else "n/a")
        lines.append(
            f"| {cfg['label']} | {cfg['split']} | {describe(cfg)} | {pct(s['overall_pass_rate'])} | {delta} | "
            f"{focus_pass} | {pct(s['retrieval'].get('context_recall'))} | "
            f"{pct(s['answers']['false_refusal_rate'])} | ${s['cost_usd']['answers_per_query']:.5f} | {latency} |")
    drafts = sum(1 for r in runs[-1][1] if r.get("review_status") == "draft")
    note = (f"\n\n_{drafts} golden items in the last run are unreviewed drafts; numbers are provisional._"
            if drafts else "")
    return "\n".join(lines) + "\n\n`*` = the 95% CI of the paired difference excludes zero." + note


def write_readme(md: str, path: Path = README) -> None:
    text = path.read_text()
    block = f"{START}\n{md}\n{END}"
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.replace("## Roadmap", f"## Results\n\n{block}\n\n## Roadmap")
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Render a results table from eval runs.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()
    md = table([load_run(r) for r in args.runs])
    print(md)
    if args.write_readme:
        write_readme(md)
        print(f"\nUpdated {README}")


if __name__ == "__main__":
    main()
