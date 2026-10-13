"""Step 7: the eval harness. Runs every golden question through the real
pipeline and scores it automatically.

    exact  items -> exact match against exact_accept (no LLM)
    open   items -> LLM judge against nuggets (critique -> per-nugget -> pass/fail)
    abstain items -> pass only if the system refused

plus section-level retrieval metrics, refusal accuracy, citation/grounding
checks, latency and cost, all with bootstrap 95% confidence intervals.

Run:  python -m docs_assistant.evaluate --split dev --label baseline
      python -m docs_assistant.evaluate --rejudge evals/runs/<run-dir>
      python -m docs_assistant.evaluate --split test --label final --confirm-test

Each run writes evals/runs/<timestamp>_<label>/ with config.json,
results.jsonl, summary.json and summary.md.
"""
import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .eval_metrics import bootstrap_ci, exact_match, mean, nugget_scores, retrieval_scores
from .golden import GOLDEN_PATH, GoldenItem, load_items, load_sections, validate
from .judge import judge_answer, judge_prompt_version
from .rag import answer_question
from .tracing import prompt_version
from .usage import USAGE_VERSION, analyze_usage


# ---------------------------------------------------------------- scoring one item

def score_item(item: GoldenItem, result, judge, k: int) -> dict:
    hits = result.retrieved
    cited = [c.number for c in result.citations]
    row = {
        "id": item.id, "split": item.split, "category": item.category, "answer_type": item.answer_type,
        "review_status": item.review.status, "question": item.question,
        "answer": result.answer, "answerable_predicted": result.answerable,
        "citations": cited, "invalid_citations": result.invalid_citations,
        "retrieved": [{"rank": h.rank, "chunk_id": h.chunk_id, "similarity": round(h.similarity, 4)} for h in hits],
        "retrieve_ms": round(result.retrieve_ms, 1), "generate_ms": round(result.generate_ms, 1),
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "answer_cost_usd": round(result.cost_usd, 6),
        "usage": analyze_usage(result.answer, {h.rank: h.document.page_content for h in hits}, cited,
                               result.answerable),
        "retrieval": None, "exact_match": None, "judge": None, "nuggets": None,
        "judge_cost_usd": 0.0, "false_refusal": None, "error": None,
    }

    if item.answer_type != "abstain":
        row["retrieval"] = retrieval_scores([h.chunk_id for h in hits], item.gold_sections, k)
        row["false_refusal"] = not result.answerable

    if item.answer_type == "abstain":
        row["passed"] = not result.answerable
    elif item.answer_type == "exact":
        row["exact_match"] = result.answerable and exact_match(result.answer, item.exact_accept)
        row["passed"] = row["exact_match"]
    elif not result.answerable:
        row["passed"] = False                 # a refusal can't contain the nuggets; don't pay to judge it
    else:
        row.update(judge_open_answer(item, result.answer, judge))
    return row


def judge_open_answer(item: GoldenItem, answer: str, judge) -> dict:
    nuggets = [n.model_dump() for n in item.nuggets]
    verdict = judge_answer(judge, question=item.question, reference=item.expected_answer,
                           nuggets=nuggets, answer=answer)
    cost = (verdict["input_tokens"] * config.JUDGE_USD_PER_1M_INPUT
            + verdict["output_tokens"] * config.JUDGE_USD_PER_1M_OUTPUT) / 1e6
    return {"judge": verdict, "nuggets": nugget_scores(nuggets, verdict["nugget_verdicts"]),
            "passed": verdict["passed"], "judge_cost_usd": round(cost, 6)}


def error_row(item: GoldenItem, error: Exception) -> dict:
    return {"id": item.id, "split": item.split, "category": item.category, "answer_type": item.answer_type,
            "review_status": item.review.status, "question": item.question,
            "error": f"{type(error).__name__}: {error}", "passed": False}


# ---------------------------------------------------------------- running

def run_items(items: list[GoldenItem], vectorstore, generator, judge, *, k: int, workers: int = 1,
              progress=print) -> list[dict]:
    def one(item: GoldenItem) -> dict:
        try:
            result = answer_question(item.question, vectorstore, generator, k=k)
            return score_item(item, result, judge, k)
        except Exception as e:                 # one failing question must not sink the run
            return error_row(item, e)

    rows: list[dict] = [None] * len(items)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, item): i for i, item in enumerate(items)}
        for done, future in enumerate(futures, start=1):
            i = futures[future]
            rows[i] = future.result()
            r = rows[i]
            mark = "ERR " if r.get("error") else ("pass" if r["passed"] else "FAIL")
            progress(f"[{done:>3}/{len(items)}] {mark}  {r['id']:<10} {r['question'][:70]}")
    return rows


def rejudge_rows(rows: list[dict], items: dict[str, GoldenItem], judge) -> list[dict]:
    """Re-score open answers from an earlier run without re-answering."""
    out = []
    for row in rows:
        item = items.get(row["id"])
        row = dict(row)
        if item and item.answer_type == "open" and not row.get("error") and row["answerable_predicted"]:
            row.update(judge_open_answer(item, row["answer"], judge))
        out.append(row)
    return out


# ---------------------------------------------------------------- summarizing

def rate(rows: list[dict], key, *, resamples: int) -> dict:
    values = [float(key(r)) for r in rows if key(r) is not None]
    ci = bootstrap_ci(values, resamples=resamples)
    return {"value": mean(values), "ci95": ci, "n": len(values)}


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, round(p * (len(values) - 1)))] if values else None


def summarize(rows: list[dict], k: int, resamples: int = config.BOOTSTRAP_RESAMPLES) -> dict:
    ok = [r for r in rows if not r.get("error")]
    answerable = [r for r in ok if r["answer_type"] != "abstain"]
    by_type = lambda t: [r for r in ok if r["answer_type"] == t]
    claims = sum(r["usage"]["n_cited_claims"] for r in ok)
    code = sum(r["usage"]["n_code_blocks"] for r in ok)
    R = lambda rs, key: rate(rs, key, resamples=resamples)

    summary = {
        "n_items": len(rows), "n_errors": len(rows) - len(ok),
        "review_status": dict(Counter(r["review_status"] for r in rows)),
        "overall_pass_rate": R(rows, lambda r: r["passed"]),
        "answers": {
            "exact_match": R(by_type("exact"), lambda r: r["exact_match"]),
            "judge_pass_rate": R(by_type("open"), lambda r: r["passed"]),
            "vital_nugget_recall": R([r for r in by_type("open") if r["nuggets"]],
                                     lambda r: r["nuggets"]["vital_strict"]),
            "correct_refusal_rate": R(by_type("abstain"), lambda r: r["passed"]),
            "false_refusal_rate": R(answerable, lambda r: r["false_refusal"]),
        },
        "retrieval": {
            "hit@1": R(answerable, lambda r: r["retrieval"]["hit@1"]),
            f"hit@{k}": R(answerable, lambda r: r["retrieval"][f"hit@{k}"]),
            f"recall@{k}": R(answerable, lambda r: r["retrieval"][f"recall@{k}"]),
            "mrr": R(answerable, lambda r: r["retrieval"]["reciprocal_rank"]),
        },
        "grounding": {
            "invalid_citation_rate": R(ok, lambda r: bool(r["invalid_citations"])),
            "unsupported_claim_rate": (sum(r["usage"]["n_unsupported_claims"] for r in ok) / claims) if claims else None,
            "ungrounded_code_block_rate": (sum(r["usage"]["n_ungrounded_code_blocks"] for r in ok) / code) if code else None,
            "judge_verdicts_overridden": sum(1 for r in ok if r.get("judge") and r["judge"]["verdict_overridden"]),
        },
        "latency_ms": {
            "retrieve_p50": percentile([r["retrieve_ms"] for r in ok], .5),
            "retrieve_p95": percentile([r["retrieve_ms"] for r in ok], .95),
            "generate_p50": percentile([r["generate_ms"] for r in ok], .5),
            "generate_p95": percentile([r["generate_ms"] for r in ok], .95),
        },
        "cost_usd": {
            "answers_total": round(sum(r["answer_cost_usd"] for r in ok), 5),
            "answers_per_query": round(mean([r["answer_cost_usd"] for r in ok]) or 0, 6),
            "judge_total": round(sum(r.get("judge_cost_usd", 0) for r in ok), 5),
        },
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == cat]
        hits = [r["retrieval"][f"hit@{k}"] for r in rs if r.get("retrieval")]
        summary["by_category"][cat] = {
            "n": len(rs), "pass_rate": mean([float(r["passed"]) for r in rs]),
            f"hit@{k}": mean([float(h) for h in hits]) if hits else None,
        }
    return summary


def fmt(metric: dict | float | None, pct: bool = True) -> str:
    if metric is None:
        return "n/a"
    if isinstance(metric, dict):
        v, ci, n = metric["value"], metric["ci95"], metric["n"]
        if v is None:
            return "n/a"
        f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
        return f"{f(v)} [{f(ci[0])}, {f(ci[1])}] (n={n})" if ci else f"{f(v)} (n={n})"
    return f"{metric:.1%}" if pct else f"{metric:.3f}"


def latency_lines(lat: dict) -> list[str]:
    if lat["retrieve_p50"] is None:
        return ["| Latency | n/a |"]
    return [f"| Latency retrieve p50 / p95 | {lat['retrieve_p50']:.0f} / {lat['retrieve_p95']:.0f} ms |",
            f"| Latency generate p50 / p95 | {lat['generate_p50']:.0f} / {lat['generate_p95']:.0f} ms |"]


def render_markdown(summary: dict, cfg: dict) -> str:
    k = cfg["k"]
    s, a, r, g, lat, cost = (summary, summary["answers"], summary["retrieval"], summary["grounding"],
                             summary["latency_ms"], summary["cost_usd"])
    drafts = s["review_status"].get("draft", 0)
    lines = [
        f"# Eval run: {cfg['label']} ({cfg['split']} split)",
        "",
        f"{cfg['timestamp']} · answer model `{cfg['chat_model']}` · judge `{cfg['judge_model']}` · "
        f"k={k} · prompt `{cfg['prompt_version']}` · judge prompt `{cfg['judge_prompt_version']}` · "
        f"golden `{cfg['golden_sha256'][:10]}` · docs `{(cfg['corpus_commit'] or '')[:10]}`",
        "",
    ]
    if drafts:
        lines += [f"> **{drafts} of {s['n_items']} golden items are unreviewed drafts.** "
                  "Numbers are provisional until the set is human-approved.", ""]
    lines += [
        "Values are means with bootstrap 95% confidence intervals over items.",
        "",
        "| Metric | Value |", "|---|---|",
        f"| **Overall pass rate** | **{fmt(s['overall_pass_rate'])}** |",
        f"| Exact match (factual) | {fmt(a['exact_match'])} |",
        f"| LLM-judge pass rate (open) | {fmt(a['judge_pass_rate'])} |",
        f"| Vital nugget recall (open) | {fmt(a['vital_nugget_recall'])} |",
        f"| Correct refusals (unanswerable) | {fmt(a['correct_refusal_rate'])} |",
        f"| False refusals (answerable) | {fmt(a['false_refusal_rate'])} |",
        f"| Retrieval hit@1 | {fmt(r['hit@1'])} |",
        f"| Retrieval hit@{k} | {fmt(r[f'hit@{k}'])} |",
        f"| Retrieval recall@{k} | {fmt(r[f'recall@{k}'])} |",
        f"| MRR@{k} | {fmt(r['mrr'], pct=False)} |",
        f"| Invalid citations | {fmt(g['invalid_citation_rate'])} |",
        f"| Unsupported claims (lexical check) | {fmt(g['unsupported_claim_rate'])} |",
        f"| Ungrounded code blocks | {fmt(g['ungrounded_code_block_rate'])} |",
        *latency_lines(lat),
        f"| Cost | ${cost['answers_per_query']} per query; ${cost['answers_total']} answers + ${cost['judge_total']} judge |",
        f"| Errors | {s['n_errors']} |",
        "",
        "## By category", "",
        f"| Category | n | Pass rate | hit@{k} |", "|---|---|---|---|",
    ]
    for cat, c in s["by_category"].items():
        lines.append(f"| {cat} | {c['n']} | {fmt(c['pass_rate'])} | {fmt(c[f'hit@{k}'])} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- run directory

def git_commit() -> str | None:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, capture_output=True, text=True)
    return out.stdout.strip() or None


def run_config(*, label: str, split: str, k: int, golden_path: Path, corpus_commit: str | None,
               extra: dict | None = None) -> dict:
    return {
        "label": label, "split": split, "k": k,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_model": config.CHAT_MODEL, "judge_model": config.JUDGE_MODEL,
        "prompt_version": prompt_version(), "judge_prompt_version": judge_prompt_version(),
        "embedding_model": config.EMBEDDING_MODEL, "collection": config.COLLECTION_NAME,
        "chunk_max_tokens": config.CHUNK_MAX_TOKENS, "usage_version": USAGE_VERSION,
        "corpus_commit": corpus_commit, "golden_path": str(golden_path.relative_to(config.ROOT))
        if golden_path.is_relative_to(config.ROOT) else str(golden_path),
        "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        "code_commit": git_commit(), "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
        **(extra or {}),
    }


def write_run(rows: list[dict], cfg: dict, runs_dir: Path = config.EVAL_RUNS_DIR) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / f"{stamp}_{cfg['label']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = summarize(rows, cfg["k"], cfg.get("bootstrap_resamples", config.BOOTSTRAP_RESAMPLES))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "summary.md").write_text(render_markdown(summary, cfg))
    return run_dir


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    return cfg, rows


# ---------------------------------------------------------------- CLI

def load_golden(path: Path) -> tuple[list[GoldenItem], str | None]:
    sections, commit = load_sections()
    items, errors = validate(load_items(path), sections, commit)
    if errors:
        raise SystemExit(f"Golden set is invalid ({len(errors)} problems). Run: python -m docs_assistant.golden validate")
    return items, commit


def main():
    parser = argparse.ArgumentParser(description="Run the eval harness.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--label", default="run")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--limit", type=int, help="only the first N items (smoke test)")
    parser.add_argument("--ids", help="comma-separated item ids")
    parser.add_argument("--workers", type=int, default=config.EVAL_WORKERS)
    parser.add_argument("--rejudge", type=Path, help="re-score an earlier run's answers with the current judge")
    parser.add_argument("--confirm-test", action="store_true", help="required to run the held-out test split")
    args = parser.parse_args()

    from .ask import explain_model_error, load_pipeline
    from .judge import build_judge, get_judge_model

    items, corpus_commit = load_golden(args.golden)
    judge = build_judge(get_judge_model())

    if args.rejudge:
        old_cfg, old_rows = load_run(args.rejudge)
        rows = rejudge_rows(old_rows, {i.id: i for i in items}, judge)
        cfg = {**old_cfg, "label": f"{old_cfg['label']}-rejudged", "judge_model": config.JUDGE_MODEL,
               "judge_prompt_version": judge_prompt_version(), "rejudged_from": str(args.rejudge)}
        run_dir = write_run(rows, cfg)
        print((run_dir / "summary.md").read_text())
        print(f"Saved to {run_dir}")
        return

    if args.split == "test" and not args.confirm_test:
        raise SystemExit("The test split is held out for final numbers. Iterate on --split dev; "
                         "add --confirm-test only for the final run.")
    if args.split != "all":
        items = [i for i in items if i.split == args.split]
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i.id in wanted]
    if args.limit:
        items = items[: args.limit]

    n_open = sum(i.answer_type == "open" for i in items)
    estimate = len(items) * 1500 * config.CHAT_USD_PER_1M_INPUT / 1e6 + n_open * (
        2000 * config.JUDGE_USD_PER_1M_INPUT + 400 * config.JUDGE_USD_PER_1M_OUTPUT) / 1e6
    print(f"Evaluating {len(items)} items ({args.split}) · answer {config.CHAT_MODEL} · judge {config.JUDGE_MODEL}"
          f" · ~${estimate:.2f} estimated\n")

    vectorstore, generator = load_pipeline()
    started = time.perf_counter()
    try:
        rows = run_items(items, vectorstore, generator, judge, k=args.k, workers=args.workers)
    except Exception as e:
        explain_model_error(e)
        raise
    cfg = run_config(label=args.label, split=args.split, k=args.k, golden_path=args.golden,
                     corpus_commit=corpus_commit, extra={"wall_seconds": round(time.perf_counter() - started, 1)})
    run_dir = write_run(rows, cfg)
    errors = [r for r in rows if r.get("error")]
    print("\n" + (run_dir / "summary.md").read_text())
    if errors:
        print(f"{len(errors)} item(s) errored, first: {errors[0]['id']}: {errors[0]['error']}")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
