"""Ragas metrics on saved eval runs: a second, independent opinion next to
our own judge, using the field's standard RAG metrics.

    python -m docs_assistant.ragas_eval evals/runs/<baseline> evals/runs/<shipped>
    python -m docs_assistant.ragas_eval evals/runs/<run> --limit 3          # smoke test

Nothing is re-answered: each run's saved answers are re-scored against exactly
what the model was shown (rebuilt from the chunk IDs, neighbors included).

| Metric                         | Question                                            | Scored on |
|--------------------------------|-----------------------------------------------------|-----------|
| faithfulness                   | Is every claim in the answer supported by the context? | answered items |
| context_recall                 | Does the context cover the reference answer?        | answerable items |
| factual_correctness (recall)   | Does the answer contain the reference's claims?     | answered, answerable items |

Factual correctness uses recall mode: our reference answers are short, so F1
would penalize correct extra detail. Refusals get no faithfulness score (there
is no claim to check). Scores are cached in <run>/ragas.jsonl; failed scores
are counted and reported, never silently dropped.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from . import config
from .doc_io import load_documents
from .eval_metrics import bootstrap_ci, mean, paired_bootstrap
from .evaluate import load_run

METRICS = ("faithfulness", "context_recall", "factual_correctness")
CALL_TIMEOUT_S = 240


# ---------------------------------------------------------------- inputs

def load_chunk_texts(path: Path = config.CHUNKS_PATH) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m docs_assistant.chunk")
    return {d.metadata["chunk_id"]: d.page_content for d in load_documents(path)}


def build_samples(rows: list[dict], golden: dict[str, dict], chunk_texts: dict[str, str]) -> list[dict]:
    """One sample per scorable row, with which metrics apply to it."""
    samples = []
    for r in rows:
        if r.get("error"):
            continue
        item = golden.get(r["id"])
        if item is None:
            continue
        missing = [h["chunk_id"] for h in r["retrieved"] if h["chunk_id"] not in chunk_texts]
        if missing:
            raise SystemExit(f"{r['id']}: chunks {missing[:3]} not in chunks.jsonl; was the corpus rebuilt "
                             "since this run? Ragas needs the exact text the model saw.")
        answerable_item = item["answer_type"] != "abstain"
        answered = bool(r.get("answerable_predicted"))
        metrics = [m for m, applies in (("faithfulness", answered),
                                        ("context_recall", answerable_item),
                                        ("factual_correctness", answered and answerable_item)) if applies]
        if metrics:
            samples.append({"id": r["id"], "category": r["category"], "user_input": r["question"],
                            "response": r["answer"], "reference": item["expected_answer"],
                            "retrieved_contexts": [chunk_texts[h["chunk_id"]] for h in r["retrieved"]],
                            "metrics": metrics})
    return samples


def estimate_cost(samples: list[dict]) -> tuple[int, int, float]:
    """Rough token estimate: faithfulness and context recall each read the context
    once; factual correctness decomposes and checks claims (several short calls)."""
    tin = tout = 0
    for s in samples:
        ctx = sum(len(c) for c in s["retrieved_contexts"]) // 4
        if "faithfulness" in s["metrics"]:
            tin, tout = tin + ctx + 1500, tout + 600
        if "context_recall" in s["metrics"]:
            tin, tout = tin + ctx + 600, tout + 300
        if "factual_correctness" in s["metrics"]:
            tin, tout = tin + 3000, tout + 700
    usd = (tin * config.RAGAS_USD_PER_1M_INPUT + tout * config.RAGAS_USD_PER_1M_OUTPUT) / 1e6
    return tin, tout, usd


# ---------------------------------------------------------------- the judge model

class UsageMeter:
    def __init__(self):
        self.input_tokens = self.output_tokens = self.calls = 0

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * config.RAGAS_USD_PER_1M_INPUT
                + self.output_tokens * config.RAGAS_USD_PER_1M_OUTPUT) / 1e6


def normalize_params(kwargs: dict) -> dict:
    """Ragas guesses API parameters from the model name, and misses names like
    'gpt-6.1-sol'. Current OpenAI reasoning models reject temperature/top_p and
    want max_completion_tokens, so normalize every call here."""
    kwargs = dict(kwargs)
    kwargs.pop("temperature", None)
    kwargs.pop("top_p", None)
    limit = max(kwargs.pop("max_tokens", 0) or 0, kwargs.pop("max_completion_tokens", 0) or 0, 8192)
    kwargs["max_completion_tokens"] = limit
    return kwargs


def make_client(meter: UsageMeter, **client_kwargs):
    """AsyncOpenAI client whose chat.completions.create normalizes parameters
    and meters token usage. Ragas wraps this exact method with instructor."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(**{"max_retries": 4, "timeout": CALL_TIMEOUT_S, **client_kwargs})
    original = client.chat.completions.create

    async def create(*args, **kwargs):
        response = await original(*args, **normalize_params(kwargs))
        usage = getattr(response, "usage", None)
        if usage is not None:
            meter.calls += 1
            meter.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            meter.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        return response

    client.chat.completions.create = create
    return client


def build_metrics(model: str, meter: UsageMeter, **client_kwargs) -> dict:
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, FactualCorrectness, Faithfulness
    llm = llm_factory(model, provider="openai", client=make_client(meter, **client_kwargs))
    return {"faithfulness": Faithfulness(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "factual_correctness": FactualCorrectness(llm=llm, mode="recall")}


# ---------------------------------------------------------------- scoring

ARGS = {
    "faithfulness": lambda s: dict(user_input=s["user_input"], response=s["response"],
                                   retrieved_contexts=s["retrieved_contexts"]),
    "context_recall": lambda s: dict(user_input=s["user_input"], retrieved_contexts=s["retrieved_contexts"],
                                     reference=s["reference"]),
    "factual_correctness": lambda s: dict(response=s["response"], reference=s["reference"]),
}


def as_float(result) -> float | None:
    value = getattr(result, "value", result)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value else value          # NaN -> None (reported as an error)


async def score_samples(samples: list[dict], metrics: dict, workers: int = 4, progress=print) -> list[dict]:
    semaphore = asyncio.Semaphore(max(1, workers))
    done = 0

    async def one_metric(sample: dict, name: str) -> tuple[float | None, str | None]:
        async with semaphore:
            try:
                result = await asyncio.wait_for(metrics[name].ascore(**ARGS[name](sample)), CALL_TIMEOUT_S)
            except Exception as e:                            # one failure must not sink the run
                return None, f"{type(e).__name__}: {str(e)[:200]}"
        value = as_float(result)
        return (value, None) if value is not None else (None, "metric returned no score (NaN)")

    async def one(sample: dict) -> dict:
        nonlocal done
        results = await asyncio.gather(*(one_metric(sample, m) for m in sample["metrics"]))
        row = {"id": sample["id"], "category": sample["category"], "errors": {}}
        for name, (value, error) in zip(sample["metrics"], results):
            row[name] = value
            if error:
                row["errors"][name] = error
        done += 1
        scores = "  ".join(f"{m[:5]} {row[m]:.2f}" if row.get(m) is not None else f"{m[:5]} ERR"
                           for m in sample["metrics"])
        progress(f"[{done:>3}/{len(samples)}] {sample['id']:<10} {scores}")
        return row

    return list(await asyncio.gather(*(one(s) for s in samples)))


def summarize(rows: list[dict], resamples: int = 2000) -> dict:
    out = {}
    for m in METRICS:
        values = [r[m] for r in rows if r.get(m) is not None]
        out[m] = {"value": mean(values), "ci95": bootstrap_ci(values, resamples=resamples), "n": len(values),
                  "errors": sum(1 for r in rows if m in r.get("errors", {}))}
    return out


def compare_runs(base: list[dict], cand: list[dict], resamples: int = 2000) -> dict:
    a, b = {r["id"]: r for r in base}, {r["id"]: r for r in cand}
    shared = sorted(a.keys() & b.keys())
    return {m: paired_bootstrap([a[i].get(m) for i in shared], [b[i].get(m) for i in shared], resamples=resamples)
            for m in METRICS}


# ---------------------------------------------------------------- run folder cache

def cached(run_dir: Path, model: str) -> list[dict] | None:
    path = run_dir / "ragas.jsonl"
    meta = run_dir / "ragas_meta.json"
    if not (path.exists() and meta.exists()):
        return None
    if json.loads(meta.read_text()).get("model") != model:
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(run_dir: Path, rows: list[dict], summary: dict, meta: dict) -> None:
    (run_dir / "ragas.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "ragas_meta.json").write_text(json.dumps({**meta, "summary": summary}, indent=2))


def render(labels: list[str], summaries: list[dict], deltas: list[dict | None], model: str) -> str:
    def cell(m: dict) -> str:
        if m["value"] is None:
            return "n/a"
        ci = m["ci95"]
        err = f", {m['errors']} err" if m["errors"] else ""
        return f"{m['value']:.3f}" + (f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else "") + f" (n={m['n']}{err})"

    names = {"faithfulness": "Faithfulness", "context_recall": "Context recall",
             "factual_correctness": "Factual correctness (recall)"}
    lines = [f"## Ragas metrics (judge `{model}`)", "", "| Metric | " + " | ".join(labels) + " |",
             "|---|" + "---|" * len(labels)]
    for m in METRICS:
        cells = []
        for s, d in zip(summaries, deltas):
            c = cell(s[m])
            if d and d.get(m):
                dm = d[m]
                c += f" · Δ {dm['diff']:+.3f}{'*' if dm['significant'] else ''}"
            cells.append(c)
        lines.append(f"| {names[m]} | " + " | ".join(cells) + " |")
    lines += ["", "Δ = paired difference vs the first run on the same items; * = 95% CI excludes zero."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Score saved eval runs with Ragas metrics.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--model", default=config.RAGAS_MODEL)
    parser.add_argument("--limit", type=int, help="only the first N items per run (smoke test)")
    parser.add_argument("--workers", type=int, default=config.EVAL_WORKERS)
    parser.add_argument("--force", action="store_true", help="re-score even if cached")
    parser.add_argument("--yes", action="store_true", help="don't ask before spending")
    args = parser.parse_args()

    from .golden import GOLDEN_PATH, load_items
    golden = {g["id"]: g for g in load_items(GOLDEN_PATH)}
    chunk_texts = load_chunk_texts()

    loaded = []
    for run_dir in args.runs:
        cfg, rows = load_run(run_dir)
        samples = build_samples(rows, golden, chunk_texts)[: args.limit] if args.limit else \
            build_samples(rows, golden, chunk_texts)
        loaded.append((run_dir, cfg, samples))

    todo = [(d, c, s) for d, c, s in loaded if args.force or args.limit or cached(d, args.model) is None]
    if todo:
        tin, tout, usd = map(sum, zip(*(estimate_cost(s) for _, _, s in todo)))
        print(f"Scoring {sum(len(s) for _, _, s in todo)} items in {len(todo)} run(s) with {args.model}: "
              f"~{tin / 1e6:.2f}M input + {tout / 1e6:.2f}M output tokens, ~${usd:.2f} estimated.")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            raise SystemExit("Cancelled.")

    labels, summaries, per_run = [], [], []
    for run_dir, cfg, samples in loaded:
        rows = None if (args.force or args.limit) else cached(run_dir, args.model)
        if rows is None:
            meter = UsageMeter()
            print(f"\n{cfg['label']}: {len(samples)} items")
            started = time.perf_counter()
            rows = asyncio.run(score_samples(samples, build_metrics(args.model, meter), args.workers))
            summary = summarize(rows)
            if not args.limit:                         # smoke tests are not cached
                save(run_dir, rows, summary, {"model": args.model, "calls": meter.calls,
                                              "input_tokens": meter.input_tokens,
                                              "output_tokens": meter.output_tokens,
                                              "cost_usd": round(meter.cost_usd, 4),
                                              "seconds": round(time.perf_counter() - started, 1)})
            print(f"{cfg['label']}: {meter.calls} calls, {meter.input_tokens:,} in / {meter.output_tokens:,} out "
                  f"tokens, ${meter.cost_usd:.2f}")
        else:
            print(f"{cfg['label']}: using cached scores")
        labels.append(f"{cfg['label']} ({cfg['split']})")
        summaries.append(summarize(rows))
        per_run.append(rows)

    deltas = [None] + [compare_runs(per_run[0], rows) if loaded[i + 1][1]["split"] == loaded[0][1]["split"]
                       else None for i, rows in enumerate(per_run[1:])]
    md = render(labels, summaries, deltas, args.model)
    print("\n" + md)
    if not args.limit:
        for (run_dir, _, _), label, s in zip(loaded, labels, summaries):
            (run_dir / "ragas.md").write_text(render([label], [s], [None], args.model))


if __name__ == "__main__":
    main()
