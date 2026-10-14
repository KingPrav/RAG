"""Retrieval-only sweep: score several retrieval strategies on the golden set
without generating answers. Costs only embeddings (+ one small rewrite call
per question), so it is the cheap inner loop of Step 8; the full eval
(evaluate.py) is reserved for the strategies that win here.

Run:  python -m docs_assistant.retrieval_eval --split dev
      python -m docs_assistant.retrieval_eval --split dev --strategies baseline,n1,rw+n1

Strategy names: baseline, n1, n2 (neighbors ±1/±2), rw (rewrite), rw+n1, rw+n2.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from langchain_core.runnables import RunnableLambda

from . import config
from .eval_metrics import bootstrap_ci, mean, retrieval_scores
from .evaluate import load_golden, percentile
from .golden import GOLDEN_PATH, GoldenItem
from .retrieval import ChunkNeighbors, RetrievalSettings, Retriever, build_rewriter

STRATEGIES = {
    "baseline": dict(neighbors=0, rewrite=False),
    "n1": dict(neighbors=1, rewrite=False),
    "n2": dict(neighbors=2, rewrite=False),
    "rw": dict(neighbors=0, rewrite=True),
    "rw+n1": dict(neighbors=1, rewrite=True),
    "rw+n2": dict(neighbors=2, rewrite=True),
}


def memoized(rewriter):
    """Rewrite each question once and reuse it across strategies, so strategies
    differ only in what they do with the rewrites (and the sweep costs less)."""
    cache: dict[str, dict] = {}

    def run(inputs):
        if inputs["question"] not in cache:
            cache[inputs["question"]] = rewriter.invoke(inputs)
            return cache[inputs["question"]]
        hit = dict(cache[inputs["question"]])
        hit["raw"] = None                     # tokens were already paid for once
        return hit
    return RunnableLambda(run)


def score_strategy(items: list[GoldenItem], retriever: Retriever, k: int, workers: int) -> list[dict]:
    def one(item: GoldenItem) -> dict:
        started = time.perf_counter()
        r = retriever(item.question)
        ms = (time.perf_counter() - started) * 1000
        return {
            "id": item.id, "category": item.category, "queries": r.queries, "rewrite_error": r.rewrite_error,
            **retrieval_scores([h.chunk_id for h in r.search_hits], item.gold_sections, k,
                               context_chunk_ids=[h.chunk_id for h in r.hits]),
            "context_ids": [h.chunk_id for h in r.hits],
            "context_tokens": sum(h.document.metadata.get("n_tokens", 0) for h in r.hits),
            "rewrite_tokens": (r.input_tokens, r.output_tokens), "ms": ms,
        }
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, items))


def summarize_strategy(rows: list[dict], k: int, focus: str) -> dict:
    def r(key, subset=rows):
        values = [float(x[key]) for x in subset]
        return {"value": mean(values), "ci95": bootstrap_ci(values, resamples=2000)}
    focus_rows = [x for x in rows if x["category"] == focus]
    tin = sum(x["rewrite_tokens"][0] for x in rows)
    tout = sum(x["rewrite_tokens"][1] for x in rows)
    return {
        "n": len(rows), "hit@1": r("hit@1"), f"hit@{k}": r(f"hit@{k}"), "mrr": r("reciprocal_rank"),
        "gold_in_context": r("context_hit"),
        "focus_gold_in_context": r("context_hit", focus_rows) if focus_rows else None,
        "context_tokens_p50": percentile([x["context_tokens"] for x in rows], .5),
        "latency_ms_p50": percentile([x["ms"] for x in rows], .5),
        "rewrite_cost_usd": (tin * config.CHAT_USD_PER_1M_INPUT + tout * config.CHAT_USD_PER_1M_OUTPUT) / 1e6,
        "rewrite_errors": sum(1 for x in rows if x["rewrite_error"]),
        "misses": [x["id"] for x in rows if not x["context_hit"]],
    }


def render(summaries: dict[str, dict], k: int, focus: str, split: str) -> str:
    pct = lambda m: "n/a" if m is None or m["value"] is None else (
        f"{m['value']:.1%}" + (f" [{m['ci95'][0]:.0%}, {m['ci95'][1]:.0%}]" if m["ci95"] else ""))
    lines = [f"# Retrieval sweep ({split} split, k={k}, {next(iter(summaries.values()))['n']} answerable items)", "",
             f"| Strategy | hit@1 | hit@{k} | MRR | Gold in context | {focus} gold in context | "
             "Context tokens p50 | Latency p50 |", "|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {pct(s['hit@1'])} | {pct(s[f'hit@{k}'])} | {s['mrr']['value']:.3f} | "
                     f"**{pct(s['gold_in_context'])}** | {pct(s['focus_gold_in_context'])} | "
                     f"{s['context_tokens_p50']} | {s['latency_ms_p50']:.0f} ms |")
    lines += ["", "hit@1 / hit@k / MRR score the search ranking; *gold in context* asks whether a gold "
              "section is anywhere in what the model is shown (neighbors included).", "", "## Still missing", ""]
    for name, s in summaries.items():
        extra = f" · rewrite cost ${s['rewrite_cost_usd']:.4f}" if s["rewrite_cost_usd"] else ""
        errors = f" · {s['rewrite_errors']} rewrite errors" if s["rewrite_errors"] else ""
        lines.append(f"- **{name}** ({len(s['misses'])}){extra}{errors}: {', '.join(s['misses']) or '-'}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval strategies without generating answers.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--strategies", default="baseline,n1,n2,rw,rw+n1")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--focus", default="vocab-mismatch", help="category to report separately")
    parser.add_argument("--workers", type=int, default=config.EVAL_WORKERS)
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()

    names = [s.strip() for s in args.strategies.split(",")]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise SystemExit(f"Unknown strategies {unknown}; choose from {list(STRATEGIES)}")
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("The test split is held out; tune on dev.")

    from .rag import get_chat_model
    from .vectorstore import get_embeddings, get_vectorstore

    items, _ = load_golden(GOLDEN_PATH)
    items = [i for i in items if i.gold_sections and (args.split == "all" or i.split == args.split)]
    vectorstore = get_vectorstore(get_embeddings())
    if vectorstore._collection.count() == 0:
        raise SystemExit("The index is empty. Run: ingest -> chunk -> index (see README).")
    needs_rewrite = any(STRATEGIES[n]["rewrite"] for n in names)
    rewriter = memoized(build_rewriter(get_chat_model())) if needs_rewrite else None
    neighbors = ChunkNeighbors.from_file() if any(STRATEGIES[n]["neighbors"] for n in names) else None

    print(f"Scoring {len(names)} strategies on {len(items)} answerable {args.split} items"
          + (f" (rewrites with {config.CHAT_MODEL})" if needs_rewrite else "") + "...")
    summaries, rows_by_strategy = {}, {}
    for name in names:
        settings = RetrievalSettings(k=args.k, **STRATEGIES[name])
        retriever = Retriever(vectorstore, settings, rewriter=rewriter if settings.rewrite else None,
                              neighbors=neighbors if settings.neighbors else None)
        rows = score_strategy(items, retriever, args.k, args.workers)
        rows_by_strategy[name] = rows
        summaries[name] = summarize_strategy(rows, args.k, args.focus)
        print(f"  {name:<9} gold in context {summaries[name]['gold_in_context']['value']:.1%}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = config.EVAL_RUNS_DIR / f"{stamp}_retrieval-sweep"
    out.mkdir(parents=True)
    md = render(summaries, args.k, args.focus, args.split)
    (out / "sweep.md").write_text(md)
    (out / "sweep.json").write_text(json.dumps({"split": args.split, "k": args.k, "summaries": summaries,
                                                "rows": rows_by_strategy}, indent=2))
    print("\n" + md + f"\nSaved to {out}")


if __name__ == "__main__":
    main()
