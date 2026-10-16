"""Retrieval-only sweep: score several retrieval strategies on the golden set
without generating answers. Costs only embeddings (+ one small rewrite call
per question), so it is the cheap inner loop of Step 8; the full eval
(evaluate.py) is reserved for the strategies that win here.

Run:  python -m docs_assistant.retrieval_eval --split dev
      python -m docs_assistant.retrieval_eval --split dev --strategies baseline,n1,rw+n1

Strategy names: baseline, n1, n2 (neighbors ±1/±2), rw (rewrite), rw+n1, rw+n2,
rw+n1+ce / rw+n1+llm (rerank top 20 with a cross-encoder / listwise LLM),
rwfast+n1, rwfast+n1+ce (rewrite with low reasoning effort).
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
from .rerank import build_reranker
from .retrieval import ChunkNeighbors, RetrievalSettings, Retriever, build_rewriter

STRATEGIES = {
    "baseline": dict(neighbors=0, rewrite=False),
    "n1": dict(neighbors=1, rewrite=False),
    "n2": dict(neighbors=2, rewrite=False),
    "rw": dict(neighbors=0, rewrite=True),
    "rw+n1": dict(neighbors=1, rewrite=True),
    "rw+n2": dict(neighbors=2, rewrite=True),
    "rw+n1+ce": dict(neighbors=1, rewrite=True, rerank="ce"),
    "rw+n1+llm": dict(neighbors=1, rewrite=True, rerank="llm"),
    "n1+llm": dict(neighbors=1, rewrite=False, rerank="llm"),          # rerank instead of rewriting (faster?)
    "rwfast+n1": dict(neighbors=1, rewrite=True, rewrite_effort="low"),
    "rwfast+n1+ce": dict(neighbors=1, rewrite=True, rewrite_effort="low", rerank="ce"),
}
DEFAULT_STRATEGIES = "baseline,rw+n1,rw+n1+ce,rw+n1+llm,n1+llm,rwfast+n1"


def memoized(rewriter, timings: dict | None = None):
    """Rewrite each question once and reuse it across strategies, so strategies
    differ only in what they do with the rewrites (and the sweep costs less).
    The real latency of the first call is recorded in `timings`, since reused
    rewrites would otherwise look instant."""
    cache: dict[str, dict] = {}

    def run(inputs):
        if inputs["question"] not in cache:
            started = time.perf_counter()
            cache[inputs["question"]] = rewriter.invoke(inputs)
            if timings is not None:
                timings[inputs["question"]] = (time.perf_counter() - started) * 1000
            return cache[inputs["question"]]
        hit = dict(cache[inputs["question"]])
        hit["raw"] = None                     # tokens were already paid for once
        return hit
    return RunnableLambda(run)


def score_strategy(items: list[GoldenItem], retriever: Retriever, k: int, workers: int,
                   rewrite_timings: dict | None = None) -> list[dict]:
    def one(item: GoldenItem) -> dict:
        started = time.perf_counter()
        r = retriever(item.question)
        ms = (time.perf_counter() - started) * 1000
        true_rewrite = (rewrite_timings or {}).get(item.question, r.rewrite_ms) if len(r.queries) > 1 or \
            r.rewrite_error else 0.0
        return {
            "id": item.id, "category": item.category, "queries": r.queries, "rewrite_error": r.rewrite_error,
            **retrieval_scores([h.chunk_id for h in r.search_hits], item.gold_sections, k,
                               context_chunk_ids=[h.chunk_id for h in r.hits]),
            "context_ids": [h.chunk_id for h in r.hits],
            "context_tokens": sum(h.document.metadata.get("n_tokens", 0) for h in r.hits),
            "rewrite_tokens": (r.input_tokens, r.output_tokens), "ms": ms,
            # end-to-end retrieval latency as a user would see it: the rewrite runs in
            # parallel with the original search, so add only the part that isn't overlapped
            "e2e_ms": ms - r.rewrite_ms + true_rewrite if true_rewrite else ms,
            "rewrite_ms": true_rewrite, "rerank_ms": r.rerank_ms, "rerank_error": r.rerank_error,
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
        "latency_ms_p50": percentile([x["e2e_ms"] for x in rows], .5),
        "rewrite_ms_p50": percentile([x["rewrite_ms"] for x in rows if x["rewrite_ms"]], .5),
        "rerank_ms_p50": percentile([x["rerank_ms"] for x in rows if x["rerank_ms"]], .5),
        "rerank_errors": sum(1 for x in rows if x["rerank_error"]),
        "rewrite_cost_usd": (tin * config.CHAT_USD_PER_1M_INPUT + tout * config.CHAT_USD_PER_1M_OUTPUT) / 1e6,
        "rewrite_errors": sum(1 for x in rows if x["rewrite_error"]),
        "misses": [x["id"] for x in rows if not x["context_hit"]],
    }


def render(summaries: dict[str, dict], k: int, focus: str, split: str) -> str:
    pct = lambda m: "n/a" if m is None or m["value"] is None else (
        f"{m['value']:.1%}" + (f" [{m['ci95'][0]:.0%}, {m['ci95'][1]:.0%}]" if m["ci95"] else ""))
    lines = [f"# Retrieval sweep ({split} split, k={k}, {next(iter(summaries.values()))['n']} answerable items)", "",
             f"| Strategy | hit@1 | hit@{k} | MRR | Gold in context | {focus} gold in context | "
             "Context tokens p50 | Retrieval p50 | Rewrite / rerank p50 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {pct(s['hit@1'])} | {pct(s[f'hit@{k}'])} | {s['mrr']['value']:.3f} | "
                     f"**{pct(s['gold_in_context'])}** | {pct(s['focus_gold_in_context'])} | "
                     f"{s['context_tokens_p50']} | {s['latency_ms_p50']:.0f} ms | "
                     f"{s['rewrite_ms_p50'] or 0:.0f} / {s['rerank_ms_p50'] or 0:.0f} ms |")
    lines += ["", "hit@1 / hit@k / MRR score the search ranking; *gold in context* asks whether a gold "
              "section is anywhere in what the model is shown (neighbors included).", "", "## Still missing", ""]
    for name, s in summaries.items():
        extra = f" · LLM cost (rewrite/rerank) ${s['rewrite_cost_usd']:.4f}" if s["rewrite_cost_usd"] else ""
        errors = f" · {s['rewrite_errors']} rewrite errors" if s["rewrite_errors"] else ""
        errors += f" · {s['rerank_errors']} rerank errors" if s.get("rerank_errors") else ""
        lines.append(f"- **{name}** ({len(s['misses'])}){extra}{errors}: {', '.join(s['misses']) or '-'}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval strategies without generating answers.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES)
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
    specs = {n: {"rerank": "none", "rewrite_effort": None, **STRATEGIES[n]} for n in names}
    # One memoized rewriter per reasoning effort (each records its true latency),
    # one reranker per kind, shared across strategies.
    rewriters, timings = {}, {}
    for effort in {sp["rewrite_effort"] for sp in specs.values() if sp["rewrite"]}:
        timings[effort] = {}
        rewriters[effort] = memoized(build_rewriter(get_chat_model(effort)), timings[effort])
    rerankers = {kind: build_reranker(kind, get_chat_model(config.RERANK_REASONING_EFFORT) if kind == "llm" else None)
                 for kind in {sp["rerank"] for sp in specs.values()} if kind != "none"}
    neighbors = ChunkNeighbors.from_file() if any(sp["neighbors"] for sp in specs.values()) else None

    print(f"Scoring {len(names)} strategies on {len(items)} answerable {args.split} items"
          + (f" (LLM calls with {config.CHAT_MODEL})" if rewriters or "llm" in rerankers else "") + "...")
    summaries, rows_by_strategy = {}, {}
    for name in names:
        sp = specs[name]
        settings = RetrievalSettings(k=args.k, **sp)
        retriever = Retriever(vectorstore, settings,
                              rewriter=rewriters.get(sp["rewrite_effort"]) if settings.rewrite else None,
                              neighbors=neighbors if settings.neighbors else None,
                              reranker=rerankers.get(sp["rerank"]))
        rows = score_strategy(items, retriever, args.k, args.workers,
                              rewrite_timings=timings.get(sp["rewrite_effort"]) if settings.rewrite else None)
        rows_by_strategy[name] = rows
        summaries[name] = summarize_strategy(rows, args.k, args.focus)
        print(f"  {name:<13} hit@1 {summaries[name]['hit@1']['value']:.1%} · "
              f"gold in context {summaries[name]['gold_in_context']['value']:.1%}")

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
