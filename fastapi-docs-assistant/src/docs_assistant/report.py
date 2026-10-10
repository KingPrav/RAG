"""Summarize query traces: the retrieval-quality picture for failure analysis.

Run:  python -m docs_assistant.report
      python -m docs_assistant.report --prompt-version 3f2a9c1b07   (one prompt only)
"""
import argparse
import json
import statistics
from collections import Counter

from . import config
from .tracing import TRACES_PATH, load_traces
from .usage import USAGE_VERSION, analyze_usage

SIMILARITY_BUCKETS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, round(p * (len(values) - 1)))]


def bucket_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}-{min(hi, 1.0):.1f}"


def rescore_usage(traces: list[dict], chunk_texts: dict[str, str]) -> int:
    """Recompute the usage signals of traces logged with an older version of
    the heuristics, using the chunk texts by ID. Returns how many were updated.
    Traces whose chunks no longer exist (re-chunked corpus) are left as is."""
    updated = 0
    for t in traces:
        if t["usage"].get("usage_version") == USAGE_VERSION:
            continue
        results = t["retrieval"]["results"]
        if not all(r["chunk_id"] in chunk_texts for r in results):
            continue
        g = t["generation"]
        t["usage"] = analyze_usage(g["answer"], {r["rank"]: chunk_texts[r["chunk_id"]] for r in results},
                                   g["citations"], g["answerable"])
        updated += 1
    return updated


def load_chunk_texts(path=config.PROCESSED_DIR / "chunks.jsonl") -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {(d := json.loads(line))["id"]: d["page_content"] for line in f}


def summarize(traces: list[dict], worst_n: int = 10) -> dict:
    n = len(traces)
    if n == 0:
        return {"traces": 0}

    top1 = [t["retrieval"]["results"][0]["similarity"] for t in traces if t["retrieval"]["results"]]
    histogram = {bucket_label(lo, hi): sum(lo <= s < hi for s in top1)
                 for lo, hi in zip(SIMILARITY_BUCKETS, SIMILARITY_BUCKETS[1:])}

    gen = [t["generation"] for t in traces]
    use = [t["usage"] for t in traces]
    retrieved_counter = Counter(r["chunk_id"] for t in traces for r in t["retrieval"]["results"])
    cited_counter = Counter(t["retrieval"]["results"][c - 1]["chunk_id"]
                            for t in traces for c in t["generation"]["citations"]
                            if 0 < c <= len(t["retrieval"]["results"]))
    cited_claims = sum(u["n_cited_claims"] for u in use)
    code_blocks = sum(u["n_code_blocks"] for u in use)

    worst = sorted((t for t in traces if t["retrieval"]["results"]),
                   key=lambda t: t["retrieval"]["results"][0]["similarity"])[:worst_n]

    return {
        "traces": n,
        "prompt_versions": dict(Counter(g["prompt_version"] for g in gen)),
        "models": dict(Counter(g["model"] for g in gen)),
        "latency_ms": {
            "retrieval_p50": percentile([t["retrieval"]["latency_ms"] for t in traces], .5),
            "retrieval_p95": percentile([t["retrieval"]["latency_ms"] for t in traces], .95),
            "generation_p50": percentile([g["latency_ms"] for g in gen], .5),
            "generation_p95": percentile([g["latency_ms"] for g in gen], .95),
        },
        "cost_usd": {"total": round(sum(g["cost_usd"] for g in gen), 5),
                     "per_query": round(statistics.mean(g["cost_usd"] for g in gen), 6)},
        "tokens_per_query": {"input": round(statistics.mean(g["input_tokens"] for g in gen)),
                             "output": round(statistics.mean(g["output_tokens"] for g in gen)),
                             "context": round(statistics.mean(g["context_tokens"] for g in gen))},
        "top1_similarity": {"p10": percentile(top1, .1), "median": percentile(top1, .5),
                            "p90": percentile(top1, .9), "histogram": histogram},
        "answerable_rate": round(sum(g["answerable"] for g in gen) / n, 3),
        "context_utilization_mean": round(statistics.mean(u["context_utilization"] for u in use), 3),
        "invalid_citation_rate": round(sum(bool(g["invalid_citations"]) for g in gen) / n, 3),
        "unsupported_claim_rate": round(sum(u["n_unsupported_claims"] for u in use) / cited_claims, 3)
                                  if cited_claims else None,
        "answers_with_uncited_claims": round(
            sum(1 for g, u in zip(gen, use) if g["answerable"] and u["n_uncited_claims"]) / n, 3),
        "ungrounded_code_block_rate": round(sum(u["n_ungrounded_code_blocks"] for u in use) / code_blocks, 3)
                                      if code_blocks else None,
        "most_retrieved_chunks": retrieved_counter.most_common(10),
        "retrieved_but_never_cited": sorted(c for c in retrieved_counter if c not in cited_counter)[:15],
        "lowest_top1_queries": [{"question": t["question"],
                                 "top1": t["retrieval"]["results"][0]["similarity"],
                                 "top1_chunk": t["retrieval"]["results"][0]["chunk_id"],
                                 "answerable": t["generation"]["answerable"]} for t in worst],
    }


def print_report(s: dict) -> None:
    if s["traces"] == 0:
        print(f"No traces yet in {TRACES_PATH}. Ask some questions first.")
        return
    lat, cost, tok, top = s["latency_ms"], s["cost_usd"], s["tokens_per_query"], s["top1_similarity"]
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"
    print(f"Traces: {s['traces']}   models: {s['models']}   prompt versions: {s['prompt_versions']}\n")
    print("Latency (ms)       retrieval p50 {:.0f} / p95 {:.0f}   generation p50 {:.0f} / p95 {:.0f}".format(
        lat["retrieval_p50"], lat["retrieval_p95"], lat["generation_p50"], lat["generation_p95"]))
    print(f"Cost               ${cost['total']} total, ${cost['per_query']} per query")
    print(f"Tokens per query   input {tok['input']}, output {tok['output']}, context {tok['context']}\n")
    print(f"Top-1 similarity   p10 {top['p10']:.3f}   median {top['median']:.3f}   p90 {top['p90']:.3f}")
    for label, count in top["histogram"].items():
        print(f"  {label}  {'#' * count} {count}")
    print()
    print(f"Answerable rate                  {pct(s['answerable_rate'])}")
    print(f"Context utilization (mean)       {pct(s['context_utilization_mean'])} of retrieved chunks cited")
    print(f"Invalid citations                {pct(s['invalid_citation_rate'])} of answers")
    print(f"Unsupported claims               {pct(s['unsupported_claim_rate'])} of cited claims")
    print(f"Answers with uncited claims      {pct(s['answers_with_uncited_claims'])}")
    print(f"Ungrounded code blocks           {pct(s['ungrounded_code_block_rate'])}\n")
    print("Most retrieved chunks:")
    for chunk_id, count in s["most_retrieved_chunks"]:
        print(f"  {count:>3}x  {chunk_id}")
    print("\nLowest top-1 similarity (weakest retrieval):")
    for q in s["lowest_top1_queries"]:
        flag = "" if q["answerable"] else "  [not answerable]"
        print(f"  {q['top1']:.3f}  {q['question'][:70]}{flag}\n         -> {q['top1_chunk']}")


def main():
    parser = argparse.ArgumentParser(description="Summarize query traces.")
    parser.add_argument("--prompt-version", help="only include traces from this prompt version")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()

    traces = load_traces()
    rescored = rescore_usage(traces, load_chunk_texts())
    if rescored and not args.json:
        print(f"(re-scored {rescored} older traces with usage heuristics v{USAGE_VERSION})\n")
    if args.prompt_version:
        traces = [t for t in traces if t["generation"]["prompt_version"] == args.prompt_version]
    summary = summarize(traces)
    print(json.dumps(summary, indent=2) if args.json else "", end="")
    if not args.json:
        print_report(summary)


if __name__ == "__main__":
    main()
