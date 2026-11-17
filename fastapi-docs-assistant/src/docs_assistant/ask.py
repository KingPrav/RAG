"""Ask the FastAPI docs assistant a question. Every question is traced to
data/logs/traces.jsonl (disable with --no-log).

By default questions go through the serving layer (semantic cache + router,
see serve.py). --direct runs the pipeline directly with the retrieval flags
below, as in Phases 1-3 (no cache, no routing).

Run:  python -m docs_assistant.ask "how do I declare a request body?"
      python -m docs_assistant.ask "..." --show-retrieved
      python -m docs_assistant.ask "..." --route fast       # force a path (skips the cache)
      python -m docs_assistant.ask "..." --no-cache
      python -m docs_assistant.ask "..." --direct --rerank ce
"""
import argparse

from . import config
from .rag import PROMPTS, answer_question, build_generator, get_chat_model
from .retrieval import RetrievalSettings, build_retriever
from .tracing import log_result
from .vectorstore import get_embeddings, get_vectorstore

MODEL_LIST_CMD = 'python -c "from openai import OpenAI; print(sorted(m.id for m in OpenAI().models.list()))"'


def load_pipeline(settings: RetrievalSettings | None = None, prompt: str = config.ANSWER_PROMPT):
    """Vector store, answer generator and retriever. Retrieval settings default
    to config (RETRIEVAL_NEIGHBORS, QUERY_REWRITE in .env)."""
    vectorstore = get_vectorstore(get_embeddings())
    if vectorstore._collection.count() == 0:
        raise SystemExit("The index is empty. Run: ingest -> chunk -> index (see README).")
    llm = get_chat_model()
    retriever = build_retriever(vectorstore, settings or RetrievalSettings.shipped(), llm)
    return vectorstore, build_generator(llm, prompt), retriever


def add_retrieval_args(parser) -> None:
    parser.add_argument("-k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--neighbors", type=int, default=config.RETRIEVAL_NEIGHBORS,
                        help="also send N sections either side of each hit (0 = off)")
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=config.QUERY_REWRITE,
                        help="rewrite the question into FastAPI vocabulary before searching")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default=config.ANSWER_PROMPT,
                        help="answer prompt version (see rag.PROMPTS)")
    parser.add_argument("--rerank", choices=["none", "ce", "llm"], default=config.RERANK,
                        help="rerank the top candidates: local cross-encoder or listwise LLM")
    parser.add_argument("--rerank-candidates", type=int, default=config.RERANK_CANDIDATES)
    parser.add_argument("--rewrite-effort", default=config.REWRITE_REASONING_EFFORT,
                        help="reasoning effort for the rewrite call, e.g. low (default: model default)")


def settings_from_args(args) -> RetrievalSettings:
    return RetrievalSettings(k=args.k, neighbors=args.neighbors, rewrite=args.rewrite, rerank=args.rerank,
                             rerank_candidates=args.rerank_candidates, rewrite_effort=args.rewrite_effort or None)


def explain_model_error(e: Exception) -> None:
    if type(e).__name__ == "NotFoundError":
        raise SystemExit(f"Model '{config.CHAT_MODEL}' was not found for your API key.\n"
                         f"Set CHAT_MODEL in .env to one of the models listed by:\n  {MODEL_LIST_CMD}")


def print_answer(answer: str, answerable: bool, citations: list[dict], invalid: list[int]) -> None:
    print(answer.strip())
    if not answerable:
        print("\n(The assistant marked this question as not answerable from the docs.)")
    if citations:
        print("\nSources:")
        for c in citations:
            print(f"  [{c['number']}] {c['url']}   (similarity {c['similarity']:.3f})")
    if invalid:
        print(f"\nWARNING: cited sources that were never retrieved: {invalid}")


def print_retrieved(result) -> None:
    cited = {c.chunk_id for c in result.citations}
    print("\nRetrieved:")
    if len(result.queries) > 1:
        print("Searched for:\n" + "\n".join(f"  - {q}" for q in result.queries))
    for h in result.retrieved:
        mark = "cited" if h.chunk_id in cited else "     "
        via = "" if h.source == "search" else f"  ({h.source})"
        via += f"  rerank {h.rerank_score:.3f}" if h.rerank_score is not None else ""
        print(f"  [{h.rank}] {h.similarity:.3f}  {mark}  {h.chunk_id}{via}")


def print_usage_line(trace: dict | None) -> None:
    if trace and trace.get("usage"):
        u = trace["usage"]
        code = (f" · code grounded {u['n_code_blocks'] - u['n_ungrounded_code_blocks']}/{u['n_code_blocks']}"
                if u["n_code_blocks"] else "")
        print(f"trace {trace['trace_id'][:8]} · used {u['n_cited']}/{u['n_retrieved']} chunks"
              f" · unsupported claims {u['n_unsupported_claims']}{code}")
    elif trace:
        print(f"trace {trace['trace_id'][:8]}")


def run_direct(args) -> None:
    settings = settings_from_args(args)
    vectorstore, generator, retriever = load_pipeline(settings, args.prompt)
    try:
        result = answer_question(args.question, vectorstore, generator, k=args.k, retriever=retriever,
                                 prompt=args.prompt)
    except Exception as e:
        explain_model_error(e)
        raise
    print_answer(result.answer, result.answerable, [c.__dict__ for c in result.citations], result.invalid_citations)
    if args.show_retrieved:
        print_retrieved(result)
    print(f"\n{result.model} · retrieve {result.retrieve_ms:.0f} ms · generate {result.generate_ms:.0f} ms"
          f" · {result.input_tokens}+{result.output_tokens} tokens · ${result.cost_usd:.5f}")
    if not args.no_log:
        print_usage_line(log_result(result, args.k, settings=settings.describe()))


def run_served(args) -> None:
    from .serve import build_assistant
    from .tracing import log_served
    assistant = build_assistant(use_cache=not args.no_cache, routing=not args.no_routing, prompt=args.prompt)
    try:
        served = assistant.answer(args.question, force_route=args.route)
    except Exception as e:
        explain_model_error(e)
        raise
    invalid = served.result.invalid_citations if served.result else []
    print_answer(served.answer, served.answerable, served.citations, invalid)
    if args.show_retrieved and served.result:
        print_retrieved(served.result)

    t = served.timings_ms
    if served.route == "cache":
        c = served.cache
        how = "exact repeat" if c["exact"] else f"similarity {c['similarity']:.3f}"
        print(f"\ncache hit ({how}) · answered earlier as: \"{c['source_question']}\""
              f" · {t['total']:.0f} ms · ${served.cost_usd:.7f}")
    else:
        r = served.result
        why = "; ".join(served.decision["reasons"])
        print(f"\nroute {served.route} ({why}) · {r.model}")
        print(f"total {t['total']:.0f} ms (embed {t.get('embed', 0):.0f} · cache {t.get('cache_lookup', 0):.0f}"
              f" · route {t.get('route', 0):.0f} · retrieve {t['retrieve']:.0f} · generate {t['generate']:.0f})"
              f" · {r.input_tokens}+{r.output_tokens} tokens · ${served.cost_usd:.5f}")
        c = served.cache
        if c.get("best_similarity") is not None:
            print(f"cache miss (nearest {c['best_similarity']:.3f}: \"{c['nearest_question']}\")")
        if c.get("skip_reason"):
            print(f"not cached: {c['skip_reason']}")
    if not args.no_log:
        print_usage_line(log_served(served, args.k, assistant.fingerprint))


def main():
    parser = argparse.ArgumentParser(description="Ask a question about FastAPI.")
    parser.add_argument("question")
    add_retrieval_args(parser)
    parser.add_argument("--show-retrieved", action="store_true", help="list every retrieved chunk")
    parser.add_argument("--no-log", action="store_true", help="don't write a trace")
    serving = parser.add_argument_group("serving (cache + routing)")
    serving.add_argument("--direct", action="store_true",
                         help="bypass cache and router; use the retrieval flags above")
    serving.add_argument("--no-cache", action="store_true", help="don't read or write the semantic cache")
    serving.add_argument("--no-routing", action="store_true", help="send every cache miss to the standard path")
    serving.add_argument("--route", choices=["fast", "standard", "escalate"], help="force a path (skips the cache)")
    args = parser.parse_args()
    (run_direct if args.direct else run_served)(args)


if __name__ == "__main__":
    main()
