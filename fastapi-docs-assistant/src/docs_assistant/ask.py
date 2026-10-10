"""Ask the FastAPI docs assistant a question. Every question is traced to
data/logs/traces.jsonl (disable with --no-log).

Run:  python -m docs_assistant.ask "how do I declare a request body?"
      python -m docs_assistant.ask "..." --show-retrieved
"""
import argparse

from . import config
from .rag import answer_question, build_generator, get_chat_model
from .tracing import log_result
from .vectorstore import get_embeddings, get_vectorstore

MODEL_LIST_CMD = 'python -c "from openai import OpenAI; print(sorted(m.id for m in OpenAI().models.list()))"'


def load_pipeline():
    vectorstore = get_vectorstore(get_embeddings())
    if vectorstore._collection.count() == 0:
        raise SystemExit("The index is empty. Run: ingest -> chunk -> index (see README).")
    return vectorstore, build_generator(get_chat_model())


def explain_model_error(e: Exception) -> None:
    if type(e).__name__ == "NotFoundError":
        raise SystemExit(f"Model '{config.CHAT_MODEL}' was not found for your API key.\n"
                         f"Set CHAT_MODEL in .env to one of the models listed by:\n  {MODEL_LIST_CMD}")


def main():
    parser = argparse.ArgumentParser(description="Ask a question about FastAPI.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--show-retrieved", action="store_true", help="list every retrieved chunk")
    parser.add_argument("--no-log", action="store_true", help="don't write a trace")
    args = parser.parse_args()

    vectorstore, generator = load_pipeline()
    try:
        result = answer_question(args.question, vectorstore, generator, k=args.k)
    except Exception as e:
        explain_model_error(e)
        raise

    print(result.answer.strip())
    if not result.answerable:
        print("\n(The assistant marked this question as not answerable from the docs.)")

    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  [{c.number}] {c.url}   (similarity {c.similarity:.3f})")
    if result.invalid_citations:
        print(f"\nWARNING: cited sources that were never retrieved: {result.invalid_citations}")

    if args.show_retrieved:
        cited = {c.chunk_id for c in result.citations}
        print("\nRetrieved:")
        for h in result.retrieved:
            mark = "cited" if h.chunk_id in cited else "     "
            print(f"  [{h.rank}] {h.similarity:.3f}  {mark}  {h.chunk_id}")

    print(f"\n{result.model} · retrieve {result.retrieve_ms:.0f} ms · generate {result.generate_ms:.0f} ms"
          f" · {result.input_tokens}+{result.output_tokens} tokens · ${result.cost_usd:.5f}")

    if not args.no_log:
        trace = log_result(result, args.k)
        if trace:
            u = trace["usage"]
            code = (f" · code grounded {u['n_code_blocks'] - u['n_ungrounded_code_blocks']}/{u['n_code_blocks']}"
                    if u["n_code_blocks"] else "")
            print(f"trace {trace['trace_id'][:8]} · used {u['n_cited']}/{u['n_retrieved']} chunks"
                  f" · unsupported claims {u['n_unsupported_claims']}{code}")


if __name__ == "__main__":
    main()
