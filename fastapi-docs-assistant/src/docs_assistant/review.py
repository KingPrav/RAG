"""Print logged questions in full, for hand review during failure analysis.

Run:  python -m docs_assistant.review 32                 (by position in the log, 0-based)
      python -m docs_assistant.review "API key"          (questions containing the text)
      python -m docs_assistant.review --refused          (every answer marked not answerable)
      python -m docs_assistant.review --flagged          (unsupported claims or ungrounded code)
"""
import argparse

from .report import load_chunk_texts, rescore_usage
from .tracing import load_traces


def show(i: int, t: dict) -> None:
    g, u, r = t["generation"], t["usage"], t["retrieval"]
    print(f"{'=' * 100}\n#{i}  {t['question']}")
    print(f"answerable={g['answerable']}  model={g['model']}  prompt={g['prompt_version']}  "
          f"retrieve {r['latency_ms']:.0f} ms · generate {g['latency_ms']:.0f} ms · ${g['cost_usd']}")
    print("\nRetrieved (* = cited):")
    for res in r["results"]:
        mark = "*" if res["rank"] in g["citations"] else " "
        print(f"  [{res['rank']}] {res['similarity']:.3f} {mark} {res['chunk_id']}")
    print("\nAnswer:\n  " + g["answer"].strip().replace("\n", "\n  "))
    print(f"\nUsage: used {u['n_cited']}/{u['n_retrieved']} chunks · claims {u.get('n_claims')} "
          f"(uncited {u.get('n_uncited_claims')}, unsupported {u.get('n_unsupported_claims')}) · "
          f"claim support {u.get('claim_support')} · code grounding {u['code_grounding']}")
    if g["invalid_citations"]:
        print(f"Invalid citations: {g['invalid_citations']}")


def main():
    parser = argparse.ArgumentParser(description="Print logged questions in full.")
    parser.add_argument("query", nargs="?", help="log position (number) or text to search for")
    parser.add_argument("--refused", action="store_true", help="answers marked not answerable")
    parser.add_argument("--flagged", action="store_true", help="unsupported claims or ungrounded code")
    args = parser.parse_args()

    traces = load_traces()
    rescore_usage(traces, load_chunk_texts())
    selected = list(enumerate(traces))
    if args.query and args.query.isdigit():
        selected = [(int(args.query), traces[int(args.query)])]
    elif args.query:
        selected = [(i, t) for i, t in selected if args.query.lower() in t["question"].lower()]
    if args.refused:
        selected = [(i, t) for i, t in selected if not t["generation"]["answerable"]]
    if args.flagged:
        selected = [(i, t) for i, t in selected
                    if t["usage"].get("n_unsupported_claims") or t["usage"]["n_ungrounded_code_blocks"]]
    if not selected:
        print("No matching traces.")
    for i, t in selected:
        show(i, t)


if __name__ == "__main__":
    main()
