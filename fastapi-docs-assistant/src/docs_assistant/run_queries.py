"""Ask a list of questions and trace every answer.

Run:  python -m docs_assistant.run_queries queries/probe_questions.txt
      python -m docs_assistant.report

The file has one question per line; blank lines and lines starting with #
are ignored.
"""
import argparse
import time
from pathlib import Path

from . import config
from .ask import explain_model_error, load_pipeline
from .rag import answer_question
from .tracing import log_result


def read_questions(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [q for q in lines if q and not q.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Run and trace a list of questions.")
    parser.add_argument("file", type=Path)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    questions = read_questions(args.file)
    vectorstore, generator = load_pipeline()
    print(f"Running {len(questions)} questions with {config.CHAT_MODEL} (k={args.k})...\n")

    started, failures, cost = time.perf_counter(), 0, 0.0
    for i, question in enumerate(questions, start=1):
        try:
            result = answer_question(question, vectorstore, generator, k=args.k)
        except Exception as e:                   # one bad question must not stop the batch
            explain_model_error(e)
            failures += 1
            print(f"[{i:>3}/{len(questions)}] ERROR  {question[:60]}  ({type(e).__name__}: {e})")
            continue
        log_result(result, args.k)
        cost += result.cost_usd
        top1 = result.retrieved[0].similarity if result.retrieved else 0.0
        status = "ok " if result.answerable else "n/a"
        print(f"[{i:>3}/{len(questions)}] {status}  top1 {top1:.3f}  cited {len(result.citations)}"
              f"  {result.total_ms:>5.0f} ms  {question[:60]}")

    print(f"\nDone in {time.perf_counter() - started:.0f}s · {failures} failed · ${cost:.4f}")
    print("Next: python -m docs_assistant.report")


if __name__ == "__main__":
    main()
