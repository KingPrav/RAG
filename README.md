# RAG

## [FastAPI Docs Assistant](fastapi-docs-assistant/)

A production-style RAG system over the FastAPI documentation: an internal developer-support assistant that answers with citations to the exact docs section.

- **Retrieval:** LLM query rewriting + reciprocal rank fusion + neighbor-section expansion over Chroma (`text-embedding-3-small`)
- **Instrumentation:** every query, retrieved chunk, similarity score and whether the answer used it, logged as JSONL traces
- **Evaluation:** 80-question golden set targeting measured failure modes; exact match for factual answers, LLM-as-judge for open-ended ones, bootstrap confidence intervals
- **CI:** offline tests on every push, plus an eval gate that blocks regressions against a committed baseline

Dev pass rate 82.9% → 90.2% and right-section-in-context 77% → 94% after Step 8 (provisional; see the [project README](fastapi-docs-assistant/README.md) for the method, results and limitations).

Built with LangChain, OpenAI and Chroma.
