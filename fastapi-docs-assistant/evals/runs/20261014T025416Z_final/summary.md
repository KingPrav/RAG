# Eval run: final (test split)

2026-10-14T02:54:16+00:00 · answer model `gpt-6-luna` · judge `openai:gpt-6.1-sol` · k=5 · prompt `df19b7bc93` · judge prompt `e6a451cb83` · golden `6b2672c521` · docs `5f9fc5c59a`

Retrieval: neighbors ±1 · query rewrite on · context p50 2262 tokens

> **39 of 39 golden items are unreviewed drafts.** Numbers are provisional until the set is human-approved.

Values are means with bootstrap 95% confidence intervals over items.

| Metric | Value |
|---|---|
| **Overall pass rate** | **87.2% [76.9%, 97.4%] (n=39)** |
| Exact match (factual) | 85.7% [57.1%, 100.0%] (n=7) |
| LLM-judge pass rate (open) | 85.7% [71.4%, 96.4%] (n=28) |
| Vital nugget recall (open) | 98.1% [94.2%, 100.0%] (n=26) |
| Correct refusals (unanswerable) | 100.0% [100.0%, 100.0%] (n=4) |
| False refusals (answerable) | 8.6% [0.0%, 20.0%] (n=35) |
| Retrieval hit@1 | 42.9% [25.7%, 60.0%] (n=35) |
| Retrieval hit@5 | 85.7% [74.3%, 97.1%] (n=35) |
| Retrieval recall@5 | 78.8% [66.2%, 89.8%] (n=35) |
| MRR@5 | 0.619 [0.495, 0.743] (n=35) |
| Gold section in context | 91.4% [80.0%, 100.0%] (n=35) |
| Invalid citations | 0.0% [0.0%, 0.0%] (n=39) |
| Unsupported claims (lexical check) | 6.8% |
| Ungrounded code blocks | 5.9% |
| Latency retrieve p50 / p95 | 2306 / 3830 ms |
| Latency generate p50 / p95 | 2176 / 6710 ms |
| Cost | $0.000954 per query; $0.03719 answers + $0.16345 judge |
| Errors | 0 |

## By category

| Category | n | Pass rate | hit@5 |
|---|---|---|---|
| code | 5 | 100.0% | 80.0% |
| exact-fact | 7 | 85.7% | 71.4% |
| exact-term | 4 | 100.0% | 100.0% |
| multi-section | 4 | 100.0% | 100.0% |
| ranking | 5 | 100.0% | 100.0% |
| unanswerable-oos | 2 | 100.0% | n/a |
| unanswerable-trap | 2 | 100.0% | n/a |
| vocab-mismatch | 10 | 60.0% | 80.0% |
