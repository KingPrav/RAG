# Eval run: baseline (dev split)

2026-10-13T17:11:02+00:00 · answer model `gpt-6-luna` · judge `openai:gpt-6.1-sol` · k=5 · prompt `df19b7bc93` · judge prompt `e6a451cb83` · golden `6b2672c521` · docs `5f9fc5c59a`

> **41 of 41 golden items are unreviewed drafts.** Numbers are provisional until the set is human-approved.

Values are means with bootstrap 95% confidence intervals over items.

| Metric | Value |
|---|---|
| **Overall pass rate** | **82.9% [70.7%, 92.7%] (n=41)** |
| Exact match (factual) | 100.0% [100.0%, 100.0%] (n=7) |
| LLM-judge pass rate (open) | 75.0% [57.1%, 89.3%] (n=28) |
| Vital nugget recall (open) | 85.3% [70.7%, 96.0%] (n=25) |
| Correct refusals (unanswerable) | 100.0% [100.0%, 100.0%] (n=6) |
| False refusals (answerable) | 8.6% [0.0%, 17.1%] (n=35) |
| Retrieval hit@1 | 51.4% [34.3%, 68.6%] (n=35) |
| Retrieval hit@5 | 77.1% [62.9%, 91.4%] (n=35) |
| Retrieval recall@5 | 70.0% [55.7%, 83.3%] (n=35) |
| MRR@5 | 0.615 [0.478, 0.752] (n=35) |
| Invalid citations | 0.0% [0.0%, 0.0%] (n=41) |
| Unsupported claims (lexical check) | 5.0% |
| Ungrounded code blocks | 5.6% |
| Latency retrieve p50 / p95 | 246 / 1467 ms |
| Latency generate p50 / p95 | 2600 / 8384 ms |
| Cost | $0.000501 per query; $0.02053 answers + $0.15065 judge |
| Errors | 0 |

## By category

| Category | n | Pass rate | hit@5 |
|---|---|---|---|
| code | 5 | 100.0% | 80.0% |
| exact-fact | 7 | 100.0% | 100.0% |
| exact-term | 4 | 100.0% | 100.0% |
| multi-section | 4 | 75.0% | 75.0% |
| ranking | 5 | 80.0% | 80.0% |
| unanswerable-oos | 3 | 100.0% | n/a |
| unanswerable-trap | 3 | 100.0% | n/a |
| vocab-mismatch | 10 | 50.0% | 50.0% |
