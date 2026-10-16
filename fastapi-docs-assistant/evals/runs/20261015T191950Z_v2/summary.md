# Eval run: v2 (dev split)

2026-10-15T19:19:50+00:00 · answer model `gpt-6-luna` · judge `openai:gpt-6.1-sol` · k=5 · prompt `492e346962` · judge prompt `e6a451cb83` · golden `6b2672c521` · docs `5f9fc5c59a`

Answer prompt v2 · retrieval: neighbors ±1 · query rewrite on · context p50 2364 tokens

> **41 of 41 golden items are unreviewed drafts.** Numbers are provisional until the set is human-approved.

Values are means with bootstrap 95% confidence intervals over items.

| Metric | Value |
|---|---|
| **Overall pass rate** | **82.9% [70.7%, 92.7%] (n=41)** |
| Exact match (factual) | 100.0% [100.0%, 100.0%] (n=7) |
| LLM-judge pass rate (open) | 92.9% [82.1%, 100.0%] (n=28) |
| Vital nugget recall (open) | 94.6% [85.7%, 100.0%] (n=28) |
| Correct refusals (unanswerable) | 16.7% [0.0%, 50.0%] (n=6) |
| False refusals (answerable) | 0.0% [0.0%, 0.0%] (n=35) |
| Retrieval hit@1 | 57.1% [40.0%, 74.3%] (n=35) |
| Retrieval hit@5 | 85.7% [74.3%, 97.1%] (n=35) |
| Retrieval recall@5 | 81.0% [68.6%, 91.9%] (n=35) |
| MRR@5 | 0.682 [0.550, 0.808] (n=35) |
| Gold section in context | 94.3% [85.7%, 100.0%] (n=35) |
| Invalid citations | 0.0% [0.0%, 0.0%] (n=41) |
| Unsupported claims (lexical check) | 14.5% |
| Ungrounded code blocks | 15.4% |
| **Latency end-to-end p50 / p95** | **6075 / 10264 ms** |
| Latency retrieve p50 / p95 | 2730 / 4238 ms |
| Latency generate p50 / p95 | 3612 / 6898 ms |
| Latency inside retrieval p50 | rewrite 2296 ms |
| Cost | $0.001074 per query; $0.04404 answers + $0.17646 judge |
| Errors | 0 |

## By category

| Category | n | Pass rate | hit@5 |
|---|---|---|---|
| code | 5 | 100.0% | 80.0% |
| exact-fact | 7 | 100.0% | 100.0% |
| exact-term | 4 | 100.0% | 100.0% |
| multi-section | 4 | 75.0% | 75.0% |
| ranking | 5 | 100.0% | 100.0% |
| unanswerable-oos | 3 | 0.0% | n/a |
| unanswerable-trap | 3 | 33.3% | n/a |
| vocab-mismatch | 10 | 90.0% | 70.0% |
