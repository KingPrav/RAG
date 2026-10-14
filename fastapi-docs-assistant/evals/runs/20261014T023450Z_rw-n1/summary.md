# Eval run: rw-n1 (dev split)

2026-10-14T02:34:50+00:00 · answer model `gpt-6-luna` · judge `openai:gpt-6.1-sol` · k=5 · prompt `df19b7bc93` · judge prompt `e6a451cb83` · golden `6b2672c521` · docs `5f9fc5c59a`

Retrieval: neighbors ±1 · query rewrite on · context p50 2231 tokens

> **41 of 41 golden items are unreviewed drafts.** Numbers are provisional until the set is human-approved.

Values are means with bootstrap 95% confidence intervals over items.

| Metric | Value |
|---|---|
| **Overall pass rate** | **90.2% [80.5%, 97.6%] (n=41)** |
| Exact match (factual) | 100.0% [100.0%, 100.0%] (n=7) |
| LLM-judge pass rate (open) | 85.7% [71.4%, 96.4%] (n=28) |
| Vital nugget recall (open) | 89.3% [76.8%, 98.2%] (n=28) |
| Correct refusals (unanswerable) | 100.0% [100.0%, 100.0%] (n=6) |
| False refusals (answerable) | 0.0% [0.0%, 0.0%] (n=35) |
| Retrieval hit@1 | 57.1% [40.0%, 74.3%] (n=35) |
| Retrieval hit@5 | 82.9% [68.6%, 94.3%] (n=35) |
| Retrieval recall@5 | 76.7% [63.8%, 88.6%] (n=35) |
| MRR@5 | 0.674 [0.537, 0.803] (n=35) |
| Gold section in context | 94.3% [85.7%, 100.0%] (n=35) |
| Invalid citations | 0.0% [0.0%, 0.0%] (n=41) |
| Unsupported claims (lexical check) | 8.0% |
| Ungrounded code blocks | 0.0% |
| Latency retrieve p50 / p95 | 2377 / 3370 ms |
| Latency generate p50 / p95 | 2074 / 5939 ms |
| Cost | $0.000985 per query; $0.0404 answers + $0.17742 judge |
| Errors | 0 |

## By category

| Category | n | Pass rate | hit@5 |
|---|---|---|---|
| code | 5 | 100.0% | 60.0% |
| exact-fact | 7 | 100.0% | 100.0% |
| exact-term | 4 | 75.0% | 100.0% |
| multi-section | 4 | 75.0% | 75.0% |
| ranking | 5 | 100.0% | 100.0% |
| unanswerable-oos | 3 | 100.0% | n/a |
| unanswerable-trap | 3 | 100.0% | n/a |
| vocab-mismatch | 10 | 80.0% | 70.0% |
