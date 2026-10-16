# Eval run: v3 (dev split)

2026-10-15T21:49:20+00:00 · answer model `gpt-6-luna` · judge `openai:gpt-6.1-sol` · k=5 · prompt `101ff39c93` · judge prompt `e6a451cb83` · golden `6b2672c521` · docs `5f9fc5c59a`

Answer prompt v3 · retrieval: neighbors ±1 · query rewrite on · context p50 2154 tokens

> **41 of 41 golden items are unreviewed drafts.** Numbers are provisional until the set is human-approved.

Values are means with bootstrap 95% confidence intervals over items.

| Metric | Value |
|---|---|
| **Overall pass rate** | **92.7% [82.9%, 100.0%] (n=41)** |
| Exact match (factual) | 100.0% [100.0%, 100.0%] (n=7) |
| LLM-judge pass rate (open) | 89.3% [75.0%, 100.0%] (n=28) |
| Vital nugget recall (open) | 91.1% [80.4%, 100.0%] (n=28) |
| Correct refusals (unanswerable) | 100.0% [100.0%, 100.0%] (n=6) |
| False refusals (answerable) | 0.0% [0.0%, 0.0%] (n=35) |
| Retrieval hit@1 | 57.1% [40.0%, 74.3%] (n=35) |
| Retrieval hit@5 | 85.7% [74.3%, 97.1%] (n=35) |
| Retrieval recall@5 | 78.8% [66.2%, 90.5%] (n=35) |
| MRR@5 | 0.689 [0.559, 0.814] (n=35) |
| Gold section in context | 97.1% [91.4%, 100.0%] (n=35) |
| Invalid citations | 0.0% [0.0%, 0.0%] (n=41) |
| Unsupported claims (lexical check) | 10.3% |
| Ungrounded code blocks | 4.2% |
| **Latency end-to-end p50 / p95** | **4732 / 9381 ms** |
| Latency retrieve p50 / p95 | 1864 / 3460 ms |
| Latency generate p50 / p95 | 2347 / 7412 ms |
| Latency inside retrieval p50 | rewrite 1583 ms |
| Cost | $0.001036 per query; $0.04248 answers + $0.17831 judge |
| Errors | 0 |

## By category

| Category | n | Pass rate | hit@5 |
|---|---|---|---|
| code | 5 | 80.0% | 80.0% |
| exact-fact | 7 | 100.0% | 100.0% |
| exact-term | 4 | 100.0% | 100.0% |
| multi-section | 4 | 100.0% | 75.0% |
| ranking | 5 | 100.0% | 80.0% |
| unanswerable-oos | 3 | 100.0% | n/a |
| unanswerable-trap | 3 | 100.0% | n/a |
| vocab-mismatch | 10 | 80.0% | 80.0% |
