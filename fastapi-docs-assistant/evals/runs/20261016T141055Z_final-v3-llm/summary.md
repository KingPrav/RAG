# Eval run: final-v3-llm (test split)

2026-10-16T14:10:55+00:00 · answer model `gpt-6-luna` · judge `openai:gpt-6.1-sol` · k=5 · prompt `101ff39c93` · judge prompt `e6a451cb83` · golden `6b2672c521` · docs `5f9fc5c59a`

Answer prompt v3 · retrieval: neighbors ±1 · query rewrite on · rerank llm (top 20) · context p50 2448 tokens

> **39 of 39 golden items are unreviewed drafts.** Numbers are provisional until the set is human-approved.

Values are means with bootstrap 95% confidence intervals over items.

| Metric | Value |
|---|---|
| **Overall pass rate** | **97.4% [92.3%, 100.0%] (n=39)** |
| Exact match (factual) | 100.0% [100.0%, 100.0%] (n=7) |
| LLM-judge pass rate (open) | 96.4% [89.3%, 100.0%] (n=28) |
| Vital nugget recall (open) | 98.2% [94.6%, 100.0%] (n=28) |
| Correct refusals (unanswerable) | 100.0% [100.0%, 100.0%] (n=4) |
| False refusals (answerable) | 0.0% [0.0%, 0.0%] (n=35) |
| Retrieval hit@1 | 80.0% [65.7%, 91.4%] (n=35) |
| Retrieval hit@5 | 97.1% [91.4%, 100.0%] (n=35) |
| Retrieval recall@5 | 88.8% [80.0%, 96.0%] (n=35) |
| MRR@5 | 0.872 [0.778, 0.952] (n=35) |
| Gold section in context | 97.1% [91.4%, 100.0%] (n=35) |
| Invalid citations | 0.0% [0.0%, 0.0%] (n=39) |
| Unsupported claims (lexical check) | 8.3% |
| Ungrounded code blocks | 5.0% |
| **Latency end-to-end p50 / p95** | **7739 / 12274 ms** |
| Latency retrieve p50 / p95 | 4500 / 6896 ms |
| Latency generate p50 / p95 | 3209 / 7831 ms |
| Latency inside retrieval p50 | rewrite 2056 ms · rerank 2132 ms |
| Cost | $0.002081 per query; $0.08116 answers + $0.16885 judge |
| Errors | 0 |

## By category

| Category | n | Pass rate | hit@5 |
|---|---|---|---|
| code | 5 | 100.0% | 100.0% |
| exact-fact | 7 | 100.0% | 85.7% |
| exact-term | 4 | 100.0% | 100.0% |
| multi-section | 4 | 100.0% | 100.0% |
| ranking | 5 | 100.0% | 100.0% |
| unanswerable-oos | 2 | 100.0% | n/a |
| unanswerable-trap | 2 | 100.0% | n/a |
| vocab-mismatch | 10 | 90.0% | 100.0% |
