# Eval run: v3-llm (dev split)

2026-10-15T21:50:58+00:00 · answer model `gpt-6-luna` · judge `openai:gpt-6.1-sol` · k=5 · prompt `101ff39c93` · judge prompt `e6a451cb83` · golden `6b2672c521` · docs `5f9fc5c59a`

Answer prompt v3 · retrieval: neighbors ±1 · query rewrite on · rerank llm (top 20) · context p50 2230 tokens

> **41 of 41 golden items are unreviewed drafts.** Numbers are provisional until the set is human-approved.

Values are means with bootstrap 95% confidence intervals over items.

| Metric | Value |
|---|---|
| **Overall pass rate** | **100.0% [100.0%, 100.0%] (n=41)** |
| Exact match (factual) | 100.0% [100.0%, 100.0%] (n=7) |
| LLM-judge pass rate (open) | 100.0% [100.0%, 100.0%] (n=28) |
| Vital nugget recall (open) | 100.0% [100.0%, 100.0%] (n=28) |
| Correct refusals (unanswerable) | 100.0% [100.0%, 100.0%] (n=6) |
| False refusals (answerable) | 0.0% [0.0%, 0.0%] (n=35) |
| Retrieval hit@1 | 80.0% [65.7%, 91.4%] (n=35) |
| Retrieval hit@5 | 94.3% [85.7%, 100.0%] (n=35) |
| Retrieval recall@5 | 89.8% [80.7%, 97.1%] (n=35) |
| MRR@5 | 0.867 [0.767, 0.952] (n=35) |
| Gold section in context | 97.1% [91.4%, 100.0%] (n=35) |
| Invalid citations | 0.0% [0.0%, 0.0%] (n=41) |
| Unsupported claims (lexical check) | 4.8% |
| Ungrounded code blocks | 4.2% |
| **Latency end-to-end p50 / p95** | **6484 / 10794 ms** |
| Latency retrieve p50 / p95 | 3311 / 6042 ms |
| Latency generate p50 / p95 | 2470 / 6598 ms |
| Latency inside retrieval p50 | rewrite 1623 ms · rerank 1358 ms |
| Cost | $0.002052 per query; $0.08414 answers + $0.17044 judge |
| Errors | 0 |

## By category

| Category | n | Pass rate | hit@5 |
|---|---|---|---|
| code | 5 | 100.0% | 100.0% |
| exact-fact | 7 | 100.0% | 100.0% |
| exact-term | 4 | 100.0% | 100.0% |
| multi-section | 4 | 100.0% | 100.0% |
| ranking | 5 | 100.0% | 100.0% |
| unanswerable-oos | 3 | 100.0% | n/a |
| unanswerable-trap | 3 | 100.0% | n/a |
| vocab-mismatch | 10 | 100.0% | 80.0% |
