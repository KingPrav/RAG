# Eval run: v3-llm-norw (dev split)

2026-10-15T21:52:16+00:00 · answer model `gpt-6-luna` · judge `openai:gpt-6.1-sol` · k=5 · prompt `101ff39c93` · judge prompt `e6a451cb83` · golden `6b2672c521` · docs `5f9fc5c59a`

Answer prompt v3 · retrieval: neighbors ±1 · query rewrite off · rerank llm (top 20) · context p50 2377 tokens

> **41 of 41 golden items are unreviewed drafts.** Numbers are provisional until the set is human-approved.

Values are means with bootstrap 95% confidence intervals over items.

| Metric | Value |
|---|---|
| **Overall pass rate** | **97.6% [92.7%, 100.0%] (n=41)** |
| Exact match (factual) | 100.0% [100.0%, 100.0%] (n=7) |
| LLM-judge pass rate (open) | 96.4% [89.3%, 100.0%] (n=28) |
| Vital nugget recall (open) | 100.0% [100.0%, 100.0%] (n=27) |
| Correct refusals (unanswerable) | 100.0% [100.0%, 100.0%] (n=6) |
| False refusals (answerable) | 2.9% [0.0%, 8.6%] (n=35) |
| Retrieval hit@1 | 77.1% [62.9%, 91.4%] (n=35) |
| Retrieval hit@5 | 88.6% [77.1%, 97.1%] (n=35) |
| Retrieval recall@5 | 81.9% [70.0%, 91.9%] (n=35) |
| MRR@5 | 0.824 [0.705, 0.929] (n=35) |
| Gold section in context | 97.1% [91.4%, 100.0%] (n=35) |
| Invalid citations | 0.0% [0.0%, 0.0%] (n=41) |
| Unsupported claims (lexical check) | 3.4% |
| Ungrounded code blocks | 8.3% |
| **Latency end-to-end p50 / p95** | **4381 / 9790 ms** |
| Latency retrieve p50 / p95 | 1631 / 3372 ms |
| Latency generate p50 / p95 | 2297 / 8169 ms |
| Latency inside retrieval p50 | rerank 1331 ms |
| Cost | $0.00191 per query; $0.07831 answers + $0.16427 judge |
| Errors | 0 |

## By category

| Category | n | Pass rate | hit@5 |
|---|---|---|---|
| code | 5 | 100.0% | 100.0% |
| exact-fact | 7 | 100.0% | 100.0% |
| exact-term | 4 | 100.0% | 100.0% |
| multi-section | 4 | 100.0% | 100.0% |
| ranking | 5 | 100.0% | 80.0% |
| unanswerable-oos | 3 | 100.0% | n/a |
| unanswerable-trap | 3 | 100.0% | n/a |
| vocab-mismatch | 10 | 90.0% | 70.0% |
