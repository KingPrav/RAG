# Step 5: Failure analysis of the baseline

**Data:** 40 probe questions (`queries/probe_questions.txt`) run through the baseline
(`text-embedding-3-small`, k=5, `gpt-6-luna`, prompt `df19b7bc93`, docs commit `5f9fc5c`).
**Labels:** `analysis/failure_labels.csv`, one row per question with the correct section(s),
the rank at which they were retrieved, a verdict and a failure mode.

**Status:** draft labels by Claude, **pending human review** (column `human_review`).
Treat the numbers below as preliminary until the review is done.

## Method

Error analysis as described by Husain & Shankar: read every trace end to end (**open coding**:
a note on the *first* thing that went wrong), then group the notes into failure modes and count
them (**axial coding**). Failure modes are mapped to the RAG failure points of Barnett et al.
(2024). Two of their seven don't apply here: FP3 (retrieved but dropped during consolidation),
because there is no consolidation step, and FP5 (wrong format), because the output is structured.

Every correct answer was checked against the corpus text, not against prior knowledge of FastAPI.

## Headline results

| | Result |
|---|---|
| Verdicts | **37 pass, 1 partial, 2 fail** of 40 |
| Correct refusals | **6 / 6** questions the docs can't answer were refused (4 out-of-scope, plus #10 and #36) |
| False refusals | **2 / 35** answerable questions refused (#32, #35) |
| Invalid citations (cites a source it wasn't given) | **0 / 40** |
| Unsupported claims (human judgment) | **0** found |
| Retrieval hit@5 (answerable questions) | **33 / 35 = 94.3%** |
| Retrieval hit@1 | **23 / 35 = 65.7%** |
| MRR@5 | **0.765** |
| Cost / latency | $0.00052 per question; generation p50 2.9 s (p95 6.3 s), retrieval p50 0.2 s |

## Failure modes

| # | Failure mode | Barnett | Count | Examples | Root cause | Step 8 candidate fix |
|---|---|---|---|---|---|---|
| 1 | **Vocabulary-mismatch retrieval miss → false refusal** | FP2 | 2 / 35 answerable; **2 of the 4 vocabulary-mismatch probes** | #32 "stop users calling my API without logging in" (docs: *security*, `get_current_user`); #35 "check the API key in a header on every request" (docs: *dependencies in the decorator*, *global dependencies*, `X-Key`) | The user describes intent in their own words; the docs use framework terms. The embedding doesn't bridge the gap, so no relevant section reaches the top 5, and the grounded prompt (correctly) refuses. | Query rewriting / multi-query / HyDE to translate intent into docs vocabulary; larger k (first check whether the right section is at rank 6-20) |
| 2 | **Right section retrieved but not ranked first** | (ranking) | 10 / 33 hits at rank 2-5 | #18 `handling-errors#add-custom-headers` at rank 2 behind `response-headers`; #3, #13, #25 | Semantic neighbors (same words, different topic) outrank the precise section | Cross-encoder reranking (hit@1 is the headroom: 65.7%) |
| 3 | **Simpler, better section ignored** | FP6 | 1 | #1: `query-params#defaults` (the direct answer) ranked #1 but the answer used an advanced `Query(min_length=3)` example | The model chose the more elaborate source | Prompt: prefer the most direct source; reranking |
| 4 | **Code adapted instead of copied** | (prompt adherence) | 1 | #27: renamed `create_multiple_images(images)` to `create_items(items)` | The prompt says copy code exactly; the model generalized names | Prompt emphasis; already detected automatically by code grounding |
| 5 | **Probe question assumed a fact the docs don't state** | FP1 (eval-design error) | 1 | #10 "what status code on validation failure?": the docs never state 422 in prose, so refusing was **correct** | Question written from knowledge of FastAPI, not from the corpus | Step 6 rule: every expected answer must be verified in the corpus text |

## Assumptions the data disproved

1. **"Exact API names are where vector search struggles."** Not in this sample: the 7 exact-term
   questions (`Depends with yield`, `jsonable_encoder`, `OAuth2PasswordBearer tokenUrl`, …) all
   passed with the right section in the top 5. The real weakness is **paraphrased intent**
   (failure mode 1). Hybrid BM25 search may therefore help less than query rewriting. Both are
   still tested in Step 8, but the hypothesis order changes.
2. **"Repeated code examples will crowd retrieval"** (known issue from Step 1). Not observed: the
   most-retrieved chunk appeared 3 times in 40 questions.
3. **"A similarity threshold can filter bad retrievals."** It can't, at least not with these scores:

   | Case | Top-1 similarity | Outcome |
   |---|---|---|
   | #16 Depends with yield | 0.427 | correct section, correct answer |
   | #33 send the email later | 0.445 | correct section, correct answer |
   | #4 upload a file | 0.466 | correct section, correct answer |
   | #32 without logging in | 0.383 | **wrong sections**, false refusal |
   | #35 API key header | 0.435 | **wrong sections**, false refusal |
   | #10 validation status code | 0.561 | correct refusal |

   Good and bad retrievals overlap in score, so refusal decisions stay with the grounded prompt.

## Calibrating the "did the answer use the chunks?" checks

The first version of the deterministic checks (`usage.py` v1) disagreed badly with the hand labels.
Three causes were found by reading the flagged cases, and fixed in v3:

| Cause | Example | Fix |
|---|---|---|
| Citations after a full stop were attached to the *next* sentence | `...form data. [1] Then define a route` | Citation-only fragments attach to the preceding claim |
| The model cites once per paragraph, the check judged each sentence | `Set the default... A parameter with a default is optional. [2]` | Claims are paragraphs / list items |
| Multi-source claims were scored against one chunk at a time; paraphrase scored low | #31 cites 5 chunks for one paragraph; "adds an expiration" vs "add the expiration" | Score against the union of cited chunks; light stemming; refusals not scored |

| Signal | v1 | v3 | Hand labels |
|---|---|---|---|
| Answers with uncited claims | 30/40 (75%) | **0/40** | 0 |
| Unsupported claims | 12/24 (50%) | **2/43 (4.7%)** | 0 (both v3 flags are paraphrase false alarms: #21, #26) |
| Ungrounded code blocks | 2/24 | **1/23** | 1 (#27; the other v1 flag was a directory tree, now excluded) |

**Limitation:** the sample has no truly unsupported claim, so the check's *recall* (does it catch
a real one?) is still unmeasured. Step 6 includes adversarial cases and Step 7 adds an LLM judge
for faithfulness; the lexical check remains a cheap triage signal. Stored traces are re-scored
automatically when the heuristics change (`usage_version`).

## Implications for Step 6 (ground-truth set)

Weight the 50-100 questions toward what this analysis found, not toward what already works:

- **Paraphrased intent / vocabulary mismatch (largest share):** questions phrased the way a user
  would ask, whose answer lives under different framework terms (auth, API keys, permissions,
  "run something later", "return a file", …).
- **Ranking-sensitive questions:** a neighboring section shares the words but the precise section
  is elsewhere (headers in errors vs. responses, status codes, testing variants).
- **Exact facts with verified answers:** each expected value (port, URL, package, default) located
  in the corpus text, so exact match is fair.
- **Unanswerable traps:** out of scope, and *plausible but not stated in the docs* (like #10).
- **Code questions** where code grounding can be checked.
- **Gold labels at section level** (`page#anchor`), so they survive re-chunking in Step 8.

## Open items

- [ ] **Human review** of `analysis/failure_labels.csv` (accept/correct each draft label).
- [ ] For #32 and #35, check whether the right section is just below the cutoff:
      `python -m docs_assistant.search "How do I check the API key in a header on every request?" -k 20`.
      If it is at rank 6-20, larger k plus reranking may fix it; if absent, query rewriting is needed.

## References

- Husain & Shankar, *Why is error analysis so important in AI evals?* — hamel.dev
- Barnett et al., *Seven Failure Points When Engineering a Retrieval Augmented Generation System*, CAIN 2024 — arXiv:2401.05856
