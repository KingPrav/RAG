# Retrieval sweep (dev split, k=5, 35 answerable items)

| Strategy | hit@1 | hit@5 | MRR | Gold in context | vocab-mismatch gold in context | Context tokens p50 | Retrieval p50 | Rewrite / rerank p50 |
|---|---|---|---|---|---|---|---|---|
| baseline | 51.4% [34%, 69%] | 77.1% [63%, 91%] | 0.615 | **77.1% [63%, 91%]** | 50.0% [20%, 80%] | 1088 | 510 ms | 0 / 0 ms |
| rw+n1 | 57.1% [40%, 74%] | 80.0% [66%, 91%] | 0.664 | **94.3% [86%, 100%]** | 80.0% [50%, 100%] | 2268 | 2997 ms | 2084 / 0 ms |
| rw+n1+ce | 54.3% [37%, 71%] | 82.9% [69%, 94%] | 0.650 | **91.4% [80%, 100%]** | 70.0% [40%, 100%] | 2616 | 5939 ms | 2084 / 2376 ms |
| rw+n1+llm | 82.9% [71%, 94%] | 94.3% [86%, 100%] | 0.881 | **97.1% [91%, 100%]** | 90.0% [70%, 100%] | 2177 | 5742 ms | 2084 / 1748 ms |
| rwfast+n1 | 54.3% [37%, 71%] | 77.1% [63%, 91%] | 0.649 | **97.1% [91%, 100%]** | 90.0% [70%, 100%] | 2205 | 2569 ms | 2042 / 0 ms |
| rwfast+n1+ce | 54.3% [37%, 71%] | 82.9% [69%, 94%] | 0.645 | **91.4% [80%, 100%]** | 70.0% [40%, 100%] | 2502 | 5597 ms | 2042 / 2639 ms |

hit@1 / hit@k / MRR score the search ranking; *gold in context* asks whether a gold section is anywhere in what the model is shown (neighbors included).

## Still missing

- **baseline** (8): vocab-001, vocab-009, vocab-011, vocab-013, vocab-019, rank-005, code-009, multi-003
- **rw+n1** (2) · LLM cost (rewrite/rerank) $0.0052: vocab-001, vocab-017
- **rw+n1+ce** (3) · 3 rerank errors: vocab-001, vocab-009, vocab-019
- **rw+n1+llm** (1) · LLM cost (rewrite/rerank) $0.0355: vocab-001
- **rwfast+n1** (1) · LLM cost (rewrite/rerank) $0.0044: vocab-001
- **rwfast+n1+ce** (3): vocab-001, vocab-009, vocab-019
