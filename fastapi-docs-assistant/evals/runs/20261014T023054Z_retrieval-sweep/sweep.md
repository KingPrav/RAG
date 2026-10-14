# Retrieval sweep (dev split, k=5, 35 answerable items)

| Strategy | hit@1 | hit@5 | MRR | Gold in context | vocab-mismatch gold in context | Context tokens p50 | Latency p50 |
|---|---|---|---|---|---|---|---|
| baseline | 51.4% [34%, 69%] | 77.1% [63%, 91%] | 0.615 | **77.1% [63%, 91%]** | 50.0% [20%, 80%] | 1088 | 264 ms |
| n1 | 51.4% [34%, 69%] | 77.1% [63%, 91%] | 0.615 | **85.7% [74%, 97%]** | 60.0% [30%, 90%] | 2427 | 243 ms |
| n2 | 51.4% [34%, 69%] | 77.1% [63%, 91%] | 0.615 | **91.4% [80%, 100%]** | 70.0% [40%, 100%] | 3282 | 289 ms |
| rw | 62.9% [46%, 77%] | 80.0% [66%, 91%] | 0.706 | **80.0% [66%, 91%]** | 70.0% [40%, 100%] | 1239 | 2438 ms |
| rw+n1 | 62.9% [46%, 77%] | 80.0% [66%, 91%] | 0.706 | **94.3% [86%, 100%]** | 80.0% [50%, 100%] | 2246 | 280 ms |

hit@1 / hit@k / MRR score the search ranking; *gold in context* asks whether a gold section is anywhere in what the model is shown (neighbors included).

## Still missing

- **baseline** (8): vocab-001, vocab-009, vocab-011, vocab-013, vocab-019, rank-005, code-009, multi-003
- **n1** (5): vocab-001, vocab-009, vocab-013, vocab-019, multi-003
- **n2** (3): vocab-001, vocab-009, vocab-013
- **rw** (7) · rewrite cost $0.0051: vocab-001, vocab-011, vocab-017, rank-005, code-005, code-009, multi-005
- **rw+n1** (2): vocab-001, vocab-017
