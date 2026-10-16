"""The experiment summary renders comparisons from saved runs (no model calls)."""
from docs_assistant.experiments import ROUND, render


def rows(passes):
    return [{"id": f"q{i}", "category": "c", "answer_type": "abstain" if i == 0 else "exact", "review_status": "draft",
             "passed": p, "exact_match": p, "answerable_predicted": not p if i == 0 else True, "invalid_citations": [],
             "usage": {"n_cited_claims": 0, "n_unsupported_claims": 0, "n_code_blocks": 0,
                       "n_ungrounded_code_blocks": 0},
             "answer_cost_usd": 0.001, "retrieve_ms": 2000.0, "generate_ms": 2000.0, "false_refusal": False,
             "retrieval": None if i == 0 else {"hit@1": p, "hit@5": True, "recall@5": 1.0,
                                               "reciprocal_rank": 1.0, "context_hit": True}}
            for i, p in enumerate(passes)]


def test_round_is_defined_and_summary_renders_deltas():
    assert set(ROUND) == {"v3", "v3-llm", "v3-llm-norw"}
    cfg = {"k": 5, "label": "rw-n1"}
    md = render((cfg, rows([True] * 8 + [False] * 2)), [("v3", (dict(cfg, label="v3"), rows([True] * 10)))])
    assert "| baseline (rw-n1) | 80.0%" in md and "| v3 | 100.0% | +20.0%" in md and "| 2 / 0 |" in md
    assert "v3** fixed: q8, q9" in md
