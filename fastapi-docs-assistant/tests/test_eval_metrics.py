import pytest

from docs_assistant.eval_metrics import (bootstrap_ci, exact_match, normalize_answer, nugget_scores,
                                         paired_bootstrap, retrieval_scores, section_of)


# ---------------------------------------------------------------- exact match

@pytest.mark.parametrize("answer, accepted, expected", [
    ("The dev server uses port **8000** by default. [1]", ["8000"], True),
    ("Docs are at `http://127.0.0.1:8000/docs`.", ["/docs"], True),
    ("Install `python-multipart` first.", ["python-multipart"], True),
    ("It defaults to `['GET']`.", ["GET"], True),
    ("Signed with HS256.", ["hs256"], True),                       # case-insensitive
    ("Responses under 1500 bytes are skipped.", ["500"], False),   # numbers match whole
    ("It caches for 5000 seconds.", ["500"], False),
    ("Responses under 500 bytes are not compressed.", ["500"], True),
    ("It listens on 127.0.0.1.", ["0.0.0.0"], False),
    ("Served at /redoc.", ["/docs"], False),
])
def test_exact_match(answer, accepted, expected):
    assert exact_match(answer, accepted) is expected


def test_normalize_strips_markdown_but_keeps_identifiers():
    assert normalize_answer("Use **`response_model_exclude_unset`**  now") == "use response_model_exclude_unset now"


# ---------------------------------------------------------------- retrieval

def test_section_of_strips_part_suffix():
    assert section_of("tutorial/body#create-your-data-model/2") == "tutorial/body#create-your-data-model"
    assert section_of("index#run-it") == "index#run-it"


def test_retrieval_scores_at_section_level():
    retrieved = ["a#x", "b#y/1", "c#z", "b#y/2", "d#w"]
    s = retrieval_scores(retrieved, ["b#y", "e#q"], k=5)
    assert s == {"context_hit": True, "hit@1": False, "hit@5": True, "recall@5": 0.5, "reciprocal_rank": 0.5,
                 "first_gold_rank": 2}


def test_context_hit_counts_neighbor_sections_but_ranking_does_not():
    s = retrieval_scores(["a#x", "b#y"], ["c#z"], k=2, context_chunk_ids=["a#x", "c#z/1", "b#y"])
    assert s["context_hit"] and not s["hit@2"] and s["reciprocal_rank"] == 0.0


def test_retrieval_miss_and_rank_one_hit():
    assert retrieval_scores(["a#x", "b#y"], ["z#z"], k=2)["reciprocal_rank"] == 0.0
    hit = retrieval_scores(["a#x/3", "b#y"], ["a#x"], k=2)
    assert hit["hit@1"] and hit["reciprocal_rank"] == 1.0


def test_retrieval_respects_k():
    assert not retrieval_scores(["a", "b", "c#gold"], ["c#gold"], k=2)["hit@2"]


# ---------------------------------------------------------------- nuggets

def test_nugget_scores_only_full_support_counts():
    nuggets = [{"vital": True}, {"vital": True}, {"vital": False}]
    s = nugget_scores(nuggets, ["support", "partial", "support"])
    assert s == {"vital_strict": 0.5, "all_strict": pytest.approx(2 / 3), "partial": 1}


def test_nugget_scores_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        nugget_scores([{"vital": True}], [])


# ---------------------------------------------------------------- statistics

def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [1.0] * 30 + [0.0] * 10                       # mean 0.75
    lo, hi = bootstrap_ci(values, resamples=2000)
    assert lo < 0.75 < hi and 0.10 < hi - lo < 0.40         # ~±13 points at n=40
    assert bootstrap_ci(values, resamples=2000) == (lo, hi)


def test_bootstrap_ci_needs_two_values():
    assert bootstrap_ci([1.0]) is None and bootstrap_ci([]) is None


def test_paired_bootstrap_detects_a_consistent_improvement():
    a = [0.0] * 20 + [1.0] * 20
    b = [1.0] * 12 + [0.0] * 8 + [1.0] * 20                 # fixes 12, breaks none
    d = paired_bootstrap(a, b, resamples=2000)
    assert d["diff"] == pytest.approx(0.3) and d["significant"] and d["ci"][0] > 0


def test_paired_bootstrap_calls_noise_not_significant():
    a = [1.0, 0.0] * 20
    b = [0.0, 1.0] * 20                                     # same rate, different items
    d = paired_bootstrap(a, b, resamples=2000)
    assert d["diff"] == 0 and not d["significant"]


def test_paired_bootstrap_skips_missing_values():
    d = paired_bootstrap([1.0, None, 0.0, 1.0], [1.0, 1.0, None, 0.0], resamples=500)
    assert d["n"] == 2
