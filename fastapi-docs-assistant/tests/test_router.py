"""Router: text signals, features from search hits, and the decision order."""
import pytest

from docs_assistant.router import (ESCALATE, FAST, STANDARD, RouterPolicy, extract_features, forced, route)

CONFIDENT = [("tutorial/cors", 0.74), ("tutorial/cors", 0.66), ("tutorial/cors", 0.61),
             ("advanced/middleware", 0.55), ("tutorial/cors", 0.52)]
SCATTERED = [("a", 0.58), ("b", 0.57), ("c", 0.565), ("d", 0.56), ("e", 0.55)]


def decide(question, hits=CONFIDENT, **policy):
    return route(extract_features(question, hits), RouterPolicy(**policy))


def test_features_from_hits():
    f = extract_features("How do I add CORS?", CONFIDENT)
    assert f.top1 == 0.74 and f.margin == pytest.approx(0.08)
    assert f.top_page_share == 0.8 and f.n_pages == 2
    assert f.n_words == 5 and not (f.comparison or f.multi_part or f.specific_value)


def test_no_hits_does_not_crash_and_is_not_fast():
    d = route(extract_features("How do I add CORS?", []))
    assert d.route == STANDARD


def test_confident_single_page_question_goes_fast():
    d = decide("How do I add CORS?")
    assert d.route == FAST


@pytest.mark.parametrize("question", [
    "What is the difference between Depends and Security?",
    "Should I use BackgroundTasks vs Celery?",
    "Pydantic models compared to dataclasses for request bodies",
])
def test_comparisons_escalate(question):
    assert decide(question).route == ESCALATE


def test_comparisons_stay_standard_when_escalation_is_off():
    assert decide("Depends vs Security?", escalate=False).route == STANDARD


def test_multi_part_escalates_only_when_scattered():
    q = "How do I set a cookie and how do I read it back?"
    assert decide(q, SCATTERED).route == ESCALATE
    d = decide(q, CONFIDENT)                          # one page holds both answers
    assert d.route == STANDARD and "multi-part" in d.reasons


@pytest.mark.parametrize("question, reason", [
    ("What is the default value of response_model_exclude_unset?", "specific value"),
    ("Which version of FastAPI added lifespan events?", "specific value"),
    ("How do I add CORS to my FastAPI application when it runs behind a proxy with a path prefix?", "long"),
])
def test_text_blockers_keep_questions_off_fast(question, reason):
    d = decide(question)
    assert d.route == STANDARD and any(reason in r for r in d.reasons)


@pytest.mark.parametrize("hits, reason", [
    ([("p", 0.55), ("p", 0.40), ("p", 0.39)], "weak top hit"),
    ([("p", 0.75), ("p", 0.74), ("p", 0.60)], "no clear winner"),
    ([("p", 0.75), ("q", 0.60), ("r", 0.59), ("s", 0.58), ("t", 0.57)], "spread out"),
])
def test_retrieval_blockers(hits, reason):
    d = decide("How do I add CORS?", hits)
    assert d.route == STANDARD and any(reason in r for r in d.reasons)


def test_policy_is_tunable():
    hits = [("p", 0.58), ("p", 0.40), ("p", 0.39)]
    assert decide("How do I add CORS?", hits).route == STANDARD
    assert decide("How do I add CORS?", hits, fast_min_top1=0.5).route == FAST


def test_decision_serialises_for_traces():
    d = decide("How do I add CORS?").to_dict()
    assert d["route"] == FAST and d["features"]["top1"] == 0.74


def test_forced_routes():
    assert forced("escalate").route == ESCALATE
    with pytest.raises(ValueError):
        forced("turbo")
