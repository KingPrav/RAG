"""The golden-set validator must catch every kind of bad item, not just pass good ones."""
import copy
import json

import pytest

from docs_assistant.golden import GOLDEN_PATH, load_items, load_sections, validate

SECTIONS = {
    "tutorial/first-steps#interactive-api-docs": "### Interactive API docs\n\nNow go to  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).",
    "tutorial/cors#use-corsmiddleware": "* `max_age` - Sets a maximum time in seconds for browsers to cache CORS responses. Defaults to `600`.",
}
COMMIT = "abc123"

EXACT = {
    "id": "fact-001", "split": "dev", "category": "exact-fact", "answer_type": "exact",
    "question": "At what path are the Swagger UI docs served?", "expected_answer": "/docs",
    "exact_accept": ["/docs"], "gold_sections": ["tutorial/first-steps#interactive-api-docs"],
    "evidence": [{"section": "tutorial/first-steps#interactive-api-docs",
                  "quote": "Now go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)."}],
    "corpus_commit": COMMIT,
}
OPEN = {
    "id": "term-001", "split": "test", "category": "exact-term", "answer_type": "open",
    "question": "What does max_age do in CORSMiddleware?", "expected_answer": "Caches CORS responses for 600s.",
    "nuggets": [{"fact": "Sets how long browsers cache CORS responses", "vital": True}],
    "gold_sections": ["tutorial/cors#use-corsmiddleware"],
    "evidence": [{"section": "tutorial/cors#use-corsmiddleware", "quote": "Defaults to `600`."}],
    "corpus_commit": COMMIT,
}
ABSTAIN = {
    "id": "trap-001", "split": "dev", "category": "unanswerable-trap", "answer_type": "abstain",
    "question": "What is the default request timeout?", "expected_answer": "Not in the docs.",
    "abstain_reason": "Never mentioned.", "corpus_commit": COMMIT,
}


def errors_for(*items):
    return validate(list(items), SECTIONS, COMMIT)[1]


def with_changes(item, **changes):
    out = copy.deepcopy(item)
    out.update(changes)
    return out


def test_valid_items_pass_and_quotes_ignore_whitespace():
    # The section text has a double space ("go to  [http"); the quote has one.
    assert errors_for(EXACT, OPEN, ABSTAIN) == []


@pytest.mark.parametrize("item, expected_error", [
    (with_changes(EXACT, evidence=[{"section": "tutorial/first-steps#interactive-api-docs",
                                    "quote": "The docs are served at /swagger by default."}]), "quote not found verbatim"),
    (with_changes(EXACT, exact_accept=["422"]), "none of exact_accept"),
    (with_changes(EXACT, gold_sections=["tutorial/does-not-exist#nope"],
                  evidence=[{"section": "tutorial/does-not-exist#nope", "quote": "anything at all"}]), "gold section not in corpus"),
    (with_changes(EXACT, evidence=[{"section": "tutorial/cors#use-corsmiddleware", "quote": "Defaults to `600`."}]),
     "is not one of the gold sections"),
    (with_changes(EXACT, evidence=[]), "need at least one evidence quote"),
    (with_changes(EXACT, exact_accept=[]), "exact items need exact_accept"),
    (with_changes(OPEN, nuggets=[{"fact": "optional detail", "vital": False}]), "at least one vital nugget"),
    (with_changes(ABSTAIN, gold_sections=["tutorial/cors#use-corsmiddleware"]), "must not have gold_sections"),
    (with_changes(ABSTAIN, abstain_reason=""), "need abstain_reason"),
    (with_changes(EXACT, category="unanswerable-trap"), "must have answer_type 'abstain'"),
    (with_changes(EXACT, corpus_commit="other999"), "!= indexed corpus"),
    (with_changes(EXACT, id="Fact_1"), "id must look like"),
    (with_changes(EXACT, split="train"), "schema"),
])
def test_bad_items_are_rejected(item, expected_error):
    errors = errors_for(item, OPEN, ABSTAIN)
    assert any(expected_error in e for e in errors), errors


def test_duplicate_ids_and_questions_are_rejected():
    dupe = with_changes(OPEN, id="fact-001", question=EXACT["question"])
    errors = errors_for(EXACT, dupe)
    assert any("duplicate id" in e for e in errors)
    assert any("duplicate question" in e for e in errors)


def test_split_imbalance_within_a_category_is_rejected():
    items = [with_changes(EXACT, id=f"fact-00{i}", question=f"Question number {i}?") for i in range(1, 4)]
    errors = errors_for(*items)                                    # 3 dev, 0 test
    assert any("split imbalance in exact-fact" in e for e in errors)


def test_committed_golden_set_is_valid_against_the_indexed_corpus():
    """The real check: runs whenever chunks.jsonl exists (after ingest + chunk)."""
    from docs_assistant import config
    chunks = config.PROCESSED_DIR / "chunks.jsonl"
    if not chunks.exists() or not GOLDEN_PATH.exists():
        pytest.skip("corpus not built yet (run ingest + chunk)")
    sections, commit = load_sections(chunks)
    items, errors = validate(load_items(), sections, commit)
    assert errors == []
    assert len(items) >= 50
    assert len({i.id for i in items}) == len(items)
    assert all(json.dumps(i.model_dump()) for i in items)
