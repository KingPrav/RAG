"""Ragas integration: metric inputs, scoring with failures, the OpenAI client
wrapper (via a mocked endpoint), and golden-candidate conversion and merging.
All offline; the Ragas-dependent tests skip if Ragas isn't installed."""
import asyncio
import json

import pytest
from langchain_core.documents import Document

from docs_assistant.golden import validate
from docs_assistant.ragas_eval import (build_samples, compare_runs, normalize_params, score_samples,
                                       summarize)
from docs_assistant.ragas_testset import (best_evidence, format_nuggets, merge, parse_nuggets, sample_chunks,
                                          to_golden)

CHUNKS = {
    "advanced/events#lifespan": "FastAPI docs > Advanced > Lifespan Events > Lifespan\n\n"
                                "You can define logic that runs before the application starts. "
                                "Use the lifespan parameter of the FastAPI app with an async context manager.",
    "advanced/events#use-case": "FastAPI docs > Advanced > Lifespan Events > Use case\n\n"
                                "Imagine you want to load a machine learning model once and share it.",
}
GOLDEN = {
    "vocab-007": {"answer_type": "open", "expected_answer": "Use the lifespan parameter with an async context manager."},
    "trap-001": {"answer_type": "abstain", "expected_answer": "Not in the docs."},
}


def row(id_, answered=True, error=None, retrieved=("advanced/events#use-case", "advanced/events#lifespan")):
    return {"id": id_, "category": "vocab-mismatch", "question": "load a model at startup?",
            "answer": "Use lifespan [2].", "answerable_predicted": answered, "error": error,
            "retrieved": [{"chunk_id": c} for c in retrieved]}


# ---------------------------------------------------------------- metric inputs

def test_metrics_apply_only_where_they_make_sense():
    samples = {s["id"]: s for s in build_samples(
        [row("vocab-007"), row("trap-001", answered=False), row("x", error="boom")], GOLDEN, CHUNKS)}
    assert samples["vocab-007"]["metrics"] == ["faithfulness", "context_recall", "factual_correctness"]
    assert "trap-001" not in samples                       # refused + unanswerable: nothing to score
    assert samples["vocab-007"]["retrieved_contexts"][1] == CHUNKS["advanced/events#lifespan"]
    refused = build_samples([row("vocab-007", answered=False)], GOLDEN, CHUNKS)[0]
    assert refused["metrics"] == ["context_recall"]        # a refusal has no claims to check


def test_missing_chunk_text_is_an_error_not_a_silent_skip():
    with pytest.raises(SystemExit):
        build_samples([row("vocab-007", retrieved=("gone#section",))], GOLDEN, CHUNKS)


# ---------------------------------------------------------------- scoring

class Fake:
    def __init__(self, value=None, fail=False):
        self.value, self.fail, self.calls = value, fail, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("rate limited")
        return type("R", (), {"value": self.value})()


def test_scoring_records_failures_and_nan_per_metric():
    samples = build_samples([row("vocab-007")], GOLDEN, CHUNKS)
    metrics = {"faithfulness": Fake(0.75), "context_recall": Fake(fail=True), "factual_correctness": Fake(float("nan"))}
    rows = asyncio.run(score_samples(samples, metrics, workers=2, progress=lambda _: None))
    r = rows[0]
    assert r["faithfulness"] == 0.75 and r["context_recall"] is None and r["factual_correctness"] is None
    assert "RuntimeError" in r["errors"]["context_recall"] and "NaN" in r["errors"]["factual_correctness"]
    assert set(metrics["faithfulness"].calls[0]) == {"user_input", "response", "retrieved_contexts"}
    assert set(metrics["factual_correctness"].calls[0]) == {"response", "reference"}
    s = summarize(rows, resamples=100)
    assert s["faithfulness"]["value"] == 0.75 and s["context_recall"]["errors"] == 1


def test_paired_comparison_on_shared_items():
    a = [{"id": f"q{i}", "faithfulness": 0.6} for i in range(20)]
    b = [{"id": f"q{i}", "faithfulness": 0.9} for i in range(20)] + [{"id": "extra", "faithfulness": 0.0}]
    d = compare_runs(a, b, resamples=500)["faithfulness"]
    assert d["diff"] == pytest.approx(0.3) and d["n"] == 20 and d["significant"]


def test_params_are_normalized_for_reasoning_models():
    out = normalize_params({"model": "gpt-6.1-sol", "temperature": 0.01, "top_p": 0.1, "max_tokens": 1024})
    assert out == {"model": "gpt-6.1-sol", "max_completion_tokens": 8192}


def test_real_ragas_faithfulness_runs_through_the_wrapped_client():
    pytest.importorskip("ragas")
    httpx = pytest.importorskip("httpx")
    from docs_assistant.ragas_eval import UsageMeter, build_metrics
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        text = " ".join(m["content"] for m in body["messages"] if isinstance(m.get("content"), str))
        statement = "Use the lifespan parameter."
        content = ({"statements": [statement]} if "verdict" not in text.lower()
                   else {"statements": [{"statement": statement, "reason": "in context", "verdict": 1}]})
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}})

    meter = UsageMeter()
    metrics = build_metrics("gpt-6.1-sol", meter, api_key="test",
                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = asyncio.run(metrics["faithfulness"].ascore(
        user_input="How do I run code at startup?", response="Use the lifespan parameter. [1]",
        retrieved_contexts=[CHUNKS["advanced/events#lifespan"]]))
    assert float(result.value) == 1.0
    assert meter.calls == len(sent) >= 2 and meter.input_tokens == 1000 * meter.calls
    assert "temperature" not in sent[0] and sent[0]["max_completion_tokens"] == 8192


# ---------------------------------------------------------------- candidate generation

def test_sampling_is_stratified_length_filtered_and_deterministic():
    docs = [Document(page_content="x", metadata={"chunk_id": f"p{p}#s{i}", "doc_id": f"p{p}", "n_tokens": n})
            for p in range(5) for i, n in enumerate([50, 200, 300, 600])]
    picked = sample_chunks(docs, 5, seed=1)
    assert len({d.metadata["doc_id"] for d in picked}) == 5                 # one per page before repeats
    assert all(120 <= d.metadata["n_tokens"] <= 500 for d in picked)
    assert [d.metadata["chunk_id"] for d in picked] == [d.metadata["chunk_id"] for d in sample_chunks(docs, 5, seed=1)]


def test_evidence_is_a_verbatim_sentence_sharing_the_most_words():
    quote = best_evidence(CHUNKS["advanced/events#lifespan"], "lifespan parameter async context manager")
    assert quote == "Use the lifespan parameter of the FastAPI app with an async context manager."
    assert quote in CHUNKS["advanced/events#lifespan"] and "FastAPI docs >" not in quote


def sections():
    return {cid: text.split("\n\n", 1)[1] for cid, text in CHUNKS.items()}


def ragas_sample(q, contexts, synth="single_hop_specific_query_synthesizer"):
    return {"user_input": q, "reference": "Use the lifespan parameter with an async context manager.",
            "reference_contexts": contexts, "synthesizer_name": synth, "persona_name": "Goal-first junior"}


def test_conversion_maps_sources_and_produces_valid_golden_items():
    by_text = {t: c for c, t in CHUNKS.items()}
    samples = [
        ragas_sample("How can I run setup code once before my app takes requests?", [CHUNKS["advanced/events#lifespan"]]),
        ragas_sample("Where do I load my ML model so it's shared?", list(CHUNKS.values()), "multi_hop_specific"),
        ragas_sample("Untraceable question about something?", ["text that is not a chunk"]),
        ragas_sample("What port does fastapi dev use?", [CHUNKS["advanced/events#lifespan"]]),   # duplicate
    ]
    items, dropped = to_golden(samples, by_text, "abc", lambda q, r: [{"fact": "use lifespan", "vital": True}],
                               existing_questions={"What port does fastapi dev use?"})
    assert [i["id"] for i in items] == ["ragas-001", "ragas-002"] and len(dropped) == 2
    single, multi = items
    assert single["category"] == "synthetic-single" and single["gold_sections"] == ["advanced/events#lifespan"]
    assert multi["category"] == "synthetic-multi" and len(multi["gold_sections"]) == 2
    assert single["source"].startswith("ragas:") and single["review"]["status"] == "draft"
    _, errors = validate(items, sections(), "abc")
    assert [e for e in errors if not e.startswith("split imbalance")] == []


def test_merge_applies_decisions_edits_and_rebalances_splits():
    by_text = {t: c for c, t in CHUNKS.items()}
    qs = [f"Question number {i} about startup code?" for i in range(4)]
    items, _ = to_golden([ragas_sample(q, [CHUNKS["advanced/events#lifespan"]]) for q in qs], by_text, "abc",
                         lambda q, r: [{"fact": "use lifespan", "vital": True}], set())
    review = [{"id": "ragas-001", "decision": "approve", "reviewer": "PR", "question": "Edited question here?",
               "expected_answer": "", "nuggets": "VITAL: use lifespan | okay: asynccontextmanager"},
              {"id": "ragas-002", "decision": "reject"},
              {"id": "ragas-003", "decision": "approve", "question": "", "expected_answer": "", "nuggets": ""},
              {"id": "ragas-004", "decision": ""}]
    approved, counts = merge(items, review)
    assert [a["id"] for a in approved] == ["ragas-001", "ragas-003"]
    assert counts == {"approve": 2, "reject": 1, "undecided": 1}
    assert approved[0]["question"] == "Edited question here?" and approved[0]["review"]["status"] == "approved"
    assert approved[0]["nuggets"] == [{"fact": "use lifespan", "vital": True},
                                      {"fact": "asynccontextmanager", "vital": False}]
    assert [a["split"] for a in approved] == ["dev", "test"]                 # rebalanced after the rejection
    _, errors = validate(approved, sections(), "abc")
    assert errors == []


def test_nugget_format_round_trips():
    nuggets = [{"fact": "use lifespan", "vital": True}, {"fact": "async context manager", "vital": False}]
    assert parse_nuggets(format_nuggets(nuggets)) == nuggets
    assert parse_nuggets("just one fact") == [{"fact": "just one fact", "vital": True}]


def test_rejected_golden_items_are_not_evaluated(monkeypatch):
    from docs_assistant import evaluate
    from docs_assistant.golden import GoldenItem
    items = [GoldenItem(id=f"vocab-00{i}", split="dev", category="vocab-mismatch", answer_type="open",
                        question=f"question {i} here?", expected_answer="answer", corpus_commit="abc",
                        review={"status": status}) for i, status in enumerate(["approved", "rejected", "draft"])]
    monkeypatch.setattr(evaluate, "load_sections", lambda: ({}, "abc"))
    monkeypatch.setattr(evaluate, "load_items", lambda path: [])
    monkeypatch.setattr(evaluate, "validate", lambda raw, s, c: (items, []))
    kept, _ = evaluate.load_golden(None)
    assert [i.id for i in kept] == ["vocab-000", "vocab-002"]
