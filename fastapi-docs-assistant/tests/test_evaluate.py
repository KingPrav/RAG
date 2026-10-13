"""End-to-end harness test: real retrieval over a tiny index, with stand-ins for
the embedding model, the answering model and the judge (offline, free)."""
import csv
import re
import zlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from docs_assistant import config
from docs_assistant.calibrate import export, score
from docs_assistant.compare import compare
from docs_assistant.evaluate import load_run, rejudge_rows, run_items, summarize, write_run
from docs_assistant.golden import GoldenItem
from docs_assistant.index import build_index
from docs_assistant.judge import JudgeVerdict, NuggetVerdict
from docs_assistant.rag import GeneratedAnswer
from docs_assistant.vectorstore import get_record_manager, get_vectorstore

TEXTS = {
    "index#run-it": "Run the server. Serving at: http://127.0.0.1:8000",
    "tutorial/cors#use-corsmiddleware": "Use CORSMiddleware and set allow_origins for the frontend.",
    "advanced/middleware#gzipmiddleware": "GZipMiddleware compresses responses above minimum_size.",
}


def chunk(section: str) -> Document:
    return Document(id=section, page_content=f"FastAPI docs > {section}\n\n{TEXTS[section]}", metadata={
        "chunk_id": section, "doc_id": section.split("#")[0], "url": f"https://x.dev/{section}",
        "breadcrumb": "FastAPI docs", "n_tokens": 20, "source_commit": "abc"})


def item(id_, category, answer_type, question, **kw) -> GoldenItem:
    return GoldenItem(id=id_, split=kw.pop("split", "dev"), category=category, answer_type=answer_type,
                      question=question, expected_answer=kw.pop("expected", "ref answer"),
                      corpus_commit="abc", **kw)


ITEMS = [
    item("fact-001", "exact-fact", "exact", TEXTS["index#run-it"], exact_accept=["8000"],
         gold_sections=["index#run-it"]),
    item("vocab-001", "vocab-mismatch", "open", TEXTS["tutorial/cors#use-corsmiddleware"],
         nuggets=[{"fact": "use CORSMiddleware", "vital": True}, {"fact": "allow_origins", "vital": False}],
         gold_sections=["tutorial/cors#use-corsmiddleware"]),
    item("vocab-002", "vocab-mismatch", "open", TEXTS["advanced/middleware#gzipmiddleware"],
         nuggets=[{"fact": "use GZipMiddleware", "vital": True}],
         gold_sections=["advanced/middleware#gzipmiddleware"]),
    item("trap-001", "unanswerable-trap", "abstain", "What is the default request timeout?",
         abstain_reason="not in docs"),
]

# What the stand-in answering model says for each question.
ANSWERS = {
    TEXTS["index#run-it"]: GeneratedAnswer(answer="It serves on port **8000**. [1]", cited_sources=[1], answerable=True),
    TEXTS["tutorial/cors#use-corsmiddleware"]: GeneratedAnswer(answer="Add CORSMiddleware. [1]", cited_sources=[1], answerable=True),
    TEXTS["advanced/middleware#gzipmiddleware"]: GeneratedAnswer(answer="Not covered.", cited_sources=[], answerable=False),
    "What is the default request timeout?": GeneratedAnswer(answer="Not in the docs.", cited_sources=[], answerable=False),
}


class BagOfWordsEmbedding(Embeddings):
    """Stand-in embedder where shared words mean closeness (unlike random fake
    vectors), so retrieval ranks are predictable in tests."""
    size = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.size] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def vectorstore(tmp_path):
    vs = get_vectorstore(BagOfWordsEmbedding(), tmp_path / "chroma", "test-docs")
    build_index([chunk(s) for s in TEXTS], vs, get_record_manager(tmp_path / "rm.sqlite", "test-docs"))
    return vs


def generator():
    def run(inputs):
        raw = AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})
        return {"raw": raw, "parsed": ANSWERS[inputs["question"]], "parsing_error": None}
    return RunnableLambda(run)


def judge(verdicts=("support", "not_support"), overall="pass", calls=None):
    def run(inputs):
        if calls is not None:
            calls.append(inputs)
        n = inputs["nuggets"].count("\n") + 1
        parsed = JudgeVerdict(critique="Covers the middleware.", verdict=overall,
                              nuggets=[NuggetVerdict(nugget=i + 1, verdict=verdicts[i]) for i in range(n)])
        raw = AIMessage(content="", usage_metadata={"input_tokens": 2000, "output_tokens": 300, "total_tokens": 2300})
        return {"raw": raw, "parsed": parsed, "parsing_error": None}
    return RunnableLambda(run)


@pytest.fixture
def rows(vectorstore):
    return run_items(ITEMS, vectorstore, generator(), judge(), k=3, workers=2, progress=lambda _: None)


def by_id(rows):
    return {r["id"]: r for r in rows}


def test_each_answer_type_is_scored_its_own_way(rows):
    r = by_id(rows)
    assert r["fact-001"]["exact_match"] is True and r["fact-001"]["passed"]          # **8000** normalized
    assert r["vocab-001"]["passed"] and r["vocab-001"]["nuggets"]["vital_strict"] == 1.0
    assert r["vocab-001"]["judge"]["critique"] == "Covers the middleware."
    assert not r["vocab-002"]["passed"] and r["vocab-002"]["false_refusal"]          # refused an answerable one
    assert r["vocab-002"]["judge"] is None                                           # refusals aren't judged
    assert r["trap-001"]["passed"] and r["trap-001"]["retrieval"] is None            # correct refusal


def test_retrieval_is_scored_against_gold_sections(rows):
    r = by_id(rows)["fact-001"]["retrieval"]
    assert r["hit@1"] and r["reciprocal_rank"] == 1.0                                # exact text -> rank 1


def test_results_keep_order_and_record_cost(rows):
    assert [r["id"] for r in rows] == [i.id for i in ITEMS]
    assert by_id(rows)["vocab-001"]["judge_cost_usd"] == pytest.approx(
        (2000 * config.JUDGE_USD_PER_1M_INPUT + 300 * config.JUDGE_USD_PER_1M_OUTPUT) / 1e6)


def test_judge_saying_pass_with_a_missing_vital_nugget_is_overridden(vectorstore):
    rows = run_items(ITEMS[1:2], vectorstore, generator(), judge(("not_support", "support"), "pass"), k=3,
                     progress=lambda _: None)
    assert not rows[0]["passed"] and rows[0]["judge"]["verdict_overridden"]


def test_one_failing_question_does_not_sink_the_run(vectorstore):
    broken = RunnableLambda(lambda inputs: (_ for _ in ()).throw(RuntimeError("API down")))
    rows = run_items(ITEMS[:2], vectorstore, broken, judge(), k=3, progress=lambda _: None)
    assert all(r["error"] == "RuntimeError: API down" and r["passed"] is False for r in rows)
    assert summarize(rows, k=3, resamples=200)["n_errors"] == 2


def test_summary_metrics(rows):
    s = summarize(rows, k=3, resamples=500)
    assert s["overall_pass_rate"]["value"] == 0.75 and s["overall_pass_rate"]["n"] == 4
    assert s["answers"]["exact_match"]["value"] == 1.0
    assert s["answers"]["judge_pass_rate"]["value"] == 0.5                          # 1 of 2 open items
    assert s["answers"]["correct_refusal_rate"]["value"] == 1.0
    assert s["answers"]["false_refusal_rate"]["value"] == pytest.approx(1 / 3)
    assert s["retrieval"]["hit@3"]["n"] == 3                                        # abstain items excluded
    assert s["by_category"]["vocab-mismatch"]["pass_rate"] == 0.5
    assert s["review_status"] == {"draft": 4}


def test_run_directory_is_complete_and_reloadable(rows, tmp_path):
    cfg = {"label": "baseline", "split": "dev", "k": 3, "timestamp": "t", "chat_model": "stub",
           "judge_model": "stub-judge", "prompt_version": "p1", "judge_prompt_version": "j1",
           "golden_sha256": "f" * 64, "corpus_commit": "abc", "bootstrap_resamples": 200}
    run_dir = write_run(rows, cfg, runs_dir=tmp_path)
    assert {p.name for p in run_dir.iterdir()} == {"config.json", "results.jsonl", "summary.json", "summary.md"}
    md = (run_dir / "summary.md").read_text()
    assert "Overall pass rate" in md and "4 of 4 golden items are unreviewed drafts" in md
    loaded_cfg, loaded_rows = load_run(run_dir)
    assert loaded_cfg["label"] == "baseline" and [r["id"] for r in loaded_rows] == [i.id for i in ITEMS]


def test_rejudge_rescores_only_answered_open_items(rows):
    calls = []
    out = rejudge_rows(rows, {i.id: i for i in ITEMS}, judge(("not_support", "support"), "fail", calls))
    assert len(calls) == 1                                                          # vocab-001 only
    assert not by_id(out)["vocab-001"]["passed"] and by_id(out)["fact-001"]["passed"]


def test_compare_runs_item_by_item(rows):
    cfg_a = {"k": 3, "label": "a", "prompt_version": "p1"}
    better = [dict(r, passed=True) for r in rows]
    result = compare((cfg_a, rows), (dict(cfg_a, label="b", prompt_version="p2"), better), resamples=500)
    assert result["n_shared"] == 4 and result["fixed"] == ["vocab-002"] and result["broken"] == []
    assert result["config_changes"] == {"prompt_version": ("p1", "p2")}
    assert result["deltas"]["pass"]["diff"] == pytest.approx(0.25)


def test_calibration_export_and_tpr_tnr(rows, tmp_path):
    cfg = {"label": "c", "split": "dev", "k": 3, "timestamp": "t", "chat_model": "s", "judge_model": "j",
           "prompt_version": "p", "judge_prompt_version": "j", "golden_sha256": "0" * 64, "corpus_commit": "abc",
           "bootstrap_resamples": 100}
    run_dir = write_run(rows, cfg, runs_dir=tmp_path)
    out = tmp_path / "calib.csv"
    golden = {i.id: i.model_dump() for i in ITEMS}
    assert export(run_dir, golden, out) == 1                                        # only judged answers
    exported = list(csv.DictReader(out.open()))
    assert exported[0]["judge_verdict"] == "pass" and exported[0]["human_verdict"] == ""

    labels = [{"id": "a", "judge_verdict": "pass", "human_verdict": "pass"},
              {"id": "b", "judge_verdict": "pass", "human_verdict": "fail"},
              {"id": "c", "judge_verdict": "fail", "human_verdict": "fail"},
              {"id": "d", "judge_verdict": "pass", "human_verdict": ""}]               # unlabeled: ignored
    s = score(labels)
    assert (s["n_labeled"], s["tpr"], s["tnr"]) == (3, 1.0, 0.5)
    assert s["disagreements"] == ["b"]
