"""Phase 1 experiments: prompt versions, rerankers, parallel rewrite + cache.
Offline, with stand-ins for the models (and FlashRank's ranker)."""
import threading
import time

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from docs_assistant.compare import compare
from docs_assistant.index import build_index
from docs_assistant.rag import PROMPTS, GeneratedAnswer, answer_question, build_generator, prompt_hash
from docs_assistant.rerank import CrossEncoderReranker, LLMReranker, Ranking, order_from_ranking
from docs_assistant.retrieval import QueryRewrites, RetrievalSettings, Retriever
from docs_assistant.vectorstore import get_record_manager, get_vectorstore

SECTIONS = [
    ("advanced/events#lifespan", "Define a lifespan function with asynccontextmanager to load a model at startup."),
    ("advanced/events#use-case", "Use case: load a machine learning model once and share it between requests."),
    ("tutorial/middleware#create-a-middleware", "Middleware runs code before and after each request; time requests."),
    ("advanced/custom-response#redirectresponse", "RedirectResponse sends the user to another URL with a redirect."),
    ("tutorial/cors#use-corsmiddleware", "CORSMiddleware lets a frontend on another origin call the API."),
]


def docs():
    return [Document(id=cid, page_content=f"FastAPI docs > {cid}\n\n{text}", metadata={
        "chunk_id": cid, "doc_id": cid.split("#")[0], "url": f"https://x.dev/{cid}", "breadcrumb": "FastAPI docs",
        "n_tokens": 20, "source_commit": "abc"}) for cid, text in SECTIONS]


@pytest.fixture
def vectorstore(tmp_path, bow_embedding):
    vs = get_vectorstore(bow_embedding, tmp_path / "chroma", "test-docs")
    build_index(docs(), vs, get_record_manager(tmp_path / "rm.sqlite", "test-docs"))
    return vs


def usage(i, o):
    return AIMessage(content="", usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o})


# ---------------------------------------------------------------- 1. prompt versions

def test_v1_prompt_is_byte_identical_to_the_baseline_runs():
    assert prompt_hash("v1") == "df19b7bc93"                      # hash recorded in every Step 7-8 run
    assert prompt_hash("v2") != prompt_hash("v1")
    assert "Answer completely, then concisely" in PROMPTS["v2"] and "Be concise" in PROMPTS["v1"]
    assert "keep answerable true" in PROMPTS["v2"]
    # v3 keeps v2's completeness rule but ties answerable to the specific thing asked
    assert "Answer completely, then concisely" in PROMPTS["v3"]
    assert "Decide answerable by the specific thing the question asks for" in PROMPTS["v3"]
    assert len({prompt_hash(v) for v in PROMPTS}) == len(PROMPTS)


def test_generator_uses_the_chosen_prompt_and_results_record_it(vectorstore):
    seen = {}

    def llm_stub(messages):
        seen["system"] = messages.to_messages()[0].content
        return {"raw": usage(10, 5), "parsed": GeneratedAnswer(answer="x [1]", cited_sources=[1], answerable=True),
                "parsing_error": None}

    class StubLLM:
        def with_structured_output(self, *a, **k):
            return RunnableLambda(llm_stub)

    result = answer_question("lifespan startup", vectorstore, build_generator(StubLLM(), "v2"), k=2, prompt="v2")
    assert "Answer completely" in seen["system"] and result.prompt == "v2"


def test_compare_reports_the_prompt_as_the_changed_setting():
    rows = [{"id": "a", "passed": True}, {"id": "b", "passed": False}]
    old = {"k": 5, "prompt_version": prompt_hash("v1")}                     # pre-Phase-1 run: no answer_prompt key
    new = {"k": 5, "prompt_version": prompt_hash("v2"), "answer_prompt": "v2", "rerank": "none"}
    changes = compare((old, rows), (new, rows), resamples=100)["config_changes"]
    assert changes == {"prompt_version": (prompt_hash("v1"), prompt_hash("v2")), "answer_prompt": ("v1", "v2")}


# ---------------------------------------------------------------- 2. rerankers

def cands():
    return [(d, 0.5 - i * 0.01) for i, d in enumerate(docs())]


def test_listwise_order_repairs_invalid_duplicate_and_missing_numbers():
    assert order_from_ranking([3, 3, 9, 1], 4) == [2, 0, 1, 3]


def test_llm_reranker_reorders_counts_tokens_and_falls_back_on_failure():
    def chain_out(ranking):
        return lambda inputs: {"raw": usage(5000, 40), "parsed": Ranking(ranking=ranking), "parsing_error": None}

    r = LLMReranker.__new__(LLMReranker)
    r.chain = RunnableLambda(chain_out([4, 2]))
    out = r("time every request", cands(), k=2)
    # model ranked passages 4 then 2 -> those come first, in that order
    assert [d.metadata["chunk_id"] for d, _, _ in out.ranked] == [SECTIONS[3][0], SECTIONS[1][0]]
    assert (out.input_tokens, out.output_tokens, out.error) == (5000, 40, None)
    assert out.ranked[0][1] == cands()[3][1]                                   # cosine similarity preserved

    r.chain = RunnableLambda(lambda inputs: (_ for _ in ()).throw(TimeoutError("slow")))
    failed = r("q", cands(), k=2)
    assert "TimeoutError" in failed.error and [d.metadata["chunk_id"] for d, _, _ in failed.ranked] == [
        SECTIONS[0][0], SECTIONS[1][0]]                                        # search order kept


def test_cross_encoder_reranks_by_score_and_falls_back_on_failure():
    pytest.importorskip("flashrank")

    class FakeRanker:
        def rerank(self, request):
            scores = {2: 0.9, 0: 0.3, 4: 0.6}
            return sorted(({"id": p["id"], "text": p["text"], "score": scores.get(p["id"], 0.0)}
                           for p in request.passages), key=lambda r: -r["score"])

    out = CrossEncoderReranker(ranker=FakeRanker())("time each request", cands(), k=3)
    assert [d.metadata["chunk_id"] for d, _, _ in out.ranked] == [SECTIONS[2][0], SECTIONS[4][0], SECTIONS[0][0]]
    assert out.ranked[0][2] == 0.9 and out.error is None

    class Broken:
        def rerank(self, request):
            raise RuntimeError("model file missing")
    failed = CrossEncoderReranker(ranker=Broken())("q", cands(), k=2)
    assert "RuntimeError" in failed.error and len(failed.ranked) == 2


def test_retriever_reranks_a_wider_pool_then_expands(vectorstore):
    seen = {}

    def fake_reranker(query, candidates, k):
        from docs_assistant.rerank import Reranked
        seen["n"] = len(candidates)
        best = [c for c in candidates if c[0].metadata["chunk_id"] == SECTIONS[2][0]]
        rest = [c for c in candidates if c not in best]
        return Reranked([(d, s, 1.0 - i / 10) for i, (d, s) in enumerate((best + rest)[:k])], ms=12.0,
                        input_tokens=100, output_tokens=5)

    settings = RetrievalSettings(k=2, neighbors=0, rewrite=False, rerank="ce", rerank_candidates=5)
    r = Retriever(vectorstore, settings, reranker=fake_reranker)("load a model at startup")
    assert seen["n"] == 5                                                      # reranker saw the wider pool
    assert r.hits[0].chunk_id == SECTIONS[2][0] and r.hits[0].rerank_score == 1.0
    assert len(r.hits) == 2 and r.rerank_ms == 12.0 and r.input_tokens == 100


def test_rerank_setting_without_a_reranker_is_rejected(vectorstore):
    with pytest.raises(ValueError):
        Retriever(vectorstore, RetrievalSettings(rerank="llm", neighbors=0, rewrite=False))


# ---------------------------------------------------------------- 3. latency

def slow_rewriter(calls, delay=0.2, queries=("RedirectResponse redirect",)):
    def run(inputs):
        calls.append(inputs["question"])
        time.sleep(delay)
        return {"raw": usage(300, 20), "parsed": QueryRewrites(queries=list(queries)), "parsing_error": None}
    return RunnableLambda(run)


def test_original_search_runs_while_the_rewrite_is_in_flight(vectorstore):
    searched_during_rewrite = threading.Event()
    calls = []
    retriever = Retriever(vectorstore, RetrievalSettings(k=2, neighbors=0, rewrite=True, rerank="none"),
                          rewriter=slow_rewriter(calls, delay=0.3))
    original_search = retriever._search

    def spy(queries, depth):
        if len(calls) == 1 and queries == ["send the user to another url"]:
            searched_during_rewrite.set()
        return original_search(queries, depth)
    retriever._search = spy
    started = time.perf_counter()
    r = retriever("send the user to another url")
    assert searched_during_rewrite.is_set()
    assert r.queries == ["send the user to another url", "RedirectResponse redirect"]
    assert r.rewrite_ms >= 300 and (time.perf_counter() - started) * 1000 < 300 + r.search_ms + 50


def test_repeat_questions_reuse_the_rewrite_and_failures_are_not_cached(vectorstore):
    calls = []
    retriever = Retriever(vectorstore, RetrievalSettings(k=2, neighbors=0, rewrite=True, rerank="none"),
                          rewriter=slow_rewriter(calls, delay=0))
    first = retriever("Send the user to another URL")
    second = retriever("  send the user   to another url ")
    assert len(calls) == 1 and second.rewrite_cached and second.input_tokens == 0
    assert second.queries[1:] == first.queries[1:]

    failing = Retriever(vectorstore, RetrievalSettings(k=2, neighbors=0, rewrite=True, rerank="none"),
                        rewriter=RunnableLambda(lambda i: (_ for _ in ()).throw(TimeoutError("x"))))
    failing("question one here")
    assert failing.cache.get("question one here") is None


def test_sweep_records_true_rewrite_latency_for_reused_rewrites():
    from docs_assistant.retrieval_eval import memoized
    timings, calls = {}, []
    rw = memoized(slow_rewriter(calls, delay=0.05), timings)
    rw.invoke({"question": "q1", "n": 3})
    started = time.perf_counter()
    rw.invoke({"question": "q1", "n": 3})
    assert len(calls) == 1 and timings["q1"] >= 50 and (time.perf_counter() - started) < 0.02


def test_cross_encoder_loads_its_model_once_under_parallel_first_use():
    import docs_assistant.rerank as rr
    loads = []

    class SlowRanker:
        def __init__(self, **kwargs):
            loads.append(1)
            time.sleep(0.05)

    pytest.importorskip("flashrank")
    import flashrank
    original = flashrank.Ranker
    flashrank.Ranker = SlowRanker
    try:
        ce = rr.CrossEncoderReranker()
        threads = [threading.Thread(target=lambda: ce.ranker) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    finally:
        flashrank.Ranker = original
    assert len(loads) == 1
