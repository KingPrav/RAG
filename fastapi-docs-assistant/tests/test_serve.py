"""Serving layer: cache -> route -> answer -> store, end to end, offline.

Real Chroma index with a deterministic fake embedding; stub generators per
path record which path answered."""
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from docs_assistant import config
from docs_assistant.index import build_index
from docs_assistant.rag import GeneratedAnswer
from docs_assistant.retrieval import RetrievalSettings, Retriever
from docs_assistant.router import RouterPolicy
from docs_assistant.semantic_cache import SemanticCache
from docs_assistant.serve import Assistant, MemoEmbeddings, Path
from docs_assistant.tracing import build_served_trace
from docs_assistant.vectorstore import get_record_manager, get_vectorstore
from test_rag import CHUNKS, QUESTION, stub_generator

GOOD = GeneratedAnswer(answer="Use a Pydantic model [1].", cited_sources=[1], answerable=True)
REFUSAL = GeneratedAnswer(answer="The docs don't say.", cited_sources=[], answerable=False)
EASY = RouterPolicy(fast_min_top1=0.5, fast_min_margin=0.0, fast_min_page_share=0.0, fast_max_words=999)
HARD = RouterPolicy(fast_min_top1=1.1)                         # nothing qualifies for fast


class CountingEmbedding(DeterministicFakeEmbedding):
    calls: int = 0

    def embed_query(self, text):
        object.__setattr__(self, "calls", self.calls + 1)
        return super().embed_query(text)


@pytest.fixture
def env(tmp_path):
    base = CountingEmbedding(size=64)
    memo = MemoEmbeddings(base)
    vs = get_vectorstore(memo, tmp_path / "chroma", "test-docs")
    build_index(CHUNKS, vs, get_record_manager(tmp_path / "rm.sqlite", "test-docs"))
    return {"vs": vs, "memo": memo, "base": base, "tmp": tmp_path}


def make(env, *, answer=GOOD, policy=EASY, cache=True, calls=None):
    calls = calls if calls is not None else {"fast": [], "standard": [], "escalate": []}
    retriever = Retriever(env["vs"], RetrievalSettings(k=3, neighbors=0, rewrite=False, rerank="none"))
    paths = {name: Path(retriever, stub_generator(answer, calls=calls[name]), model)
             for name, model in [("fast", config.CHAT_MODEL), ("standard", config.CHAT_MODEL),
                                 ("escalate", config.ESCALATE_MODEL)]}
    c = SemanticCache(env["memo"].embed_query, path=env["tmp"] / "cache.sqlite") if cache else None
    return Assistant(env["vs"], paths, cache=c, policy=policy, pipeline={"test": 1}, k=3), calls


def test_miss_routes_answers_and_stores_then_repeat_is_a_cache_hit(env):
    a, calls = make(env)
    first = a.answer(QUESTION)
    assert first.route == "fast" and len(calls["fast"]) == 1
    assert first.cache["stored"] and first.cache["best_similarity"] is None      # empty cache
    assert first.decision["features"]["top1"] == pytest.approx(1.0, abs=1e-3)

    second = a.answer(QUESTION)
    assert second.route == "cache" and second.cache["exact"]
    assert second.answer == first.answer and second.citations == first.citations
    assert sum(len(v) for v in calls.values()) == 1                             # no second LLM call
    assert second.cost_usd < 1e-5


def test_one_embedding_call_per_request(env):
    a, _ = make(env)
    a.answer(QUESTION)                    # cache lookup + router probe + retriever search share one vector
    assert env["base"].calls == 1
    a.answer(QUESTION)                    # exact repeat: no embedding at all
    assert env["base"].calls == 1


def test_hard_questions_take_standard_with_reasons(env):
    a, calls = make(env, policy=HARD)
    r = a.answer(QUESTION)
    assert r.route == "standard" and len(calls["standard"]) == 1
    assert any("weak top hit" in x for x in r.decision["reasons"])


def test_routing_off_sends_everything_to_standard(env):
    a, _ = make(env, policy=None)
    assert a.answer(QUESTION).route == "standard"


def test_forced_route_bypasses_cache_and_prices_the_big_model(env):
    a, calls = make(env)
    a.answer(QUESTION)                                          # now cached
    r = a.answer(QUESTION, force_route="escalate")
    assert r.route == "escalate" and len(calls["escalate"]) == 1
    assert r.result.model == config.ESCALATE_MODEL
    in_price, out_price = config.model_price(config.ESCALATE_MODEL)
    assert r.cost_usd == pytest.approx((1200 * in_price + 150 * out_price) / 1e6)


def test_refusals_are_answered_but_not_cached(env):
    a, calls = make(env, answer=REFUSAL)
    r = a.answer(QUESTION)
    assert not r.cache["stored"] and r.cache["skip_reason"] == "not answerable"
    assert a.answer(QUESTION).route != "cache"


def test_no_cache_mode(env):
    a, calls = make(env, cache=False)
    a.answer(QUESTION)
    assert a.answer(QUESTION).route == "fast"


def test_policy_change_changes_fingerprint(env):
    assert make(env, policy=EASY)[0].fingerprint != make(env, policy=HARD)[0].fingerprint


def test_traces_for_hit_and_miss(env):
    a, _ = make(env)
    miss, hit = a.answer(QUESTION), a.answer(QUESTION)
    t_miss = build_served_trace(miss, 3, a.fingerprint)
    assert t_miss["schema_version"] == 4 and t_miss["serving"]["route"] == "fast"
    assert t_miss["retrieval"]["results"] and t_miss["usage"] is not None
    assert t_miss["serving"]["timings_ms"]["total"] > 0
    t_hit = build_served_trace(hit, 3, a.fingerprint)
    assert t_hit["serving"]["route"] == "cache" and t_hit["retrieval"] is None and t_hit["usage"] is None
    assert t_hit["serving"]["cache"]["source_question"] == QUESTION
