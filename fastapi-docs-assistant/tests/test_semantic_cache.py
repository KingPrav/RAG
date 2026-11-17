"""Semantic cache: hits, near-miss rejection, pipeline scoping, write gate, TTL, eviction.

A fake embedder maps known questions to fixed vectors so similarities are
exact and the tests need no API key."""
import numpy as np
import pytest

from docs_assistant.rag import Citation, RAGResult
from docs_assistant.semantic_cache import SemanticCache, fingerprint, normalize

# Hand-built 3-d "embeddings". cos(cors_a, cors_b) ~ 0.98 (paraphrase);
# cos(query_params, path_params) ~ 0.90 (near-duplicate wording, different answer).
VECTORS = {
    "how do i add cors?": [1.0, 0.0, 0.0],
    "enable cors in fastapi": [0.98, 0.199, 0.0],
    "how do i declare query parameters?": [0.0, 1.0, 0.0],
    "how do i declare path parameters?": [0.0, 0.9, 0.436],
    "unrelated": [0.0, 0.0, 1.0],
}


class FakeEmbed:
    def __init__(self):
        self.calls = 0

    def __call__(self, text: str):
        self.calls += 1
        return VECTORS[text.lower()]


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def result(question="how do i add cors?", **overrides) -> RAGResult:
    base = dict(question=question, answer="Use CORSMiddleware [1].", answerable=True,
                citations=[Citation(1, "tutorial/cors#use-corsmiddleware", "https://x/cors", 0.8)],
                retrieved=[], model="m", prompt="v3")
    base.update(overrides)
    return RAGResult(**base)


FP = fingerprint({"prompt": "v3", "model": "m", "commit": "abc"})


@pytest.fixture
def cache(tmp_path):
    return SemanticCache(FakeEmbed(), path=tmp_path / "c.sqlite", threshold=0.95, clock=Clock())


def test_fingerprint_is_order_independent_and_sensitive():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_normalize_ignores_case_whitespace_and_trailing_punctuation():
    assert normalize("  How do I   add CORS? ") == normalize("how do i add cors")


def test_exact_repeat_is_served_without_embedding(cache):
    assert cache.store("how do i add cors?", FP, result())[0]
    calls = cache.embed.calls
    hit, info = cache.lookup("How do I add CORS", FP)
    assert hit and hit.exact and info["exact"]
    assert cache.embed.calls == calls                    # free path


def test_paraphrase_above_threshold_hits_and_records_source(cache):
    cache.store("how do i add cors?", FP, result())
    hit, info = cache.lookup("enable cors in fastapi", FP)
    assert hit is not None and not hit.exact
    assert hit.source_question == "how do i add cors?"
    assert hit.similarity == pytest.approx(0.98, abs=0.01)
    assert hit.citations[0]["chunk_id"] == "tutorial/cors#use-corsmiddleware"


def test_near_duplicate_with_different_meaning_is_a_miss(cache):
    # The false-hit case the threshold exists for.
    cache.store("how do i declare query parameters?", FP, result("how do i declare query parameters?"))
    hit, info = cache.lookup("how do i declare path parameters?", FP)
    assert hit is None
    assert 0.85 < info["best_similarity"] < 0.95             # visible in traces for tuning
    assert info["nearest_question"] == "how do i declare query parameters?"


def test_miss_returns_vector_so_store_does_not_embed_again(cache):
    hit, info = cache.lookup("how do i add cors?", FP)
    assert hit is None and cache.embed.calls == 1
    cache.store("how do i add cors?", FP, result(), vector=info["vector"])
    assert cache.embed.calls == 1


def test_other_pipeline_never_sees_entries(cache):
    cache.store("how do i add cors?", FP, result())
    other = fingerprint({"prompt": "v4", "model": "m", "commit": "abc"})
    assert cache.lookup("how do i add cors?", other)[0] is None
    assert cache.lookup("enable cors in fastapi", other)[0] is None


@pytest.mark.parametrize("overrides, reason", [
    (dict(answerable=False), "not answerable"),
    (dict(citations=[]), "no citations"),
    (dict(invalid_citations=[7]), "invalid citations"),
    (dict(rerank_error="timeout"), "degraded"),
])
def test_unclean_answers_are_not_cached(cache, overrides, reason):
    ok, why = cache.store("how do i add cors?", FP, result(**overrides))
    assert not ok and reason in why
    assert cache.lookup("how do i add cors?", FP)[0] is None


def test_entries_expire_after_ttl(tmp_path):
    clock = Clock()
    c = SemanticCache(FakeEmbed(), path=tmp_path / "c.sqlite", ttl_days=1, clock=clock)
    c.store("how do i add cors?", FP, result())
    clock.t += 2 * 86400
    assert c.lookup("how do i add cors?", FP)[0] is None
    assert c.lookup("enable cors in fastapi", FP)[0] is None


def test_lru_eviction_keeps_recently_used(tmp_path):
    clock = Clock()
    c = SemanticCache(FakeEmbed(), path=tmp_path / "c.sqlite", max_entries=2, clock=clock)
    c.store("how do i add cors?", FP, result()); clock.t += 1
    c.store("how do i declare query parameters?", FP, result("how do i declare query parameters?")); clock.t += 1
    c.lookup("how do i add cors?", FP); clock.t += 1                  # touch cors
    c.store("unrelated", FP, result("unrelated"))                    # evicts query params
    assert c.stats()["entries"] == 2
    assert c.lookup("how do i add cors?", FP)[0] is not None
    assert c.lookup("how do i declare query parameters?", FP)[0] is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "c.sqlite"
    SemanticCache(FakeEmbed(), path=path, clock=Clock()).store("how do i add cors?", FP, result())
    hit, _ = SemanticCache(FakeEmbed(), path=path, clock=Clock()).lookup("enable cors in fastapi", FP)
    assert hit is not None and hit.hits == 1


def test_vectors_are_normalised(cache):
    cache.store("how do i add cors?", FP, result(), vector=np.array([3.0, 0.0, 0.0]))
    assert cache.lookup("enable cors in fastapi", FP)[0].similarity <= 1.0
