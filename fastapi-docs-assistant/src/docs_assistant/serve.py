"""Phase 4c: the serving layer. One entry point for a question:

    embed once ─► cache lookup ──hit──► cached answer
                     │ miss
                     ▼
                 route-probe search (same vector, local index, ~20 ms)
                     ▼
                 router ─► fast | standard | escalate  ─► answer_question()
                     ▼
                 cache.store (clean answers only) ─► ServeResult (+ trace)

The question is embedded exactly once per request: the cache lookup, the
router's probe search and the retriever's own search all reuse the vector
(MemoEmbeddings), so cache + routing add ~one local search of overhead on a miss.
"""
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings

from . import config
from .rag import RAGResult, answer_question
from .router import STANDARD, Decision, RouterPolicy, extract_features, forced, route
from .semantic_cache import CachedAnswer, SemanticCache, fingerprint

CACHE = "cache"                  # the route name recorded for a cache hit


class MemoEmbeddings(Embeddings):
    """Wraps the index's embedding model and remembers recent query vectors,
    so every component that embeds the same question pays once."""

    def __init__(self, base: Embeddings, size: int = 2048):
        self.base, self.size = base, size
        self._memo: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.calls = 0                                   # real embedding calls (for tests/metrics)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            if text in self._memo:
                self._memo.move_to_end(text)
                return self._memo[text]
        vector = self.base.embed_query(text)
        with self._lock:
            self.calls += 1
            self._memo[text] = vector
            while len(self._memo) > self.size:
                self._memo.popitem(last=False)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents(texts)          # rewrites / corpus: not worth memoising


# ---------------------------------------------------------------- results

@dataclass
class Path:
    """One way of answering: a retriever, a generator and the model behind it."""
    retriever: object
    generator: object
    model: str


@dataclass
class ServeResult:
    question: str
    route: str                                 # cache | fast | standard | escalate
    answer: str
    answerable: bool
    citations: list[dict]                      # [{number, chunk_id, url, similarity}]
    decision: dict | None = None               # router decision + features (None on a cache hit)
    cache: dict = field(default_factory=dict)  # lookup info, hit details, store outcome
    result: RAGResult | None = None            # full pipeline result on a miss
    timings_ms: dict = field(default_factory=dict)
    cost_usd: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.timings_ms.get("total", 0.0)


# ---------------------------------------------------------------- the assistant

class Assistant:
    def __init__(self, vectorstore, paths: dict[str, Path], *, cache: SemanticCache | None,
                 policy: RouterPolicy | None, pipeline: dict, k: int = config.TOP_K,
                 prompt: str = config.ANSWER_PROMPT, probe_k: int = 5):
        """policy=None turns routing off: every miss takes the standard path."""
        self.vectorstore, self.paths, self.cache, self.policy = vectorstore, paths, cache, policy
        self.k, self.prompt, self.probe_k = k, prompt, probe_k
        # The cache is shared only by requests that would get the same answer
        # from the same system, router policy included.
        self.pipeline = {**pipeline, "router": policy.describe() if policy else None}
        self.fingerprint = fingerprint(self.pipeline)

    def _probe(self, vector) -> list[tuple[str, float]]:
        results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(vector, k=self.probe_k)
        return [(doc.metadata.get("doc_id", ""), 1.0 - distance) for doc, distance in results]

    def _decide(self, question: str, vector, force_route: str | None) -> Decision:
        if force_route:
            return forced(force_route)
        if self.policy is None:
            return Decision(STANDARD, ["routing off"])
        return route(extract_features(question, self._probe(vector)), self.policy)

    def answer(self, question: str, *, force_route: str | None = None, use_cache: bool = True) -> ServeResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        info: dict = {}
        if self.cache is not None and use_cache and force_route is None:
            # Exact repeats are found before any embedding call; on a miss the
            # cache embeds through the shared memo, so the vector is reused below.
            hit, info = self.cache.lookup(question, self.fingerprint)
            timings["embed"] = info.get("embed_ms", 0.0)
            timings["cache_lookup"] = info.get("lookup_ms", 0.0) - timings["embed"]
            if hit is not None:
                timings["total"] = (time.perf_counter() - t0) * 1000
                return self._from_cache(question, hit, info, timings)
        t = time.perf_counter()
        vector = self.vectorstore.embeddings.embed_query(question)        # memo hit after a cache miss
        timings["embed"] = timings.get("embed", 0.0) + (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        decision = self._decide(question, vector, force_route)
        timings["route"] = (time.perf_counter() - t) * 1000

        path = self.paths[decision.route]
        result = answer_question(question, self.vectorstore, path.generator, k=self.k, model_name=path.model,
                                 retriever=path.retriever, prompt=self.prompt)
        timings.update(retrieve=result.retrieve_ms, generate=result.generate_ms,
                       rewrite=result.rewrite_ms, rerank=result.rerank_ms)

        cache_info = {"hit": False, "best_similarity": info.get("best_similarity"),
                      "nearest_question": info.get("nearest_question"), "stored": False, "skip_reason": None}
        if self.cache is not None and use_cache and force_route is None:
            stored, reason = self.cache.store(question, self.fingerprint, result, vector=vector)
            cache_info.update(stored=stored, skip_reason=reason or None)
        timings["total"] = (time.perf_counter() - t0) * 1000

        return ServeResult(question=question, route=decision.route, answer=result.answer,
                           answerable=result.answerable, citations=[c.__dict__ for c in result.citations],
                           decision=decision.to_dict(), cache=cache_info, result=result,
                           timings_ms={k: round(v, 1) for k, v in timings.items()}, cost_usd=result.cost_usd)

    def _from_cache(self, question: str, hit: CachedAnswer, info: dict, timings: dict) -> ServeResult:
        return ServeResult(
            question=question, route=CACHE, answer=hit.answer, answerable=hit.answerable, citations=hit.citations,
            cache={"hit": True, "exact": hit.exact, "similarity": round(hit.similarity, 4),
                   "source_question": hit.source_question, "times_served": hit.hits, "answer_model": hit.model},
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            # One embedding call, ~20 tokens: rounds to $0 but is counted for honesty.
            cost_usd=len(question.split()) * 1.3 * config.EMBEDDING_USD_PER_1M_TOKENS / 1e6)


# ---------------------------------------------------------------- wiring

def fast_settings():
    """The fast path: no rewrite, no rerank; neighbor expansion stays (it is a
    local lookup, no LLM, and it fixed 'right page, wrong section')."""
    from .retrieval import RetrievalSettings
    return RetrievalSettings(rewrite=False, rerank="none")


def build_assistant(*, use_cache: bool = config.CACHE_ENABLED, routing: bool = config.ROUTING != "off",
                    policy: RouterPolicy | None = None, prompt: str = config.ANSWER_PROMPT,
                    cache_threshold: float = config.CACHE_THRESHOLD, cache_path=config.CACHE_DB) -> Assistant:
    """Everything from config: index, the three paths, cache and router."""
    from .rag import build_generator, get_chat_model, prompt_hash
    from .retrieval import RetrievalSettings, build_retriever
    from .vectorstore import get_embeddings, get_vectorstore

    embeddings = MemoEmbeddings(get_embeddings())
    vectorstore = get_vectorstore(embeddings)
    if vectorstore._collection.count() == 0:
        raise SystemExit("The index is empty. Run: ingest -> chunk -> index (see README).")

    small = get_chat_model()
    standard_settings, fast = RetrievalSettings.shipped(), fast_settings()
    standard_retriever = build_retriever(vectorstore, standard_settings, small)
    small_generator = build_generator(small, prompt)
    paths = {
        "fast": Path(build_retriever(vectorstore, fast, small), small_generator, config.CHAT_MODEL),
        "standard": Path(standard_retriever, small_generator, config.CHAT_MODEL),
        "escalate": Path(standard_retriever, build_generator(get_chat_model(model=config.ESCALATE_MODEL), prompt),
                         config.ESCALATE_MODEL),
    }
    pipeline = {
        "corpus_commit": config.FASTAPI_COMMIT, "collection": config.COLLECTION_NAME,
        "embedding_model": config.EMBEDDING_MODEL, "prompt": prompt, "prompt_hash": prompt_hash(prompt),
        "chat_model": config.CHAT_MODEL, "escalate_model": config.ESCALATE_MODEL,
        "standard": standard_settings.describe(), "fast": fast.describe(),
    }
    cache = SemanticCache(embeddings.embed_query, path=cache_path, threshold=cache_threshold) if use_cache else None
    if routing and policy is None:
        policy = RouterPolicy.load(config.ROOT / "evals" / "router_policy.json")
    return Assistant(vectorstore, paths, cache=cache, policy=policy if routing else None,
                     pipeline=pipeline, prompt=prompt)
