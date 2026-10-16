"""Phase 1, experiment 2: rerank the top candidates before picking the final k.

Vector search ranks by embedding similarity, which put the right section first
only 57% of the time (hit@1) even when it was in the top 5. A reranker reads
the question and each candidate together and re-orders them.

Two rerankers, chosen between by the retrieval sweep rather than on faith:

- "ce":  a cross-encoder (FlashRank, ms-marco-MiniLM-L-12-v2, ~25 MB ONNX, runs
         locally on CPU). Free and fast (tens of ms for 20 passages). Downloaded
         once from Hugging Face into data/models/.
- "llm": listwise LLM reranking (RankGPT-style): one call to the cheap model with
         all candidates, returning them in order. Can reason across passages, but
         costs a call (~5K tokens) and adds latency.

Both return (document, cosine similarity, rerank score) for the top k; the
cosine score is kept so Step 4 traces stay comparable.
"""
import threading
import time
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from . import config

Candidate = tuple[Document, float]                       # (document, cosine similarity)


@dataclass
class Reranked:
    ranked: list[tuple[Document, float, float]]           # (document, cosine similarity, rerank score)
    ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


def fallback(candidates: list[Candidate], k: int, ms: float, error: str) -> Reranked:
    """Never fail a question because reranking failed: keep the search order."""
    return Reranked([(d, s, s) for d, s in candidates[:k]], ms=ms, error=error)


# ---------------------------------------------------------------- cross-encoder

class CrossEncoderReranker:
    name = "ce"

    def __init__(self, model_name: str = config.RERANK_CE_MODEL, cache_dir=config.MODELS_DIR, ranker=None):
        self.model_name, self.cache_dir = model_name, cache_dir
        self._ranker = ranker                              # injectable for tests
        self._lock = threading.Lock()

    @property
    def ranker(self):
        # The first call downloads and unzips the model. Without the lock, parallel
        # eval workers raced on the half-written zip (3 FileNotFoundErrors in the
        # first sweep).
        with self._lock:
            if self._ranker is None:
                from flashrank import Ranker
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._ranker = Ranker(model_name=self.model_name, cache_dir=str(self.cache_dir), max_length=512)
        return self._ranker

    def __call__(self, query: str, candidates: list[Candidate], k: int) -> Reranked:
        started = time.perf_counter()
        try:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": doc.page_content} for i, (doc, _) in enumerate(candidates)]
            results = self.ranker.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            return fallback(candidates, k, (time.perf_counter() - started) * 1000, f"{type(e).__name__}: {e}")
        ranked = [(candidates[r["id"]][0], candidates[r["id"]][1], float(r["score"])) for r in results[:k]]
        return Reranked(ranked, ms=(time.perf_counter() - started) * 1000)


# ---------------------------------------------------------------- LLM listwise

LISTWISE_SYSTEM = """You rank documentation passages by how useful they are for answering a developer's question about the FastAPI web framework.

The developer may describe what they want in their own words rather than FastAPI's terms; judge by meaning, not word overlap. A passage that contains the answer, or the code that implements it, ranks above one that only mentions the topic.

Return the passage numbers from most to least useful. Include every passage number exactly once."""

LISTWISE_HUMAN = "Question: {question}\n\nPassages:\n{passages}"
LISTWISE_PROMPT = ChatPromptTemplate.from_messages([("system", LISTWISE_SYSTEM), ("human", LISTWISE_HUMAN)])
PASSAGE_CHARS = 1500                                      # cap per passage to bound cost


class Ranking(BaseModel):
    ranking: list[int] = Field(description="Passage numbers, most useful first.")


def order_from_ranking(ranking: list[int], n: int) -> list[int]:
    """0-based order from the model's 1-based list: drop invalid and repeated
    numbers, then append anything it left out in the original search order."""
    seen, order = set(), []
    for num in ranking:
        i = num - 1
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    return order + [i for i in range(n) if i not in seen]


class LLMReranker:
    name = "llm"

    def __init__(self, llm):
        self.chain = LISTWISE_PROMPT | llm.with_structured_output(Ranking, include_raw=True)

    def __call__(self, query: str, candidates: list[Candidate], k: int) -> Reranked:
        started = time.perf_counter()
        passages = "\n\n".join(f"[{i}] {doc.page_content[:PASSAGE_CHARS]}"
                               for i, (doc, _) in enumerate(candidates, start=1))
        try:
            out = self.chain.invoke({"question": query, "passages": passages})
        except Exception as e:
            return fallback(candidates, k, (time.perf_counter() - started) * 1000, f"{type(e).__name__}: {e}")
        ms = (time.perf_counter() - started) * 1000
        usage = getattr(out.get("raw"), "usage_metadata", None) or {}
        parsed = out.get("parsed")
        if parsed is None:
            r = fallback(candidates, k, ms, f"unparseable ranking: {out.get('parsing_error')}")
            r.input_tokens, r.output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            return r
        order = order_from_ranking(parsed.ranking, len(candidates))
        n = len(candidates)
        ranked = [(candidates[i][0], candidates[i][1], float(n - pos)) for pos, i in enumerate(order[:k])]
        return Reranked(ranked, ms=ms, input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0))


def build_reranker(kind: str, llm=None):
    if kind in (None, "", "none"):
        return None
    if kind == "ce":
        return CrossEncoderReranker()
    if kind == "llm":
        if llm is None:
            raise ValueError("the llm reranker needs a chat model")
        return LLMReranker(llm)
    raise ValueError(f"unknown reranker {kind!r} (choose none, ce or llm)")
