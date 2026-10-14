"""Step 8: the retrieval stage, with the two improvements that the failure
analysis and offline experiments pointed to.

    question
      -> [8b] LLM rewrites it into FastAPI's vocabulary (optional)
      -> vector search for the original + each rewrite
      -> reciprocal rank fusion -> top k            (original always included)
      -> [8a] add the sections just before/after each hit on the same page
      -> numbered context for the answering model

Why not hybrid BM25 + vectors, the textbook fix? It was tested offline on the
dev set and rescued none of the misses: paraphrased questions fail because the
user's words are absent from the docs, which keyword matching cannot fix.

Fusion is implemented here (a few lines) rather than with LangChain's
EnsembleRetriever / MultiQueryRetriever because those drop the similarity
scores that the Step 4 instrumentation logs for every chunk.
"""
import time
from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from . import config
from .doc_io import load_documents
from .search import Hit


@dataclass(frozen=True)
class RetrievalSettings:
    k: int = config.TOP_K
    neighbors: int = config.RETRIEVAL_NEIGHBORS
    rewrite: bool = config.QUERY_REWRITE
    n_rewrites: int = config.N_REWRITES
    fusion_depth: int = config.FUSION_DEPTH
    context_max_tokens: int = config.CONTEXT_MAX_TOKENS

    def describe(self) -> dict:
        return asdict(self)


@dataclass
class Retrieved:
    hits: list[Hit]                    # the context, numbered 1..n in the order the model sees it
    queries: list[str]                 # original first, then rewrites
    rewrite_ms: float = 0.0
    search_ms: float = 0.0
    input_tokens: int = 0              # rewrite call usage (same model as answering)
    output_tokens: int = 0
    rewrite_error: str | None = None

    @property
    def search_hits(self) -> list[Hit]:
        """Hits found by search, in search order: what retrieval metrics score."""
        return [h for h in self.hits if h.source == "search"]


# ---------------------------------------------------------------- 8b: query rewriting

REWRITE_SYSTEM = """You turn a developer's question into search queries for the official FastAPI documentation.

Developers describe what they want in their own words; the docs use FastAPI's names for things. Write {n} short search queries that use the terms the documentation would use: class, function, parameter and feature names (for example RedirectResponse, lifespan, APIRouter, middleware, Depends, status_code).

Each query should take a different angle. Do not answer the question. Do not add requirements the developer did not state."""

REWRITE_PROMPT = ChatPromptTemplate.from_messages([("system", REWRITE_SYSTEM), ("human", "{question}")])


class QueryRewrites(BaseModel):
    queries: list[str] = Field(description="Search queries in FastAPI documentation vocabulary.")


def build_rewriter(llm: BaseChatModel) -> Runnable:
    return REWRITE_PROMPT | llm.with_structured_output(QueryRewrites, include_raw=True)


def rewrite_query(rewriter: Runnable, question: str, n: int) -> tuple[list[str], dict, str | None]:
    """Returns (rewrites, usage, error). Never raises: if rewriting fails the
    pipeline falls back to the original question alone."""
    try:
        out = rewriter.invoke({"question": question, "n": n})
    except Exception as e:
        return [], {}, f"{type(e).__name__}: {e}"
    usage = getattr(out.get("raw"), "usage_metadata", None) or {}
    parsed = out.get("parsed")
    if parsed is None:
        return [], usage, f"unparseable rewrite: {out.get('parsing_error')}"
    seen, rewrites = {question.strip().lower()}, []
    for q in parsed.queries:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            rewrites.append(q)
    return rewrites[:n], usage, None


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[Document, float]]], k: int,
                           c: int = config.RRF_K) -> list[tuple[Document, float]]:
    """Score each document by sum(1 / (c + rank)) across lists; a document that
    several phrasings agree on rises. Returns (document, best cosine
    similarity seen) for the top k."""
    scores: dict[str, float] = {}
    best: dict[str, tuple[Document, float]] = {}
    for results in ranked_lists:
        for rank, (doc, similarity) in enumerate(results, start=1):
            cid = doc.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + rank)
            if cid not in best or similarity > best[cid][1]:
                best[cid] = (doc, similarity)
    order = sorted(scores, key=lambda cid: (-scores[cid], -best[cid][1]))
    return [best[cid] for cid in order[:k]]


# ---------------------------------------------------------------- 8a: neighbor expansion

class ChunkNeighbors:
    """Chunks in page order, so a hit's adjacent sections can be looked up."""

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.position = {c.metadata["chunk_id"]: i for i, c in enumerate(chunks)}

    @classmethod
    def from_file(cls, path=config.CHUNKS_PATH) -> "ChunkNeighbors":
        if not path.exists():
            raise SystemExit(f"{path} not found; neighbor expansion needs it. Run: python -m docs_assistant.chunk")
        return cls(load_documents(path))

    def around(self, chunk_id: str, window: int) -> list[Document]:
        """The chunk and up to `window` chunks either side of it on the same page, in page order."""
        i = self.position.get(chunk_id)
        if i is None:
            return []
        doc_id = self.chunks[i].metadata["doc_id"]
        return [self.chunks[j] for j in range(max(0, i - window), min(len(self.chunks), i + window + 1))
                if self.chunks[j].metadata["doc_id"] == doc_id]


def expand_with_neighbors(hits: list[Hit], neighbors: ChunkNeighbors, window: int,
                          max_tokens: int) -> list[Hit]:
    """Each search hit becomes [previous section, hit, next section] in page
    order, so the model reads them as continuous text. Duplicates are dropped
    and neighbors stop being added once the token budget is reached (search
    hits are always kept)."""
    if window <= 0:
        return hits
    tokens = sum(h.document.metadata.get("n_tokens", 0) for h in hits)
    included = {h.chunk_id for h in hits}
    out: list[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        for doc in neighbors.around(hit.chunk_id, window) or [hit.document]:
            cid = doc.metadata["chunk_id"]
            if cid in seen:
                continue
            if cid == hit.chunk_id:
                out.append(hit)
            elif cid in included:
                continue                       # another search hit; it gets its own slot
            else:
                n = doc.metadata.get("n_tokens", 0)
                if tokens + n > max_tokens:
                    continue
                tokens += n
                out.append(Hit(rank=0, similarity=hit.similarity, document=doc, source="neighbor"))
            seen.add(cid)
    for rank, h in enumerate(out, start=1):
        h.rank = rank
    return out


# ---------------------------------------------------------------- the retriever

@dataclass
class Retriever:
    vectorstore: object
    settings: RetrievalSettings = field(default_factory=RetrievalSettings)
    rewriter: Runnable | None = None
    neighbors: ChunkNeighbors | None = None

    def __post_init__(self):
        if self.settings.rewrite and self.rewriter is None:
            raise ValueError("rewrite is on but no rewriter was given")
        if self.settings.neighbors and self.neighbors is None:
            raise ValueError("neighbor expansion is on but no ChunkNeighbors was given")

    def _search(self, queries: list[str], depth: int) -> list[list[tuple[Document, float]]]:
        embeddings = self.vectorstore.embeddings
        vectors = embeddings.embed_documents(queries) if len(queries) > 1 else [embeddings.embed_query(queries[0])]
        return [[(doc, 1.0 - distance) for doc, distance in
                 self.vectorstore.similarity_search_by_vector_with_relevance_scores(v, k=depth)]
                for v in vectors]

    def __call__(self, question: str) -> Retrieved:
        s = self.settings
        queries, usage, error, rewrite_ms = [question], {}, None, 0.0
        if s.rewrite:
            started = time.perf_counter()
            rewrites, usage, error = rewrite_query(self.rewriter, question, s.n_rewrites)
            rewrite_ms = (time.perf_counter() - started) * 1000
            queries += rewrites

        started = time.perf_counter()
        if len(queries) == 1:
            ranked = self._search(queries, s.k)[0]
        else:
            ranked = reciprocal_rank_fusion(self._search(queries, max(s.fusion_depth, s.k)), s.k)
        hits = [Hit(rank=i, similarity=sim, document=doc) for i, (doc, sim) in enumerate(ranked, start=1)]
        if s.neighbors:
            hits = expand_with_neighbors(hits, self.neighbors, s.neighbors, s.context_max_tokens)
        search_ms = (time.perf_counter() - started) * 1000

        return Retrieved(hits=hits, queries=queries, rewrite_ms=rewrite_ms, search_ms=search_ms,
                         input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                         rewrite_error=error)


def build_retriever(vectorstore, settings: RetrievalSettings, llm: BaseChatModel | None = None) -> Retriever:
    """Wire up whatever the settings need (the rewrite model, the chunk file)."""
    rewriter = None
    if settings.rewrite:
        if llm is None:
            from .rag import get_chat_model
            llm = get_chat_model()
        rewriter = build_rewriter(llm)
    neighbors = ChunkNeighbors.from_file() if settings.neighbors else None
    return Retriever(vectorstore, settings, rewriter=rewriter, neighbors=neighbors)
