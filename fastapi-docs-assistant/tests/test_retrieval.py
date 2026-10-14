"""Step 8 retrieval: rank fusion, neighbor expansion and query rewriting,
over a tiny real index with a word-overlap embedder and a stub rewriter."""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from docs_assistant.index import build_index
from docs_assistant.rag import GeneratedAnswer, answer_question
from docs_assistant.retrieval import (ChunkNeighbors, QueryRewrites, RetrievalSettings, Retriever,
                                      expand_with_neighbors, reciprocal_rank_fusion, rewrite_query)
from docs_assistant.search import Hit
from docs_assistant.vectorstore import get_record_manager, get_vectorstore

# Two pages, sections in page order. The redirect page uses docs vocabulary
# that a paraphrased question ("send the user to another URL") doesn't share.
PAGES = {
    "advanced/events": [("use-case", "Load a machine learning model before the app starts taking requests.", 30),
                        ("lifespan", "Define a lifespan function with asynccontextmanager and pass lifespan=.", 40),
                        ("alternative-events", "The deprecated startup and shutdown events.", 20)],
    "advanced/custom-response": [("jsonresponse", "JSONResponse returns JSON.", 10),
                                 ("redirectresponse", "RedirectResponse returns an HTTP redirect, 307 by default.", 15)],
}


def chunks() -> list[Document]:
    out = []
    for doc_id, sections in PAGES.items():
        for anchor, text, n in sections:
            cid = f"{doc_id}#{anchor}"
            out.append(Document(id=cid, page_content=f"FastAPI docs > {doc_id}\n\n{text}", metadata={
                "chunk_id": cid, "doc_id": doc_id, "url": f"https://x.dev/{cid}", "breadcrumb": "FastAPI docs",
                "n_tokens": n, "source_commit": "abc"}))
    return out


@pytest.fixture
def vectorstore(tmp_path, bow_embedding):
    vs = get_vectorstore(bow_embedding, tmp_path / "chroma", "test-docs")
    build_index(chunks(), vs, get_record_manager(tmp_path / "rm.sqlite", "test-docs"))
    return vs


def hit(cid: str, rank: int = 1, sim: float = 0.5) -> Hit:
    doc = next(c for c in chunks() if c.metadata["chunk_id"] == cid)
    return Hit(rank=rank, similarity=sim, document=doc)


def stub_rewriter(queries=None, fail=False, calls=None):
    def run(inputs):
        if calls is not None:
            calls.append(inputs)
        if fail:
            raise TimeoutError("rewrite timed out")
        raw = AIMessage(content="", usage_metadata={"input_tokens": 300, "output_tokens": 40, "total_tokens": 340})
        return {"raw": raw, "parsed": QueryRewrites(queries=queries or []), "parsing_error": None}
    return RunnableLambda(run)


# ---------------------------------------------------------------- fusion

def test_rrf_rewards_agreement_and_keeps_best_similarity():
    a, b, c = (Document(page_content=x, metadata={"chunk_id": x}) for x in "abc")
    fused = reciprocal_rank_fusion([[(a, .9), (b, .5)], [(b, .7), (c, .6)], [(b, .4)]], k=3)
    assert [d.metadata["chunk_id"] for d, _ in fused] == ["b", "a", "c"]     # b appears in all three
    assert dict((d.metadata["chunk_id"], s) for d, s in fused)["b"] == .7


# ---------------------------------------------------------------- neighbors

def test_neighbors_stay_on_the_same_page_in_page_order():
    n = ChunkNeighbors(chunks())
    assert [d.metadata["chunk_id"] for d in n.around("advanced/events#alternative-events", 1)] == [
        "advanced/events#lifespan", "advanced/events#alternative-events"]          # next page not included


def test_expansion_wraps_each_hit_and_renumbers():
    out = expand_with_neighbors([hit("advanced/events#use-case", sim=.56)], ChunkNeighbors(chunks()), 1, 6000)
    assert [(h.rank, h.chunk_id, h.source) for h in out] == [
        (1, "advanced/events#use-case", "search"), (2, "advanced/events#lifespan", "neighbor")]
    assert out[1].similarity == .56                                                  # inherits the hit's score


def test_expansion_does_not_duplicate_other_search_hits():
    hits = [hit("advanced/events#use-case", 1), hit("advanced/events#lifespan", 2)]
    out = expand_with_neighbors(hits, ChunkNeighbors(chunks()), 1, 6000)
    ids = [h.chunk_id for h in out]
    assert len(ids) == len(set(ids)) == 3
    assert [h.source for h in out].count("search") == 2


def test_expansion_respects_the_token_budget_but_keeps_search_hits():
    hits = [hit("advanced/events#lifespan")]                                        # 40 tokens
    out = expand_with_neighbors(hits, ChunkNeighbors(chunks()), 1, max_tokens=65)    # room for one neighbor (20)
    assert [h.chunk_id for h in out] == ["advanced/events#lifespan", "advanced/events#alternative-events"]
    assert [h.chunk_id for h in expand_with_neighbors(hits, ChunkNeighbors(chunks()), 1, 10)] == [
        "advanced/events#lifespan"]


# ---------------------------------------------------------------- rewriting

def test_rewrite_query_drops_duplicates_and_blanks():
    rw = stub_rewriter(["RedirectResponse", "  redirectresponse ", "", "send the user to another url"])
    queries, usage, error = rewrite_query(rw, "Send the user to another URL", 3)
    assert queries == ["RedirectResponse"] and usage["input_tokens"] == 300 and error is None


def test_rewrite_rescues_a_paraphrase_and_counts_its_cost(vectorstore):
    question = "after the app gets a form, send the user to another url"
    plain = Retriever(vectorstore, RetrievalSettings(k=1, neighbors=0, rewrite=False))(question)
    assert plain.hits[0].chunk_id == "advanced/events#use-case"                     # shares "the app": wrong page

    calls, rewrites = [], ["RedirectResponse", "HTTP redirect response"]
    # fusion_depth=1: with only 5 sections every one would appear in every list at
    # depth 10, which flattens RRF; in the real 1,030-chunk index they don't.
    r = Retriever(vectorstore, RetrievalSettings(k=1, neighbors=0, rewrite=True, fusion_depth=1),
                  rewriter=stub_rewriter(rewrites, calls=calls))(question)
    assert r.queries == [question, *rewrites] and calls[0]["n"] == 3
    assert r.hits[0].chunk_id == "advanced/custom-response#redirectresponse"
    assert (r.input_tokens, r.output_tokens) == (300, 40)


def test_failed_rewrite_falls_back_to_the_original_question(vectorstore):
    r = Retriever(vectorstore, RetrievalSettings(k=2, neighbors=0, rewrite=True),
                  rewriter=stub_rewriter(fail=True))("lifespan function")
    assert r.queries == ["lifespan function"] and "TimeoutError" in r.rewrite_error
    assert r.hits[0].chunk_id == "advanced/events#lifespan"


def test_settings_without_their_dependencies_are_rejected(vectorstore):
    with pytest.raises(ValueError):
        Retriever(vectorstore, RetrievalSettings(rewrite=True))
    with pytest.raises(ValueError):
        Retriever(vectorstore, RetrievalSettings(neighbors=1, rewrite=False))


# ---------------------------------------------------------------- end to end

def test_answer_includes_neighbors_in_context_and_rewrite_tokens_in_cost(vectorstore):
    seen = {}

    def generate(inputs):
        seen["context"] = inputs["context"]
        raw = AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})
        return {"raw": raw, "parsed": GeneratedAnswer(answer="Use lifespan [2].", cited_sources=[2], answerable=True),
                "parsing_error": None}

    retriever = Retriever(vectorstore, RetrievalSettings(k=1, neighbors=1, rewrite=True),
                          rewriter=stub_rewriter(["lifespan function", "asynccontextmanager lifespan"]),
                          neighbors=ChunkNeighbors(chunks()))
    result = answer_question("load a machine learning model before the app starts", vectorstore,
                             RunnableLambda(generate), retriever=retriever)
    # the rewrites agree on lifespan; expansion adds the sections around it, in page order
    assert [h.chunk_id for h in result.search_hits] == ["advanced/events#lifespan"]
    assert [(h.chunk_id, h.source) for h in result.retrieved] == [
        ("advanced/events#use-case", "neighbor"), ("advanced/events#lifespan", "search"),
        ("advanced/events#alternative-events", "neighbor")]
    assert "[2] (id: advanced/events#lifespan)" in seen["context"]
    assert result.citations[0].chunk_id == "advanced/events#lifespan"
    assert (result.input_tokens, result.output_tokens) == (1300, 140)              # rewrite tokens included
    assert len(result.queries) == 3


# ---------------------------------------------------------------- retrieval-only sweep

def test_sweep_scores_strategies_and_rewrites_each_question_once(vectorstore):
    from docs_assistant.golden import GoldenItem
    from docs_assistant.retrieval_eval import memoized, render, score_strategy, summarize_strategy

    items = [GoldenItem(id="vocab-001", split="dev", category="vocab-mismatch", answer_type="open",
                        question="before the app starts load a machine learning model", expected_answer="Use lifespan.",
                        corpus_commit="abc", nuggets=[{"fact": "lifespan", "vital": True}],
                        gold_sections=["advanced/events#lifespan"])]
    calls = []
    rewriter = memoized(stub_rewriter(["lifespan function"], calls=calls))
    summaries = {}
    for name, s in {"baseline": RetrievalSettings(k=1, neighbors=0, rewrite=False),
                    "n1": RetrievalSettings(k=1, neighbors=1, rewrite=False),
                    "rw": RetrievalSettings(k=1, neighbors=0, rewrite=True),
                    "rw+n1": RetrievalSettings(k=1, neighbors=1, rewrite=True)}.items():
        retriever = Retriever(vectorstore, s, rewriter=rewriter if s.rewrite else None,
                              neighbors=ChunkNeighbors(chunks()) if s.neighbors else None)
        summaries[name] = summarize_strategy(score_strategy(items, retriever, 1, workers=1), 1, "vocab-mismatch")

    assert summaries["baseline"]["gold_in_context"]["value"] == 0.0                # finds the use-case section
    assert summaries["n1"]["gold_in_context"]["value"] == 1.0                      # its neighbor is the answer
    assert summaries["n1"]["hit@1"]["value"] == 0.0                                # ranking itself unchanged
    assert summaries["rw+n1"]["gold_in_context"]["value"] == 1.0
    assert len(calls) == 1                                                         # rewrite reused across strategies
    assert summaries["rw"]["rewrite_cost_usd"] > 0 and summaries["rw+n1"]["rewrite_cost_usd"] == 0
    md = render(summaries, 1, "vocab-mismatch", "dev")
    assert "| n1 |" in md and "**baseline** (1)" in md
