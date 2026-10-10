"""Baseline RAG chain tests. A stub generator stands in for the LLM so the
tests are offline and free, and it records exactly what the model was sent."""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from docs_assistant import config
from docs_assistant.index import build_index
from docs_assistant.rag import (NO_SOURCES_ANSWER, PROMPT, GeneratedAnswer, answer_question,
                                format_context)
from docs_assistant.search import retrieve
from docs_assistant.vectorstore import get_record_manager, get_vectorstore


def chunk(doc_id: str, anchor: str, text: str) -> Document:
    cid = f"{doc_id}#{anchor}"
    crumb = f"FastAPI docs > {doc_id}"
    return Document(id=cid, page_content=f"{crumb}\n\n{text}", metadata={
        "chunk_id": cid, "doc_id": doc_id, "url": f"https://x.dev/{doc_id}/#{anchor}",
        "breadcrumb": crumb, "n_tokens": len(text.split()), "source_commit": "abc"})


CHUNKS = [
    chunk("tutorial/body", "request-body", "Declare a request body with a Pydantic model."),
    chunk("tutorial/body", "create-your-data-model", "Inherit from BaseModel to create a data model."),
    chunk("tutorial/query-params", "query-parameters", "Function parameters not in the path are query parameters."),
]


@pytest.fixture
def vectorstore(tmp_path):
    vs = get_vectorstore(DeterministicFakeEmbedding(size=64), tmp_path / "chroma", "test-docs")
    build_index(CHUNKS, vs, get_record_manager(tmp_path / "rm.sqlite", "test-docs"))
    return vs


def stub_generator(parsed: GeneratedAnswer | None, tokens=(1200, 150), calls: list | None = None):
    """Mimics `prompt | llm.with_structured_output(..., include_raw=True)`."""
    def run(inputs):
        if calls is not None:
            calls.append(inputs)
        raw = AIMessage(content="", usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                                    "total_tokens": sum(tokens)})
        return {"raw": raw, "parsed": parsed, "parsing_error": None if parsed else "bad json"}
    return RunnableLambda(run)


QUESTION = CHUNKS[0].page_content           # exact text -> retrieved first by the fake embedding


def test_citations_map_to_retrieved_chunks_and_urls(vectorstore):
    parsed = GeneratedAnswer(answer="Use a Pydantic model [1][2].", cited_sources=[1, 2], answerable=True)
    result = answer_question(QUESTION, vectorstore, stub_generator(parsed), k=3, model_name="stub")
    assert result.answer == "Use a Pydantic model [1][2]."
    assert [c.chunk_id for c in result.citations] == ["tutorial/body#request-body",
                                                     result.retrieved[1].chunk_id]
    assert result.citations[0].url == "https://x.dev/tutorial/body/#request-body"
    assert result.citations[0].similarity == pytest.approx(1.0, abs=1e-3)
    assert len(result.retrieved) == 3 and result.invalid_citations == []


def test_hallucinated_citation_numbers_are_removed_and_counted(vectorstore):
    parsed = GeneratedAnswer(answer="See [1] and [7].", cited_sources=[1, 7, 1], answerable=True)
    result = answer_question(QUESTION, vectorstore, stub_generator(parsed), k=3)
    assert [c.number for c in result.citations] == [1]         # deduplicated, valid only
    assert result.invalid_citations == [7]


def test_model_receives_numbered_sources_with_ids_and_breadcrumbs(vectorstore):
    calls = []
    parsed = GeneratedAnswer(answer="x", cited_sources=[], answerable=False)
    answer_question("request body", vectorstore, stub_generator(parsed, calls=calls), k=2)
    sent = calls[0]
    assert sent["question"] == "request body"
    assert sent["context"].count("[1] (id: ") == 1 and "[2] (id: " in sent["context"]
    assert "FastAPI docs > tutorial/" in sent["context"]
    system = PROMPT.format_messages(**sent)[0].content
    assert "ONLY the numbered sources" in system and sent["context"] in system


def test_unanswerable_is_reported(vectorstore):
    parsed = GeneratedAnswer(answer="The sources don't cover GraphQL subscriptions.", cited_sources=[],
                             answerable=False)
    result = answer_question("graphql subscriptions?", vectorstore, stub_generator(parsed), k=3)
    assert result.answerable is False and result.citations == []


def test_tokens_latency_and_cost_are_recorded(vectorstore, monkeypatch):
    monkeypatch.setattr(config, "CHAT_USD_PER_1M_INPUT", 0.20)
    monkeypatch.setattr(config, "CHAT_USD_PER_1M_OUTPUT", 1.20)
    parsed = GeneratedAnswer(answer="x [1]", cited_sources=[1], answerable=True)
    result = answer_question(QUESTION, vectorstore, stub_generator(parsed, tokens=(1000, 500)), k=3)
    assert (result.input_tokens, result.output_tokens) == (1000, 500)
    assert result.cost_usd == pytest.approx((1000 * 0.20 + 500 * 1.20) / 1e6)
    assert result.retrieve_ms > 0 and result.generate_ms > 0
    assert result.total_ms == pytest.approx(result.retrieve_ms + result.generate_ms)


def test_unparseable_model_output_raises(vectorstore):
    with pytest.raises(RuntimeError, match="could not be parsed"):
        answer_question(QUESTION, vectorstore, stub_generator(None), k=3)


def test_empty_index_skips_the_llm_call(tmp_path):
    empty = get_vectorstore(DeterministicFakeEmbedding(size=64), tmp_path / "empty", "empty-docs")
    calls = []
    result = answer_question("anything", empty, stub_generator(None, calls=calls))
    assert calls == [] and result.answer == NO_SOURCES_ANSWER and result.answerable is False


def test_format_context_numbers_hits_in_rank_order(vectorstore):
    hits = retrieve(vectorstore, QUESTION, k=3)
    ctx = format_context(hits)
    assert ctx.index("[1] (id: tutorial/body#request-body)") < ctx.index("[2] (id: ") < ctx.index("[3] (id: ")
