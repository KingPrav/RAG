"""Trace records, the JSONL log, the batch-question reader and the report."""
import json

from langchain_core.documents import Document

from docs_assistant import config
from docs_assistant.rag import Citation, RAGResult
from docs_assistant.report import summarize
from docs_assistant.run_queries import read_questions
from docs_assistant.search import Hit
from docs_assistant.tracing import build_trace, load_traces, log_result, prompt_version


def hit(rank: int, chunk_id: str, sim: float, text: str) -> Hit:
    return Hit(rank, sim, Document(id=chunk_id, page_content=text, metadata={
        "chunk_id": chunk_id, "doc_id": chunk_id.split("#")[0], "url": f"https://x.dev/{chunk_id}",
        "n_tokens": 40, "source_commit": "abc123"}))


def make_result(question="How do I declare a request body?", top=0.62, answerable=True) -> RAGResult:
    hits = [hit(1, "tutorial/body#request-body", top, "Use Pydantic models that inherit from BaseModel."),
            hit(2, "tutorial/query-params#query", top - 0.1, "Query parameters come from the signature.")]
    return RAGResult(
        question=question,
        answer="Use Pydantic models that inherit from BaseModel [1]." if answerable else "Not covered.",
        answerable=answerable,
        citations=[Citation(1, hits[0].chunk_id, "u", top)] if answerable else [],
        retrieved=hits, invalid_citations=[], model="stub-model",
        retrieve_ms=120.0, generate_ms=900.0, input_tokens=1000, output_tokens=100)


def test_trace_has_separate_retrieval_and_generation_sections():
    t = build_trace(make_result(), k=5)
    assert t["schema_version"] == 1 and len(t["trace_id"]) == 32
    assert t["corpus"]["source_commit"] == "abc123"
    assert t["corpus"]["collection"] == config.COLLECTION_NAME
    r = t["retrieval"]
    assert r["k"] == 5 and r["latency_ms"] == 120.0
    assert [x["chunk_id"] for x in r["results"]] == ["tutorial/body#request-body", "tutorial/query-params#query"]
    assert r["results"][0] == {"rank": 1, "chunk_id": "tutorial/body#request-body", "doc_id": "tutorial/body",
                               "similarity": 0.62, "n_tokens": 40}
    g = t["generation"]
    assert g["prompt_version"] == prompt_version() and len(g["prompt_version"]) == 10
    assert (g["input_tokens"], g["output_tokens"], g["context_tokens"]) == (1000, 100, 80)
    assert g["citations"] == [1] and g["answerable"] is True
    assert t["usage"]["n_cited"] == 1 and t["usage"]["context_utilization"] == 0.5


def test_trace_stores_ids_not_chunk_text():
    serialized = json.dumps(build_trace(make_result(), k=5))
    assert "Query parameters come from the signature" not in serialized


def test_log_appends_jsonl_and_loads_back(tmp_path):
    path = tmp_path / "logs" / "traces.jsonl"
    log_result(make_result(), 5, path)
    log_result(make_result("second?"), 5, path)
    traces = load_traces(path)
    assert [t["question"] for t in traces] == ["How do I declare a request body?", "second?"]


def test_logging_failure_never_raises(tmp_path, capsys):
    bad_path = tmp_path / "file.txt"
    bad_path.write_text("x")
    assert log_result(make_result(), 5, bad_path / "traces.jsonl") is None      # parent is a file
    assert "could not write trace" in capsys.readouterr().out


def test_read_questions_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text("# header\n\nFirst?\n  # indented comment\nSecond?\n")
    assert read_questions(f) == ["First?", "Second?"]


def test_report_summarizes_traces():
    traces = [build_trace(make_result(f"q{i}", top=0.35 + i * 0.1, answerable=i != 0), k=2) for i in range(4)]
    s = summarize(traces, worst_n=2)
    assert s["traces"] == 4
    assert s["answerable_rate"] == 0.75
    assert s["cost_usd"]["per_query"] == round(make_result().cost_usd, 6)
    assert sum(s["top1_similarity"]["histogram"].values()) == 4
    assert s["most_retrieved_chunks"][0] == ("tutorial/body#request-body", 4)
    assert [q["question"] for q in s["lowest_top1_queries"]] == ["q0", "q1"]
    assert s["lowest_top1_queries"][0]["answerable"] is False
    assert s["latency_ms"]["generation_p50"] == 900.0


def test_report_handles_no_traces():
    assert summarize([]) == {"traces": 0}


def test_report_rescores_traces_logged_with_older_heuristics():
    from docs_assistant.report import rescore_usage
    from docs_assistant.usage import USAGE_VERSION
    old = build_trace(make_result(), k=2)
    old["usage"] = {"usage_version": 1, "n_uncited_sentences": 1}                # v1 shape
    texts = {"tutorial/body#request-body": "Use Pydantic models that inherit from BaseModel.",
             "tutorial/query-params#query": "Query parameters come from the signature."}
    missing_chunk = build_trace(make_result(), k=2)
    missing_chunk["usage"] = {"usage_version": 1}
    missing_chunk["retrieval"]["results"][0]["chunk_id"] = "gone#after-rechunking"
    assert rescore_usage([old, missing_chunk], texts) == 1
    assert old["usage"]["usage_version"] == USAGE_VERSION and old["usage"]["n_cited_claims"] == 1
    assert missing_chunk["usage"] == {"usage_version": 1}                     # left untouched
