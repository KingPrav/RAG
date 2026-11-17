"""Structured query traces: one JSON line per question.

Schema (internal, versioned; map to OpenTelemetry GenAI conventions at export
time once they are stable):

    trace_id, timestamp, schema_version, question
    corpus     {source_commit, collection, embedding_model, chunk_settings}
    retrieval  {k, latency_ms, results: [{rank, chunk_id, doc_id, similarity, n_tokens}]}
    generation {model, prompt_version, latency_ms, input_tokens, output_tokens,
                context_tokens, cost_usd, answer, answerable, citations, invalid_citations}
    usage      {context utilization, citation support, code grounding}  (see usage.py)
    serving    {route, decision {reasons, features}, cache, timings_ms, cost_usd,
                pipeline_fingerprint}   (v4, requests answered through serve.Assistant)

A cache hit has no retrieval and no usage block (nothing was retrieved); its
generation block describes the cached answer and its original model.

Stores chunk IDs and scores rather than chunk text: IDs are enough to replay
a query against the index, and they keep the log small.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .rag import RAGResult, prompt_hash
from .usage import analyze_usage

SCHEMA_VERSION = 4          # 2: settings, queries, per-chunk source; 3: prompt id, rerank, rewrite cache
                            # 4: serving block (route, router features, cache, per-step timings, cost)
TRACES_PATH = config.DATA_DIR / "logs" / "traces.jsonl"


def prompt_version(prompt: str = config.ANSWER_PROMPT) -> str:
    """Short hash of the system prompt, so every trace says which prompt made it."""
    return prompt_hash(prompt)


def build_trace(result: RAGResult, k: int, settings: dict | None = None) -> dict:
    hits = result.retrieved
    first = hits[0].document.metadata if hits else {}
    chunks_shown = {h.rank: h.document.page_content for h in hits}
    return {
        "trace_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "schema_version": SCHEMA_VERSION,
        "question": result.question,
        "corpus": {
            "source_commit": first.get("source_commit"),
            "collection": config.COLLECTION_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
        },
        "retrieval": {
            "k": k,
            "settings": settings,
            "queries": result.queries,
            "rewrite_ms": round(result.rewrite_ms, 1),
            "rewrite_error": result.rewrite_error,
            "rewrite_cached": result.rewrite_cached,
            "rerank_ms": round(result.rerank_ms, 1),
            "rerank_error": result.rerank_error,
            "latency_ms": round(result.retrieve_ms, 1),
            "results": [
                {"rank": h.rank, "chunk_id": h.chunk_id, "doc_id": h.document.metadata.get("doc_id"),
                 "similarity": round(h.similarity, 4), "n_tokens": h.document.metadata.get("n_tokens"),
                 "source": h.source, "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 4)}
                for h in hits
            ],
        },
        "generation": {
            "model": result.model,
            "prompt_version": prompt_version(result.prompt),
            "prompt": result.prompt,
            "latency_ms": round(result.generate_ms, 1),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "context_tokens": sum(h.document.metadata.get("n_tokens", 0) for h in hits),
            "cost_usd": round(result.cost_usd, 6),
            "answer": result.answer,
            "answerable": result.answerable,
            "citations": [c.number for c in result.citations],
            "invalid_citations": result.invalid_citations,
        },
        "usage": analyze_usage(result.answer, chunks_shown, [c.number for c in result.citations],
                               answerable=result.answerable),
    }


def log_result(result: RAGResult, k: int, path: Path = TRACES_PATH, settings: dict | None = None) -> dict | None:
    """Build and append a trace. Logging must never break answering, so any
    failure is reported and swallowed."""
    try:
        trace = build_trace(result, k, settings)
        append_trace(trace, path)
        return trace
    except Exception as e:                       # noqa: BLE001 - deliberate
        print(f"WARNING: could not write trace: {e}")
        return None


def append_trace(trace: dict, path: Path = TRACES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(trace) + "\n")


def load_traces(path: Path = TRACES_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- served requests (v4)

def build_served_trace(served, k: int, fingerprint: str, settings: dict | None = None) -> dict:
    """Trace for a request answered by serve.Assistant: the usual blocks for a
    pipeline answer, plus `serving`. Cache hits get a compact trace."""
    serving = {"route": served.route, "decision": served.decision, "cache": served.cache,
               "timings_ms": served.timings_ms, "cost_usd": round(served.cost_usd, 8),
               "pipeline_fingerprint": fingerprint}
    if served.result is not None:
        trace = build_trace(served.result, k, settings)
        trace["serving"] = serving
        return trace
    return {
        "trace_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "schema_version": SCHEMA_VERSION,
        "question": served.question,
        "corpus": {"collection": config.COLLECTION_NAME, "embedding_model": config.EMBEDDING_MODEL},
        "retrieval": None,
        "generation": {"model": served.cache.get("answer_model"), "latency_ms": 0.0, "input_tokens": 0,
                       "output_tokens": 0, "cost_usd": 0.0, "answer": served.answer,
                       "answerable": served.answerable, "citations": [c["number"] for c in served.citations],
                       "invalid_citations": []},
        "usage": None,
        "serving": serving,
    }


def log_served(served, k: int, fingerprint: str, path: Path = TRACES_PATH, settings: dict | None = None) -> dict | None:
    try:
        trace = build_served_trace(served, k, fingerprint, settings)
        append_trace(trace, path)
        return trace
    except Exception as e:                       # noqa: BLE001 - logging must never break answering
        print(f"WARNING: could not write trace: {e}")
        return None
