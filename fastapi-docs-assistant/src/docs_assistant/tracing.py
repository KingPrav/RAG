"""Structured query traces: one JSON line per question.

Schema (internal, versioned; map to OpenTelemetry GenAI conventions at export
time once they are stable):

    trace_id, timestamp, schema_version, question
    corpus     {source_commit, collection, embedding_model, chunk_settings}
    retrieval  {k, latency_ms, results: [{rank, chunk_id, doc_id, similarity, n_tokens}]}
    generation {model, prompt_version, latency_ms, input_tokens, output_tokens,
                context_tokens, cost_usd, answer, answerable, citations, invalid_citations}
    usage      {context utilization, citation support, code grounding}  (see usage.py)

Stores chunk IDs and scores rather than chunk text: IDs are enough to replay
a query against the index, and they keep the log small.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .rag import SYSTEM_PROMPT, RAGResult
from .usage import analyze_usage

SCHEMA_VERSION = 1
TRACES_PATH = config.DATA_DIR / "logs" / "traces.jsonl"


def prompt_version() -> str:
    """Short hash of the system prompt, so every trace says which prompt made it."""
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:10]


def build_trace(result: RAGResult, k: int) -> dict:
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
            "latency_ms": round(result.retrieve_ms, 1),
            "results": [
                {"rank": h.rank, "chunk_id": h.chunk_id, "doc_id": h.document.metadata.get("doc_id"),
                 "similarity": round(h.similarity, 4), "n_tokens": h.document.metadata.get("n_tokens")}
                for h in hits
            ],
        },
        "generation": {
            "model": result.model,
            "prompt_version": prompt_version(),
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


def log_result(result: RAGResult, k: int, path: Path = TRACES_PATH) -> dict | None:
    """Build and append a trace. Logging must never break answering, so any
    failure is reported and swallowed."""
    try:
        trace = build_trace(result, k)
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
