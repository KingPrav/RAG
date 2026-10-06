"""Step 2b: embed chunks and store them in Chroma.

Uses LangChain's indexing API (index() + a record manager) so re-runs are
incremental:
  - unchanged chunks are skipped (no API cost)
  - changed chunks are re-embedded and their old versions deleted
  - chunks from pages that no longer exist are deleted (cleanup="full")

Run:  python -m docs_assistant.index
"""
import json
import time
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index

from . import config
from .doc_io import load_documents
from .vectorstore import get_embeddings, get_record_manager, get_vectorstore


def build_index(chunks: list[Document], vectorstore, record_manager) -> dict:
    # source_id_key: chunks are grouped by page, so when a page changes its
    # outdated chunks are found and deleted.
    # key_encoder: index() hashes content to detect changes; SHA-256 instead
    # of the default SHA-1, which is not collision-resistant.
    return index(
        chunks,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="doc_id",
        key_encoder="sha256",
        batch_size=config.INDEX_BATCH_SIZE,
    )


def run(chunks: list[Document], embeddings: Embeddings, vectorstore, record_manager) -> dict:
    started = time.perf_counter()
    result = build_index(chunks, vectorstore, record_manager)
    seconds = time.perf_counter() - started

    total_tokens = sum(c.metadata["n_tokens"] for c in chunks)
    # index() does not say which chunks were embedded, only how many, so the
    # cost is pro-rated from the average chunk size.
    embedded_tokens = round(total_tokens * result["num_added"] / max(len(chunks), 1))
    chunk_manifest_path = config.PROCESSED_DIR / "chunks_manifest.json"
    chunk_settings = (json.loads(chunk_manifest_path.read_text())["settings"]
                      if chunk_manifest_path.exists() else None)

    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": getattr(embeddings, "model", type(embeddings).__name__),
        "collection": vectorstore._collection.name,
        "distance": "cosine",
        "source_commit": chunks[0].metadata["source_commit"] if chunks else None,
        "chunk_settings": chunk_settings,
        "chunks_in_input": len(chunks),
        "vectors_in_store": vectorstore._collection.count(),
        "this_run": {**result, "seconds": round(seconds, 1),
                     "approx_tokens_embedded": embedded_tokens,
                     "approx_cost_usd": round(embedded_tokens / 1e6 * config.EMBEDDING_USD_PER_1M_TOKENS, 4)},
    }


def main():
    src = config.PROCESSED_DIR / "chunks.jsonl"
    if not src.exists():
        raise SystemExit("chunks.jsonl not found. Run: python -m docs_assistant.chunk")
    chunks = load_documents(src)

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    record_manager = get_record_manager()

    print(f"Indexing {len(chunks)} chunks into '{config.COLLECTION_NAME}' with {config.EMBEDDING_MODEL}...")
    manifest = run(chunks, embeddings, vectorstore, record_manager)
    (config.INDEX_DIR / "index_manifest.json").write_text(json.dumps(manifest, indent=2))

    r = manifest["this_run"]
    print(f"  embedded:  {r['num_added']}   skipped (unchanged): {r['num_skipped']}   deleted: {r['num_deleted']}")
    print(f"  vectors in store: {manifest['vectors_in_store']}")
    print(f"  time: {r['seconds']}s   approx cost: ${r['approx_cost_usd']}")
    if manifest["vectors_in_store"] != len(chunks):
        print("  WARNING: store size does not match the number of chunks")


if __name__ == "__main__":
    main()
