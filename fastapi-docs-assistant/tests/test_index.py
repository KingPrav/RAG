"""Indexing and retrieval tests with a fake embedding model: offline, free,
and deterministic (identical text -> identical vector)."""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from docs_assistant.index import build_index
from docs_assistant.search import retrieve
from docs_assistant.vectorstore import get_record_manager, get_vectorstore


def chunk(doc_id: str, anchor: str, text: str) -> Document:
    cid = f"{doc_id}#{anchor}"
    return Document(id=cid, page_content=text, metadata={
        "chunk_id": cid, "doc_id": doc_id, "url": f"https://x.dev/{doc_id}/#{anchor}",
        "breadcrumb": f"FastAPI docs > {doc_id}", "n_tokens": len(text.split()),
        "source_commit": "abc"})


@pytest.fixture
def store(tmp_path):
    embeddings = DeterministicFakeEmbedding(size=64)
    vs = get_vectorstore(embeddings, persist_dir=tmp_path / "chroma", collection_name="test-docs")
    rm = get_record_manager(db_path=tmp_path / "rm.sqlite", collection_name="test-docs")
    return vs, rm


CHUNKS = [
    chunk("tutorial/body", "request-body", "Declare a request body with a Pydantic model."),
    chunk("tutorial/body", "create-your-data-model", "Inherit from BaseModel to create a data model."),
    chunk("tutorial/query-params", "query-parameters", "Function parameters not in the path are query parameters."),
]


def test_collection_uses_cosine_distance(store):
    vs, _ = store
    assert vs._collection.configuration["hnsw"]["space"] == "cosine"


def test_first_run_embeds_everything(store):
    vs, rm = store
    result = build_index(CHUNKS, vs, rm)
    assert result == {"num_added": 3, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}
    assert vs._collection.count() == 3


def test_rerun_with_no_changes_embeds_nothing(store):
    vs, rm = store
    build_index(CHUNKS, vs, rm)
    result = build_index(CHUNKS, vs, rm)
    assert result["num_added"] == 0 and result["num_skipped"] == 3      # no API cost
    assert vs._collection.count() == 3


def test_changed_chunk_is_reembedded_and_old_version_deleted(store):
    vs, rm = store
    build_index(CHUNKS, vs, rm)
    edited = [*CHUNKS[:2], chunk("tutorial/query-params", "query-parameters", "Query params, now reworded.")]
    result = build_index(edited, vs, rm)
    assert (result["num_added"], result["num_skipped"], result["num_deleted"]) == (1, 2, 1)
    assert vs._collection.count() == 3


def test_removed_page_is_deleted_from_store(store):
    vs, rm = store
    build_index(CHUNKS, vs, rm)
    result = build_index(CHUNKS[:2], vs, rm)                            # query-params page gone
    assert result["num_deleted"] == 1
    assert vs._collection.count() == 2


def test_retrieve_ranks_exact_match_first_with_cosine_similarity(store):
    vs, rm = store
    build_index(CHUNKS, vs, rm)
    hits = retrieve(vs, CHUNKS[1].page_content, k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].chunk_id == "tutorial/body#create-your-data-model"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[0].similarity > hits[1].similarity >= hits[2].similarity
    assert hits[0].document.metadata["url"] == "https://x.dev/tutorial/body/#create-your-data-model"
