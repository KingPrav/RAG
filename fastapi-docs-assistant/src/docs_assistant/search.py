"""Semantic search over the index: the retrieval half of RAG, on its own.

Run:  python -m docs_assistant.search "how do I declare a request body?"
      python -m docs_assistant.search "status code for validation errors" -k 10

retrieve() is the function Steps 3 and 4 build on.
"""
import argparse
import time
from dataclasses import dataclass

from langchain_core.documents import Document

from . import config
from .vectorstore import get_embeddings, get_vectorstore


@dataclass
class Hit:
    rank: int
    similarity: float          # cosine similarity, 1.0 = same direction
    document: Document

    @property
    def chunk_id(self) -> str:
        return self.document.metadata["chunk_id"]


def retrieve(vectorstore, query: str, k: int = config.TOP_K) -> list[Hit]:
    # similarity_search_with_score returns cosine *distance*; convert it here
    # rather than using LangChain's relevance-score helper, which can emit
    # values outside 0..1 and warn.
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [Hit(rank=i, similarity=1.0 - distance, document=doc)
            for i, (doc, distance) in enumerate(results, start=1)]


def snippet(doc: Document, width: int = 160) -> str:
    body = doc.page_content.split("\n\n", 1)[-1]          # drop the breadcrumb line
    return " ".join(body.split())[:width]


def main():
    parser = argparse.ArgumentParser(description="Search the FastAPI docs index.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    vectorstore = get_vectorstore(get_embeddings())
    if vectorstore._collection.count() == 0:
        raise SystemExit(f"The index '{config.COLLECTION_NAME}' is empty. "
                         "Run the pipeline first: ingest -> chunk -> index (see README).")
    started = time.perf_counter()
    hits = retrieve(vectorstore, args.query, args.k)
    ms = (time.perf_counter() - started) * 1000

    print(f'Query: "{args.query}"   ({len(hits)} results in {ms:.0f} ms)\n')
    for h in hits:
        m = h.document.metadata
        print(f"{h.rank}. [{h.similarity:.3f}] {m['chunk_id']}")
        print(f"   {m['breadcrumb']}")
        print(f"   {m['url']}")
        print(f"   {snippet(h.document)}...\n")


if __name__ == "__main__":
    main()
