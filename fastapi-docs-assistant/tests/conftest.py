"""Shared test setup.

Chroma caches one client (with its open SQLite and index files) per directory
for the life of the process. Each test builds its own temporary index, so
without cleanup the open files pile up until macOS's low default limit (256)
is hit and Chroma fails with "unable to open database file". Clearing the
cache after every test closes them.
"""
import re
import zlib

import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_core.embeddings import Embeddings


@pytest.fixture(autouse=True)
def _release_chroma_clients():
    yield
    SharedSystemClient.clear_system_cache()


class BagOfWordsEmbedding(Embeddings):
    """Stand-in embedder where shared words mean closeness (unlike random fake
    vectors), so retrieval ranks are predictable in tests."""
    size = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.size] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def bow_embedding():
    return BagOfWordsEmbedding()
