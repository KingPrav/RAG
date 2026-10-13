"""Shared test setup.

Chroma caches one client (with its open SQLite and index files) per directory
for the life of the process. Each test builds its own temporary index, so
without cleanup the open files pile up until macOS's low default limit (256)
is hit and Chroma fails with "unable to open database file". Clearing the
cache after every test closes them.
"""
import pytest
from chromadb.api.shared_system_client import SharedSystemClient


@pytest.fixture(autouse=True)
def _release_chroma_clients():
    yield
    SharedSystemClient.clear_system_cache()
