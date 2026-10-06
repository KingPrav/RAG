"""Factories for the embedding model, vector store and record manager.

Everything that touches OpenAI or Chroma is created here, so tests can swap
in a fake embedding model and a temporary directory.
"""
import os
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from . import config


def get_embeddings() -> Embeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env (see .env.example).")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, max_retries=6)


def get_vectorstore(embeddings: Embeddings, persist_dir: Path = config.CHROMA_DIR,
                    collection_name: str = config.COLLECTION_NAME) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        # Chroma defaults to squared-L2 distance. Cosine gives an interpretable
        # similarity (1 - distance) that query logs and evals can threshold on.
        collection_configuration={"hnsw": {"space": "cosine"}},
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True,
                                 persist_directory=str(persist_dir)),
    )


def get_record_manager(db_path: Path = config.RECORD_MANAGER_DB,
                       collection_name: str = config.COLLECTION_NAME):
    """Tracks a content hash for every indexed chunk, so re-runs only embed
    what changed and delete what disappeared."""
    from langchain_classic.indexes import SQLRecordManager

    db_path.parent.mkdir(parents=True, exist_ok=True)
    manager = SQLRecordManager(f"chroma/{collection_name}", db_url=f"sqlite:///{db_path}")
    manager.create_schema()
    return manager
