"""Single source of truth for paths and settings.

Every tunable lives here so later experiments (chunk size, top-k, models) are
compared by changing one file and re-running the evals.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPO_DIR = DATA_DIR / "raw" / "fastapi"          # sparse clone of the FastAPI repo
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"                   # vector store + record manager

# ---- Step 1: corpus
# The corpus is pinned to one commit, so eval numbers stay comparable.
# Bump it deliberately, re-run ingestion and evals, and note it in the results.
FASTAPI_REPO_URL = "https://github.com/fastapi/fastapi.git"
FASTAPI_COMMIT = "5f9fc5c59a9bb54608aa35376715f3ba9708188e"

DOCS_BASE_URL = "https://fastapi.tiangolo.com"

# Pages shorter than this after cleaning are landing-page stubs with no answers.
MIN_DOC_CHARS = 300

# Excluded on purpose. Each reason is written to the manifest.
EXCLUDED_PAGES = {
    "release-notes.md": "changelog; ~710 KB, would be half the corpus and dominate retrieval",
    "fastapi-people.md": "community page, not product documentation",
    "management.md": "repository governance, not product documentation",
    "contributing.md": "contributor guide, not product documentation",
    "help-fastapi.md": "community page, not product documentation",
    "newsletter.md": "signup page",
    "external-links.md": "link list with no answer content",
    "translations.md": "translation process, not product documentation",
    "translation-banner.md": "UI fragment",
    "_llm-test.md": "internal test page for translation tooling",
}
EXCLUDED_DIRS = {
    "reference": "API reference is generated from source docstrings at build time; "
                 "the Markdown holds only '::: fastapi.X' directives",
}

# ---- Step 2a: chunking (values chosen from the section-size distribution;
# see README). These are the first knobs to tune in Step 8.
CHUNK_SPLIT_LEVEL = 3      # start a new chunk at every H1/H2/H3 heading
CHUNK_MAX_TOKENS = 512     # ~90% of H3-level sections fit without splitting
CHUNK_MIN_TOKENS = 50      # smaller sections are merged into a neighbour
TOKENIZER_ENCODING = "cl100k_base"   # the tokenizer used by text-embedding-3-*

# ---- Step 2b: embeddings + vector store
EMBEDDING_MODEL = "text-embedding-3-small"    # 1,536 dims, unit-length vectors
# Approximate list price, used only for the cost estimate in the index manifest.
# Check OpenAI's pricing page; update if it changes.
EMBEDDING_USD_PER_1M_TOKENS = 0.02

# One collection per (model, chunk size), so Step 8 experiments can sit side by
# side without overwriting each other.
COLLECTION_NAME = f"fastapi-docs-te3small-c{CHUNK_MAX_TOKENS}"
CHROMA_DIR = INDEX_DIR / "chroma"
RECORD_MANAGER_DB = INDEX_DIR / "record_manager.sqlite"
INDEX_BATCH_SIZE = 100

# ---- Retrieval
TOP_K = 5

# ---- Step 3: generation
# OpenAI's pages disagree on the exact identifier of the cheapest current tier
# ("Luna"), so the model is configurable. Override with CHAT_MODEL in .env.
# List the models your key can use:
#   python -c "from openai import OpenAI; print(sorted(m.id for m in OpenAI().models.list()))"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-6-luna")
# Approximate list prices for the cost estimate (USD per 1M tokens). Check
# OpenAI's pricing page and update these if you change model.
CHAT_USD_PER_1M_INPUT = float(os.getenv("CHAT_USD_PER_1M_INPUT", "0.20"))
CHAT_USD_PER_1M_OUTPUT = float(os.getenv("CHAT_USD_PER_1M_OUTPUT", "1.20"))
