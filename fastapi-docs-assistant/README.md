# FastAPI Docs Assistant

An internal developer-support assistant: ask a question about FastAPI and get
an accurate answer with citations to the exact docs section. Built as a
production-style RAG system with retrieval instrumentation and an automated
eval harness, so every quality claim is backed by a number.

## Architecture

Built on **LangChain** interfaces, with custom components where the built-ins
don't fit this corpus:

| Stage | Component | LangChain interface |
|---|---|---|
| Load | `FastAPIDocsLoader` (custom: resolves code includes, keeps section anchors) | `BaseLoader` → `Document` |
| Chunk | `SectionChunker` (custom: code-safe, token-bounded) | `BaseDocumentTransformer` |
| Embed + store | `text-embedding-3-small` → Chroma (cosine), incremental via `index()` + `SQLRecordManager` | `OpenAIEmbeddings`, `Chroma`, indexing API |
| Retrieve | `retrieve()` → top-k chunks with cosine similarity | `VectorStore` |
| Generate | *(next step)* | LCEL chain |

Why custom loader and chunker: no built-in loader resolves FastAPI's `{* ... *}` code
includes, and `MarkdownHeaderTextSplitter` strips indentation inside code blocks
(`    return item` → `return item`), corrupting the Python examples. A regression test
(`test_langchain_markdown_splitter_breaks_code`) documents this.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
python -m pytest -q
```

## Step 1: build the corpus

```bash
python -m docs_assistant.fetch     # sparse download of the docs at the pinned commit
python -m docs_assistant.ingest    # clean + structure -> data/processed/
```

Outputs:
- `data/processed/documents.jsonl`: one record per page, with clean text,
  section headings (offset + anchor URL), section and source commit
- `data/processed/manifest.json`: counts, include-resolution stats and the
  reason each page was excluded

### Corpus at commit `5f9fc5c`

| | |
|---|---|
| Pages indexed | 119 (tutorial 51, advanced 35, general 12, how-to 12, deployment 9) |
| Size | ~923K chars (~231K tokens) |
| Section headings | 1,088 |
| Code blocks | 788 (449/449 code includes resolved) |
| Pages excluded | 37 (24 API reference, 10 non-docs, 3 stubs) |

### Ingestion decisions

- **Code examples are inlined.** The docs reference code via `{* ../../docs_src/... *}`
  markers. Stripping them would silently drop every example, so each is resolved
  (including `ln[...]` line ranges) and any failure is reported in the manifest.
- **Headings are parsed outside code blocks only.** Over 1,000 Python comments
  (`# A Pydantic model`) would otherwise be read as section titles.
- **Section anchors are kept** so answers can cite `.../tutorial/body/#create-your-data-model`,
  not just the page.
- **Markup becomes text.** `/// tip` becomes `Tip:`, `<abbr title="Object-Relational Mapper">ORM</abbr>`
  becomes `ORM (Object-Relational Mapper)`, and terminal colour tags are stripped from console blocks.
- **Release notes are excluded** (~710 KB, half the raw corpus) so they don't dominate retrieval.
- **API reference pages are excluded.** They are generated from docstrings at build time and
  contain only `::: fastapi.X` directives in Markdown.
- **Embedded binary is removed.** One example holds a 4 KB base64 image; long base64 runs in
  code become `<base64 data, N chars omitted>`.
- **Known issue (to measure in Step 5):** a page that highlights different lines of the same
  example repeats the full file each time.

## Step 2a: chunking

```bash
python -m docs_assistant.chunk     # -> data/processed/chunks.jsonl + chunks_manifest.json
```

Structure-aware, token-bounded chunking. Settings live in `config.py` and were chosen
from the measured section sizes:

| Split pages at | Sections | Median tokens | p90 | Max | Over 800 tokens |
|---|---|---|---|---|---|
| H2 | 681 | 188 | 826 | 4,462 | 72 |
| H2 + H3 | 1,034 | 159 | 444 | 1,761 | 29 |

- **One chunk per H1/H2/H3 section**, so every retrieved chunk maps to a citable URL anchor.
- **512-token cap.** ~90% of sections fit whole; larger ones split at paragraph boundaries into
  balanced parts. Code blocks are never cut unless one alone exceeds the cap, in which case each
  piece is re-fenced so it stays valid Markdown.
- **Sections under 50 tokens are merged forward**, so a heading like `## WebSockets client` opens
  the chunk holding its content instead of becoming a near-empty chunk that matches everything.
- **Contextual chunk headers.** Each chunk is embedded with its breadcrumb, e.g.
  `FastAPI docs > Tutorial > Request Body > Create your data model`.
- **Stable IDs** (`tutorial/body#create-your-data-model`, `.../2` for split parts) so logs and
  evals can refer to chunks across runs.

## Step 2b: embeddings + vector store

```bash
python -m docs_assistant.index                                   # embed + store (incremental)
python -m docs_assistant.search "how do I declare a request body?"   # inspect retrieval
```

- **Model:** `text-embedding-3-small` (1,536 dims, unit-length vectors). Embedding the full
  corpus (~250K tokens) costs about half a cent.
- **Store:** Chroma, persisted locally in `data/index/`, configured for **cosine** distance
  (Chroma's default is squared L2) so every result carries an interpretable similarity score.
- **Incremental indexing** with LangChain's indexing API: a record manager keeps a SHA-256
  hash of every chunk. Re-running with no changes embeds nothing; edited chunks are
  re-embedded and their old versions deleted; chunks from removed pages are deleted
  (`cleanup="full"`, grouped by `doc_id`).
- **One collection per configuration** (`fastapi-docs-te3small-c512`), so Step 8 experiments
  don't overwrite each other.
- `data/index/index_manifest.json` records the model, docs commit, chunk settings, counts,
  time and approximate cost of every indexing run.

## Roadmap

1. Corpus ingestion ✅
2. Chunking ✅, embeddings + vector store ✅
3. Baseline RAG pipeline
4. Retrieval instrumentation: queries, chunks, similarity scores, whether the answer used them
5. Failure analysis
6. Ground-truth set: 50–100 Q&A pairs targeting the weak spots
7. Eval harness: exact match for factual answers, LLM-as-judge for open-ended ones
8. Improvements, measured against the baseline
9. CI eval gate + results table
