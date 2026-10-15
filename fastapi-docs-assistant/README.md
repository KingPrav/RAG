# FastAPI Docs Assistant

An internal developer-support assistant: ask a question about FastAPI and get
an accurate answer with citations to the exact docs section. Built as a
production-style RAG system with retrieval instrumentation and an automated
eval harness, so every quality claim is backed by a number.

**At a glance** (provisional until the golden set is human-reviewed; full table in [Results](#results)):

| | Baseline (dev) | Shipped (dev) | Shipped (held-out test) |
|---|---|---|---|
| Overall pass rate | 82.9% | **90.2%** | **87.2%** |
| Paraphrased questions passed | 5/10 | **8/10** | 6/10 |
| Right section in the model's context | 77.1% | **94.3%** | 91.4% |
| Invalid citations | 0% | 0% | 0% |
| Cost / p50 latency per question | $0.0005 / 2.8 s | $0.0010 / 4.5 s | $0.0010 / 4.5 s |

Pipeline: FastAPI docs (pinned commit) → custom loader and code-safe chunker → `text-embedding-3-small` + Chroma → LLM query rewrite + rank fusion + neighbor-section expansion → grounded answer with checked citations → JSONL traces. An 80-question golden set, LLM-as-judge harness and CI gate measure every change.

## Architecture

Built on **LangChain** interfaces, with custom components where the built-ins
don't fit this corpus:

| Stage | Component | LangChain interface |
|---|---|---|
| Load | `FastAPIDocsLoader` (custom: resolves code includes, keeps section anchors) | `BaseLoader` → `Document` |
| Chunk | `SectionChunker` (custom: code-safe, token-bounded) | `BaseDocumentTransformer` |
| Embed + store | `text-embedding-3-small` → Chroma (cosine), incremental via `index()` + `SQLRecordManager` | `OpenAIEmbeddings`, `Chroma`, indexing API |
| Retrieve | `Retriever` (Step 8): LLM rewrites the question into docs vocabulary → vector search for original + rewrites → reciprocal rank fusion → top-k → adjacent sections added (±1) | `VectorStore`, `ChatPromptTemplate` \| `with_structured_output` |
| Generate | Retrieve → numbered sources → structured answer → citation check | `ChatPromptTemplate` \| `ChatOpenAI.with_structured_output` |
| Trace | One JSON line per question: queries, chunks, scores, sources, usage signals, latency, cost | internal schema (v2) |
| Evaluate | Golden set → exact match / LLM judge / refusal check → bootstrap CIs → CI gate | `init_chat_model` (pluggable judge) |

Why custom loader and chunker: no built-in loader resolves FastAPI's `{* ... *}` code
includes, and `MarkdownHeaderTextSplitter` strips indentation inside code blocks
(`    return item` → `return item`), corrupting the Python examples. A regression test
(`test_langchain_markdown_splitter_breaks_code`) documents this.

<details>
<summary>Module map (<code>src/docs_assistant/</code>)</summary>

| Module | Role |
|---|---|
| `config.py` | Every setting and price in one place |
| `fetch.py`, `markdown_clean.py`, `loader.py`, `ingest.py` | Step 1: download the docs at a pinned commit and turn them into clean `Document`s |
| `tokens.py`, `chunk.py` | Step 2a: section-aware, token-bounded chunking |
| `vectorstore.py`, `index.py`, `search.py` | Step 2b: embeddings, Chroma, incremental indexing, plain search |
| `retrieval.py` | Step 8: query rewriting, rank fusion, neighbor expansion |
| `rag.py`, `ask.py` | Step 3: the answer chain and the CLI |
| `tracing.py`, `usage.py`, `report.py`, `run_queries.py`, `review.py` | Step 4–5: traces, "did the answer use the chunks?" signals, reports |
| `golden.py` | Step 6: golden-set schema and validator |
| `eval_metrics.py`, `judge.py`, `evaluate.py`, `compare.py`, `calibrate.py` | Step 7: scoring, LLM judge, eval runs, paired comparison, judge calibration |
| `retrieval_eval.py` | Step 8: cheap retrieval-only sweep |
| `gate.py`, `results.py` | Step 9: CI gate and the generated results table |
| `ragas_eval.py`, `ragas_testset.py` | Ragas metrics on saved runs; Ragas-generated golden candidates |

</details>

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
python -m pytest -q          # 134 offline tests: no API key, stand-in models
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

## Step 3: baseline RAG chain

```bash
python -m docs_assistant.ask "how do I declare a request body?"
python -m docs_assistant.ask "..." --show-retrieved     # also list every retrieved chunk
```

- **Two-step chain, not an agent:** one retrieval, one LLM call. Predictable cost and latency,
  and every later improvement can be attributed to a specific change.
- **Grounded prompt:** answer only from the numbered sources, cite every claim as `[n]`,
  copy code exactly, and say so when the sources don't contain the answer.
- **Structured output** (`answer`, `cited_sources`, `answerable`) instead of free text, so
  citations are machine-checkable. Cited numbers that were never retrieved are removed and
  counted as `invalid_citations`.
- **No similarity cutoff yet:** good matches score ~0.5-0.6 with `text-embedding-3-small`, so a
  threshold will be set from logged data in Step 4, not guessed.
- Every answer returns a `RAGResult`: retrieved chunks and similarities, citations, retrieval
  and generation latency, tokens and cost. Step 4 logs these.
- **Model:** configurable via `CHAT_MODEL` in `.env`; prices for the cost estimate live in
  `config.py`.

## Step 4: retrieval instrumentation

```bash
python -m docs_assistant.ask "..."                                   # every question is traced
python -m docs_assistant.run_queries queries/probe_questions.txt     # trace 40 probe questions
python -m docs_assistant.report                                      # summarize the traces
```

Every question writes one JSON line to `data/logs/traces.jsonl`:

| Section | Fields |
|---|---|
| `retrieval` | k, settings, **queries searched** (original + rewrites), rewrite latency/errors, latency, and per result: rank, chunk ID, doc ID, cosine similarity, tokens, **source** (`search` or `neighbor`) |
| `generation` | model, **prompt version** (hash of the system prompt), latency, input/output/context tokens, cost, answer, citations, invalid citations, answerable |
| `usage` | did the answer actually use the chunks? (below) |
| `corpus` | docs commit, collection, embedding model, chunk size |

- **Retrieval and generation are logged separately**, each with its own latency, so a bad or
  slow answer can be attributed to search or to the model.
- **IDs and scores, not chunk text**: enough to replay a query against the index, and small.
- **Own schema, versioned** (v2 since Step 8 added queries and per-chunk source). OpenTelemetry's GenAI semantic conventions are still in
  *Development* status, so traces use an internal schema (`schema_version`) that can be mapped
  to OTel at export time. LangSmith tracing can be enabled separately via environment variables.

**Did the answer use the retrieved chunks?** Three deterministic signals per answer (`usage.py`):

| Signal | Question it answers |
|---|---|
| Context utilization | What share of retrieved chunks were cited? |
| Claim support | For each claim (paragraph or list item) citing `[n][m]`, how much of its vocabulary appears in the cited chunks combined? (< 0.5 = unsupported) |
| Code grounding | What share of each answer code block's lines appear verbatim in the retrieved chunks? (< 0.8 = ungrounded, i.e. possibly invented API) |

These are lexical triage signals, calibrated in Step 5 against 40 hand-graded answers: they can
miss heavy paraphrase and reward copying, so they flag answers to inspect rather than grade them.
Step 7 adds an LLM judge for faithfulness.

`report` summarizes: latency p50/p95 per stage, cost and tokens per query, top-1 similarity
distribution, answerable rate, context utilization, invalid/unsupported citation rates,
ungrounded code, the most-retrieved chunks and the weakest-retrieval queries.

## Step 5: failure analysis

```bash
python -m docs_assistant.review --refused     # print refused answers in full
python -m docs_assistant.review "API key"     # or search by question text / log position
```

Full write-up: [`analysis/FAILURE_ANALYSIS.md`](analysis/FAILURE_ANALYSIS.md). Labels:
[`analysis/failure_labels.csv`](analysis/failure_labels.csv).

- **Method:** open coding of every trace, then axial coding into counted failure modes (Husain &
  Shankar), mapped to Barnett et al.'s seven RAG failure points.
- **Baseline on 40 probe questions:** 37 pass / 1 partial / 2 fail; retrieval hit@5 94.3%,
  hit@1 65.7%, MRR@5 0.765; 0 invalid citations; 6/6 correct refusals.
- **Main failure mode:** paraphrased intent ("API key on every request", "without logging in")
  doesn't retrieve the docs' framework vocabulary, causing false refusals. Exact API names, the
  assumed weak spot, all passed.
- **Similarity scores don't separate good from bad retrievals**, so there is no score threshold.
- **Instrumentation calibrated:** the v1 "unsupported citation" (50%) and "uncited sentence" (75%)
  signals were mostly false alarms; v3 claim-level checks agree with the hand labels (4.7% / 0%).

## Step 6: ground-truth eval set

```bash
python evals/build_golden_v1.py                # regenerate evals/golden_v1.jsonl (+ review sheet)
python -m docs_assistant.golden validate       # prove every item against the indexed corpus
```

**80 items** in [`evals/golden_v1.jsonl`](evals/golden_v1.jsonl), weighted toward the weak spots
found in Step 5:

| Category | Items | Purpose |
|---|---|---|
| vocab-mismatch | 20 | User intent in everyday words; answer under framework terms (the main Step 5 failure) |
| ranking | 10 | A neighboring section shares the words (hit@1 was 65.7%) |
| exact-fact | 14 | Exact match: ports, URLs, defaults, status codes, packages |
| exact-term | 8 | Exact API names (confirm they stay strong) |
| code | 10 | Code answers, checked with code grounding |
| multi-section | 8 | Answers that need several sections |
| unanswerable (oos + trap) | 10 | Correct refusal, including plausible-but-not-in-the-docs traps (e.g. "422") |

Each item has a split (`dev` for Step 8 iteration, `test` held out for final numbers), an
answer type (`exact` → exact match on `exact_accept`; `open` → LLM judge on vital/okay
**nuggets**; `abstain` → must refuse), **gold sections** at `page#anchor` level (stable across
re-chunking), and **verbatim evidence quotes**.

`golden validate` (exit code 1 on failure, CI-ready) proves every item against the corpus:
gold sections exist, every quote appears verbatim in its section, every exact answer appears in
the evidence, refusal items have no evidence, IDs/questions are unique and splits are balanced per
category. The validator itself is tested against 13 kinds of bad items.

**Review:** items are written as drafts and approved by a human in
[`evals/golden_v1_review.csv`](evals/golden_v1_review.csv) (decision: approve / reject / edit),
merged back by the build script. Items marked rejected stay in the file for the record but are skipped by every eval command. Gold-section lists may be incomplete where several sections
answer a question; review should add any missing ones.

## Step 7: eval harness

Every golden question goes through the real pipeline (retrieve, then answer), and the output is scored automatically:

| Answer type | How it is scored |
|---|---|
| `exact` (factual) | Exact match: does an accepted value (`8000`, `/docs`, `HS256`) appear in the answer? Markdown is stripped; numbers match whole (`500` ≠ `1500`) |
| `open` | LLM-as-judge (`judge.py`): it writes a critique first, then gives a per-nugget verdict (support / partial / not_support) and a binary pass/fail. A missing vital nugget forces a fail even if the judge says pass |
| `abstain` | Passes only if the assistant refuses |

The judge is a different model from the answering model (`JUDGE_MODEL`, default `openai:gpt-6.1-sol`), which avoids self-preference bias. Refusals of answerable questions are counted as **false refusals** and are never sent to the judge.

Each run also records:
- retrieval at section level (hit@1, hit@k, recall@k, MRR);
- grounding (invalid citations, unsupported claims, ungrounded code);
- latency p50/p95;
- cost.

Every headline number carries a **95% bootstrap CI**.

```bash
python -m docs_assistant.evaluate --split dev --label baseline        # -> evals/runs/<stamp>_baseline/
python -m docs_assistant.evaluate --rejudge evals/runs/<run>           # new judge prompt, same answers
python -m docs_assistant.compare evals/runs/<A> evals/runs/<B>         # paired bootstrap, fixed/broken items
python -m docs_assistant.calibrate export evals/runs/<run>             # label judge outputs yourself
python -m docs_assistant.calibrate score                               # judge TPR / TNR vs. you
```

Each run folder holds four files:
- `config.json`: models, prompt hashes, golden-set hash, corpus and code commits;
- `results.jsonl`: one row per question;
- `summary.json`;
- `summary.md`.

The `test` split is sealed. It runs only with `--confirm-test` and is meant for the final number, not for tuning.

## Step 8: retrieval improvements, each one measured

The Step 7 baseline put the weak spot on paraphrased ("vocab-mismatch") questions: they passed 5 of 10. Reading the misses one by one showed two different problems:

- **Right page, wrong section** (most misses). Search found the "use case" intro of the lifespan page, while the code was in the next section. The model then refused because "the excerpt doesn't include the code".
- **True vocabulary gap.** "Redirect after a form" never mentions `RedirectResponse`, and "time every request" never mentions middleware.

Candidate fixes were tested **offline first** on the dev set, before spending anything on API calls:

| Strategy (top 5) | Gold section in the model's context | Paraphrased questions |
|---|---|---|
| Baseline (vectors) | 77.1% | 50% |
| BM25 keyword search | 57.1% | 40% |
| Hybrid BM25 + vectors (RRF) | 74.3% | 40% |
| Vectors + neighbor sections ±1 | **85.7%** | **60%** |

**Hybrid search, the textbook fix, was rejected.** It rescued none of the misses. Paraphrases fail because the user's words are *absent* from the docs, and keyword matching can't bridge that.

What was built (`retrieval.py`). Both changes were switched on as defaults after the dev eval below; set `RETRIEVAL_NEIGHBORS=0 QUERY_REWRITE=0` to reproduce the baseline:

| Change | What it does | Cost |
|---|---|---|
| **8a Neighbor expansion** (`--neighbors 1`) | Each search hit is sent with the sections just before and after it on the same page, in page order, under a 6,000-token cap | No extra calls; about 2x context tokens |
| **8b Query rewriting** (`--rewrite`) | An LLM rewrites the question into FastAPI's vocabulary (3 queries). Original + rewrites are searched and fused with reciprocal rank fusion. The original is always kept, and a failed rewrite falls back to it | 1 small LLM call (~$0.0002, ~1 s) |

Fusion is about ten lines written by hand rather than LangChain's `EnsembleRetriever`, because that drops the similarity scores the Step 4 traces log. Traces now also record which queries were searched and whether each chunk came from search or neighbor expansion. Retrieval metrics are split into two kinds:
- **Ranking** (hit@k, MRR): did *search* put the right section near the top?
- **Gold in context**: did the model *see* it, neighbors included?

```bash
python -m docs_assistant.retrieval_eval --split dev                      # cheap: retrieval only, 5 strategies
python -m docs_assistant.evaluate --split dev --label n1 --neighbors 1   # full eval of a candidate
python -m docs_assistant.compare evals/runs/<baseline> evals/runs/<n1>   # paired: real gain or noise?
```

A change ships only if `compare` shows the gain is real on dev. Then its setting becomes the default in `config.py`, its run becomes the gate baseline, and the sealed test split is run once for the headline number.

### What the measurements showed (dev, 41 items)

Retrieval-only sweep (`retrieval_eval`, about $0.01):

| Strategy | hit@1 | Gold in context | Paraphrased gold in context | Context tokens p50 |
|---|---|---|---|---|
| baseline | 51.4% | 77.1% | 50% | 1,088 |
| neighbors ±1 | 51.4% | 85.7% | 60% | 2,427 |
| neighbors ±2 | 51.4% | 91.4% | 70% | 3,282 |
| rewrite | 62.9% | 80.0% | 70% | 1,239 |
| **rewrite + neighbors ±1** | **62.9%** | **94.3%** | **80%** | 2,246 |

- **Rewriting alone is not enough.** It improves ranking (hit@1 +11 points) and rescues 4 misses, but loses 3 sections that plain search had found (the original query's results get diluted in fusion). Combined with neighbor expansion it comes out ahead on everything.
- **Neighbors ±2 is the cheaper-latency alternative** (91.4%, no extra LLM call) at about 45% more context tokens. Not shipped: it doesn't fix the true vocabulary gaps.

Full evals compared item by item against the baseline (`compare`):

| Change | Pass rate | Gold in context | Fixed | Broken |
|---|---|---|---|---|
| neighbors ±1 | +0.0 pts | +8.6 pts | 2 | 2 |
| **rewrite + neighbors ±1 (shipped)** | **+7.3 pts** (CI −7 to +22) | **+17.1 pts, significant** | 6 | 3 |

- **Shipped:** rewrite + neighbors ±1, now the default.
- **Getting the right section in front of the model** improved significantly.
- **The pass-rate gain is not yet significant** at 41 items; a larger golden set is needed to prove it.
- **All 3 regressions had the right section in context.** The model wrote incomplete answers instead (dropped a vital fact, or produced a mismatched URL in example code). That is a generation problem, and the prompt's "Be concise" rule is the next thing to test.
- **Trade-off:** cost per question doubles ($0.0005 → $0.0010) and p50 latency rises about 1.6 s, almost all from the rewrite call.

## Step 9: CI eval gate + results table

`.github/workflows/fastapi-docs-assistant.yml` (at the repo root) has two jobs:

| Job | When | What |
|---|---|---|
| `tests` | Every push / PR touching the project | 130+ offline tests. No API key, free |
| `eval-gate` | PRs from this repo, manual runs | Rebuild the index (cached; re-runs embed nothing), run the dev eval with the real models (~$0.20), then `python -m docs_assistant.gate`. The report lands on the run's summary page and the run folder is uploaded as an artifact |

The gate (`evals/gate.json`) blocks a merge on any of three kinds of check:

1. **Infrastructure:** errored items, or baseline items that were never evaluated. Missing results are failures, never silently dropped.
2. **Hard floors:** any invalid citation, any unanswerable trap that was answered, or an overall pass rate below 75%.
3. **Regression:** a **statistically significant** drop against the committed baseline (currently the shipped `rw-n1` dev run), on the same items (paired bootstrap on pass/fail and gold-in-context).

When an improvement ships, its dev run replaces `baseline_run` in `evals/gate.json` in the same reviewed commit.

**Why "significant drop" and not a fixed margin?** With about 40 dev items, ordinary run-to-run noise (a few answers flipping each way) gives a confidence interval roughly ±12 points wide. A tight margin would block good changes constantly. The gate catches real regressions (several items broken, few fixed) without crying wolf; a test proves both cases.

The results table below is generated, not typed:

```bash
python -m docs_assistant.results evals/runs/*_baseline evals/runs/*_n1 evals/runs/*_rw-n1 evals/runs/*_final --write-readme
```

## Ragas: second-opinion metrics and generated questions

[Ragas](https://docs.ragas.io) is the field's standard RAG-evaluation library. It's used here in two ways, alongside the project's own judge rather than instead of it.

```bash
python -m pip install -e ".[ragas]"
python -m docs_assistant.ragas_eval evals/runs/*_baseline evals/runs/*_rw-n1      # metrics on saved runs
python -m docs_assistant.ragas_testset generate --size 30                          # candidate questions
python -m docs_assistant.ragas_testset merge                                       # after review
```

**1. Metrics on saved runs** (`ragas_eval.py`). Nothing is re-answered: each run's saved answers are scored against exactly what the model was shown, rebuilt from the logged chunk IDs with neighbors included.

| Metric | Question | Scored on |
|---|---|---|
| Faithfulness | Is every claim in the answer supported by the retrieved text? | Answered items |
| Context recall | Does the retrieved text cover the reference answer? | Answerable items |
| Factual correctness (recall mode) | Does the answer contain the reference's claims? | Answered, answerable items |

- **Recall mode for factual correctness:** our reference answers are short, so F1 would penalize correct extra detail.
- **Judge:** the eval judge (`RAGAS_MODEL`), never the answering model.
- **Cost:** printed as an estimate before running. Real token usage is metered and saved as `ragas_meta.json` in the run folder.
- **Caching and failures:** scores are cached in `ragas.jsonl`. Failed or NaN scores are counted, never dropped.
- **Comparisons:** paired against the first run, with bootstrap CIs.

**2. Question generation** (`ragas_testset.py`) to grow the golden set:
1. A stratified sample of real chunks (300 across all pages) becomes a Ragas knowledge graph. The cheap model extracts summaries, entities and themes and links related sections.
2. The strong model writes single- and multi-section questions **in the voice of custom personas aimed at the known weak spot:** a Flask migrant using Flask terms, a junior developer who describes goals in everyday words, and a detail hunter. Default Ragas questions echo the docs' wording, which would miss the paraphrases that broke retrieval in Step 5.
3. Each question is converted to the golden schema:
   - gold sections are mapped back from the chunks it was written from;
   - evidence is a sentence copied verbatim from that chunk;
   - nuggets come from a small LLM call.

   Only items that pass `golden validate` are offered for review.
4. Nothing enters the golden set until it's approved in `evals/golden_ragas_review.csv`. `merge` writes approved items (with any in-sheet edits) to `evals/golden_ragas.jsonl`, rebalancing dev/test. `build_golden_v1.py` appends them as the categories `synthetic-single` / `synthetic-multi` (`source: ragas:*`), so they can be reported separately from hand-written items.

**Compatibility note:** a plain `pip install ragas` quietly breaks this project. It resolves an old `instructor` that downgrades `langchain-openai` and the `openai` SDK, and Ragas 0.4.3 imports a module that `langchain-community` 0.4.2 removed. The `[ragas]` extra pins around both. Ragas also guesses API parameters from model names and misses `gpt-6.1-sol`, so every call goes through a client wrapper that normalizes parameters for reasoning models and meters tokens.

## Results

<!-- results:start -->
| Run | Split | Retrieval | Pass rate [95% CI] | Δ vs first | Paraphrased pass | Gold in context | False refusals | $/query | p50 latency |
|---|---|---|---|---|---|---|---|---|---|
| baseline | dev | k=5 | 82.9% [71%–93%] | - | 5/10 | 77.1% [63%–91%] | 8.6% [0%–20%] | $0.00050 | 2,847 ms |
| n1 | dev | k=5, neighbors ±1 | 82.9% [71%–93%] | +0.0% | 5/10 | 85.7% [74%–97%] | 2.9% [0%–9%] | $0.00084 | 2,834 ms |
| rw-n1 | dev | k=5, neighbors ±1, query rewrite | 90.2% [80%–98%] | +7.3% | 8/10 | 94.3% [86%–100%] | 0.0% [0%–0%] | $0.00098 | 4,451 ms |
| final | test | k=5, neighbors ±1, query rewrite | 87.2% [74%–97%] | - | 6/10 | 91.4% [80%–100%] | 8.6% [0%–20%] | $0.00095 | 4,482 ms |

`*` = the 95% CI of the paired difference excludes zero.

_39 golden items in the last run are unreviewed drafts; numbers are provisional._
<!-- results:end -->

**Reading the test row.** The test split was never used for tuning; it ran once, with the shipped settings.
- **Pass rate:** 87.2% on test against 90.2% on dev. That is within each other's confidence intervals, which is the expected small drop on unseen questions.
- **Paraphrased questions:** the gain is smaller on unseen ones (6/10 against 8/10 on dev), and test false refusals are 8.6% against 0% on dev. Paraphrase handling is still the main weak spot.
- **No baseline run on test yet**, so the test split shows the shipped system's level, not its improvement. Running the baseline settings on test (`--neighbors 0 --no-rewrite`) would give a paired before/after on unseen questions.

## Known limitations

- **Golden set not yet human-reviewed.** All items are drafts, so every number above is provisional. Review happens in `evals/golden_v1_review.csv`.
- **Judge not yet calibrated.** Its agreement with human labels (TPR/TNR via `calibrate`) is unmeasured.
- **Small eval set** (about 40 items per split). Confidence intervals are about ±10 points, so only large effects reach significance.
- **Generation completeness.** Some answers drop a vital fact even when the right section was retrieved.
- **Latency.** Query rewriting adds about 1.6 s at p50. A faster rewrite model, or neighbors ±2 without rewriting, are the options.

## Roadmap

1. Corpus ingestion ✅
2. Chunking ✅, embeddings + vector store ✅
3. Baseline RAG pipeline ✅
4. Retrieval instrumentation: queries, chunks, similarity scores, whether the answer used them ✅
5. Failure analysis ✅ (labels pending human review)
6. Ground-truth set: 80 verified Q&A pairs targeting the weak spots ✅ (pending human review)
7. Eval harness: exact match for factual answers, LLM-as-judge for open-ended ones ✅
8. Improvements, measured against the baseline ✅ (shipped: rewrite + neighbors ±1)
9. CI eval gate + results table ✅

Next:
- Human review of the golden set.
- Judge calibration.
- Baseline run on the test split.
- Test the prompt's "Be concise" rule against answer completeness.
- Grow the golden set to 100+ with reviewed Ragas questions, so smaller effects can be proven.
- Ragas faithfulness on baseline vs shipped runs.
