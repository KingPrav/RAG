# ClaimCheck: evidence-grounded supplement claim checker

Paste a gym claim ("creatine damages your kidneys") and get a verdict
(supported / mixed / not supported) backed by cited research.
Scope: protein, creatine, caffeine. Corpus: PubMed abstracts plus the ISSN
position stands.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then set NCBI_EMAIL (and optionally NCBI_API_KEY)
pytest -q
```

## Step 1: build the corpus

```bash
python -m claimcheck.ingest
```

Outputs:
- `data/raw/pubmed/<topic>/` cached search results and raw XML (re-runs are free)
- `data/processed/documents.jsonl` one cleaned record per paper
- `data/processed/manifest.json` queries, counts and drop reasons for this build

## Roadmap

1. Corpus ingestion ✅
2. Chunking + embeddings + vector store
3. Baseline RAG pipeline
4. Retrieval instrumentation (queries, chunks, scores, citation usage)
5. Failure analysis
6. Ground-truth set from the position stands (50–100 claims)
7. Eval harness: exact match on verdicts, LLM-as-judge on explanations
8. Improvements, measured against the baseline
9. CI eval gate + results table
