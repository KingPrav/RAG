"""Step 1: build the corpus.

    pinned FastAPI repo -> FastAPIDocsLoader -> LangChain Documents
    -> data/processed/documents.jsonl + manifest.json

Run:  python -m docs_assistant.fetch    (once)
      python -m docs_assistant.ingest
"""
import json
from collections import Counter
from datetime import datetime, timezone

from . import config
from .doc_io import save_documents
from .loader import FastAPIDocsLoader, git_commit


def main():
    commit = git_commit(config.REPO_DIR)
    if commit != config.FASTAPI_COMMIT:
        print(f"WARNING: repo is at {commit[:10]}, config pins {config.FASTAPI_COMMIT[:10]}")

    loader = FastAPIDocsLoader(config.REPO_DIR, commit=commit)
    docs = loader.load()
    stats = loader.include_stats
    save_documents(docs, config.PROCESSED_DIR / "documents.jsonl")

    total_chars = sum(len(d.page_content) for d in docs)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": commit,
        "documents": len(docs),
        "by_section": dict(Counter(d.metadata["section"] for d in docs).most_common()),
        "total_chars": total_chars,
        "approx_tokens": total_chars // 4,
        "headings": sum(len(d.metadata["headings"]) for d in docs),
        "code_blocks": sum(d.metadata["n_code_blocks"] for d in docs),
        "code_includes_resolved": stats.resolved,
        "code_includes_unresolved": stats.unresolved,
        "excluded": loader.excluded,
    }
    (config.PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(docs)} documents to data/processed/documents.jsonl")
    print(f"  by section:      {manifest['by_section']}")
    print(f"  size:            {total_chars:,} chars (~{manifest['approx_tokens']:,} tokens)")
    print(f"  headings:        {manifest['headings']}")
    print(f"  code blocks:     {manifest['code_blocks']} "
          f"({stats.resolved} includes resolved, {len(stats.unresolved)} unresolved)")
    print(f"  excluded pages:  {len(loader.excluded)} (reasons in manifest.json)")
    if stats.unresolved:
        print(f"  UNRESOLVED INCLUDES: {stats.unresolved[:5]}")


if __name__ == "__main__":
    main()
