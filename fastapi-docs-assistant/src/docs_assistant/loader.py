"""A LangChain document loader for FastAPI's MkDocs documentation.

No built-in loader handles this source: pages pull their code examples in from
docs_src/ via {* ... *} markers, which generic Markdown loaders leave as
literal text (or drop), losing every code example. See markdown_clean.py.

Usage:
    loader = FastAPIDocsLoader(repo_dir)
    docs = loader.load()          # list[Document]
    loader.excluded, loader.include_stats   # what was skipped and why
"""
import subprocess
from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from . import config
from .markdown_clean import IncludeStats, clean_markdown


def page_url(rel_path: Path) -> str:
    """MkDocs directory URLs: tutorial/body.md -> /tutorial/body/, x/index.md -> /x/."""
    parts = list(rel_path.with_suffix("").parts)
    if parts[-1] == "index":
        parts = parts[:-1]
    path = "/".join(parts)
    return f"{config.DOCS_BASE_URL}/{path}/" if path else f"{config.DOCS_BASE_URL}/"


def exclusion_reason(rel_path: Path) -> str | None:
    if rel_path.parts[0] in config.EXCLUDED_DIRS:
        return config.EXCLUDED_DIRS[rel_path.parts[0]]
    return config.EXCLUDED_PAGES.get(rel_path.as_posix())


def git_commit(repo_dir: Path) -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


class FastAPIDocsLoader(BaseLoader):
    def __init__(self, repo_dir: Path, commit: str | None = None):
        self.repo_dir = Path(repo_dir)
        self.docs_dir = self.repo_dir / "docs" / "en" / "docs"
        self.includes_base = self.repo_dir / "docs" / "en"     # include paths are relative to docs/en
        self.commit = commit
        self.excluded: list[dict] = []
        self.include_stats = IncludeStats()

    def lazy_load(self) -> Iterator[Document]:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"{self.docs_dir} not found. Run: python -m docs_assistant.fetch")
        commit = self.commit or git_commit(self.repo_dir)
        self.excluded, self.include_stats = [], IncludeStats()

        for path in sorted(self.docs_dir.rglob("*.md")):
            rel = path.relative_to(self.docs_dir)
            reason = exclusion_reason(rel)
            if reason:
                self.excluded.append({"page": rel.as_posix(), "reason": reason})
                continue

            text, headings = clean_markdown(path.read_text(encoding="utf-8"), self.includes_base,
                                            self.include_stats)
            if len(text) < config.MIN_DOC_CHARS:
                self.excluded.append({"page": rel.as_posix(),
                                      "reason": f"stub (<{config.MIN_DOC_CHARS} chars after cleaning)"})
                continue

            url = page_url(rel)
            h1 = next((h for h in headings if h["level"] == 1), None)
            doc_id = rel.with_suffix("").as_posix()              # stable and human-readable
            yield Document(
                id=doc_id,
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "title": h1["title"] if h1 else rel.stem.replace("-", " ").title(),
                    "section": rel.parts[0] if len(rel.parts) > 1 else "general",
                    "source_path": f"docs/en/docs/{rel.as_posix()}",
                    "url": url,
                    # Heading offsets let the chunker cut exactly at sections.
                    "headings": [{**h, "url": f"{url}#{h['anchor']}"} for h in headings],
                    "n_code_blocks": text.count("```") // 2,
                    "source_commit": commit,
                },
            )
