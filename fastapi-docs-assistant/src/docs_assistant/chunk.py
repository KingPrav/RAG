"""Step 2a: split documents into retrievable chunks.

`SectionChunker` is a LangChain BaseDocumentTransformer, so it is used like
any LangChain splitter:  chunks = SectionChunker(...).split_documents(docs)

Why not LangChain's MarkdownHeaderTextSplitter? It strips leading whitespace
from every line, including inside code blocks, which destroys Python
indentation (`    return item` -> `return item`), and it drops the blank lines
between paragraphs. On a docs corpus that is mostly code, that corrupts the
most valuable content (see tests/test_chunk.py::test_langchain_markdown_splitter_breaks_code).

Strategy (structure-aware, token-bounded):
  1. Cut each page at its H1/H2/H3 headings, so a chunk is a citable section.
  2. Merge sections smaller than min_tokens into the next one.
  3. Split sections larger than max_tokens at paragraph boundaries into
     balanced parts. Code blocks are atomic; one that alone exceeds the limit
     is split by lines and each piece is re-fenced so it stays valid Markdown.
  4. Prefix every chunk with its breadcrumb ("FastAPI docs > Tutorial >
     Request Body > Create your data model"), so it makes sense on its own.

Run:  python -m docs_assistant.chunk
"""
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document

from . import config
from .doc_io import load_documents, save_documents

FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
SAFETY_MARGIN = 8          # token counts are not exactly additive when joining strings
TokenCounter = Callable[[str], int]


@dataclass
class Section:
    anchor: str
    url: str
    path: list[str]          # heading titles from the page title down
    text: str


# ---------------------------------------------------------------- sections

def split_sections(doc: Document, split_level: int) -> list[Section]:
    meta, text = doc.metadata, doc.page_content
    headings = [h for h in meta["headings"] if h["level"] <= split_level]
    sections, stack = [], []

    if not headings or headings[0]["offset"] > 0:              # text before the first heading
        end = headings[0]["offset"] if headings else len(text)
        if text[:end].strip():
            sections.append(Section("", meta["url"], [meta["title"]], text[:end].strip()))

    for i, h in enumerate(headings):
        stack = [s for s in stack if s["level"] < h["level"]] + [h]
        end = headings[i + 1]["offset"] if i + 1 < len(headings) else len(text)
        path = [s["title"] for s in stack]
        if path[0] != meta["title"]:
            path = [meta["title"], *path]
        sections.append(Section(h["anchor"], h["url"], path, text[h["offset"]:end].strip()))
    return sections


def merge_small(sections: list[Section], count: TokenCounter, min_tokens: int) -> list[Section]:
    """A tiny section (often a heading plus one line, e.g. a parent H2 whose
    content lives in its H3s) is always merged forward, even if that makes the
    result too big: pack() splits it afterwards, and the heading then opens the
    first part instead of becoming a near-empty chunk of its own. The merged
    chunk keeps the first section's anchor, since that is where it starts."""
    def join(a: Section, b: Section) -> Section:
        return Section(a.anchor, a.url, a.path, f"{a.text}\n\n{b.text}")

    out, pending = [], None
    for sec in sections:
        if pending is not None:
            sec, pending = join(pending, sec), None
        if count(sec.text) < min_tokens:
            pending = sec
        else:
            out.append(sec)
    if pending is not None:                                     # tiny last section: merge backward
        if out:
            out[-1] = join(out[-1], pending)
        else:
            out.append(pending)
    return out


# ---------------------------------------------------------------- splitting

def split_blocks(text: str) -> list[str]:
    """Paragraphs separated by blank lines; fenced code blocks kept whole even
    though they contain blank lines."""
    blocks, current, fence = [], [], None
    for line in text.split("\n"):
        match = FENCE.match(line)
        if fence:
            current.append(line)
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
                blocks.append("\n".join(current))
                current, fence = [], None
        elif match:
            if current:
                blocks.append("\n".join(current))
            current, fence = [line], match.group(1)
        elif not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def split_oversized_block(block: str, count: TokenCounter, max_tokens: int) -> list[str]:
    lines = block.split("\n")
    is_code = bool(FENCE.match(lines[0])) and len(lines) > 1
    opener, closer = (lines[0], lines[-1]) if is_code else ("", "")
    body = lines[1:-1] if is_code else lines
    budget = max_tokens - (count(opener + "\n" + closer) if is_code else 0)

    pieces, current = [], []
    for line in body:
        if current and count("\n".join(current + [line])) > budget:
            pieces.append(current)
            current = []
        current.append(line)
    if current:
        pieces.append(current)
    if is_code:
        return ["\n".join([opener, *p, closer]) for p in pieces]
    return ["\n".join(p) for p in pieces]


def pack(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Pack whole blocks into balanced parts of at most max_tokens.

    Parts aim for total/n tokens each rather than filling to the max, which
    would leave a tiny, context-free tail (e.g. 480 + 480 + 30)."""
    total = count(text)
    if total <= max_tokens:
        return [text]
    blocks = []
    for block in split_blocks(text):
        blocks.extend([block] if count(block) <= max_tokens else split_oversized_block(block, count, max_tokens))

    target = total / -(-total // max_tokens)          # total / ceil(total / max)
    parts, current = [], []
    for block in blocks:
        if current and (count("\n\n".join(current + [block])) > max_tokens
                        or count("\n\n".join(current)) >= target):
            parts.append("\n\n".join(current))
            current = []
        current.append(block)
    if current:
        parts.append("\n\n".join(current))
    return parts


# ---------------------------------------------------------------- transformer

def breadcrumb(section_area: str, path: list[str]) -> str:
    area = section_area.replace("-", " ").title()
    return " > ".join(["FastAPI docs"] + ([area] if area != "General" else []) + path)


class SectionChunker(BaseDocumentTransformer):
    """Structure-aware, token-bounded chunker for documents from FastAPIDocsLoader.

    Output Documents are ready for a LangChain vector store:
      - page_content: breadcrumb + section text (what gets embedded and shown to the LLM)
      - id: stable chunk id, e.g. "tutorial/body#create-your-data-model" (".../2" for parts),
        so re-indexing upserts instead of duplicating
      - metadata: flat str/int/bool values only, which every vector store accepts
    """

    def __init__(self, count_tokens: TokenCounter, *, split_level: int = config.CHUNK_SPLIT_LEVEL,
                 min_tokens: int = config.CHUNK_MIN_TOKENS, max_tokens: int = config.CHUNK_MAX_TOKENS):
        self.count = count_tokens
        self.split_level, self.min_tokens, self.max_tokens = split_level, min_tokens, max_tokens

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        return list(self.transform_documents(documents))

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        chunks = [c for doc in documents for c in self._chunk(doc)]
        duplicates = [i for i, n in Counter(c.id for c in chunks).items() if n > 1]
        if duplicates:
            raise ValueError(f"duplicate chunk ids: {duplicates[:5]}")
        return chunks

    def _chunk(self, doc: Document) -> list[Document]:
        meta = doc.metadata
        sections = merge_small(split_sections(doc, self.split_level), self.count, self.min_tokens)
        chunks = []
        for sec in sections:
            context = breadcrumb(meta["section"], sec.path)
            # The breadcrumb is prepended to every part, so the text budget shrinks by its size.
            parts = pack(sec.text, self.count, self.max_tokens - self.count(context) - SAFETY_MARGIN)
            for i, part in enumerate(parts, start=1):
                chunk_id = f"{meta['doc_id']}#{sec.anchor or 'top'}" + (f"/{i}" if len(parts) > 1 else "")
                content = f"{context}\n\n{part}"
                chunks.append(Document(
                    id=chunk_id,
                    page_content=content,
                    metadata={
                        "chunk_id": chunk_id,
                        "doc_id": meta["doc_id"],
                        "part": i,
                        "n_parts": len(parts),
                        "title": meta["title"],
                        "section": meta["section"],
                        "breadcrumb": context,
                        "url": sec.url,
                        "n_tokens": self.count(content),
                        "has_code": "```" in part,
                        "source_commit": meta["source_commit"],
                    },
                ))
        return chunks


# ---------------------------------------------------------------- CLI

def percentile(values: list[int], p: float) -> int:
    values = sorted(values)
    return values[round(p * (len(values) - 1))]


def summarize(chunks: list[Document], n_docs: int, max_tokens: int) -> dict:
    sizes = [c.metadata["n_tokens"] for c in chunks]
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {"split_level": config.CHUNK_SPLIT_LEVEL, "min_tokens": config.CHUNK_MIN_TOKENS,
                     "max_tokens": max_tokens, "tokenizer": config.TOKENIZER_ENCODING},
        "source_commit": chunks[0].metadata["source_commit"] if chunks else None,
        "documents": n_docs,
        "chunks": len(chunks),
        "tokens_total": sum(sizes),
        "tokens_per_chunk": {"min": min(sizes), "p10": percentile(sizes, .1), "p50": percentile(sizes, .5),
                             "p90": percentile(sizes, .9), "max": max(sizes)},
        "chunks_over_max": sum(s > max_tokens for s in sizes),
        "sections_split": len({c.id.rsplit("/", 1)[0] for c in chunks if c.metadata["n_parts"] > 1}),
        "chunks_with_code": sum(c.metadata["has_code"] for c in chunks),
        "by_section": dict(Counter(c.metadata["section"] for c in chunks).most_common()),
    }


def main():
    from .tokens import count_tokens

    src = config.PROCESSED_DIR / "documents.jsonl"
    if not src.exists():
        raise SystemExit("documents.jsonl not found. Run: python -m docs_assistant.ingest")
    docs = load_documents(src)

    chunks = SectionChunker(count_tokens).split_documents(docs)
    save_documents(chunks, config.PROCESSED_DIR / "chunks.jsonl")
    manifest = summarize(chunks, len(docs), config.CHUNK_MAX_TOKENS)
    (config.PROCESSED_DIR / "chunks_manifest.json").write_text(json.dumps(manifest, indent=2))

    t = manifest["tokens_per_chunk"]
    print(f"Wrote {manifest['chunks']} chunks from {manifest['documents']} documents to data/processed/chunks.jsonl")
    print(f"  tokens/chunk:   p10 {t['p10']} | median {t['p50']} | p90 {t['p90']} | max {t['max']}")
    print(f"  total tokens:   {manifest['tokens_total']:,} (what embedding will cost)")
    print(f"  over max:       {manifest['chunks_over_max']}")
    print(f"  sections split: {manifest['sections_split']}")
    print(f"  with code:      {manifest['chunks_with_code']}")


if __name__ == "__main__":
    main()
