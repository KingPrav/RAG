"""Chunking tests. A whitespace word counter stands in for the tokenizer so
the tests run offline and the arithmetic is easy to follow."""
from langchain_core.documents import BaseDocumentTransformer, Document

from docs_assistant.chunk import (Section, SectionChunker, merge_small, pack, split_blocks,
                                  split_sections)


def words(text: str) -> int:
    return len(text.split())


def make_doc(text: str) -> Document:
    """Build a Document the way FastAPIDocsLoader does (headings with offsets)."""
    headings, offset = [], 0
    for line in text.split("\n"):
        if line.startswith("#"):
            hashes, title = line.split(" ", 1)
            anchor = title.lower().replace(" ", "-")
            headings.append({"level": len(hashes), "title": title, "anchor": anchor, "offset": offset,
                             "url": f"https://x.dev/page/#{anchor}"})
        offset += len(line) + 1
    return Document(id="tutorial/page", page_content=text, metadata={
        "doc_id": "tutorial/page", "title": "Page", "section": "tutorial",
        "url": "https://x.dev/page/", "headings": headings, "source_commit": "abc"})


PROSE = " ".join(["word"] * 30)

DOC = make_doc(f"""# Page

{PROSE}

## Setup

Short intro.

### Install

{PROSE}

#### Detail stays inside its H3

{PROSE}

## Usage

{PROSE}""")


def chunker(**kw) -> SectionChunker:
    return SectionChunker(words, **({"split_level": 3, "min_tokens": 10, "max_tokens": 500} | kw))


# ---------------------------------------------------------------- sections

def test_sections_cut_at_h1_to_h3_with_breadcrumb_paths():
    sections = split_sections(DOC, split_level=3)
    assert [s.anchor for s in sections] == ["page", "setup", "install", "usage"]
    assert sections[2].path == ["Page", "Setup", "Install"]
    assert sections[3].path == ["Page", "Usage"]                       # H3 popped off the stack
    assert "#### Detail stays inside its H3" in sections[2].text


def test_tiny_section_merges_forward_and_keeps_its_anchor():
    merged = merge_small(split_sections(DOC, 3), words, min_tokens=10)
    assert [s.anchor for s in merged] == ["page", "setup", "usage"]
    setup = merged[1]
    assert setup.text.startswith("## Setup\n\nShort intro.\n\n### Install")
    assert setup.url.endswith("#setup")


def test_tiny_last_section_merges_backward():
    secs = [Section("a", "u#a", ["P"], PROSE), Section("b", "u#b", ["P"], "tiny")]
    merged = merge_small(secs, words, min_tokens=10)
    assert len(merged) == 1 and merged[0].text.endswith("tiny")


def test_heading_only_section_always_attaches_even_if_result_is_large():
    # Merging may exceed the max; pack() splits it later, with the heading
    # opening the first part instead of becoming a near-empty chunk.
    big = " ".join(["w"] * 1000)
    secs = [Section("a", "u#a", ["P"], "## WebSockets client"), Section("b", "u#b", ["P"], big)]
    merged = merge_small(secs, words, min_tokens=10)
    assert len(merged) == 1
    assert merged[0].anchor == "a" and merged[0].text.startswith("## WebSockets client")


# ---------------------------------------------------------------- splitting

def test_code_block_with_blank_lines_is_one_block():
    text = "Intro.\n\n```python\nimport x\n\n\ndef f():\n    pass\n```\n\nOutro."
    assert split_blocks(text) == ["Intro.", "```python\nimport x\n\n\ndef f():\n    pass\n```", "Outro."]


def test_pack_splits_on_block_boundaries_and_respects_max():
    para = " ".join(["w"] * 40)
    parts = pack("\n\n".join([para] * 5), words, max_tokens=100)
    assert len(parts) == 3
    assert all(words(p) <= 100 for p in parts)


def test_pack_balances_parts_instead_of_leaving_a_tiny_tail():
    paras = [" ".join(["w"] * 10)] * 11                                # 110 words
    parts = pack("\n\n".join(paras), words, max_tokens=100)
    assert [words(p) for p in parts] == [60, 50]                       # greedy would give 100 + 10


def test_oversized_code_block_is_split_and_refenced():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(60)) + "\n```"
    parts = pack(code, words, max_tokens=50)
    assert len(parts) > 1
    for p in parts:
        assert p.startswith("```python\n") and p.endswith("\n```")      # every piece is valid Markdown
        assert words(p) <= 50


# ---------------------------------------------------------------- transformer

def test_chunker_is_a_langchain_transformer_producing_store_ready_documents():
    c = chunker()
    assert isinstance(c, BaseDocumentTransformer)
    chunks = c.split_documents([DOC])
    assert [d.id for d in chunks] == ["tutorial/page#page", "tutorial/page#setup", "tutorial/page#usage"]
    usage = chunks[2]
    assert usage.page_content.startswith("FastAPI docs > Tutorial > Page > Usage\n\n## Usage")
    assert usage.metadata["url"] == "https://x.dev/page/#usage"
    assert usage.metadata["n_tokens"] == words(usage.page_content)
    # Vector stores only accept flat metadata values.
    for d in chunks:
        assert all(isinstance(v, (str, int, float, bool)) for v in d.metadata.values())


def test_split_section_gets_numbered_ids_and_fits_with_breadcrumb():
    long_doc = make_doc("# Page\n\n" + "\n\n".join([" ".join(["w"] * 40)] * 6))
    chunks = chunker(max_tokens=100).split_documents([long_doc])
    assert [d.id for d in chunks] == [f"tutorial/page#page/{i}" for i in range(1, len(chunks) + 1)]
    assert all(d.metadata["n_tokens"] <= 100 for d in chunks)          # breadcrumb counted too
    assert all(d.metadata["n_parts"] == len(chunks) for d in chunks)


def test_code_indentation_survives_chunking():
    doc = make_doc("# Page\n\n" + PROSE + "\n\n```python\nclass Item(BaseModel):\n    name: str\n\n\n"
                   "def f():\n    return 1\n```")
    content = chunker().split_documents([doc])[0].page_content
    assert "class Item(BaseModel):\n    name: str\n\n\ndef f():\n    return 1" in content


def test_langchain_markdown_splitter_breaks_code():
    """Regression evidence for not using MarkdownHeaderTextSplitter: it strips
    indentation inside code blocks. If a future LangChain version fixes this,
    this test fails and we can reconsider."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    md = "## Model\n\n```python\nclass Item(BaseModel):\n    name: str\n```"
    out = MarkdownHeaderTextSplitter([("##", "h2")], strip_headers=False).split_text(md)[0].page_content
    assert "    name: str" not in out
    assert "\nname: str" in out
