from pathlib import Path

import pytest

from langchain_core.documents import Document

from docs_assistant.doc_io import load_documents, save_documents
from docs_assistant.loader import FastAPIDocsLoader, page_url
from docs_assistant.markdown_clean import IncludeStats, clean_markdown, select_lines

REPO = Path(__file__).parent / "fixtures" / "repo"
DOCS = REPO / "docs" / "en" / "docs"
INCLUDES_BASE = REPO / "docs" / "en"


@pytest.fixture(scope="module")
def body():
    stats = IncludeStats()
    text, headings = clean_markdown((DOCS / "tutorial" / "body.md").read_text(), INCLUDES_BASE, stats)
    return text, headings, stats


def test_code_includes_are_inlined(body):
    text, _, stats = body
    assert "```python\nfrom fastapi import FastAPI" in text          # {* ... *} full file
    assert "async def create_item(item: Item):" in text
    assert text.count("class Item(BaseModel):") == 3                 # ln-range + full + {!> !}
    assert stats.resolved == 3
    assert stats.unresolved == ["../../docs_src/body/missing.py"]    # reported, not silent
    assert "[missing code example: ../../docs_src/body/missing.py]" in text


def test_line_range_selection():
    code = "a\nb\nc\nd\ne\nf\ng"
    assert select_lines(code, "1:2,5:7") == "a\nb\n# ...\ne\nf\ng"
    assert select_lines(code, "3") == "c"


def test_line_range_include_only_keeps_selected_lines(body):
    text, _, _ = body
    section = text.split("## Create your data model")[1].split("## Full example")[0]
    assert "from fastapi import FastAPI\nfrom pydantic import BaseModel\n# ...\nclass Item" in section
    assert "app = FastAPI()" not in section


def test_markup_becomes_readable_text(body):
    text, _, _ = body
    assert "ORM (Object-Relational Mapper)-free" in text
    assert "Note: Technical Details" in text
    assert "Python 3.10+:" in text
    for junk in ("///", "<abbr", "![diagram]", "<div", "<font", "{ #"):
        assert junk not in text
    assert "$ fastapi dev main.py" in text                            # colour tags stripped
    assert "{%" not in text                                           # mkdocs-macros escaping
    # Runs of blank lines in prose collapse to one...
    assert "Use `POST` to send data.\n\n## Create your data model" in text
    # ...but code is untouched (PEP 8's two blank lines survive).
    assert "from pydantic import BaseModel\n\n\nclass Item" in text


def test_headings_skip_code_comments_and_offsets_are_exact(body):
    text, headings, _ = body
    assert [(h["level"], h["title"], h["anchor"]) for h in headings] == [
        (1, "Request Body", "request-body"),
        (2, "Create your data model", "create-your-data-model"),
        (2, "Full example", "full-example"),                         # slug fallback
    ]
    for h in headings:
        assert text[h["offset"]:].startswith("#" * h["level"] + " " + h["title"])


def test_page_urls():
    assert page_url(Path("tutorial/body.md")) == "https://fastapi.tiangolo.com/tutorial/body/"
    assert page_url(Path("advanced/index.md")) == "https://fastapi.tiangolo.com/advanced/"
    assert page_url(Path("index.md")) == "https://fastapi.tiangolo.com/"


def test_loader_yields_langchain_documents_and_explains_exclusions():
    loader = FastAPIDocsLoader(REPO, commit="abc123")
    docs = loader.load()
    assert [d.id for d in docs] == ["tutorial/body"]
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.page_content.startswith("# Request Body")
    assert doc.metadata["title"] == "Request Body"
    assert doc.metadata["section"] == "tutorial"
    assert doc.metadata["headings"][1]["url"] == "https://fastapi.tiangolo.com/tutorial/body/#create-your-data-model"
    assert doc.metadata["source_commit"] == "abc123"
    assert loader.include_stats.resolved == 3

    reasons = {e["page"]: e["reason"] for e in loader.excluded}
    assert set(reasons) == {"index.md", "reference/fastapi.md", "release-notes.md"}
    assert reasons["index.md"].startswith("stub")
    assert "docstrings" in reasons["reference/fastapi.md"]


def test_base64_blobs_in_code_are_replaced():
    blob = "iVBORw0KGgo" + "A" * 400
    raw = f'# T {{ #t }}\n\n```python\nimage = "{blob}"\nx = 1\n```'
    text, _ = clean_markdown(raw, INCLUDES_BASE, IncludeStats())
    assert blob not in text
    assert 'image = "<base64 data, 411 chars omitted>"' in text
    assert "x = 1" in text


def test_documents_round_trip_through_jsonl(tmp_path):
    docs = FastAPIDocsLoader(REPO, commit="abc123").load()
    save_documents(docs, tmp_path / "docs.jsonl")
    assert load_documents(tmp_path / "docs.jsonl") == docs
