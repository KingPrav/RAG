"""Save and load LangChain Documents as JSONL, so each pipeline stage can be
re-run on its own and its output inspected by hand."""
import json
from pathlib import Path

from langchain_core.documents import Document


def save_documents(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "page_content": d.page_content, "metadata": d.metadata}) + "\n")


def load_documents(path: Path) -> list[Document]:
    with path.open() as f:
        return [Document(**json.loads(line)) for line in f]
