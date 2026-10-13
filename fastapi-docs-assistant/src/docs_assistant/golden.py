"""Step 6: the golden (ground-truth) eval set: schema, loading and validation.

Every item is a *verified hypothesis* about correct behavior:
  - answer_type "exact": a short factual answer, scored by exact match against
    `exact_accept` (any accepted string, case-insensitive, must appear in the answer)
  - answer_type "open": scored by an LLM judge against `nuggets` (atomic facts;
    vital ones must be present)
  - answer_type "abstain": the docs do not answer it; the correct behavior is to refuse

Evidence is a verbatim quote from a named section, so the validator can prove
that the expected answer is actually in the corpus (the Step 5 "422" lesson).
Gold sections are `page#anchor`, independent of chunking, so they survive Step 8.

Run:  python -m docs_assistant.golden validate        (exit code 1 on any error)
      python -m docs_assistant.golden summary
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from . import config

GOLDEN_PATH = config.ROOT / "evals" / "golden_v1.jsonl"
CATEGORIES = ("vocab-mismatch", "ranking", "exact-fact", "exact-term", "code", "multi-section",
              "unanswerable-oos", "unanswerable-trap")
ID_PATTERN = re.compile(r"^[a-z]+-\d{3}$")


class Nugget(BaseModel):
    fact: str = Field(min_length=3)
    vital: bool = True


class Evidence(BaseModel):
    section: str
    quote: str = Field(min_length=8)


class Review(BaseModel):
    status: Literal["draft", "approved", "rejected"] = "draft"
    reviewer: str = ""
    note: str = ""


class GoldenItem(BaseModel):
    id: str
    split: Literal["dev", "test"]
    category: Literal[CATEGORIES]  # type: ignore[valid-type]
    answer_type: Literal["exact", "open", "abstain"]
    question: str = Field(min_length=8)
    expected_answer: str = Field(min_length=2)
    exact_accept: list[str] = []
    nuggets: list[Nugget] = []
    gold_sections: list[str] = []
    evidence: list[Evidence] = []
    abstain_reason: str = ""
    corpus_commit: str
    source: str = "authored"
    review: Review = Review()


def normalize(text: str) -> str:
    """Whitespace-insensitive comparison; quotes must otherwise match exactly."""
    return re.sub(r"\s+", " ", text).strip()


def load_sections(chunks_path: Path = config.PROCESSED_DIR / "chunks.jsonl") -> tuple[dict[str, str], str | None]:
    """Section text keyed by `page#anchor`, rebuilt from chunks (parts joined,
    breadcrumb line dropped), plus the corpus commit the chunks came from."""
    parts: dict[str, list[str]] = {}
    commit = None
    with chunks_path.open() as f:
        for line in f:
            chunk = json.loads(line)
            commit = chunk["metadata"].get("source_commit", commit)
            section = re.sub(r"/\d+$", "", chunk["id"])
            body = chunk["page_content"].split("\n\n", 1)[-1]
            parts.setdefault(section, []).append(body)
    return {k: "\n\n".join(v) for k, v in parts.items()}, commit


def load_items(path: Path = GOLDEN_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(raw_items: list[dict], sections: dict[str, str], corpus_commit: str | None) -> tuple[list[GoldenItem], list[str]]:
    errors: list[str] = []
    items: list[GoldenItem] = []
    normalized = {k: normalize(v) for k, v in sections.items()}

    for n, raw in enumerate(raw_items, start=1):
        label = raw.get("id", f"line {n}")
        try:
            item = GoldenItem(**raw)
        except ValidationError as e:
            errors.append(f"{label}: schema: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")
            continue
        items.append(item)
        err = lambda msg: errors.append(f"{item.id}: {msg}")

        if not ID_PATTERN.match(item.id):
            err("id must look like 'golden-001'")
        if corpus_commit and item.corpus_commit != corpus_commit:
            err(f"corpus_commit {item.corpus_commit[:10]} != indexed corpus {corpus_commit[:10]}")

        if item.answer_type == "abstain":
            if item.gold_sections or item.evidence:
                err("abstain items must not have gold_sections or evidence")
            if not item.abstain_reason:
                err("abstain items need abstain_reason (why the docs don't answer it)")
            if not item.category.startswith("unanswerable"):
                err("abstain items must use an unanswerable-* category")
            continue

        if item.category.startswith("unanswerable"):
            err("unanswerable-* categories must have answer_type 'abstain'")
        if not item.gold_sections:
            err("answerable items need at least one gold section")
        if not item.evidence:
            err("answerable items need at least one evidence quote")
        for section in item.gold_sections:
            if section not in sections:
                err(f"gold section not in corpus: {section}")
        for ev in item.evidence:
            if ev.section not in item.gold_sections:
                err(f"evidence section {ev.section} is not one of the gold sections")
            elif ev.section in normalized and normalize(ev.quote) not in normalized[ev.section]:
                err(f"quote not found verbatim in {ev.section}: {ev.quote[:60]!r}")

        if item.answer_type == "exact":
            if not item.exact_accept:
                err("exact items need exact_accept")
            quotes = " ".join(normalize(ev.quote).lower() for ev in item.evidence)
            if item.exact_accept and not any(a.lower() in quotes for a in item.exact_accept):
                err(f"none of exact_accept {item.exact_accept} appears in the evidence quotes")
        if item.answer_type == "open" and not any(nug.vital for nug in item.nuggets):
            err("open items need at least one vital nugget")

    ids = Counter(i.id for i in items)
    errors += [f"{i}: duplicate id" for i, c in ids.items() if c > 1]
    questions = Counter(normalize(i.question).lower() for i in items)
    errors += [f"duplicate question: {q[:60]!r}" for q, c in questions.items() if c > 1]
    by_cat = Counter((i.category, i.split) for i in items)
    for cat in {i.category for i in items}:
        if abs(by_cat[(cat, "dev")] - by_cat[(cat, "test")]) > 1:
            errors.append(f"split imbalance in {cat}: dev {by_cat[(cat, 'dev')]} vs test {by_cat[(cat, 'test')]}")
    return items, errors


def summary(items: list[GoldenItem]) -> str:
    lines = [f"{len(items)} items   splits {dict(Counter(i.split for i in items))}   "
             f"review {dict(Counter(i.review.status for i in items))}",
             f"{'category':<20}{'dev':>5}{'test':>6}   answer types"]
    for cat in CATEGORIES:
        rows = [i for i in items if i.category == cat]
        if rows:
            dev = sum(i.split == "dev" for i in rows)
            lines.append(f"{cat:<20}{dev:>5}{len(rows) - dev:>6}   {dict(Counter(i.answer_type for i in rows))}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate or summarize the golden eval set.")
    parser.add_argument("command", choices=["validate", "summary"])
    parser.add_argument("--path", type=Path, default=GOLDEN_PATH)
    args = parser.parse_args()

    sections, commit = load_sections()
    items, errors = validate(load_items(args.path), sections, commit)
    print(summary(items))
    if args.command == "validate":
        if errors:
            print(f"\n{len(errors)} problem(s):")
            for e in errors:
                print(f"  - {e}")
            raise SystemExit(1)
        print("\nOK: every item is well-formed, every gold section exists, every quote is verbatim.")


if __name__ == "__main__":
    main()
