"""Grow the golden set with Ragas-generated questions, as drafts for human review.

    python -m docs_assistant.ragas_testset generate --size 30      # -> evals/golden_ragas_candidates.jsonl + review CSV
    (review evals/golden_ragas_review.csv: decision = approve / reject; you may edit question,
     expected_answer and nuggets in place)
    python -m docs_assistant.ragas_testset merge                   # approved -> evals/golden_ragas.jsonl
    python evals/build_golden_v1.py && python -m docs_assistant.golden validate

How it works:
1. A sample of real chunks (stratified across pages) becomes a Ragas knowledge
   graph: a cheap model extracts summaries, entities and themes and links related
   sections (many small calls).
2. A strong model writes single-section and multi-section questions from it,
   in the voice of personas chosen to target this system's known weak spot:
   developers who describe goals in their own words rather than FastAPI's.
   Ragas' default personas tend to echo the docs' wording, which would miss
   exactly the paraphrased questions that broke retrieval in Step 5.
3. Each question is converted to the golden schema: gold sections are mapped
   back from the chunks it was written from, evidence is a sentence copied
   verbatim from that chunk, and nuggets are extracted by a small LLM call.
   Only items that pass `golden validate` are offered for review.

Nothing enters the golden set until a human approves it.
"""
import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from . import config
from .doc_io import load_documents
from .eval_metrics import section_of
from .golden import GOLDEN_PATH, load_items, load_sections, validate

CANDIDATES_PATH = config.ROOT / "evals" / "golden_ragas_candidates.jsonl"
REVIEW_PATH = config.ROOT / "evals" / "golden_ragas_review.csv"
APPROVED_PATH = config.ROOT / "evals" / "golden_ragas.jsonl"
REVIEW_FIELDS = ["id", "split", "category", "question", "expected_answer", "nuggets", "gold_sections",
                 "synthesizer", "persona", "decision", "reviewer", "note"]

PERSONAS = [
    ("Flask migrant", "A backend developer moving a Flask app to FastAPI. Describes what they want in Flask "
                      "terms (blueprints, before_request, flash messages, app factories) and doesn't know "
                      "FastAPI's names for things."),
    ("Goal-first junior", "A junior developer who knows the behavior they want ('only logged-in users', "
                          "'send the user somewhere else', 'run something when the server starts') but not "
                          "the framework vocabulary, so they describe goals in everyday words."),
    ("Detail hunter", "An experienced FastAPI developer who wants one precise fact: a default value, a "
                      "parameter name, a status code or the exact behavior of an option."),
]


# ---------------------------------------------------------------- 1. sampling

def sample_chunks(chunks: list, n: int, seed: int = 0, min_tokens: int = 120, max_tokens: int = 500) -> list:
    """Round-robin across pages so every part of the docs is represented, using
    chunks long enough for Ragas' summary extraction (>100 tokens)."""
    by_doc: dict[str, list] = defaultdict(list)
    for c in chunks:
        if min_tokens <= c.metadata.get("n_tokens", 0) <= max_tokens:
            by_doc[c.metadata["doc_id"]].append(c)
    rng = random.Random(seed)
    pools = [rng.sample(v, len(v)) for _, v in sorted(by_doc.items())]
    rng.shuffle(pools)
    picked = []
    while len(picked) < n and any(pools):
        for pool in pools:
            if pool and len(picked) < n:
                picked.append(pool.pop())
    return picked


# ---------------------------------------------------------------- 3. conversion

def body_of(chunk_text: str) -> str:
    return chunk_text.split("\n\n", 1)[-1]                 # drop the breadcrumb line


def best_evidence(context: str, reference: str, min_chars: int = 20) -> str | None:
    """The sentence or line of the chunk that shares the most words with the
    reference answer, copied verbatim (so the validator can prove it)."""
    ref = set(re.findall(r"[a-z0-9_]+", reference.lower()))
    candidates = []
    for line in body_of(context).splitlines():
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            sentence = sentence.strip()
            if len(sentence) >= min_chars and not sentence.startswith("```"):
                words = set(re.findall(r"[a-z0-9_]+", sentence.lower()))
                candidates.append((len(words & ref), len(sentence), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2][:400]


class NuggetOut(BaseModel):
    fact: str = Field(description="One atomic fact a correct answer must state.")
    vital: bool = Field(description="True if an answer missing this fact is wrong.")


class NuggetList(BaseModel):
    nuggets: list[NuggetOut] = Field(description="1-3 atomic facts, most important first.")


NUGGET_PROMPT = ("Split this reference answer to a FastAPI question into 1-3 atomic facts that a correct "
                 "answer must contain. Mark a fact vital if an answer without it would be wrong. At least "
                 "one fact must be vital.\n\nQuestion: {question}\nReference answer: {reference}")


def extract_nuggets(llm, question: str, reference: str) -> list[dict]:
    """LLM nugget extraction; falls back to the whole reference as one vital nugget."""
    try:
        out = llm.with_structured_output(NuggetList).invoke(NUGGET_PROMPT.format(question=question,
                                                                                   reference=reference))
        nuggets = [{"fact": n.fact.strip(), "vital": n.vital} for n in out.nuggets if len(n.fact.strip()) >= 3]
    except Exception:
        nuggets = []
    if not nuggets:
        nuggets = [{"fact": reference.strip()[:300], "vital": True}]
    if not any(n["vital"] for n in nuggets):
        nuggets[0]["vital"] = True
    return nuggets[:3]


def to_golden(samples: list[dict], chunk_ids_by_text: dict[str, str], commit: str, nugget_fn,
              existing_questions: set[str], start: int = 1) -> tuple[list[dict], list[str]]:
    """Ragas samples -> golden-schema drafts. Returns (items, reasons for dropped samples)."""
    items, dropped = [], []
    counters: dict[str, int] = defaultdict(int)
    seen = {q.strip().lower() for q in existing_questions}
    for s in samples:
        question, reference = (s.get("user_input") or "").strip(), (s.get("reference") or "").strip()
        if len(question) < 8 or len(reference) < 2:
            dropped.append(f"empty question/reference: {question[:60]!r}")
            continue
        if question.lower() in seen:
            dropped.append(f"duplicate question: {question[:60]!r}")
            continue
        contexts = s.get("reference_contexts") or []
        chunk_ids = [chunk_ids_by_text[c] for c in contexts if c in chunk_ids_by_text]
        if not chunk_ids:
            dropped.append(f"source text not traceable to a chunk: {question[:60]!r}")
            continue
        sections = list(dict.fromkeys(section_of(c) for c in chunk_ids))
        evidence = []
        for cid in chunk_ids:
            quote = best_evidence(next(c for c in contexts if chunk_ids_by_text.get(c) == cid), reference)
            if quote and section_of(cid) not in {e["section"] for e in evidence}:
                evidence.append({"section": section_of(cid), "quote": quote})
        if not evidence:
            dropped.append(f"no quotable evidence: {question[:60]!r}")
            continue
        multi = "multi" in (s.get("synthesizer_name") or "") or len(sections) > 1
        category = "synthetic-multi" if multi else "synthetic-single"
        counters[category] += 1
        seen.add(question.lower())
        items.append({
            "id": f"ragas-{start + len(items):03d}",
            "split": "dev" if counters[category] % 2 else "test",
            "category": category, "answer_type": "open", "question": question,
            "expected_answer": reference, "exact_accept": [], "nuggets": nugget_fn(question, reference),
            "gold_sections": sections, "evidence": evidence, "abstain_reason": "", "corpus_commit": commit,
            "source": f"ragas:{s.get('synthesizer_name', '')}",
            "review": {"status": "draft", "reviewer": "", "note": f"persona: {s.get('persona_name', '')}; "
                                                                   f"style: {s.get('query_style', '')}"},
        })
    return items, dropped


def format_nuggets(nuggets: list[dict]) -> str:
    return " | ".join(f"{'VITAL' if n['vital'] else 'okay'}: {n['fact']}" for n in nuggets)


def parse_nuggets(text: str) -> list[dict]:
    out = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        label, _, fact = part.partition(":")
        if fact and label.strip().lower() in ("vital", "okay"):
            out.append({"fact": fact.strip(), "vital": label.strip().lower() == "vital"})
        else:
            out.append({"fact": part, "vital": True})
    return out


def write_review(items: list[dict], path: Path = REVIEW_PATH) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for it in items:
            w.writerow({"id": it["id"], "split": it["split"], "category": it["category"],
                        "question": it["question"], "expected_answer": it["expected_answer"],
                        "nuggets": format_nuggets(it["nuggets"]), "gold_sections": "; ".join(it["gold_sections"]),
                        "synthesizer": it["source"].removeprefix("ragas:"),
                        "persona": it["review"]["note"].split(";")[0].removeprefix("persona: "),
                        "decision": "", "reviewer": "", "note": ""})


def merge(candidates: list[dict], review_rows: list[dict]) -> tuple[list[dict], dict]:
    """Approved candidates (with any in-sheet edits) become golden items."""
    by_id = {c["id"]: c for c in candidates}
    approved, counts = [], defaultdict(int)
    for row in review_rows:
        decision = row.get("decision", "").strip().lower()
        counts[decision or "undecided"] += 1
        if decision not in ("approve", "approved") or row["id"] not in by_id:
            continue
        item = json.loads(json.dumps(by_id[row["id"]]))
        item["question"] = row.get("question", "").strip() or item["question"]
        item["expected_answer"] = row.get("expected_answer", "").strip() or item["expected_answer"]
        if row.get("nuggets", "").strip():
            item["nuggets"] = parse_nuggets(row["nuggets"])
        item["review"] = {"status": "approved", "reviewer": row.get("reviewer", "").strip(),
                          "note": row.get("note", "").strip() or item["review"]["note"]}
        approved.append(item)
    # Rebalance dev/test within each category: rejections can leave one split
    # heavier, and `golden validate` requires balance.
    seen: dict[str, int] = defaultdict(int)
    for item in approved:
        seen[item["category"]] += 1
        item["split"] = "dev" if seen[item["category"]] % 2 else "test"
    return approved, dict(counts)


# ---------------------------------------------------------------- 2. generation (calls OpenAI)

def generate(size: int, n_chunks: int, seed: int) -> None:
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator
    from ragas.testset.persona import Persona

    chunks = load_documents(config.CHUNKS_PATH)
    picked = sample_chunks(chunks, n_chunks, seed)
    docs = [c.__class__(page_content=c.page_content, metadata={"chunk_id": c.metadata["chunk_id"],
                                                               "filename": c.metadata["doc_id"]}) for c in picked]
    usage = UsageMetadataCallbackHandler()
    kg_llm = ChatOpenAI(model=config.RAGAS_KG_MODEL, max_retries=4, timeout=120, callbacks=[usage])
    q_llm = ChatOpenAI(model=config.RAGAS_QUESTION_MODEL, max_retries=4, timeout=180, callbacks=[usage])
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=config.EMBEDDING_MODEL))
    # bypass_*: current OpenAI reasoning models reject temperature and n.
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(q_llm, bypass_temperature=True, bypass_n=True),
        embedding_model=embeddings,
        persona_list=[Persona(name=n, role_description=d) for n, d in PERSONAS],
        llm_context="Questions are asked by developers using the FastAPI Python web framework.",
    )
    print(f"Building a knowledge graph from {len(docs)} chunks across "
          f"{len({d.metadata['filename'] for d in docs})} pages ({config.RAGAS_KG_MODEL}), then writing "
          f"{size} questions ({config.RAGAS_QUESTION_MODEL})...")
    testset = generator.generate_with_langchain_docs(
        docs, testset_size=size,
        transforms_llm=LangchainLLMWrapper(kg_llm, bypass_temperature=True, bypass_n=True),
        transforms_embedding_model=embeddings, raise_exceptions=False)

    samples = []
    for s in testset.samples:
        e = s.eval_sample
        samples.append({"user_input": e.user_input, "reference": e.reference,
                        "reference_contexts": e.reference_contexts or [], "synthesizer_name": s.synthesizer_name,
                        "persona_name": getattr(e, "persona_name", ""), "query_style": getattr(e, "query_style", "")})

    sections, commit = load_sections()
    golden = load_items(GOLDEN_PATH)
    previous = [json.loads(l) for l in CANDIDATES_PATH.read_text().splitlines()] if CANDIDATES_PATH.exists() else []
    start = 1 + max([int(c["id"].split("-")[1]) for c in previous] or [0])
    chunk_ids_by_text = {c.page_content: c.metadata["chunk_id"] for c in chunks}
    nugget_llm = ChatOpenAI(model=config.RAGAS_QUESTION_MODEL, max_retries=4, timeout=120, callbacks=[usage])
    items, dropped = to_golden(samples, chunk_ids_by_text, commit,
                               lambda q, r: extract_nuggets(nugget_llm, q, r),
                               {g["question"] for g in golden} | {c["question"] for c in previous}, start)

    # Offer only items that pass the same validator as the hand-written set.
    valid, errors = validate(items, sections, commit)
    bad = {e.split(":")[0] for e in errors}
    keep = [i for i in items if i["id"] not in bad]
    dropped += [e for e in errors if not e.startswith("split imbalance")]

    all_candidates = previous + keep
    CANDIDATES_PATH.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in all_candidates))
    write_review(all_candidates)

    cost = 0.0
    for model, u in usage.usage_metadata.items():
        cin, cout = ((config.CHAT_USD_PER_1M_INPUT, config.CHAT_USD_PER_1M_OUTPUT) if model.startswith(config.RAGAS_KG_MODEL)
                     else (config.RAGAS_USD_PER_1M_INPUT, config.RAGAS_USD_PER_1M_OUTPUT))
        cost += (u.get("input_tokens", 0) * cin + u.get("output_tokens", 0) * cout) / 1e6
        print(f"  {model}: {u.get('input_tokens', 0):,} in / {u.get('output_tokens', 0):,} out tokens")
    print(f"\nRagas wrote {len(samples)} questions; {len(keep)} passed conversion + validation "
          f"({len(dropped)} dropped). Approx cost ${cost:.2f}.")
    for reason in dropped[:10]:
        print(f"  dropped: {reason}")
    print(f"\nReview {REVIEW_PATH.relative_to(config.ROOT)} (decision: approve / reject), then run:\n"
          "  python -m docs_assistant.ragas_testset merge")


def main():
    parser = argparse.ArgumentParser(description="Ragas question generation for the golden set.")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--size", type=int, default=30, help="questions to generate")
    g.add_argument("--chunks", type=int, default=300, help="chunks sampled into the knowledge graph")
    g.add_argument("--seed", type=int, default=0)
    sub.add_parser("merge")
    args = parser.parse_args()

    if args.command == "generate":
        generate(args.size, args.chunks, args.seed)
        return

    if not (CANDIDATES_PATH.exists() and REVIEW_PATH.exists()):
        raise SystemExit("No candidates yet. Run: python -m docs_assistant.ragas_testset generate")
    candidates = [json.loads(l) for l in CANDIDATES_PATH.read_text().splitlines() if l.strip()]
    with REVIEW_PATH.open(newline="") as f:
        approved, counts = merge(candidates, list(csv.DictReader(f)))
    sections, commit = load_sections()
    _, errors = validate(approved, sections, commit)
    errors = [e for e in errors if not e.startswith("split imbalance")]
    if errors:
        raise SystemExit("Approved items fail validation (fix them in the review sheet):\n  " + "\n  ".join(errors))
    APPROVED_PATH.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in approved))
    print(f"Decisions: {counts}. Wrote {len(approved)} approved items to {APPROVED_PATH.relative_to(config.ROOT)}.")
    print("Next: python evals/build_golden_v1.py && python -m docs_assistant.golden validate")


if __name__ == "__main__":
    main()
