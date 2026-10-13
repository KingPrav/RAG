"""Step 1: build the corpus.

    PubMed search -> raw XML (cached) -> parsed records -> merge + quality
    filters -> data/processed/documents.jsonl + manifest.json

Run:  python -m claimcheck.ingest            (uses cache if present)
      python -m claimcheck.ingest --refresh  (re-download everything)
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone

from . import config
from .pubmed import PubMedClient, parse_pubmed_xml
from .topics import POSITION_STAND_QUERY, TOPIC_QUERIES, build_query, topics_for_title

BATCH_SIZE = 200


def download(client, name: str, query: str, max_docs: int, refresh: bool) -> tuple[int, list[dict]]:
    """Search, then fetch in batches. Raw XML is cached on disk so re-runs are
    free and the exact source data behind any eval run can be inspected."""
    folder = config.RAW_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    ids_file = folder / "pmids.json"

    if ids_file.exists() and not refresh:
        cached = json.loads(ids_file.read_text())
        total, pmids = cached["total_matches"], cached["pmids"]
    else:
        total, pmids = client.search(query, retmax=max_docs)
        ids_file.write_text(json.dumps({"query": query, "total_matches": total, "pmids": pmids}, indent=1))

    records = []
    for start in range(0, len(pmids), BATCH_SIZE):
        batch_file = folder / f"batch_{start // BATCH_SIZE:03d}.xml"
        if refresh or not batch_file.exists():
            batch_file.write_text(client.fetch_xml(pmids[start:start + BATCH_SIZE]))
        records.extend(parse_pubmed_xml(batch_file.read_text()))
    print(f"  {name:<16} {total:>6} matches, kept top {len(pmids)}, parsed {len(records)}")
    return total, records


def merge_and_filter(by_topic: dict[str, list[dict]], position_stands: list[dict]):
    """One record per paper. A paper found under two topics keeps both tags,
    so it is embedded once but can be filtered by either topic later."""
    docs: dict[str, dict] = {}
    for topic, records in by_topic.items():
        for rec in records:
            doc = docs.setdefault(rec["pmid"], {**rec, "topics": [], "doc_type": "study"})
            if topic not in doc["topics"]:
                doc["topics"].append(topic)

    for rec in position_stands:
        doc = docs.setdefault(rec["pmid"], {**rec, "topics": []})
        doc["doc_type"] = "position_stand"
        doc["evidence_level"] = "position_stand"
        doc["topics"] = sorted(set(doc["topics"]) | set(topics_for_title(rec["title"])))

    kept, dropped = [], Counter()
    for doc in docs.values():
        if any("Retract" in p for p in doc["pub_types"]):
            dropped["retracted"] += 1
        elif doc["doc_type"] == "study" and len(doc["abstract"]) < config.MIN_ABSTRACT_CHARS:
            dropped["abstract_too_short"] += 1
        elif not doc["title"]:
            dropped["missing_title"] += 1
        else:
            kept.append(doc)
    kept.sort(key=lambda d: int(d["pmid"]))
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-per-topic", type=int, default=config.MAX_PER_TOPIC)
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    args = parser.parse_args()

    client = PubMedClient(config.NCBI_EMAIL, config.NCBI_API_KEY)
    print("Downloading from PubMed...")

    by_topic, totals = {}, {}
    for topic in TOPIC_QUERIES:
        totals[topic], by_topic[topic] = download(
            client, topic, build_query(topic), args.max_per_topic, args.refresh)
    _, stands = download(client, "position_stands", POSITION_STAND_QUERY, 50, args.refresh)

    docs, dropped = merge_and_filter(by_topic, stands)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (config.PROCESSED_DIR / "documents.jsonl").open("w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "max_per_topic": args.max_per_topic,
        "queries": {t: build_query(t) for t in TOPIC_QUERIES} | {"position_stands": POSITION_STAND_QUERY},
        "pubmed_total_matches": totals,
        "documents": len(docs),
        "by_topic": dict(Counter(t for d in docs for t in d["topics"])),
        "by_evidence_level": dict(Counter(d["evidence_level"] for d in docs)),
        "position_stands": [d["title"] for d in docs if d["doc_type"] == "position_stand"],
        "dropped": dict(dropped),
        "abstract_chars_total": sum(len(d["abstract"]) for d in docs),
    }
    (config.PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(docs)} documents to data/processed/documents.jsonl")
    for key in ("by_topic", "by_evidence_level", "dropped"):
        print(f"  {key}: {manifest[key]}")
    print(f"  position stands: {len(manifest['position_stands'])}")


if __name__ == "__main__":
    main()
