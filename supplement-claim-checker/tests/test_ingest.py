from pathlib import Path

from claimcheck.ingest import merge_and_filter
from claimcheck.pubmed import parse_pubmed_xml

XML = (Path(__file__).parent / "fixtures" / "sample_pubmed.xml").read_text()


def records():
    return {r["pmid"]: r for r in parse_pubmed_xml(XML)}


def test_parses_structured_abstract_and_inline_markup():
    r = records()["11111111"]
    assert r["title"] == "Effects of creatine supplementation on lean mass: a meta-analysis"
    assert r["abstract"].startswith("BACKGROUND: Creatine")
    assert "\nRESULTS: Across 35" in r["abstract"]
    assert r["evidence_level"] == "meta_analysis"
    assert r["mesh_terms"] == ["Creatine", "Resistance Training"]
    assert r["doi"] == "10.1000/example.1"
    assert r["year"] == 2021


def test_handles_medline_date_and_pmc_id():
    r = records()["22222222"]
    assert r["year"] == 2019
    assert r["pmcid"] == "PMC999"
    assert r["evidence_level"] == "rct"


def test_merge_dedupes_and_keeps_all_topics():
    r = records()
    by_topic = {"creatine": [r["11111111"]], "protein": [r["11111111"]], "caffeine": [r["22222222"]]}
    docs, _ = merge_and_filter(by_topic, [])
    by_id = {d["pmid"]: d for d in docs}
    assert len(docs) == 2
    assert by_id["11111111"]["topics"] == ["creatine", "protein"]


def test_filters_retracted_but_keeps_short_position_stand():
    r = records()
    docs, dropped = merge_and_filter({"creatine": [r["33333333"]]}, [r["44444444"]])
    assert dropped == {"retracted": 1}
    assert len(docs) == 1
    stand = docs[0]
    assert stand["doc_type"] == "position_stand"
    assert stand["evidence_level"] == "position_stand"
    assert stand["topics"] == ["creatine"]
