"""Thin client for NCBI E-utilities (esearch + efetch) and an XML parser.

Network code and parsing are kept separate so the parser can be unit-tested
offline against fixture XML.
"""
import re
import time
import xml.etree.ElementTree as ET

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    def __init__(self, email: str, api_key: str | None = None, session=None):
        if not email:
            raise ValueError("Set NCBI_EMAIL in .env (NCBI requires it).")
        self.session = session or requests.Session()
        self.base_params = {"tool": "claimcheck", "email": email}
        if api_key:
            self.base_params["api_key"] = api_key
        # NCBI limits: 3 req/s without a key, 10 req/s with one.
        self.min_interval = 0.11 if api_key else 0.34
        self._last_call = 0.0

    def _get(self, endpoint: str, params: dict) -> requests.Response:
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self.session.get(
                    f"{EUTILS}/{endpoint}",
                    params={**self.base_params, **params},
                    timeout=60,
                )
            except requests.RequestException:
                time.sleep(2**attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2**attempt)  # back off on rate limits / server errors
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"PubMed {endpoint} failed after retries")

    def search(self, term: str, retmax: int) -> tuple[int, list[str]]:
        """Return (total matches, PMIDs sorted by relevance)."""
        data = self._get(
            "esearch.fcgi",
            {"db": "pubmed", "term": term, "retmax": retmax,
             "retmode": "json", "sort": "relevance"},
        ).json()["esearchresult"]
        return int(data["count"]), data["idlist"]

    def fetch_xml(self, pmids: list[str]) -> str:
        return self._get(
            "efetch.fcgi",
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        ).text


# ---------------------------------------------------------------- parsing

def _text(el) -> str:
    """All text inside an element, including inline tags like <i> or <sub>."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _year(article) -> int | None:
    pub_date = article.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year = pub_date.findtext("Year")
        if year:
            return int(year)
        medline = pub_date.findtext("MedlineDate") or ""  # e.g. "2019 Jan-Feb"
        match = re.search(r"\d{4}", medline)
        if match:
            return int(match.group())
    year = article.findtext("ArticleDate/Year")
    return int(year) if year else None


def evidence_level(pub_types: list[str]) -> str:
    """Rank study design. Later, the verdict step will weigh a meta-analysis
    above a single small trial, so we capture this at ingestion time."""
    pt = set(pub_types)
    if pt & {"Meta-Analysis", "Systematic Review"}:
        return "meta_analysis"
    if "Randomized Controlled Trial" in pt:
        return "rct"
    if any("Clinical Trial" in p for p in pt):
        return "clinical_trial"
    if "Review" in pt:
        return "review"
    return "other"


def parse_pubmed_xml(xml: str) -> list[dict]:
    root = ET.fromstring(xml)
    records = []
    for node in root.iter("PubmedArticle"):
        citation = node.find("MedlineCitation")
        article = citation.find("Article")

        sections = []
        for part in article.findall("Abstract/AbstractText"):
            text = _text(part)
            if not text:
                continue
            label = part.get("Label")
            sections.append(f"{label}: {text}" if label else text)

        pub_types = [_text(p) for p in article.findall("PublicationTypeList/PublicationType")]
        ids = {i.get("IdType"): _text(i) for i in node.findall("PubmedData/ArticleIdList/ArticleId")}
        pmid = _text(citation.find("PMID"))

        records.append({
            "doc_id": f"pmid:{pmid}",
            "pmid": pmid,
            "title": _text(article.find("ArticleTitle")),
            "abstract": "\n".join(sections),
            "journal": _text(article.find("Journal/Title")),
            "year": _year(article),
            "pub_types": pub_types,
            "evidence_level": evidence_level(pub_types),
            "mesh_terms": [_text(d) for d in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")],
            "doi": ids.get("doi"),
            "pmcid": ids.get("pmc"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return records
