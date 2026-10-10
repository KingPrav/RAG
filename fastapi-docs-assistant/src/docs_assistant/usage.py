"""Did the answer actually use the retrieved chunks? (context utilization)

Three cheap, deterministic signals, computed for every answer:

1. Citation use   - which retrieved chunks were cited at all.
2. Claim support  - the answer is split into claims (paragraphs and list
                    items, which is how the model places its citations).
                    For each claim citing [n][m], the share of its vocabulary
                    found in the cited chunks combined. Low support means
                    "cites [2], but [2] doesn't say that".
3. Code grounding - for every code block in the answer, the share of its
                    lines found verbatim in the retrieved chunks. Low
                    grounding means adapted or invented code.

Calibrated in Step 5 against 40 hand-graded answers (analysis/FAILURE_ANALYSIS.md).
Limitations: lexical overlap misses heavy paraphrase and rewards copying, so
these are triage signals, not grades. Step 7 adds an LLM judge for faithfulness.
"""
import re

CITATION = re.compile(r"\[(\d+)\]")
CITATIONS_ONLY = re.compile(r"^[\s.]*(\[\d+\]\s*)+[\s.]*$")
FENCED_BLOCK = re.compile(r"```([^\n]*)\n(.*?)```", re.S)
# Illustrations (directory trees, diagrams) are not code: the model shortens
# them and box-drawing characters shift, so they are not checked for grounding.
NOT_CODE_LANGS = {"text", "txt", "plaintext", "tree"}
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d+")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# Versioned so stored traces can be re-scored when the heuristics change
# (report.py re-scores traces whose usage_version differs).
#   v1: sentence units, one chunk at a time.
#   v2: citations stay with the preceding sentence; stemming; refusals not scored.
#   v3: claim units = paragraphs / list items, scored against the union of their
#       cited chunks. Matches how the model actually cites (once per paragraph).
#       ```text blocks (directory trees) are not checked as code.
USAGE_VERSION = 3
MIN_CONTENT_TOKENS = 4          # shorter units (headings, "For example:") are skipped
SUPPORTED_THRESHOLD = 0.5       # a claim shares >= half its content tokens with its cited chunks
CODE_GROUNDED_THRESHOLD = 0.8   # >= 80% of a code block's lines appear in retrieved chunks

STOPWORDS = frozenset("""
the and for are but not you your with this that from have has can will would should could
use used using into than then them they their there these those what when where which while
also just only any all one each other such its it's was were been being does did done how why
about over under more most some very via per our out get set may might must
""".split())


def stem(word: str) -> str:
    """Deliberately crude suffix stripping: enough to match add/adds/added,
    sign/signs/signed and expire/expiration, without a dependency."""
    for suffix in ("ation", "ing", "ed", "es", "s"):
        if suffix == "s" and word.endswith("ss"):          # class, access, pass
            continue
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> set[str]:
    words = {t.lower() for t in TOKEN.findall(CITATION.sub(" ", text))} - STOPWORDS
    return {stem(w) for w in words}


def split_prose_and_code(answer: str) -> tuple[str, list[str]]:
    code_blocks = [body for lang, body in FENCED_BLOCK.findall(answer)
                   if lang.strip().lower() not in NOT_CODE_LANGS]
    prose = FENCED_BLOCK.sub("\n\n", answer)
    return prose, code_blocks


def claims(prose: str) -> list[tuple[str, list[int]]]:
    """(claim text, cited numbers) for each paragraph or list item long enough
    to judge. A unit that is only citations (e.g. "[3]" after a code block)
    is credited to the claim before it."""
    units: list[str] = []
    for para in PARAGRAPH_BREAK.split(prose):
        lines = [ln for ln in para.splitlines() if ln.strip()]
        if lines and sum(bool(LIST_ITEM.match(ln)) for ln in lines) >= 2:
            units.extend(LIST_ITEM.sub("", ln) for ln in lines)     # each bullet is a claim
        else:
            units.append(" ".join(ln.strip() for ln in lines))

    out: list[tuple[str, list[int]]] = []
    for unit in units:
        text = unit.strip(" >\t")
        if not text:
            continue
        numbers = [int(n) for n in CITATION.findall(text)]
        if CITATIONS_ONLY.match(text):
            if out:
                prev, prev_numbers = out[-1]
                out[-1] = (prev, prev_numbers + numbers)
            continue
        if len(content_tokens(text)) < MIN_CONTENT_TOKENS:
            continue
        out.append((text, numbers))
    return out


def support(text: str, chunk_texts: list[str]) -> float:
    """Share of the claim's content tokens found in the given chunks combined."""
    claim = content_tokens(text)
    if not claim:
        return 0.0
    available = set().union(*(content_tokens(t) for t in chunk_texts)) if chunk_texts else set()
    return len(claim & available) / len(claim)


def code_grounding(block: str, chunk_texts: list[str]) -> float:
    """Share of meaningful code lines found verbatim (ignoring indentation)
    in any retrieved chunk."""
    available = {line.strip() for text in chunk_texts for line in text.splitlines()}
    lines = [ln.strip() for ln in block.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 4 and ln not in ("...", "# ...")]
    if not lines:
        return 1.0
    return sum(ln in available for ln in lines) / len(lines)


def analyze_usage(answer: str, chunks: dict[int, str], cited: list[int], answerable: bool = True) -> dict:
    """chunks: {source number: chunk text} as shown to the model.
    cited: validated citation numbers from the structured output.
    answerable: refusals ("the sources don't cover X [1]") make no factual
    claim to support, so their text is not scored."""
    prose, code_blocks = split_prose_and_code(answer)
    units = claims(prose) if answerable else []

    claim_scores: list[float] = []
    per_chunk_support: dict[int, list[float]] = {}
    n_uncited = 0
    for text, numbers in units:
        valid = [n for n in dict.fromkeys(numbers) if n in chunks]
        if not valid:
            n_uncited += 1
            continue
        claim_scores.append(support(text, [chunks[n] for n in valid]))
        for n in valid:
            per_chunk_support.setdefault(n, []).append(support(text, [chunks[n]]))

    texts = list(chunks.values())
    code_scores = [code_grounding(b, texts) for b in code_blocks]
    n_cited = len(set(cited) & set(chunks))

    return {
        # 1. citation use
        "n_retrieved": len(chunks),
        "n_cited": n_cited,
        "context_utilization": round(n_cited / len(chunks), 3) if chunks else 0.0,
        # 2. claim support
        "n_claims": len(units),
        "n_uncited_claims": n_uncited,
        "n_cited_claims": len(claim_scores),
        "n_unsupported_claims": sum(s < SUPPORTED_THRESHOLD for s in claim_scores),
        "claim_support": [round(s, 3) for s in claim_scores],
        # 3. code grounding
        "n_code_blocks": len(code_blocks),
        "code_grounding": [round(s, 3) for s in code_scores],
        "n_ungrounded_code_blocks": sum(s < CODE_GROUNDED_THRESHOLD for s in code_scores),
        "per_chunk": {n: {"cited": n in cited,
                          "max_claim_support": round(max(per_chunk_support[n]), 3) if n in per_chunk_support else None}
                      for n in chunks},
        "usage_version": USAGE_VERSION,
        "thresholds": {"supported": SUPPORTED_THRESHOLD, "code_grounded": CODE_GROUNDED_THRESHOLD,
                       "min_content_tokens": MIN_CONTENT_TOKENS},
    }
