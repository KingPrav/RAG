"""Did the answer actually use the retrieved chunks? (context utilization)

Three cheap, deterministic signals, computed for every answer:

1. Citation use     - which retrieved chunks were cited at all.
2. Citation support - for every sentence that cites [n], how much of its
                      vocabulary appears in chunk n. Low support means
                      "cites [2], but [2] doesn't say that".
3. Code grounding   - for every code block in the answer, the share of its
                      lines found verbatim in the retrieved chunks. Low
                      grounding means invented APIs or parameters.

Limitations: lexical overlap misses paraphrase and rewards copying. These are
triage signals for failure analysis, not final grades; Step 7 adds an
LLM judge for faithfulness.
"""
import re

CITATION = re.compile(r"\[(\d+)\]")
FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.S)
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n+")

# Thresholds are starting heuristics, recorded with each trace; Step 5 tunes them.
MIN_CONTENT_TOKENS = 4          # shorter sentences (headings, "For example:") are skipped
SUPPORTED_THRESHOLD = 0.5       # a citing sentence shares >= half its content tokens with the chunk
CODE_GROUNDED_THRESHOLD = 0.8   # >= 80% of a code block's lines appear in retrieved chunks

STOPWORDS = frozenset("""
the and for are but not you your with this that from have has can will would should could
use used using into than then them they their there these those what when where which while
also just only any all one each other such its it's was were been being does did done how why
about over under more most some very via per our out get set via may might must
""".split())


def content_tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(CITATION.sub(" ", text))} - STOPWORDS


def split_prose_and_code(answer: str) -> tuple[str, list[str]]:
    code_blocks = FENCED_BLOCK.findall(answer)
    prose = FENCED_BLOCK.sub(" ", answer)
    return prose, code_blocks


def citing_sentences(prose: str) -> list[tuple[str, list[int]]]:
    """(sentence, cited numbers) for each sentence long enough to judge."""
    out = []
    for raw in SENTENCE_SPLIT.split(prose):
        sentence = raw.strip(" -*>\t")
        if len(content_tokens(sentence)) < MIN_CONTENT_TOKENS:
            continue
        out.append((sentence, [int(n) for n in CITATION.findall(sentence)]))
    return out


def support(sentence: str, chunk_text: str) -> float:
    """Share of the sentence's content tokens that appear in the chunk."""
    s = content_tokens(sentence)
    return len(s & content_tokens(chunk_text)) / len(s) if s else 0.0


def code_grounding(block: str, chunk_texts: list[str]) -> float:
    """Share of meaningful code lines found verbatim (ignoring indentation)
    in any retrieved chunk."""
    available = {line.strip() for text in chunk_texts for line in text.splitlines()}
    lines = [ln.strip() for ln in block.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 4 and ln not in ("...", "# ...")]
    if not lines:
        return 1.0
    return sum(ln in available for ln in lines) / len(lines)


def analyze_usage(answer: str, chunks: dict[int, str], cited: list[int]) -> dict:
    """chunks: {source number: chunk text} as shown to the model.
    cited: validated citation numbers from the structured output."""
    prose, code_blocks = split_prose_and_code(answer)
    sentences = citing_sentences(prose)

    # 2. Citation support, per (sentence, cited chunk) pair.
    pair_scores = []
    per_chunk_support: dict[int, list[float]] = {}
    for sentence, numbers in sentences:
        for n in numbers:
            if n in chunks:
                score = support(sentence, chunks[n])
                pair_scores.append(score)
                per_chunk_support.setdefault(n, []).append(score)

    # 3. Code grounding, per block.
    texts = list(chunks.values())
    code_scores = [code_grounding(b, texts) for b in code_blocks]

    per_chunk = {
        n: {
            "cited": n in cited,
            "max_sentence_support": round(max(per_chunk_support[n]), 3) if n in per_chunk_support else None,
        }
        for n in chunks
    }
    return {
        # 1. citation use
        "n_retrieved": len(chunks),
        "n_cited": len(set(cited) & set(chunks)),
        "context_utilization": round(len(set(cited) & set(chunks)) / len(chunks), 3) if chunks else 0.0,
        # 2. citation support
        "n_sentences": len(sentences),
        "n_uncited_sentences": sum(1 for _, nums in sentences if not nums),
        "n_citation_pairs": len(pair_scores),
        "n_unsupported_citations": sum(s < SUPPORTED_THRESHOLD for s in pair_scores),
        "mean_citation_support": round(sum(pair_scores) / len(pair_scores), 3) if pair_scores else None,
        # 3. code grounding
        "n_code_blocks": len(code_blocks),
        "code_grounding": [round(s, 3) for s in code_scores],
        "n_ungrounded_code_blocks": sum(s < CODE_GROUNDED_THRESHOLD for s in code_scores),
        "per_chunk": per_chunk,
        "thresholds": {"supported": SUPPORTED_THRESHOLD, "code_grounded": CODE_GROUNDED_THRESHOLD,
                       "min_content_tokens": MIN_CONTENT_TOKENS},
    }
