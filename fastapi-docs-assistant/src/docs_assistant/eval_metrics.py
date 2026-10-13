"""Scoring functions for the eval harness (pure, deterministic, unit-tested).

Retrieval metrics are computed at *section* level (`page#anchor`): a chunk
`tutorial/body#create-your-data-model/2` counts as the section
`tutorial/body#create-your-data-model`, so gold labels survive re-chunking.
"""
import random
import re
import statistics
from collections.abc import Callable, Sequence

# ---------------------------------------------------------------- answers

def normalize_answer(text: str) -> str:
    """Lowercase, drop backticks and asterisks (markdown code/bold) and collapse
    whitespace, so `**8000**` and "8000" match. Underscores are kept: they are
    part of identifiers like response_model_exclude_unset, and / . : - stay so
    URLs, paths and names like /docs, 127.0.0.1, python-multipart still match."""
    text = re.sub(r"[`*]+", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def exact_match(answer: str, accepted: Sequence[str]) -> bool:
    """True if any accepted string appears in the answer (after normalization).
    Containment rather than equality: answers are sentences, not bare values.
    Accepted strings that are pure numbers must match as whole numbers, so
    "500" does not match inside "1500" or "5000"."""
    norm = normalize_answer(answer)
    for acc in accepted:
        a = normalize_answer(acc)
        if a.isdigit():
            if re.search(rf"(?<![\d.]){re.escape(a)}(?![\d])", norm):
                return True
        elif a in norm:
            return True
    return False


# ---------------------------------------------------------------- retrieval

def section_of(chunk_id: str) -> str:
    return re.sub(r"/\d+$", "", chunk_id)


def retrieval_scores(retrieved_chunk_ids: Sequence[str], gold_sections: Sequence[str], k: int) -> dict:
    """hit@1, hit@k, recall@k and reciprocal rank for one question.
    A question with several gold sections is a hit if *any* is retrieved
    (any of them answers it); recall@k says what share of them were retrieved."""
    sections = [section_of(c) for c in retrieved_chunk_ids[:k]]
    gold = set(gold_sections)
    ranks = [i for i, s in enumerate(sections, start=1) if s in gold]
    first = ranks[0] if ranks else None
    return {
        "hit@1": first == 1,
        f"hit@{k}": first is not None,
        f"recall@{k}": len(gold & set(sections)) / len(gold) if gold else 0.0,
        "reciprocal_rank": 1.0 / first if first else 0.0,
        "first_gold_rank": first,
    }


# ---------------------------------------------------------------- nuggets

def nugget_scores(nuggets: Sequence[dict], verdicts: Sequence[str]) -> dict:
    """Nugget evaluation (Lin et al., 2025): only full support counts.
    vital_strict = share of vital nuggets fully supported (primary);
    all_strict   = share of all nuggets fully supported."""
    if len(nuggets) != len(verdicts):
        raise ValueError(f"{len(verdicts)} verdicts for {len(nuggets)} nuggets")
    hits = [v == "support" for v in verdicts]
    vital = [h for h, n in zip(hits, nuggets) if n["vital"]]
    return {
        "vital_strict": sum(vital) / len(vital) if vital else None,
        "all_strict": sum(hits) / len(hits) if hits else None,
        "partial": sum(v == "partial" for v in verdicts),
    }


# ---------------------------------------------------------------- statistics

def mean(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def bootstrap_ci(values: Sequence[float], *, resamples: int = 5000, confidence: float = 0.95,
                 seed: int = 0, statistic: Callable[[list], float] = statistics.fmean) -> tuple[float, float] | None:
    """Percentile bootstrap over items: resample items with replacement,
    recompute the statistic, take the middle `confidence` of the results."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    lo = stats[int((1 - confidence) / 2 * resamples)]
    hi = stats[int((1 + confidence) / 2 * resamples) - 1]
    return lo, hi


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, resamples: int = 5000,
                     confidence: float = 0.95, seed: int = 0) -> dict | None:
    """Difference b - a on the *same* items. Resampling item indices (not each
    run separately) cancels out per-item difficulty, which is what makes a
    small set able to detect a real improvement."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    diffs = sorted(statistics.fmean(pairs[j][1] - pairs[j][0] for j in (rng.randrange(n) for _ in range(n)))
                   for _ in range(resamples))
    observed = statistics.fmean(y - x for x, y in pairs)
    lo = diffs[int((1 - confidence) / 2 * resamples)]
    hi = diffs[int((1 + confidence) / 2 * resamples) - 1]
    return {"diff": observed, "ci": (lo, hi), "n": n,
            "significant": lo > 0 or hi < 0}
