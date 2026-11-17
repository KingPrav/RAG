"""Phase 4b: cost-aware routing.

Every question that misses the cache is sent down one of three paths:

    fast      no rewrite, no rerank, small model    ~3 s, ~1 LLM call
    standard  the shipped Phase 1 pipeline          ~7.7 s, 3 LLM calls
    escalate  standard retrieval + a larger model   for hard multi-part questions

The router never calls an LLM, since that would spend the latency it is meant
to save. It reads two cheap things:

1. The question text: is it multi-part, a comparison, or asking for a
   specific value/version (the "trap" questions where the v3 prompt's careful
   answerable rule matters)?
2. One vector search on the raw question, which the cache lookup's embedding
   makes nearly free (~20 ms against the local index). A question whose top
   hit is strong AND clearly ahead of the runner-up, with the top results on
   one page, is one the rewrite and rerank steps rarely change.

Thresholds live in RouterPolicy and are chosen by cost_eval (simulated on
per-path eval results), not guessed.
"""
import re
from dataclasses import asdict, dataclass, field

FAST, STANDARD, ESCALATE = "fast", "standard", "escalate"
ROUTES = (FAST, STANDARD, ESCALATE)

COMPARISON = re.compile(r"\b(vs\.?|versus|difference between|compared? (?:to|with)|differ|instead of|"
                        r"which is better|pros and cons|when should i use .+ or)\b", re.I)
SPECIFIC_VALUE = re.compile(r"\b(which version|what version|since version|default (?:value|is)|"
                            r"what is the default|how many|maximum|minimum|limit)\b", re.I)
CONJUNCTION = re.compile(r"\b(and also|and then|as well as|and how|and what|and why|and can)\b", re.I)
WORD = re.compile(r"\w+")


# ---------------------------------------------------------------- features

@dataclass
class Features:
    n_words: int
    comparison: bool
    specific_value: bool
    multi_part: bool                  # a conjunction joining two asks, or more than one "?"
    top1: float                       # cosine similarity of the best chunk for the raw question
    margin: float                     # top1 - top2: how clearly one chunk wins
    top_page_share: float             # share of the top-n hits on the same page as the best one
    n_pages: int                      # distinct pages among the top-n hits

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def extract_features(question: str, hits: list[tuple[str, float]]) -> Features:
    """`hits` = [(doc_id, similarity)] from one search on the raw question, best first."""
    sims = [s for _, s in hits]
    pages = [d for d, _ in hits]
    top1 = sims[0] if sims else 0.0
    return Features(
        n_words=len(WORD.findall(question)),
        comparison=bool(COMPARISON.search(question)),
        specific_value=bool(SPECIFIC_VALUE.search(question)),
        multi_part=bool(CONJUNCTION.search(question)) or question.count("?") > 1,
        top1=top1,
        margin=top1 - sims[1] if len(sims) > 1 else top1,
        top_page_share=pages.count(pages[0]) / len(pages) if pages else 0.0,
        n_pages=len(set(pages)),
    )


# ---------------------------------------------------------------- policy

@dataclass(frozen=True)
class RouterPolicy:
    """Starting values are deliberately conservative (few questions go fast);
    cost_eval sweeps them on dev and reports the cheapest policy whose pass
    rate is not worse than always-standard."""
    fast_min_top1: float = 0.62
    fast_min_margin: float = 0.03
    fast_min_page_share: float = 0.6
    fast_max_words: int = 18
    escalate: bool = True                 # False: hard questions stay on standard (no big model)
    escalate_min_pages: int = 3           # a multi-part question escalates if its hits span this many pages

    def describe(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path) -> "RouterPolicy":
        """The policy chosen by `cost_eval simulate` on dev, if one was saved;
        otherwise the conservative defaults above."""
        import json
        from pathlib import Path
        path = Path(path)
        return cls(**json.loads(path.read_text())["policy"]) if path.exists() else cls()


@dataclass
class Decision:
    route: str
    reasons: list[str] = field(default_factory=list)
    features: Features | None = None

    def to_dict(self) -> dict:
        return {"route": self.route, "reasons": self.reasons,
                "features": self.features.to_dict() if self.features else None}


def route(f: Features, policy: RouterPolicy = RouterPolicy()) -> Decision:
    """Pure function of the features, so eval replays are exact."""
    # 1. Hard questions: comparisons, or multi-part asks whose evidence is scattered.
    if policy.escalate and (f.comparison or (f.multi_part and f.n_pages >= policy.escalate_min_pages)):
        why = "comparison" if f.comparison else f"multi-part across {f.n_pages} pages"
        return Decision(ESCALATE, [why], f)

    # 2. Easy questions: every fast-path condition must hold; record each that fails.
    blockers = []
    if f.comparison or f.multi_part:
        blockers.append("multi-part")
    if f.specific_value:
        blockers.append("asks for a specific value/version")
    if f.n_words > policy.fast_max_words:
        blockers.append(f"long ({f.n_words} words)")
    if f.top1 < policy.fast_min_top1:
        blockers.append(f"weak top hit ({f.top1:.2f} < {policy.fast_min_top1})")
    if f.margin < policy.fast_min_margin:
        blockers.append(f"no clear winner (margin {f.margin:.3f})")
    if f.top_page_share < policy.fast_min_page_share:
        blockers.append(f"hits spread out ({f.top_page_share:.0%} on top page)")
    if not blockers:
        return Decision(FAST, ["confident single-page match"], f)

    # 3. Everything else: the shipped pipeline.
    return Decision(STANDARD, blockers, f)


def forced(route_name: str) -> Decision:
    """For --route overrides and per-path evals."""
    if route_name not in ROUTES:
        raise ValueError(f"unknown route {route_name!r}; expected one of {ROUTES}")
    return Decision(route_name, ["forced"])
