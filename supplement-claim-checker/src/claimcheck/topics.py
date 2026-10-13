"""The corpus definition: which PubMed searches make up each topic.

Queries are versioned in code (not typed ad hoc) so the corpus is reproducible.
If you change a query, the manifest records it, and your eval numbers stay
comparable only across runs built from the same queries.
"""

# Only human studies in English with an abstract, and only study designs that
# carry evidential weight. Narrative opinion pieces are excluded on purpose.
STUDY_FILTER = (
    '(randomized controlled trial[pt] OR meta-analysis[pt] OR '
    'systematic review[pt] OR clinical trial[pt]) '
    'AND humans[mh] AND english[la] AND hasabstract'
)

TOPIC_QUERIES = {
    "protein": (
        '("dietary proteins"[mh] OR "whey proteins"[mh] OR '
        '"protein supplement*"[tiab] OR "protein intake"[tiab]) AND '
        '(exercise[mh] OR "resistance training"[mh] OR athlet*[tiab] OR '
        '"muscle protein synthesis"[tiab] OR "lean mass"[tiab])'
    ),
    "creatine": '(creatine[ti]) AND (supplement*[tiab] OR loading[tiab])',
    "caffeine": (
        '(caffeine[ti] OR coffee[ti]) AND '
        '(exercise[mh] OR "athletic performance"[mh] OR performance[tiab] OR '
        'endurance[tiab] OR strength[tiab])'
    ),
}

# The ISSN position stands are the expert verdicts we will later turn into
# ground truth (Step 6). They are fetched separately, without STUDY_FILTER.
POSITION_STAND_QUERY = (
    '"J Int Soc Sports Nutr"[ta] AND "position stand"[ti] AND '
    '(protein[ti] OR creatine[ti] OR caffeine[ti])'
)


def build_query(topic: str) -> str:
    return f"({TOPIC_QUERIES[topic]}) AND {STUDY_FILTER}"


def topics_for_title(title: str) -> list[str]:
    """Assign a position stand to topics by its title."""
    t = title.lower()
    return [name for name in TOPIC_QUERIES if name in t]
