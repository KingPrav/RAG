"""Context-utilization signals: citation use, citation support, code grounding."""
from docs_assistant.usage import (analyze_usage, citing_sentences, code_grounding, content_tokens,
                                  split_prose_and_code, support)

CHUNKS = {
    1: "FastAPI docs > Tutorial > Request Body\n\nTo declare a request body, use Pydantic models "
       "that inherit from BaseModel.\n\n```python\nclass Item(BaseModel):\n    name: str\n    price: float\n```",
    2: "FastAPI docs > Tutorial > Query Parameters\n\nFunction parameters that are not part of the "
       "path are interpreted as query parameters.",
    3: "FastAPI docs > Deployment > Docker\n\nBuild a container image with a Dockerfile.",
}


def test_content_tokens_ignore_citations_stopwords_and_short_words():
    assert content_tokens("Use the BaseModel class [1] to do it") == {"basemodel", "class"}


def test_sentences_carry_their_citation_numbers():
    prose = "Declare a request body using Pydantic models [1]. Query parameters come from the function signature [2][1]."
    assert [nums for _, nums in citing_sentences(prose)] == [[1], [2, 1]]


def test_short_fragments_are_not_judged():
    assert citing_sentences("For example:\n\nSee [1].") == []


def test_prose_and_code_are_separated():
    prose, code = split_prose_and_code("Text [1].\n\n```python\nx = 1\n```\n\nMore.")
    assert code == ["x = 1\n"] and "x = 1" not in prose


def test_supported_vs_unsupported_citation():
    good = "To declare a request body, use Pydantic models that inherit from BaseModel [1]."
    bad = "Kubernetes Helm charts configure autoscaling replicas automatically [1]."
    assert support(good, CHUNKS[1]) > 0.8
    assert support(bad, CHUNKS[1]) == 0.0


def test_code_copied_from_sources_is_grounded_and_invented_code_is_not():
    copied = "class Item(BaseModel):\n        name: str\n    price: float\n"   # indentation differs
    invented = "app.add_autoscaler(replicas=3)\nItem.validate_strict(True)\n"
    assert code_grounding(copied, list(CHUNKS.values())) == 1.0
    assert code_grounding(invented, list(CHUNKS.values())) == 0.0


def test_ellipsis_lines_do_not_count_against_grounding():
    assert code_grounding("class Item(BaseModel):\n    ...\n", list(CHUNKS.values())) == 1.0


def test_analyze_usage_full_answer():
    answer = (
        "To declare a request body, use Pydantic models that inherit from BaseModel [1].\n\n"
        "```python\nclass Item(BaseModel):\n    name: str\n    price: float\n```\n\n"
        "Kubernetes Helm charts configure autoscaling replicas automatically [2].\n\n"
        "This sentence has no citation at all but is long enough to judge."
    )
    u = analyze_usage(answer, CHUNKS, cited=[1, 2])
    assert (u["n_retrieved"], u["n_cited"]) == (3, 2)
    assert u["context_utilization"] == round(2 / 3, 3)
    assert u["n_citation_pairs"] == 2 and u["n_unsupported_citations"] == 1     # the Helm claim
    assert u["n_uncited_sentences"] == 1
    assert u["n_code_blocks"] == 1 and u["n_ungrounded_code_blocks"] == 0
    assert u["per_chunk"][1]["cited"] and not u["per_chunk"][3]["cited"]
    assert u["per_chunk"][2]["max_sentence_support"] < 0.5
    assert u["per_chunk"][3]["max_sentence_support"] is None


def test_answer_without_citations_or_code():
    u = analyze_usage("The sources do not cover Helm deployments on Kubernetes clusters.", CHUNKS, cited=[])
    assert u["context_utilization"] == 0.0 and u["mean_citation_support"] is None
    assert u["n_code_blocks"] == 0 and u["code_grounding"] == []
