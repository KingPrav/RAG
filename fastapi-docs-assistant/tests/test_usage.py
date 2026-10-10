"""Context-utilization signals: citation use, claim support, code grounding.

Several cases are regression tests for the false alarms found when the v1
heuristics were calibrated against 40 hand-graded answers (Step 5)."""
from docs_assistant.usage import (USAGE_VERSION, analyze_usage, claims, code_grounding, content_tokens,
                                  split_prose_and_code, stem, support)

CHUNKS = {
    1: "FastAPI docs > Tutorial > Request Body\n\nTo declare a request body, use Pydantic models "
       "that inherit from BaseModel.\n\n```python\nclass Item(BaseModel):\n    name: str\n    price: float\n```",
    2: "FastAPI docs > Tutorial > Query Parameters\n\nFunction parameters that are not part of the "
       "path are interpreted as query parameters.",
    3: "FastAPI docs > Security > JWT\n\nCreate a utility function to generate a new access token. "
       "It adds an expiration time and signs it with the secret key.",
}


def test_content_tokens_ignore_citations_stopwords_and_short_words():
    assert content_tokens("Use the BaseModel class [1] to do it") == {"basemodel", "class"}


def test_stemming_matches_simple_paraphrase():
    assert stem("adds") == stem("add") and stem("signs") == stem("signed") == "sign"
    assert stem("class") == "class" and stem("access") == "access"


def test_paragraph_is_one_claim_with_its_trailing_citations():
    # The model cites once at the end of a paragraph; v1 split this into sentences
    # and called the first one "uncited".
    prose = "Set the default in the parameter declaration. A parameter with a default value is optional. [2]"
    assert claims(prose) == [(prose, [2])]


def test_citation_after_code_block_credits_previous_claim():
    answer = "Declare a Pydantic model and use it as a parameter:\n\n```python\nx = 1\n```\n[3]"
    prose, _ = split_prose_and_code(answer)
    assert [nums for _, nums in claims(prose)] == [[3]]


def test_list_items_are_separate_claims():
    prose = "Steps:\n\n- Install python-multipart for form parsing [1]\n- Declare Form parameters in the function [2]"
    assert [nums for _, nums in claims(prose)] == [[1], [2]]


def test_short_fragments_are_not_judged():
    assert claims("For example:\n\nSee [1].") == []


def test_prose_and_code_are_separated_and_text_blocks_are_not_code():
    prose, code = split_prose_and_code("Text [1].\n\n```python\nx = 1\n```\n\n```text\napp/\n└── main.py\n```")
    assert code == ["x = 1\n"] and "x = 1" not in prose


def test_claim_is_scored_against_all_its_cited_chunks_combined():
    claim = "Declare a request body with Pydantic models; other function parameters become query parameters."
    assert support(claim, [CHUNKS[1]]) < 0.7                    # each chunk covers only part
    assert support(claim, [CHUNKS[1], CHUNKS[2]]) > 0.8         # together they support it


def test_paraphrase_with_inflections_is_supported():
    claim = "Define a helper that adds an expiration and signs the token with the secret key."
    assert support(claim, [CHUNKS[3]]) >= 0.5


def test_unrelated_claim_is_unsupported():
    assert support("Kubernetes Helm charts configure autoscaling replicas automatically.", [CHUNKS[1]]) == 0.0


def test_code_copied_from_sources_is_grounded_and_invented_code_is_not():
    copied = "class Item(BaseModel):\n        name: str\n    price: float\n"   # indentation differs
    invented = "app.add_autoscaler(replicas=3)\nItem.validate_strict(True)\n"
    assert code_grounding(copied, list(CHUNKS.values())) == 1.0
    assert code_grounding(invented, list(CHUNKS.values())) == 0.0


def test_ellipsis_lines_do_not_count_against_grounding():
    assert code_grounding("class Item(BaseModel):\n    ...\n", list(CHUNKS.values())) == 1.0


def test_analyze_usage_full_answer():
    answer = (
        "To declare a request body, use Pydantic models that inherit from BaseModel. [1]\n\n"
        "```python\nclass Item(BaseModel):\n    name: str\n    price: float\n```\n\n"
        "Kubernetes Helm charts configure autoscaling replicas automatically. [2]\n\n"
        "This paragraph has no citation at all but is long enough to judge."
    )
    u = analyze_usage(answer, CHUNKS, cited=[1, 2])
    assert (u["n_retrieved"], u["n_cited"]) == (3, 2)
    assert u["context_utilization"] == round(2 / 3, 3)
    assert u["n_claims"] == 3 and u["n_cited_claims"] == 2 and u["n_uncited_claims"] == 1
    assert u["n_unsupported_claims"] == 1                                    # the Helm claim
    assert u["n_code_blocks"] == 1 and u["n_ungrounded_code_blocks"] == 0
    assert u["per_chunk"][1]["cited"] and not u["per_chunk"][3]["cited"]
    assert u["per_chunk"][2]["max_claim_support"] < 0.5
    assert u["per_chunk"][3]["max_claim_support"] is None
    assert u["usage_version"] == USAGE_VERSION


def test_refusals_are_not_scored_for_support():
    refusal = "The provided sources don't explain how to connect to MongoDB or configure a replica set. [1][2]"
    u = analyze_usage(refusal, CHUNKS, cited=[1, 2], answerable=False)
    assert u["n_claims"] == 0 and u["n_unsupported_claims"] == 0
    assert u["n_cited"] == 2                                                 # citation use still counted
