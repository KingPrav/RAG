"""LLM-as-judge for open-ended answers, scored against nuggets.

Design (Husain; Lin et al. 2025):
  - the judge writes a critique *before* any verdict
  - each nugget gets support / partial / not_support (only support counts)
  - the overall verdict is binary pass/fail, never a 1-5 score
  - it judges correctness against the reference, not style or length
    (a concise correct answer must not lose to a long one: verbosity bias)

The judge model should be stronger than, and ideally from a different provider
than, the model being judged (self-preference bias). It is configurable via
JUDGE_MODEL ("provider:model", e.g. "openai:gpt-6.1-sol", "anthropic:...").
Agreement with a human is measured by `calibrate`.
"""
import hashlib
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from . import config

JUDGE_SYSTEM = """You grade answers from a FastAPI documentation assistant.

You get: a user question, a reference answer written by a FastAPI expert, a numbered list of
facts ("nuggets") that a good answer should contain, and the assistant's answer.

Grade ONLY factual content against the reference and nuggets:
- A nugget is "support" if the answer clearly states it (any wording, code counts).
  "partial" if it is only partly or vaguely stated. "not_support" if absent.
- Do not reward length, formatting or extra detail. A short correct answer is as good as a long one.
- Information beyond the reference is fine unless it contradicts the reference or is clearly wrong.

Verdict:
- "pass" only if every VITAL nugget is "support" AND the answer contains nothing that contradicts
  the reference.
- otherwise "fail".

Write the critique first (2-4 sentences: what is covered, what is missing or wrong), then the
per-nugget verdicts, then the overall verdict."""

JUDGE_HUMAN = """Question:
{question}

Reference answer:
{reference}

Nuggets:
{nuggets}

Assistant's answer:
{answer}"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])


class NuggetVerdict(BaseModel):
    nugget: int = Field(description="Nugget number as listed.")
    verdict: Literal["support", "partial", "not_support"]


class JudgeVerdict(BaseModel):
    critique: str = Field(description="2-4 sentences, written before deciding.")
    nuggets: list[NuggetVerdict]
    verdict: Literal["pass", "fail"]


def judge_prompt_version() -> str:
    return hashlib.sha256((JUDGE_SYSTEM + JUDGE_HUMAN).encode()).hexdigest()[:10]


def format_nuggets(nuggets: list[dict]) -> str:
    return "\n".join(f"{i}. [{'VITAL' if n['vital'] else 'okay'}] {n['fact']}"
                     for i, n in enumerate(nuggets, start=1))


def build_judge(llm: BaseChatModel) -> Runnable:
    return JUDGE_PROMPT | llm.with_structured_output(JudgeVerdict, include_raw=True)


def judge_answer(judge: Runnable, *, question: str, reference: str, nuggets: list[dict], answer: str) -> dict:
    out = judge.invoke({"question": question, "reference": reference,
                        "nuggets": format_nuggets(nuggets), "answer": answer})
    parsed: JudgeVerdict | None = out.get("parsed")
    if parsed is None:
        raise RuntimeError(f"Judge output could not be parsed: {out.get('parsing_error')}")
    by_number = {v.nugget: v.verdict for v in parsed.nuggets}
    # A nugget the judge skipped counts as not supported, rather than crashing the run.
    verdicts = [by_number.get(i, "not_support") for i in range(1, len(nuggets) + 1)]
    # Enforce the rule mechanically: a vital nugget not fully supported means fail,
    # whatever the judge's own verdict says.
    vital_ok = all(v == "support" for v, n in zip(verdicts, nuggets) if n["vital"])
    usage = getattr(out.get("raw"), "usage_metadata", None) or {}
    return {
        "critique": parsed.critique,
        "nugget_verdicts": verdicts,
        "judge_verdict": parsed.verdict,
        "passed": parsed.verdict == "pass" and vital_ok,
        "verdict_overridden": parsed.verdict == "pass" and not vital_ok,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def get_judge_model() -> BaseChatModel:
    from langchain.chat_models import init_chat_model
    return init_chat_model(config.JUDGE_MODEL, max_retries=3, timeout=90)
