"""Step 3: the baseline RAG chain.

    question -> retrieve top-k chunks -> number them as sources -> prompt
    -> structured answer {answer, cited sources, answerable} -> validate citations

Deliberately simple (one retrieval, one LLM call) so Steps 4-8 can measure it
and attribute every later improvement to a specific change.

Every call returns a RAGResult with what instrumentation needs: retrieved
chunks and similarities, which were cited, invalid citations, latency per
stage, token usage and cost.
"""
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from . import config
from .search import Hit, retrieve

SYSTEM_PROMPT = """You are a support assistant for developers using the FastAPI web framework.

Answer the question using ONLY the numbered sources below, which are excerpts from the official FastAPI documentation.

Rules:
- Cite every claim with the number of the source it comes from, like [1] or [2][3].
- Copy code from the sources exactly; do not invent APIs, parameters or defaults.
- Be concise: lead with the direct answer, then a short code example if one helps.
- If the sources do not contain the answer, set answerable to false and say briefly what is missing. Do not answer from general knowledge.

Sources:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])

NO_SOURCES_ANSWER = "I couldn't find anything relevant in the FastAPI documentation for this question."


class GeneratedAnswer(BaseModel):
    """Structured output requested from the model."""
    answer: str = Field(description="Markdown answer with inline [n] citations.")
    cited_sources: list[int] = Field(description="Numbers of the sources the answer relies on.")
    answerable: bool = Field(description="False if the sources do not contain the answer.")


@dataclass
class Citation:
    number: int
    chunk_id: str
    url: str
    similarity: float


@dataclass
class RAGResult:
    question: str
    answer: str
    answerable: bool
    citations: list[Citation]
    retrieved: list[Hit]
    invalid_citations: list[int] = field(default_factory=list)   # cited numbers that were never retrieved
    model: str = ""
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.generate_ms

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * config.CHAT_USD_PER_1M_INPUT
                + self.output_tokens * config.CHAT_USD_PER_1M_OUTPUT) / 1e6


def format_context(hits: list[Hit]) -> str:
    """Number each chunk so the model can cite it compactly. Each chunk's
    content already begins with its breadcrumb (e.g. FastAPI docs > Tutorial >
    Request Body > ...), which tells the model where it comes from."""
    return "\n\n".join(f"[{h.rank}] (id: {h.chunk_id})\n{h.document.page_content}" for h in hits)


def build_generator(llm: BaseChatModel) -> Runnable:
    """prompt | model with structured output. include_raw keeps the raw
    message so token usage can be read from it."""
    return PROMPT | llm.with_structured_output(GeneratedAnswer, include_raw=True)


def validate_citations(parsed: GeneratedAnswer, hits: list[Hit]) -> tuple[list[Citation], list[int]]:
    by_number = {h.rank: h for h in hits}
    citations, invalid = [], []
    for n in dict.fromkeys(parsed.cited_sources):          # dedupe, keep order
        hit = by_number.get(n)
        if hit is None:
            invalid.append(n)
            continue
        m = hit.document.metadata
        citations.append(Citation(n, m["chunk_id"], m["url"], round(hit.similarity, 4)))
    return citations, invalid


def answer_question(question: str, vectorstore, generator: Runnable, *, k: int = config.TOP_K,
                    model_name: str = config.CHAT_MODEL) -> RAGResult:
    started = time.perf_counter()
    hits = retrieve(vectorstore, question, k=k)
    retrieve_ms = (time.perf_counter() - started) * 1000

    if not hits:                       # nothing to ground on: don't spend an LLM call
        return RAGResult(question, NO_SOURCES_ANSWER, False, [], [], model=model_name,
                         retrieve_ms=retrieve_ms)

    started = time.perf_counter()
    out = generator.invoke({"question": question, "context": format_context(hits)})
    generate_ms = (time.perf_counter() - started) * 1000

    parsed: GeneratedAnswer | None = out.get("parsed")
    if parsed is None:
        raise RuntimeError(f"Model output could not be parsed: {out.get('parsing_error')}")
    usage = getattr(out.get("raw"), "usage_metadata", None) or {}
    citations, invalid = validate_citations(parsed, hits)

    return RAGResult(
        question=question,
        answer=parsed.answer,
        answerable=parsed.answerable,
        citations=citations,
        retrieved=hits,
        invalid_citations=invalid,
        model=model_name,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


def get_chat_model() -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.CHAT_MODEL, max_retries=3, timeout=60)
