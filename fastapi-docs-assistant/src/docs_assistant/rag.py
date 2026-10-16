"""Step 3: the baseline RAG chain.

    question -> retrieve top-k chunks -> number them as sources -> prompt
    -> structured answer {answer, cited sources, answerable} -> validate citations

Deliberately simple (one retrieval, one LLM call) so Steps 4-8 can measure it
and attribute every later improvement to a specific change.

Every call returns a RAGResult with what instrumentation needs: retrieved
chunks and similarities, which were cited, invalid citations, latency per
stage, token usage and cost.
"""
import hashlib
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from . import config
from .retrieval import RetrievalSettings, Retriever
from .search import Hit

SYSTEM_PROMPT_V1 = """You are a support assistant for developers using the FastAPI web framework.

Answer the question using ONLY the numbered sources below, which are excerpts from the official FastAPI documentation.

Rules:
- Cite every claim with the number of the source it comes from, like [1] or [2][3].
- Copy code from the sources exactly; do not invent APIs, parameters or defaults.
- Be concise: lead with the direct answer, then a short code example if one helps.
- If the sources do not contain the answer, set answerable to false and say briefly what is missing. Do not answer from general knowledge.

Sources:
{context}"""

# v2 (Step 8 follow-up): the v1 "be concise" rule made answers drop facts that were
# in the sources (Ragas factual correctness 0.73; all 3 rw-n1 regressions were
# incomplete answers with the right section retrieved), and refusing whenever the
# answer was partial drove false refusals on the test split.
SYSTEM_PROMPT_V2 = """You are a support assistant for developers using the FastAPI web framework.

Answer the question using ONLY the numbered sources below, which are excerpts from the official FastAPI documentation.

Rules:
- Cite every claim with the number of the source it comes from, like [1] or [2][3].
- Copy code from the sources exactly; do not invent APIs, parameters or defaults.
- Answer completely, then concisely: lead with the direct answer, then include every fact from the sources needed to fully answer it, such as how it works, defaults, required parameters or installs, and important caveats. Add a short code example if one helps. No filler.
- If the sources answer only part of the question, answer that part with citations and say briefly what the sources don't cover; keep answerable true.
- Set answerable to false only if nothing in the sources answers the question. Do not answer from general knowledge.

Sources:
{context}"""

# v3: v2's completeness rule fixed incomplete answers (multi-007, term-001, vocab-015),
# but "answer the supported part ... keep answerable true" made the model mark
# out-of-scope and trap questions answerable whenever *related* text existed
# (5 of 6 unanswerable dev items broken, though the answer text itself admitted
# the gap). v3 keeps completeness and defines answerable precisely: the sources
# must contain the specific thing asked for.
SYSTEM_PROMPT_V3 = """You are a support assistant for developers using the FastAPI web framework.

Answer the question using ONLY the numbered sources below, which are excerpts from the official FastAPI documentation.

Rules:
- Cite every claim with the number of the source it comes from, like [1] or [2][3].
- Copy code from the sources exactly; do not invent APIs, parameters or defaults.
- Answer completely, then concisely: lead with the direct answer, then include every fact from the sources needed to fully answer it, such as how it works, defaults, required parameters or installs, and important caveats. Add a short code example if one helps. No filler.
- Decide answerable by the specific thing the question asks for. If the sources do not contain it (for example a version number, a default value, or a tool, library or integration they never mention), set answerable to false, even if they contain related information; say briefly what is missing.
- If the sources answer the main question but not a minor detail, answer it, keep answerable true, and note the missing detail.
- Do not answer from general knowledge.

Sources:
{context}"""

PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3}
SYSTEM_PROMPT = PROMPTS[config.ANSWER_PROMPT]           # the active prompt (config / .env)


def build_prompt(version: str = config.ANSWER_PROMPT) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", PROMPTS[version]), ("human", "{question}")])


def prompt_hash(version: str = config.ANSWER_PROMPT) -> str:
    """Short hash of a system prompt, so every trace and eval run says which prompt made it."""
    return hashlib.sha256(PROMPTS[version].encode()).hexdigest()[:10]


PROMPT = build_prompt()

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
    input_tokens: int = 0              # answer call + query rewrite call (same model)
    output_tokens: int = 0
    queries: list[str] = field(default_factory=list)      # what was searched (original first)
    rewrite_ms: float = 0.0                                # part of retrieve_ms
    rewrite_error: str | None = None
    prompt: str = config.ANSWER_PROMPT                     # prompt id (v1/v2) that produced the answer
    rerank_ms: float = 0.0                                 # part of retrieve_ms
    rerank_error: str | None = None
    rewrite_cached: bool = False

    @property
    def search_hits(self) -> list[Hit]:
        return [h for h in self.retrieved if h.source == "search"]

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


def build_generator(llm: BaseChatModel, prompt: str = config.ANSWER_PROMPT) -> Runnable:
    """prompt | model with structured output. include_raw keeps the raw
    message so token usage can be read from it."""
    return build_prompt(prompt) | llm.with_structured_output(GeneratedAnswer, include_raw=True)


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
                    model_name: str = config.CHAT_MODEL, retriever: Retriever | None = None,
                    prompt: str = config.ANSWER_PROMPT) -> RAGResult:
    """Without a retriever this is the Step 3 baseline: plain top-k vector search."""
    retriever = retriever or Retriever(vectorstore, RetrievalSettings(k=k, neighbors=0, rewrite=False, rerank="none"))
    started = time.perf_counter()
    retrieved = retriever(question)
    hits = retrieved.hits
    retrieve_ms = (time.perf_counter() - started) * 1000
    common = dict(model=model_name, retrieve_ms=retrieve_ms, queries=retrieved.queries, prompt=prompt,
                  rewrite_ms=retrieved.rewrite_ms, rewrite_error=retrieved.rewrite_error,
                  rerank_ms=retrieved.rerank_ms, rerank_error=retrieved.rerank_error,
                  rewrite_cached=retrieved.rewrite_cached)

    if not hits:                       # nothing to ground on: don't spend an LLM call
        return RAGResult(question, NO_SOURCES_ANSWER, False, [], [], input_tokens=retrieved.input_tokens,
                         output_tokens=retrieved.output_tokens, **common)

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
        generate_ms=generate_ms,
        input_tokens=usage.get("input_tokens", 0) + retrieved.input_tokens,
        output_tokens=usage.get("output_tokens", 0) + retrieved.output_tokens,
        **common,
    )


def get_chat_model(reasoning_effort: str | None = None) -> BaseChatModel:
    """The cheap chat model. A reasoning effort (e.g. "low") is set only for
    simple auxiliary calls (rewrite, rerank) where thinking time is wasted."""
    from langchain_openai import ChatOpenAI
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    return ChatOpenAI(model=config.CHAT_MODEL, max_retries=3, timeout=60, **extra)
