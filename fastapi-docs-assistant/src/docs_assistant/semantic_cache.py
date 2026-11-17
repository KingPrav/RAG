"""Phase 4a: semantic answer cache.

A repeated or paraphrased question ("how do I add CORS?" / "enable CORS in
FastAPI") is answered from the cache instead of running rewrite -> search ->
rerank -> generate again: ~150 ms and one embedding call instead of ~7 s and
three LLM calls.

    lookup(question) -> embed -> cosine vs cached questions of THIS pipeline
                     -> best >= threshold ? cached answer : miss
    store(question, result)  only for clean answers (see is_cacheable)

Design choices:
- Scoped by a pipeline fingerprint (corpus commit, prompt, model, retrieval
  settings). Changing any of them makes old entries invisible, so a cache can
  never serve an answer the current pipeline wouldn't have produced.
- Exact-match first: a normalised-text hit is free and can't be a false hit.
- The threshold is a quality setting, calibrated by cost_eval (false-hit rate
  on near-duplicate questions that need different answers).
- SQLite for persistence + an in-memory numpy matrix for search. Thousands of
  cached questions take well under a millisecond to scan; a vector DB would be
  overkill until the cache is in the hundreds of thousands.
"""
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import config


def normalize(question: str) -> str:
    """Case and whitespace don't change a question's meaning."""
    return " ".join(question.lower().split()).rstrip("?!. ")


def fingerprint(pipeline: dict) -> str:
    """Stable short hash of everything that determines an answer. Two
    requests share cache entries only if their fingerprints match."""
    canonical = json.dumps(pipeline, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class CachedAnswer:
    """What a hit returns: enough to answer and cite, not the full context."""
    answer: str
    answerable: bool
    citations: list[dict]                 # [{number, chunk_id, url, similarity}]
    model: str
    prompt: str
    source_question: str                  # the question that produced this answer
    similarity: float = 1.0               # how close the new question was to it
    exact: bool = False                   # matched on normalised text, no embedding needed
    hits: int = 0                         # times this entry has been served
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        for transient in ("similarity", "exact", "hits"):     # per-lookup, not stored
            d.pop(transient)
        return json.dumps(d)


def is_cacheable(result) -> tuple[bool, str]:
    """Only clean answers are cached, because a cached mistake is repeated to
    every later paraphrase of the question. Returns (ok, reason) so a skip
    can be traced."""
    if not result.answerable:
        return False, "not answerable"          # may be a retrieval miss; retry it next time
    if not result.citations:
        return False, "no citations"
    if result.invalid_citations:
        return False, "invalid citations"
    if result.rewrite_error or result.rerank_error:
        return False, "degraded pipeline (rewrite/rerank error)"
    return True, ""


# ---------------------------------------------------------------- storage


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id            INTEGER PRIMARY KEY,
    fingerprint   TEXT NOT NULL,
    norm_question TEXT NOT NULL,
    question      TEXT NOT NULL,
    vector        BLOB NOT NULL,        -- float32, unit length
    payload       TEXT NOT NULL,        -- CachedAnswer JSON
    created_at    REAL NOT NULL,
    last_hit_at   REAL,
    hits          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS by_text ON entries (fingerprint, norm_question);
"""


def unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


class _Matrix:
    """All live vectors of one fingerprint, stacked, for a single matmul per lookup."""

    def __init__(self, ids: list[int], vectors: list[np.ndarray], dim: int | None = None):
        self.ids = np.asarray(ids, dtype=np.int64)
        width = vectors[0].shape[0] if vectors else (dim or 0)
        self.m = np.vstack(vectors) if vectors else np.empty((0, width), dtype=np.float32)

    def add(self, id_: int, v: np.ndarray) -> None:
        self.ids = np.append(self.ids, id_)
        self.m = np.vstack([self.m, v[None, :]]) if self.m.size else v[None, :]

    def best(self, q: np.ndarray) -> tuple[int, float] | None:
        if not len(self.ids):
            return None
        sims = self.m @ q                                   # cosine: everything is unit length
        i = int(np.argmax(sims))
        return int(self.ids[i]), float(sims[i])


# ---------------------------------------------------------------- the cache

class SemanticCache:
    """Thread-safe. `embed` maps text -> vector (the index's embedding model,
    so cached questions and the corpus live in the same space)."""

    def __init__(self, embed: Callable[[str], list[float]], *, path: Path = config.CACHE_DB,
                 threshold: float = config.CACHE_THRESHOLD, ttl_days: float = config.CACHE_TTL_DAYS,
                 max_entries: int = config.CACHE_MAX_ENTRIES, clock: Callable[[], float] = time.time):
        self.embed, self.threshold, self.max_entries, self.clock = embed, threshold, max_entries, clock
        self.ttl_s = ttl_days * 86400
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self._matrices: dict[str, _Matrix] = {}

    # -- internals
    def _live_cutoff(self) -> float:
        return self.clock() - self.ttl_s

    def _matrix(self, fp: str) -> _Matrix:
        """Lazily load one fingerprint's vectors; later stores append in place."""
        if fp not in self._matrices:
            rows = self.db.execute("SELECT id, vector FROM entries WHERE fingerprint = ? AND created_at >= ?",
                                   (fp, self._live_cutoff())).fetchall()
            self._matrices[fp] = _Matrix([r[0] for r in rows],
                                         [np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return self._matrices[fp]

    def _serve(self, row_id: int, similarity: float, exact: bool) -> CachedAnswer | None:
        row = self.db.execute("SELECT payload, hits, created_at FROM entries WHERE id = ?", (row_id,)).fetchone()
        if row is None or row[2] < self._live_cutoff():
            return None
        self.db.execute("UPDATE entries SET hits = hits + 1, last_hit_at = ? WHERE id = ?", (self.clock(), row_id))
        self.db.commit()
        return CachedAnswer(**json.loads(row[0]), similarity=similarity, exact=exact, hits=row[1] + 1)

    # -- public
    def lookup(self, question: str, fp: str, vector=None) -> tuple[CachedAnswer | None, dict]:
        """Returns (answer or None, info). info always carries the best
        similarity seen, so misses just below the threshold are visible in
        traces; that is the data for tuning it."""
        started = time.perf_counter()
        info = {"exact": False, "best_similarity": None, "nearest_question": None, "embed_ms": 0.0}
        with self.lock:
            row = self.db.execute("SELECT id FROM entries WHERE fingerprint = ? AND norm_question = ? "
                                  "AND created_at >= ? ORDER BY id DESC LIMIT 1",
                                  (fp, normalize(question), self._live_cutoff())).fetchone()
            if row:
                hit = self._serve(row[0], 1.0, exact=True)
                if hit:
                    info.update(exact=True, best_similarity=1.0, nearest_question=hit.source_question,
                                lookup_ms=(time.perf_counter() - started) * 1000)
                    return hit, info

        t = time.perf_counter()
        q = unit(vector if vector is not None else self.embed(question))
        info["embed_ms"] = (time.perf_counter() - t) * 1000
        with self.lock:
            best = self._matrix(fp).best(q)
            hit = None
            if best is not None:
                row_id, sim = best
                info["best_similarity"] = round(sim, 4)
                nearest = self.db.execute("SELECT question FROM entries WHERE id = ?", (row_id,)).fetchone()
                info["nearest_question"] = nearest[0] if nearest else None
                if sim >= self.threshold:
                    hit = self._serve(row_id, sim, exact=False)
        info["vector"] = q                       # reuse for store() on a miss: no second embedding call
        info["lookup_ms"] = (time.perf_counter() - started) * 1000
        return hit, info

    def store(self, question: str, fp: str, result, vector=None) -> tuple[bool, str]:
        ok, reason = is_cacheable(result)
        if not ok:
            return False, reason
        entry = CachedAnswer(
            answer=result.answer, answerable=result.answerable, model=result.model, prompt=result.prompt,
            citations=[{"number": c.number, "chunk_id": c.chunk_id, "url": c.url, "similarity": c.similarity}
                       for c in result.citations],
            source_question=question)
        v = unit(vector if vector is not None else self.embed(question))
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO entries (fingerprint, norm_question, question, vector, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fp, normalize(question), question, v.tobytes(), entry.to_json(), self.clock()))
            self.db.commit()
            self._matrix(fp).add(cur.lastrowid, v)
            self._evict()
        return True, ""

    def _evict(self) -> None:
        """Drop expired rows, then the least recently used beyond max_entries.
        Called under the lock; matrices are rebuilt lazily afterwards."""
        n_before = self.db.total_changes
        self.db.execute("DELETE FROM entries WHERE created_at < ?", (self._live_cutoff(),))
        self.db.execute("DELETE FROM entries WHERE id IN (SELECT id FROM entries "
                        "ORDER BY COALESCE(last_hit_at, created_at) DESC LIMIT -1 OFFSET ?)", (self.max_entries,))
        # Always commit: a DELETE that removes nothing still opens a write
        # transaction, which would lock out other processes sharing the file.
        self.db.commit()
        if self.db.total_changes != n_before:
            self._matrices.clear()

    def clear(self, fp: str | None = None) -> int:
        with self.lock:
            cur = self.db.execute("DELETE FROM entries" + (" WHERE fingerprint = ?" if fp else ""),
                                  (fp,) if fp else ())
            self.db.commit()
            self._matrices.clear()
            return cur.rowcount

    def stats(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT fingerprint, COUNT(*), SUM(hits) FROM entries GROUP BY fingerprint").fetchall()
        return {"entries": sum(r[1] for r in rows), "hits": sum(r[2] or 0 for r in rows),
                "by_fingerprint": {r[0]: {"entries": r[1], "hits": r[2] or 0} for r in rows}}
