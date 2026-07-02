"""Repo embeddings + semantic retrieval (#6).

Complements :mod:`harness.impact`, which builds an AST-based symbol
graph: this module builds a *semantic* index of code chunks so the
planner can ask "what other files in this repo discuss authentication
session storage?" and get back results that share *meaning*, not just
identifier names.

v1 backends
===========

- **TfidfBackend** (default, zero-dep, deterministic, pure Python).
  Builds a TF-IDF vocabulary over all indexed chunks; per-chunk vectors
  are stored as JSON-encoded sparse maps. Query path tokenises the
  prompt with the same rules, scores via cosine over the sparse vectors,
  returns top-K. Suitable for medium repos (~10 K chunks) without any
  vector database; for larger repos swap in the embeddings backend.

- **OpenAIEmbeddingsBackend** (opt-in via ``repo_index.backend =
  "openai_embeddings"``). Calls ``text-embedding-3-small`` over the
  existing httpx infrastructure. Vectors stored as JSON arrays of 1536
  floats. Requires ``OPENAI_API_KEY`` in env; degrades to TF-IDF
  on init failure so a missing key never breaks startup.

Both backends share:
- A SQLite store next to the checkpoint DB (``~/.harness/repo_index.db``).
- The chunker (line-window with overlap; configurable).
- The query path (cosine + top-K; filter by minimum score).
- The CLI surface (``teane index build / status / clear``).

The planner is unchanged when ``repo_index.enabled=false`` (default).
When enabled, ``planning_node`` calls :func:`query_top_chunks` once for
the initial user prompt and injects up to ``inject_max_bytes`` of
matching content as an extra system message — same shape as the
per-repo memory injection.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


_DEFAULT_INDEX_DIR = "~/.harness/repo_index"
_DEFAULT_TOP_K = 5
_DEFAULT_CHUNK_LINES = 200
_DEFAULT_CHUNK_OVERLAP = 20
_DEFAULT_INJECT_MAX_BYTES = 4_000
_DEFAULT_EXCLUDE_GLOBS = (
    ".git/**", "node_modules/**", "**/__pycache__/**", "**/.venv/**",
    "**/venv/**", "**/dist/**", "**/build/**", "**/target/**",
    "**/.tox/**", "**/coverage/**", ".pytest_cache/**", ".ruff_cache/**",
    "**/*.min.js", "**/*.min.css", "**/*.lock",
    "**/package-lock.json", "**/yarn.lock", "**/pnpm-lock.yaml",
)
_DEFAULT_TEXT_EXTENSIONS = (
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".java",
    ".sh", ".bash", ".zsh",
    ".html", ".css", ".scss",
    ".md", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml",
    ".sql",
)
_DEFAULT_MAX_FILE_BYTES = 250_000  # skip giant generated files


@dataclass
class RepoIndexConfig:
    enabled: bool = False
    backend: str = "tfidf"  # "tfidf" | "openai_embeddings"
    top_k: int = _DEFAULT_TOP_K
    chunk_lines: int = _DEFAULT_CHUNK_LINES
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP
    inject_max_bytes: int = _DEFAULT_INJECT_MAX_BYTES
    index_dir: str = _DEFAULT_INDEX_DIR
    exclude_globs: list[str] = field(default_factory=lambda: list(_DEFAULT_EXCLUDE_GLOBS))
    text_extensions: list[str] = field(default_factory=lambda: list(_DEFAULT_TEXT_EXTENSIONS))
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES
    # OpenAI backend params
    openai_model: str = "text-embedding-3-small"
    openai_api_base: str = "https://api.openai.com/v1"
    # TLS verification toggle for the embeddings backend. Defaults True;
    # callers behind an MITM corporate proxy can flip via config.
    ssl_verify: bool = True

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "RepoIndexConfig":
        section = ((config or {}).get("repo_index") or {})
        return cls(
            enabled=bool(section.get("enabled", False)),
            backend=str(section.get("backend", "tfidf")),
            top_k=int(section.get("top_k", _DEFAULT_TOP_K)),
            chunk_lines=int(section.get("chunk_lines", _DEFAULT_CHUNK_LINES)),
            chunk_overlap=int(section.get("chunk_overlap", _DEFAULT_CHUNK_OVERLAP)),
            inject_max_bytes=int(section.get("inject_max_bytes", _DEFAULT_INJECT_MAX_BYTES)),
            index_dir=str(section.get("index_dir", _DEFAULT_INDEX_DIR)),
            exclude_globs=list(section.get("exclude_globs") or _DEFAULT_EXCLUDE_GLOBS),
            text_extensions=list(section.get("text_extensions") or _DEFAULT_TEXT_EXTENSIONS),
            max_file_bytes=int(section.get("max_file_bytes", _DEFAULT_MAX_FILE_BYTES)),
            openai_model=str(section.get("openai_model", "text-embedding-3-small")),
            openai_api_base=str(section.get("openai_api_base", "https://api.openai.com/v1")),
            ssl_verify=bool(section.get("ssl_verify", True)),
        )


# ---------------------------------------------------------------------------
# 1. Data shape
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    file_path: str  # workspace-relative, posix-style
    chunk_index: int
    content: str

    def file_sha(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:32]


@dataclass
class RetrievalResult:
    file_path: str
    chunk_index: int
    score: float
    content: str

    def render(self, *, content_max_lines: int = 40) -> str:
        body = self.content.splitlines()
        if len(body) > content_max_lines:
            body = body[:content_max_lines] + ["... (truncated)"]
        return (
            f"### {self.file_path} (chunk {self.chunk_index}, score {self.score:.3f})\n"
            "```\n" + "\n".join(body) + "\n```\n"
        )


# ---------------------------------------------------------------------------
# 2. Chunker — file walker + line windowing
# ---------------------------------------------------------------------------

class Chunker:
    def __init__(self, cfg: RepoIndexConfig):
        self.cfg = cfg

    def walk(self, workspace_path: str) -> Iterable[str]:
        """Yield posix-relative file paths inside ``workspace_path`` that
        pass extension + exclude-glob + size filters. Deterministic
        (sorted) traversal so build runs are reproducible."""
        ws_real = os.path.realpath(workspace_path)
        for root, dirs, files in os.walk(ws_real):
            # Skip any directory whose POSIX-relative path matches the
            # exclude globs — keeps the walk fast (we don't descend).
            dirs.sort()
            kept_dirs: list[str] = []
            for d in dirs:
                rel = os.path.relpath(os.path.join(root, d), ws_real).replace(os.sep, "/")
                if not self._matches_any_exclude(rel + "/"):
                    kept_dirs.append(d)
            dirs[:] = kept_dirs
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(root, f), ws_real).replace(os.sep, "/")
                if not self._has_text_extension(f):
                    continue
                if self._matches_any_exclude(rel):
                    continue
                abs_path = os.path.join(root, f)
                try:
                    if os.path.getsize(abs_path) > self.cfg.max_file_bytes:
                        continue
                except OSError:
                    continue
                yield rel

    def _has_text_extension(self, filename: str) -> bool:
        lower = filename.lower()
        return any(lower.endswith(ext) for ext in self.cfg.text_extensions)

    def _matches_any_exclude(self, rel_path: str) -> bool:
        for pattern in self.cfg.exclude_globs:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

    def chunks_for_file(self, workspace_path: str, rel_path: str) -> list[Chunk]:
        abs_path = os.path.join(workspace_path, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as exc:
            logger.debug("[repo_index] cannot read %s: %s", abs_path, exc)
            return []
        lines = text.splitlines()
        if not lines:
            return []
        window = max(20, self.cfg.chunk_lines)
        overlap = max(0, min(self.cfg.chunk_overlap, window - 1))
        step = window - overlap
        out: list[Chunk] = []
        idx = 0
        i = 0
        while i < len(lines):
            chunk_lines = lines[i : i + window]
            content = "\n".join(chunk_lines)
            out.append(Chunk(file_path=rel_path, chunk_index=idx, content=content))
            idx += 1
            if i + window >= len(lines):
                break
            i += step
        return out


# ---------------------------------------------------------------------------
# 3. Tokeniser shared by TF-IDF + (future) hybrid backends
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,}")


def _tokenize(text: str) -> list[str]:
    """Identifier-friendly tokeniser. Splits on non-identifier boundaries
    and also splits CamelCase / snake_case into sub-tokens so a search
    for ``parseRequest`` matches a chunk containing ``ParseRequestBody``.
    """
    raw = _TOKEN_RE.findall(text)
    out: list[str] = []
    for tok in raw:
        # snake_case
        parts = tok.split("_")
        for p in parts:
            if not p:
                continue
            # CamelCase
            sub = re.findall(r"[A-Z][a-z]+|[A-Z]+(?![a-z])|[a-z]+|\d+", p)
            for s in sub:
                s_lower = s.lower()
                if len(s_lower) >= 2:
                    out.append(s_lower)
            if len(p) >= 2:
                out.append(p.lower())  # also keep the un-split token
    return out


# ---------------------------------------------------------------------------
# 4. Backend ABC + TF-IDF impl
# ---------------------------------------------------------------------------

class IndexBackend:
    name: str = "base"

    def fit_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Compute and return a JSON-encoded vector per chunk, in the
        same order as ``chunks``."""
        raise NotImplementedError

    def vectorize_query(self, query: str) -> str:
        """Encode a search query into the same vector space as
        ``fit_chunks`` output."""
        raise NotImplementedError

    def cosine(self, a_json: str, b_json: str) -> float:
        raise NotImplementedError


class TfidfBackend(IndexBackend):
    """Pure-Python TF-IDF backend.

    Vocabulary is built once per ``fit_chunks`` call. To search after
    indexing, the backend serialises the vocabulary's IDF weights into
    the SQLite store so ``vectorize_query`` can use the same IDF when
    the process restarts.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._vocab: dict[str, int] = {}

    # --- Fit -----------------------------------------------------------

    def fit_chunks(self, chunks: list[Chunk]) -> list[str]:
        # 1. Compute DF
        df: Counter[str] = Counter()
        tokenised: list[list[str]] = []
        for ch in chunks:
            toks = _tokenize(ch.content)
            tokenised.append(toks)
            for t in set(toks):
                df[t] += 1
        n = max(1, len(chunks))
        # 2. IDF = log((n + 1) / (df + 1)) + 1 (smooth)
        idf: dict[str, float] = {
            term: math.log((n + 1) / (df_count + 1)) + 1.0
            for term, df_count in df.items()
        }
        self._idf = idf
        self._vocab = {term: i for i, term in enumerate(sorted(idf.keys()))}
        # 3. Encode each chunk as sparse {term_idx: tfidf_weight}, L2-normalised
        vectors_json: list[str] = []
        for toks in tokenised:
            tf: Counter[str] = Counter(toks)
            length_norm = math.sqrt(sum(c * c for c in tf.values())) or 1.0
            sparse: dict[str, float] = {}
            for term, count in tf.items():
                w = (count / length_norm) * idf.get(term, 0.0)
                if w > 0:
                    sparse[term] = w
            # L2 normalisation over the term-weights
            mag = math.sqrt(sum(v * v for v in sparse.values())) or 1.0
            sparse = {t: v / mag for t, v in sparse.items()}
            vectors_json.append(json.dumps(sparse, ensure_ascii=False))
        return vectors_json

    # --- Query ---------------------------------------------------------

    def load_idf(self, idf_json: str) -> None:
        try:
            self._idf = json.loads(idf_json) if idf_json else {}
        except json.JSONDecodeError:
            self._idf = {}

    def idf_json(self) -> str:
        return json.dumps(self._idf, ensure_ascii=False)

    def vectorize_query(self, query: str) -> str:
        toks = _tokenize(query)
        if not toks:
            return "{}"
        tf: Counter[str] = Counter(toks)
        length_norm = math.sqrt(sum(c * c for c in tf.values())) or 1.0
        sparse: dict[str, float] = {}
        for term, count in tf.items():
            w = (count / length_norm) * self._idf.get(term, 0.0)
            if w > 0:
                sparse[term] = w
        mag = math.sqrt(sum(v * v for v in sparse.values())) or 1.0
        sparse = {t: v / mag for t, v in sparse.items()}
        return json.dumps(sparse, ensure_ascii=False)

    def cosine(self, a_json: str, b_json: str) -> float:
        try:
            a = json.loads(a_json) if a_json else {}
            b = json.loads(b_json) if b_json else {}
        except json.JSONDecodeError:
            return 0.0
        if not a or not b:
            return 0.0
        # Iterate over the smaller dict
        if len(a) > len(b):
            a, b = b, a
        return float(sum(a.get(k, 0.0) * v for k, v in b.items() if k in a))


# ---------------------------------------------------------------------------
# 5. OpenAI embeddings backend (opt-in)
# ---------------------------------------------------------------------------

def _track_embedding_usage(response_json: dict, openai_model: str) -> None:
    """Account a successful /v1/embeddings response into the Gateway's
    session tracker so end-of-run / status / metrics / dashboard all see
    the cost. Best-effort: if the singleton isn't injected (e.g., a test
    fixture or a cold ``teane index`` invocation that ran before the
    gateway was wired in) or the response is malformed, silently no-op.
    The index build must NEVER fail because we couldn't account it.
    """
    try:
        usage = response_json.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if prompt_tokens <= 0:
            return
        from harness.graph import get_gateway
        gw = get_gateway()
        if gw is None:
            return
        gw.track_embedding_call(f"openai:{openai_model}", prompt_tokens)
    except Exception as exc:  # noqa: BLE001 — accounting is best-effort
        logger.debug("[repo_index] Embedding cost tracking skipped: %s", exc)


class OpenAIEmbeddingsBackend(IndexBackend):
    """Calls ``/v1/embeddings`` with batches of chunk contents. Vectors
    are dense 1536-float lists for ``text-embedding-3-small``. Cosine
    similarity uses pure Python (no numpy) so we don't pull a heavy
    dependency for what's a few thousand dot products.

    Falls back to TF-IDF at construction time when ``OPENAI_API_KEY``
    is missing — a missing key must NEVER break ``teane index``.
    Callers check :attr:`available`.
    """

    name = "openai_embeddings"
    _BATCH_SIZE = 32

    def __init__(self, cfg: RepoIndexConfig):
        self.cfg = cfg
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.available = bool(self.api_key)

    def fit_chunks(self, chunks: list[Chunk]) -> list[str]:
        if not self.available:
            raise RuntimeError(
                "OPENAI_API_KEY missing — cannot use openai_embeddings "
                "backend. Set the env var or switch repo_index.backend to "
                "tfidf."
            )
        import httpx
        import time as _time
        vectors_json: list[str] = []
        # Audit §4.10: filter out empty chunks (otherwise OpenAI rejects
        # the whole batch with "input is invalid"), bound retries with
        # a backoff that tolerates transient 429/5xx, and honour the
        # operator's ssl_verify setting.
        verify = bool(getattr(self.cfg, "ssl_verify", True))
        with httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
            base_url=self.cfg.openai_api_base,
            verify=verify,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            for batch_start in range(0, len(chunks), self._BATCH_SIZE):
                batch = chunks[batch_start : batch_start + self._BATCH_SIZE]
                # Track which batch positions were blank so we can place
                # API embeddings back at their ORIGINAL positions. The
                # previous "filter inputs, append result, backfill at end"
                # shape misaligned vectors when blank chunks were
                # interleaved (not just trailing): vec_C would land at
                # batch slot 1 (chunk B's slot) and chunk C's slot got
                # the "[]" backfill. Result: every chunk after the first
                # blank held the wrong vector and retrieval scored the
                # wrong neighbours.
                non_blank_positions: list[int] = []
                inputs: list[str] = []
                for pos, c in enumerate(batch):
                    if c.content and c.content.strip():
                        non_blank_positions.append(pos)
                        inputs.append(c.content[:8000])

                # Pre-allocate this batch's slot in the output, blank by
                # default; we'll overwrite the non-blank positions below.
                batch_vectors: list[str] = ["[]"] * len(batch)
                if not inputs:
                    vectors_json.extend(batch_vectors)
                    continue
                payload = {"model": self.cfg.openai_model, "input": inputs}
                # Bounded retry loop (audit §4.10) — small embedding
                # request set, so a flat 4-attempt schedule with
                # exponential backoff is enough.
                last_exc: Optional[Exception] = None
                response = None
                for attempt in range(4):
                    try:
                        response = client.post("/embeddings", json=payload)
                        response.raise_for_status()
                        break
                    except httpx.HTTPStatusError as exc:
                        last_exc = exc
                        sc = exc.response.status_code
                        if sc != 429 and sc < 500:
                            raise
                    except (httpx.ConnectError, httpx.ReadError,
                            httpx.TimeoutException) as exc:
                        last_exc = exc
                    _time.sleep(min(60.0, 1.0 * (2 ** attempt)))
                else:
                    # Retries exhausted without a successful break. Raise
                    # whatever we caught last; if we never caught anything
                    # (defensive), raise a synthetic error rather than
                    # silently emitting blank embeddings for this batch.
                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError(
                        "OpenAI embeddings: retries exhausted with no captured exception."
                    )
                # response must be non-None here (else branch above would have raised).
                assert response is not None
                data = response.json()
                items = data.get("data", []) or []
                _track_embedding_usage(data, self.cfg.openai_model)
                # Bind each API embedding to its ORIGINAL batch position.
                # If the API returned fewer items than inputs (shouldn't
                # happen but defend), trailing positions stay blank.
                for src_idx, item in enumerate(items):
                    if src_idx >= len(non_blank_positions):
                        break
                    dest_pos = non_blank_positions[src_idx]
                    batch_vectors[dest_pos] = json.dumps(
                        item.get("embedding", []), ensure_ascii=False,
                    )
                vectors_json.extend(batch_vectors)
        return vectors_json

    def vectorize_query(self, query: str) -> str:
        if not self.available:
            return "[]"
        import httpx
        payload = {"model": self.cfg.openai_model, "input": query[:8000]}
        with httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
            base_url=self.cfg.openai_api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            response = client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            _track_embedding_usage(data, self.cfg.openai_model)
            embedding = data.get("data", [{}])[0].get("embedding", [])
        return json.dumps(embedding, ensure_ascii=False)

    def cosine(self, a_json: str, b_json: str) -> float:
        try:
            a = json.loads(a_json) if a_json else []
            b = json.loads(b_json) if b_json else []
        except json.JSONDecodeError:
            return 0.0
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        # OpenAI embeddings are pre-normalised, so dot == cosine.
        return float(dot)


def make_backend(cfg: RepoIndexConfig) -> IndexBackend:
    name = (cfg.backend or "tfidf").lower()
    if name in ("tfidf", "default"):
        return TfidfBackend()
    if name in ("openai_embeddings", "openai", "embeddings"):
        backend = OpenAIEmbeddingsBackend(cfg)
        if not backend.available:
            logger.warning(
                "[repo_index] openai_embeddings backend unavailable "
                "(OPENAI_API_KEY missing); falling back to tfidf."
            )
            return TfidfBackend()
        return backend
    raise ValueError(f"unknown repo_index.backend: {cfg.backend!r}")


# ---------------------------------------------------------------------------
# 6. Storage  ---  SQLite next to the checkpoint DB
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS repo_meta (
    workspace_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    idf_json TEXT,
    built_at TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS repo_chunks (
    workspace_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    file_sha TEXT NOT NULL,
    content TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    PRIMARY KEY (workspace_id, file_path, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_repo_chunks_ws ON repo_chunks (workspace_id);
"""


def _workspace_id(workspace_path: str) -> str:
    """16-char hex SHA-256 of the absolute workspace path."""
    return hashlib.sha256(
        os.path.abspath(workspace_path).encode("utf-8"),
    ).hexdigest()[:16]


def _db_path(cfg: RepoIndexConfig) -> str:
    base = os.path.expanduser(cfg.index_dir)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "repo_index.db")


def _open_db(cfg: RepoIndexConfig) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(cfg), timeout=10.0)
    # WAL + busy_timeout mirror the harness.storage / harness.web_state
    # pragma set so a concurrent planner + update_index_for_files run
    # against different workspaces (or the same one across two teane
    # processes) doesn't hit "database is locked".
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
    except sqlite3.DatabaseError:
        # On some platforms/filesystems WAL is unsupported (network
        # mounts, ramdisks). Fall back silently — correctness still
        # holds with the default journal.
        pass
    conn.executescript(_SCHEMA_SQL)
    return conn


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class IndexStats:
    workspace_id: str
    backend: str
    chunk_count: int
    file_count: int
    built_at: str


def build_index(
    workspace_path: str,
    cfg: Optional[RepoIndexConfig] = None,
    *,
    progress: Optional[Any] = None,
) -> IndexStats:
    """Walk the workspace, chunk every text file, fit the configured
    backend, and persist to SQLite. Replaces any prior index for the
    same workspace.

    Returns an :class:`IndexStats` summary.
    """
    cfg = cfg or RepoIndexConfig()
    backend = make_backend(cfg)
    chunker = Chunker(cfg)

    workspace_id = _workspace_id(workspace_path)
    all_chunks: list[Chunk] = []
    file_set: set[str] = set()
    for rel in chunker.walk(workspace_path):
        for ch in chunker.chunks_for_file(workspace_path, rel):
            all_chunks.append(ch)
            file_set.add(ch.file_path)
            if progress is not None and len(all_chunks) % 50 == 0:
                progress(len(all_chunks))

    vectors = backend.fit_chunks(all_chunks) if all_chunks else []

    conn = _open_db(cfg)
    try:
        with conn:
            conn.execute(
                "DELETE FROM repo_chunks WHERE workspace_id = ?",
                (workspace_id,),
            )
            conn.executemany(
                "INSERT INTO repo_chunks (workspace_id, file_path, chunk_index, "
                "file_sha, content, vector_json) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        workspace_id, ch.file_path, ch.chunk_index,
                        ch.file_sha(), ch.content, vec,
                    )
                    for ch, vec in zip(all_chunks, vectors)
                ],
            )
            idf_json = (
                backend.idf_json() if isinstance(backend, TfidfBackend) else ""
            )
            now = _now_iso()
            conn.execute(
                "REPLACE INTO repo_meta (workspace_id, backend, idf_json, "
                "built_at, chunk_count) VALUES (?, ?, ?, ?, ?)",
                (workspace_id, backend.name, idf_json, now, len(all_chunks)),
            )
    finally:
        conn.close()
    return IndexStats(
        workspace_id=workspace_id,
        backend=backend.name,
        chunk_count=len(all_chunks),
        file_count=len(file_set),
        built_at=_now_iso(),
    )


def get_stats(
    workspace_path: str, cfg: Optional[RepoIndexConfig] = None,
) -> Optional[IndexStats]:
    """Return the index summary for ``workspace_path`` if one has been
    built; ``None`` otherwise."""
    cfg = cfg or RepoIndexConfig()
    workspace_id = _workspace_id(workspace_path)
    conn = _open_db(cfg)
    try:
        row = conn.execute(
            "SELECT backend, built_at, chunk_count FROM repo_meta WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        if row is None:
            return None
        files = conn.execute(
            "SELECT COUNT(DISTINCT file_path) FROM repo_chunks WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        return IndexStats(
            workspace_id=workspace_id,
            backend=row[0],
            chunk_count=int(row[2]),
            file_count=int(files[0] if files else 0),
            built_at=row[1],
        )
    finally:
        conn.close()


def purge_workspace(
    workspace_path: str, cfg: Optional[RepoIndexConfig] = None,
) -> tuple[int, int]:
    """Delete every ``repo_meta`` and ``repo_chunks`` row tied to
    ``workspace_path``. Used by ``--new-build`` so the next session
    starts with no stale index left over from prior runs.

    Returns ``(meta_rows_deleted, chunk_rows_deleted)``. Best-effort:
    a DB open failure logs and returns ``(0, 0)`` so the broader
    new-build flow isn't aborted by a missing or corrupt index DB.
    """
    cfg = cfg or RepoIndexConfig()
    db_path = _db_path(cfg)
    if not os.path.isfile(db_path):
        return 0, 0
    workspace_id = _workspace_id(workspace_path)
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "[repo_index] Could not open index DB %s to purge workspace %s: %s",
            db_path, workspace_path, exc,
        )
        return 0, 0
    try:
        meta_cur = conn.execute(
            "DELETE FROM repo_meta WHERE workspace_id = ?", (workspace_id,),
        )
        chunk_cur = conn.execute(
            "DELETE FROM repo_chunks WHERE workspace_id = ?", (workspace_id,),
        )
        conn.commit()
        meta_n = meta_cur.rowcount or 0
        chunk_n = chunk_cur.rowcount or 0
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "[repo_index] Purge failed for workspace %s: %s",
            workspace_path, exc,
        )
        return 0, 0
    finally:
        conn.close()
    if meta_n or chunk_n:
        logger.info(
            "[repo_index] Purged index for workspace %s "
            "(meta=%d, chunks=%d).",
            workspace_path, meta_n, chunk_n,
        )
    return meta_n, chunk_n


def purge_all(cfg: Optional[RepoIndexConfig] = None) -> tuple[int, int]:
    """Delete every ``repo_meta`` and ``repo_chunks`` row across ALL
    workspaces. Used by ``teane purge --all``. Preserves the DB file so
    the schema is still ready for the next index build.

    Returns ``(meta_rows_deleted, chunk_rows_deleted)``. Best-effort: a
    missing or unreadable DB logs and returns ``(0, 0)``.
    """
    cfg = cfg or RepoIndexConfig()
    db_path = _db_path(cfg)
    if not os.path.isfile(db_path):
        return 0, 0
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "[repo_index] Could not open index DB %s for global purge: %s",
            db_path, exc,
        )
        return 0, 0
    try:
        meta_cur = conn.execute("DELETE FROM repo_meta")
        chunk_cur = conn.execute("DELETE FROM repo_chunks")
        conn.commit()
        meta_n = meta_cur.rowcount or 0
        chunk_n = chunk_cur.rowcount or 0
    except sqlite3.DatabaseError as exc:
        logger.warning("[repo_index] Global purge failed: %s", exc)
        return 0, 0
    finally:
        conn.close()
    if meta_n or chunk_n:
        logger.info(
            "[repo_index] Global purge removed meta=%d, chunks=%d.",
            meta_n, chunk_n,
        )
    return meta_n, chunk_n


def update_index_for_files(
    workspace_path: str,
    modified_files: Iterable[str],
    cfg: Optional[RepoIndexConfig] = None,
) -> int:
    """Re-chunk and re-index only the listed files. Audit §6.5.

    For TF-IDF this triggers a full rebuild because the IDF score
    depends on the whole-corpus document frequency, so per-file
    re-vectorisation would silently use stale weights. For the
    embedding backend only the modified files are re-chunked and
    re-vectorised; the rest of the index is left intact.

    Best-effort: any failure logs and returns 0 so the caller (the
    graph nodes that just landed patches) never crashes on an index
    update path. Returns the number of chunks refreshed.
    """
    cfg = cfg or RepoIndexConfig()
    file_list = [str(p) for p in modified_files if p]
    if not file_list:
        return 0
    workspace_id = _workspace_id(workspace_path)
    try:
        existing = get_stats(workspace_path, cfg)
    except Exception:  # noqa: BLE001
        existing = None
    if existing is None:
        # No prior index — nothing to update incrementally.
        return 0
    if existing.backend == "tfidf":
        # IDF is corpus-wide; correct re-fit needs the full corpus.
        try:
            stats = build_index(workspace_path, cfg)
            return stats.chunk_count
        except Exception as exc:  # noqa: BLE001
            logger.warning("[repo_index] tfidf rebuild failed: %s", exc)
            return 0
    if existing.backend != "openai_embeddings":
        return 0
    # Embedding backend: rebuild just the listed files.
    try:
        chunker = Chunker(cfg)
        backend = make_backend(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[repo_index] backend init failed: %s", exc)
        return 0
    new_chunks: list[Chunk] = []
    for rel in file_list:
        try:
            new_chunks.extend(chunker.chunks_for_file(workspace_path, rel))
        except Exception:  # noqa: BLE001 — skip unreadable files
            continue
    if not new_chunks:
        return 0
    try:
        vectors = backend.fit_chunks(new_chunks)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[repo_index] re-vectorise failed: %s", exc)
        return 0
    conn = _open_db(cfg)
    try:
        with conn:
            placeholders = ",".join("?" for _ in file_list)
            conn.execute(
                f"DELETE FROM repo_chunks WHERE workspace_id = ? AND "
                f"file_path IN ({placeholders})",
                (workspace_id, *file_list),
            )
            conn.executemany(
                "INSERT INTO repo_chunks (workspace_id, file_path, chunk_index, "
                "file_sha, content, vector_json) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        workspace_id, ch.file_path, ch.chunk_index,
                        ch.file_sha(), ch.content, vec,
                    )
                    for ch, vec in zip(new_chunks, vectors)
                ],
            )
            conn.execute(
                "UPDATE repo_meta SET built_at = ?, chunk_count = "
                "(SELECT COUNT(*) FROM repo_chunks WHERE workspace_id = ?) "
                "WHERE workspace_id = ?",
                (_now_iso(), workspace_id, workspace_id),
            )
    finally:
        conn.close()
    return len(new_chunks)


def clear_index(
    workspace_path: str, cfg: Optional[RepoIndexConfig] = None,
) -> int:
    """Wipe the index for ``workspace_path``. Returns the number of
    chunk rows deleted."""
    cfg = cfg or RepoIndexConfig()
    workspace_id = _workspace_id(workspace_path)
    conn = _open_db(cfg)
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM repo_chunks WHERE workspace_id = ?",
                (workspace_id,),
            )
            count = cur.rowcount
            conn.execute(
                "DELETE FROM repo_meta WHERE workspace_id = ?",
                (workspace_id,),
            )
        return int(count or 0)
    finally:
        conn.close()


def query_top_chunks(
    workspace_path: str,
    query: str,
    *,
    top_k: Optional[int] = None,
    cfg: Optional[RepoIndexConfig] = None,
) -> list[RetrievalResult]:
    """Return the top-K chunks most relevant to ``query`` for the
    workspace's prior-built index. Empty list when no index exists,
    when the query is empty, or when the backend init fails. Never
    raises on a query — failure logs and returns ``[]``.
    """
    cfg = cfg or RepoIndexConfig()
    k = top_k if top_k is not None else cfg.top_k
    if not query or not query.strip():
        return []
    workspace_id = _workspace_id(workspace_path)
    conn = _open_db(cfg)
    try:
        meta = conn.execute(
            "SELECT backend, idf_json FROM repo_meta WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        if meta is None:
            return []
        backend_name, idf_json = meta
        backend: IndexBackend
        if backend_name == "tfidf":
            tfidf_backend = TfidfBackend()
            tfidf_backend.load_idf(idf_json or "")
            backend = tfidf_backend
        elif backend_name == "openai_embeddings":
            openai_backend = OpenAIEmbeddingsBackend(cfg)
            if not openai_backend.available:
                logger.debug("[repo_index] openai backend unavailable for query.")
                return []
            backend = openai_backend
        else:
            return []
        try:
            query_vec = backend.vectorize_query(query)
        except Exception as exc:  # noqa: BLE001 — never explode planner path
            logger.warning("[repo_index] query vectorise failed: %s", exc)
            return []
        rows = conn.execute(
            "SELECT file_path, chunk_index, content, vector_json "
            "FROM repo_chunks WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchall()
        scored: list[RetrievalResult] = []
        for file_path, chunk_index, content, vec_json in rows:
            score = backend.cosine(query_vec, vec_json)
            if score <= 0.0:
                continue
            scored.append(RetrievalResult(
                file_path=file_path, chunk_index=int(chunk_index),
                score=score, content=content,
            ))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
    finally:
        conn.close()


def render_results_for_injection(
    results: list[RetrievalResult], *, max_bytes: int,
) -> str:
    """Render retrieval results as a single string capped at ``max_bytes``.

    Results are concatenated in score order; we stop adding once the
    next result would push us past the cap. Suitable for injecting as
    a system message in the planner.
    """
    if not results:
        return ""
    parts: list[str] = ["### Repository context — top semantic matches\n\n"]
    size = len(parts[0].encode("utf-8"))
    for r in results:
        block = r.render()
        b = len(block.encode("utf-8"))
        if size + b > max_bytes:
            break
        parts.append(block)
        size += b
    return "".join(parts) if len(parts) > 1 else ""


# ---------------------------------------------------------------------------
# 8. Misc helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Async wrapper used by planning_node so the query doesn't block the
# event loop on a large index.

async def async_query_top_chunks(
    workspace_path: str,
    query: str,
    *,
    top_k: Optional[int] = None,
    cfg: Optional[RepoIndexConfig] = None,
) -> list[RetrievalResult]:
    return await asyncio.to_thread(
        query_top_chunks, workspace_path, query, top_k=top_k, cfg=cfg,
    )
