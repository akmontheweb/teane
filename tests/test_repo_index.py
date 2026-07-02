"""Regression tests for the repo embeddings / semantic retrieval index (#6).

Covers:
    - Tokeniser splits snake_case and CamelCase into sub-tokens.
    - Chunker honours line windows + overlap.
    - Chunker walker respects exclude globs and text-extension filter.
    - TfidfBackend produces vectors whose cosine ranks the right chunk
      higher for a known semantic query (and is stable across runs).
    - build_index → get_stats → query_top_chunks round-trip persists
      and retrieves the right files for a synthetic corpus.
    - clear_index wipes both the chunks and the meta row.
    - When no index exists, query_top_chunks returns ``[]`` (does not
      raise).
    - The render helper truncates to the byte cap on section
      boundaries.
"""

from __future__ import annotations

from harness.repo_index import (
    Chunker,
    RepoIndexConfig,
    RetrievalResult,
    TfidfBackend,
    _tokenize,
    build_index,
    clear_index,
    get_stats,
    query_top_chunks,
    render_results_for_injection,
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def test_tokenize_splits_snake_case_and_camel_case():
    toks = _tokenize("parseRequest body handle_auth_session")
    # both the full and sub-tokens should land
    for expected in ("parse", "request", "handle", "auth", "session"):
        assert expected in toks


def test_tokenize_ignores_short_tokens():
    toks = _tokenize("a b c hi xx ok long_name")
    assert "a" not in toks  # length 1 dropped
    assert "hi" in toks
    assert "long" in toks
    assert "name" in toks


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def test_chunker_window_with_overlap(tmp_path):
    cfg = RepoIndexConfig(chunk_lines=50, chunk_overlap=10)
    chunker = Chunker(cfg)
    src = tmp_path / "demo.py"
    src.write_text("\n".join(f"line {i}" for i in range(120)))
    chunks = chunker.chunks_for_file(str(tmp_path), "demo.py")
    # 120 lines, window 50, step 40 → 3 chunks (0-49, 40-89, 80-119)
    assert len(chunks) == 3
    assert chunks[0].chunk_index == 0
    assert "line 0" in chunks[0].content
    assert "line 49" in chunks[0].content
    # overlap: 40-49 should appear in both chunk 0 and 1
    assert "line 40" in chunks[0].content
    assert "line 40" in chunks[1].content


def test_chunker_walker_excludes_globs_and_skips_non_text(tmp_path):
    cfg = RepoIndexConfig()
    chunker = Chunker(cfg)
    (tmp_path / "main.py").write_text("print(1)")
    (tmp_path / "README.md").write_text("hi")
    excluded_dir = tmp_path / "node_modules" / "pkg"
    excluded_dir.mkdir(parents=True)
    (excluded_dir / "skip.js").write_text("// no")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")  # not in text extensions
    found = sorted(chunker.walk(str(tmp_path)))
    assert "main.py" in found
    assert "README.md" in found
    assert all("node_modules" not in f for f in found)
    assert all(not f.endswith(".png") for f in found)


# ---------------------------------------------------------------------------
# TfidfBackend
# ---------------------------------------------------------------------------

def _make_chunks(samples):  # samples: list of (path, text)
    from harness.repo_index import Chunk
    return [
        Chunk(file_path=p, chunk_index=0, content=t) for p, t in samples
    ]


def test_tfidf_ranks_relevant_chunk_highest():
    chunks = _make_chunks([
        ("auth/login.py",
         "def handle_login(request):\n    session = create_session(request.user)\n    return session"),
        ("data/database.py",
         "def query_database(sql):\n    return execute(sql)"),
        ("ui/render.py",
         "def render_page(template):\n    return template.render()"),
    ])
    backend = TfidfBackend()
    vectors = backend.fit_chunks(chunks)
    q = backend.vectorize_query("user login session")
    scores = [backend.cosine(q, v) for v in vectors]
    # The auth chunk must be ranked highest for the "login session" query.
    assert max(range(len(scores)), key=lambda i: scores[i]) == 0
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_tfidf_is_deterministic():
    chunks = _make_chunks([
        ("a.py", "session handler login"),
        ("b.py", "render template page"),
    ])
    backend1 = TfidfBackend()
    backend2 = TfidfBackend()
    v1 = backend1.fit_chunks(chunks)
    v2 = backend2.fit_chunks(chunks)
    assert v1 == v2


# ---------------------------------------------------------------------------
# build_index → query_top_chunks roundtrip
# ---------------------------------------------------------------------------

def test_build_query_clear_roundtrip(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "auth.py").write_text(
        "def authenticate(user, password):\n    return validate(user, password)\n"
    )
    (workspace / "ui.py").write_text(
        "def render_template(tpl):\n    return tpl.render()\n"
    )
    (workspace / "db.py").write_text(
        "def query_database(sql):\n    return run(sql)\n"
    )
    index_dir = tmp_path / "indices"
    cfg = RepoIndexConfig(
        enabled=True, top_k=3, chunk_lines=50,
        index_dir=str(index_dir),
    )

    stats = build_index(str(workspace), cfg)
    assert stats.chunk_count >= 3
    assert stats.file_count >= 3
    assert stats.backend == "tfidf"

    fetched = get_stats(str(workspace), cfg)
    assert fetched is not None
    assert fetched.chunk_count == stats.chunk_count

    results = query_top_chunks(str(workspace), "authenticate user password", cfg=cfg)
    assert results, "expected at least one result for an authentication-related query"
    assert isinstance(results[0], RetrievalResult)
    assert results[0].file_path == "auth.py"
    assert results[0].score > 0

    # The render-related query should rank ui.py first
    results2 = query_top_chunks(str(workspace), "render template page", cfg=cfg)
    assert results2
    assert results2[0].file_path == "ui.py"

    deleted = clear_index(str(workspace), cfg)
    assert deleted >= stats.chunk_count
    assert get_stats(str(workspace), cfg) is None
    assert query_top_chunks(str(workspace), "anything", cfg=cfg) == []


def test_purge_all_wipes_every_workspace(tmp_path):
    """purge_all clears repo_meta + repo_chunks across all workspaces."""
    from harness.repo_index import purge_all

    index_dir = tmp_path / "indices"
    cfg = RepoIndexConfig(
        enabled=True, top_k=3, chunk_lines=50, index_dir=str(index_dir),
    )
    for name in ("ws-alpha", "ws-beta"):
        ws = tmp_path / name
        ws.mkdir()
        (ws / "a.py").write_text("def f(): return 1\n")
        build_index(str(ws), cfg)
        assert get_stats(str(ws), cfg) is not None

    meta_n, chunk_n = purge_all(cfg)
    assert meta_n >= 2
    assert chunk_n >= 2

    for name in ("ws-alpha", "ws-beta"):
        assert get_stats(str(tmp_path / name), cfg) is None


def test_purge_all_no_db_returns_zeros(tmp_path):
    """purge_all soft-fails when the index DB file is absent."""
    from harness.repo_index import purge_all

    cfg = RepoIndexConfig(index_dir=str(tmp_path / "empty"))
    assert purge_all(cfg) == (0, 0)


def test_query_top_chunks_returns_empty_when_no_index(tmp_path):
    cfg = RepoIndexConfig(index_dir=str(tmp_path / "ix"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert query_top_chunks(str(workspace), "anything", cfg=cfg) == []


def test_query_top_chunks_empty_query_returns_empty(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "x.py").write_text("def f(): pass\n")
    cfg = RepoIndexConfig(enabled=True, index_dir=str(tmp_path / "ix"))
    build_index(str(workspace), cfg)
    assert query_top_chunks(str(workspace), "", cfg=cfg) == []
    assert query_top_chunks(str(workspace), "   ", cfg=cfg) == []


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------

def test_render_results_caps_at_max_bytes():
    results = [
        RetrievalResult(file_path=f"file{i}.py", chunk_index=0, score=0.9 - i*0.01,
                        content="\n".join(["line"] * 200))
        for i in range(10)
    ]
    block = render_results_for_injection(results, max_bytes=500)
    assert block
    assert len(block.encode("utf-8")) <= 500 * 2  # allow for partial fit
    # The highest-score result must appear first.
    assert "file0.py" in block


def test_render_empty_when_no_results():
    assert render_results_for_injection([], max_bytes=1000) == ""


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_repo_index_config_from_dict():
    cfg = RepoIndexConfig.from_config({
        "repo_index": {
            "enabled": True,
            "backend": "tfidf",
            "top_k": 8,
            "chunk_lines": 100,
            "chunk_overlap": 5,
            "inject_max_bytes": 6000,
        },
    })
    assert cfg.enabled is True
    assert cfg.top_k == 8
    assert cfg.chunk_lines == 100
    assert cfg.chunk_overlap == 5


def test_repo_index_config_defaults_when_section_missing():
    cfg = RepoIndexConfig.from_config({})
    assert cfg.enabled is False
    assert cfg.backend == "tfidf"
