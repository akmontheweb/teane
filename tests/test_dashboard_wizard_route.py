"""Phase 3 regression: the /home wizard renders the correct step per
wizard_state and the two POST routes are dispatched.

Only tests the render + dispatch surface; the subprocess spawn path
is exercised in test_dashboard_app.py.
"""

from __future__ import annotations



from harness.dashboard import DashboardConfig, dispatch


def _cfg(tmp_path):
    return DashboardConfig.from_config(
        {
            "dashboard": {
                "log_dir": str(tmp_path / "logs"),
                "metrics_dir": str(tmp_path / "metrics"),
                "memory_dir": str(tmp_path / "memory"),
                "repo_index_dir": str(tmp_path / "idx"),
                "schedule_db": str(tmp_path / "schedule.db"),
                "enabled": True,
            }
        }
    )


def test_home_renders_setup_step_when_no_config(tmp_path, monkeypatch):
    from harness import cli as _cli
    monkeypatch.setattr(_cli, "_get_global_config_path",
                        lambda: str(tmp_path / "missing" / "config.json"))
    status, ctype, body = dispatch(_cfg(tmp_path), "/home")
    assert status == 200
    assert "Let's pick a provider" in body
    assert "action='/home/wizard/setup'" in body
    # Every supported provider appears in the picker.
    for p in ("anthropic", "openai", "deepseek", "ollama"):
        assert p in body


def test_home_renders_build_step_when_config_ok(tmp_path, monkeypatch):
    from harness import cli as _cli
    import shutil
    import os as _os
    cfg_path = tmp_path / "config.json"
    src = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "config", "config.json",
    )
    shutil.copy(src, cfg_path)
    monkeypatch.setattr(_cli, "_get_global_config_path", lambda: str(cfg_path))
    # Shipped config references models whose API keys must be set for
    # strict validation to pass; supply fakes so the wizard sees
    # config_ok and renders the build step.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    status, ctype, body = dispatch(_cfg(tmp_path), "/home")
    assert status == 200
    assert "What do you want to build?" in body
    assert "action='/home/wizard/start'" in body
    # The three shipped starter cards render.
    assert "Flask To-do App" in body
    assert "FastAPI Notes API" in body
    assert "Static Portfolio Site" in body


def test_home_shows_resume_block_when_sessions_exist(tmp_path, monkeypatch):
    from harness import cli as _cli
    import shutil
    import os as _os
    cfg_path = tmp_path / "config.json"
    src = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "config", "config.json",
    )
    shutil.copy(src, cfg_path)
    monkeypatch.setattr(_cli, "_get_global_config_path", lambda: str(cfg_path))
    # Shipped config references models whose API keys must be set for
    # strict validation to pass; supply fakes so the wizard sees
    # config_ok and renders the build step.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "sess-a.jsonl").write_text('{"event":"session_start"}\n')
    _, _, body = dispatch(_cfg(tmp_path), "/home")
    assert "Continue an existing session" in body
    assert "1 session(s)" in body


def test_home_setup_step_carries_csrf_token(tmp_path, monkeypatch):
    """The wizard POST is CSRF-gated; the rendered form must include
    a token or the operator's submit will 403."""
    from harness import cli as _cli
    monkeypatch.setattr(_cli, "_get_global_config_path",
                        lambda: str(tmp_path / "missing" / "config.json"))
    _, _, body = dispatch(_cfg(tmp_path), "/home")
    assert "name='csrf_token'" in body


def test_wizard_setup_route_registered_in_ROUTES_or_via_dispatch_write():
    """The setup POST is dispatched inside `_dispatch_write`. This
    test simply asserts the source references the two paths so the
    routing table stays discoverable via grep."""
    from harness import dashboard as _d
    src = open(_d.__file__).read()
    assert '"/home/wizard/setup"' in src
    assert '"/home/wizard/start"' in src


def test_home_render_html_includes_alpine_bindings(tmp_path, monkeypatch):
    """Build step uses `teaneHomeBuild()` Alpine store to sync the
    template-card click with the textarea + workspace input."""
    from harness import cli as _cli
    import shutil
    import os as _os
    cfg_path = tmp_path / "config.json"
    src = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "config", "config.json",
    )
    shutil.copy(src, cfg_path)
    monkeypatch.setattr(_cli, "_get_global_config_path", lambda: str(cfg_path))
    # Shipped config references models whose API keys must be set for
    # strict validation to pass; supply fakes so the wizard sees
    # config_ok and renders the build step.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    (tmp_path / "logs").mkdir()
    _, _, body = dispatch(_cfg(tmp_path), "/home")
    assert 'x-data="teaneHomeBuild()"' in body
    assert 'x-on:click="pickTemplate(' in body
    assert "x-model='prompt'" in body
    assert "x-model='workspace'" in body
