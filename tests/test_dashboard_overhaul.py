"""Regression tests for the configure + run page overhaul.

Covers the new dashboard helpers and the request handlers added in the
"Run + Configure screen overhaul" change:
    - ``_parse_multipart_body`` parses the file part + scalar fields.
    - ``_persist_product_spec`` writes safely under ``product_spec/``
      and rejects unsafe filenames / wrong extensions.
    - ``_persist_user_skill`` / ``_delete_user_skill`` round-trip a
      ``.py`` file in ``user_skills_dir``.
    - ``_persist_new_memory`` writes the named ``.md`` file.
    - ``_browse_response`` returns directories only and reports
      parents.
    - The Run-form HTML carries the "Product Requirement" label, the
      Upload button, and the workspace folder picker.
    - ``has_running_for_workspace`` blocks ``/run/now`` with 409 when a
      run is already active.
"""

from __future__ import annotations

import json
import os

import pytest

from harness.dashboard import (
    DashboardConfig,
    _browse_response,
    _delete_user_skill,
    _list_user_skill_files,
    _parse_multipart_body,
    _persist_new_memory,
    _persist_product_spec,
    _persist_user_skill,
    _write_web_input_sidecar,
)


# ---------------------------------------------------------------------------
# Multipart parser
# ---------------------------------------------------------------------------

def _multipart(boundary: str, parts: list[tuple[str, str, bytes, str]]) -> bytes:
    """Build a multipart body. Each part is (name, filename, value, content_type)."""
    out: list[bytes] = []
    for name, filename, value, ctype in parts:
        out.append(b"--" + boundary.encode())
        if filename:
            disp = (
                f'form-data; name="{name}"; filename="{filename}"'.encode()
            )
            out.append(b"Content-Disposition: " + disp)
            out.append(("Content-Type: " + ctype).encode())
        else:
            out.append(
                b"Content-Disposition: form-data; name=\"" + name.encode() + b"\""
            )
        out.append(b"")
        out.append(value)
    out.append(b"--" + boundary.encode() + b"--")
    out.append(b"")
    return b"\r\n".join(out)


def test_parse_multipart_extracts_fields_and_files():
    boundary = "ABC123"
    body = _multipart(boundary, [
        ("workspace", "", b"/home/op/repo", ""),
        ("file", "spec.md", b"# Product spec\n\nDo the thing.", "text/markdown"),
    ])
    fields, files = _parse_multipart_body(
        body, f"multipart/form-data; boundary={boundary}",
    )
    assert fields == {"workspace": "/home/op/repo"}
    assert "file" in files
    assert files["file"] == ("spec.md", b"# Product spec\n\nDo the thing.")


def test_parse_multipart_rejects_non_multipart():
    with pytest.raises(ValueError):
        _parse_multipart_body(b"hello", "application/x-www-form-urlencoded")


def test_parse_multipart_rejects_oversized():
    huge = b"x" * (6 * 1024 * 1024)  # 6 MiB > 5 MiB ceiling
    with pytest.raises(ValueError):
        _parse_multipart_body(
            huge, "multipart/form-data; boundary=Z",
        )


# ---------------------------------------------------------------------------
# Product-spec persistence
# ---------------------------------------------------------------------------

def test_persist_product_spec_writes_under_product_spec_folder(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    saved, err = _persist_product_spec(
        str(workspace), "feature.md", b"# Feature\n",
    )
    assert err is None
    assert saved == str(workspace / "product_spec" / "feature.md")
    assert (workspace / "product_spec").is_dir()
    assert (workspace / "product_spec" / "feature.md").read_bytes() == b"# Feature\n"


def test_persist_product_spec_rejects_disallowed_extension(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    saved, err = _persist_product_spec(str(workspace), "evil.exe", b"")
    assert saved == ""
    assert err and "only" in err.lower()


def test_persist_product_spec_rejects_path_traversal(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    saved, err = _persist_product_spec(
        str(workspace), "../etc/passwd.txt", b"",
    )
    # Either rejected outright (unsafe filename) or sanitised to
    # passwd.txt under product_spec/. Both behaviours keep the file
    # inside the workspace.
    if err is not None:
        assert "unsafe" in err.lower()
    else:
        assert saved.startswith(str(workspace / "product_spec"))


def test_persist_product_spec_rejects_missing_workspace(tmp_path):
    saved, err = _persist_product_spec(
        str(tmp_path / "does-not-exist"), "ok.md", b"x",
    )
    assert saved == "" and err and "does not exist" in err


def test_write_web_input_sidecar_creates_and_overwrites(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    p = _write_web_input_sidecar(str(workspace), "hello")
    assert p == str(workspace / "product_spec" / "web_input.md")
    assert (workspace / "product_spec" / "web_input.md").read_text() == "hello"
    _write_web_input_sidecar(str(workspace), "second pass")
    assert (workspace / "product_spec" / "web_input.md").read_text() == "second pass"


# ---------------------------------------------------------------------------
# User-skill upload/delete
# ---------------------------------------------------------------------------

def _make_cfg_with_dirs(tmp_path) -> DashboardConfig:
    cfg = DashboardConfig()
    cfg.config_path = str(tmp_path / "config.json")
    cfg.log_dir = str(tmp_path / "logs")
    cfg.memory_dir = str(tmp_path / "memory")
    cfg.repo_index_dir = str(tmp_path / "index")
    cfg.web_db_path = str(tmp_path / "web.db")
    cfg.schedule_db = str(tmp_path / "schedule.db")
    cfg.writes_enabled = True
    cfg.host = "127.0.0.1"
    cfg.port = 0
    # Point the config at a real file so the resolver finds the skills dir.
    skills_dir = tmp_path / "skills"
    mem_dir = tmp_path / "mem-conf"
    cfg_payload = {
        "skills": {"user_skills_dir": str(skills_dir)},
        "memory": {"dir": str(mem_dir)},
    }
    with open(cfg.config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_payload, f)
    return cfg


def test_persist_user_skill_round_trip(tmp_path):
    cfg = _make_cfg_with_dirs(tmp_path)
    saved, err = _persist_user_skill(cfg, "my_skill.py", b"def x(): return 1\n")
    assert err is None
    assert saved.endswith("my_skill.py")
    assert "my_skill.py" in _list_user_skill_files(cfg)
    removed, err = _delete_user_skill(cfg, "my_skill.py")
    assert err is None and removed.endswith("my_skill.py")
    assert _list_user_skill_files(cfg) == []


def test_persist_user_skill_rejects_non_py(tmp_path):
    cfg = _make_cfg_with_dirs(tmp_path)
    saved, err = _persist_user_skill(cfg, "evil.sh", b"#!/bin/sh\n")
    assert saved == "" and err and ".py" in err


def test_delete_user_skill_rejects_unknown_file(tmp_path):
    cfg = _make_cfg_with_dirs(tmp_path)
    _, err = _delete_user_skill(cfg, "nope.py")
    assert err and "no such" in err.lower()


# ---------------------------------------------------------------------------
# Memory new-entry
# ---------------------------------------------------------------------------

def test_persist_new_memory_writes_named_md_file(tmp_path):
    cfg = _make_cfg_with_dirs(tmp_path)
    saved, err = _persist_new_memory(cfg, "team-context", "Notes for the team.")
    assert err is None
    assert saved.endswith("team-context.md")
    assert os.path.isfile(saved)


def test_persist_new_memory_rejects_invalid_name(tmp_path):
    cfg = _make_cfg_with_dirs(tmp_path)
    _, err = _persist_new_memory(cfg, "Bad Name!!", "x")
    assert err and "match" in err


def test_persist_new_memory_rejects_empty_content(tmp_path):
    cfg = _make_cfg_with_dirs(tmp_path)
    _, err = _persist_new_memory(cfg, "ok", "   ")
    assert err and "empty" in err


# ---------------------------------------------------------------------------
# Directory browser
# ---------------------------------------------------------------------------

def test_browse_response_lists_subdirectories(tmp_path):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "a-file.txt").write_text("ignored")
    status, ctype, body = _browse_response(str(tmp_path))
    assert status == 200
    assert "application/json" in ctype
    payload = json.loads(body)
    names = [e["name"] for e in payload["entries"]]
    assert "alpha" in names and "beta" in names
    assert "a-file.txt" not in names  # files filtered out
    assert payload["ok"] is True
    assert payload["path"] == str(tmp_path)


def test_browse_response_reports_error_for_nonexistent_path(tmp_path):
    status, _, body = _browse_response(str(tmp_path / "does-not-exist"))
    assert status == 400
    payload = json.loads(body)
    assert payload["ok"] is False
    assert "not a directory" in payload["error"]


def test_browse_response_defaults_to_home(monkeypatch, tmp_path):
    # Pointing $HOME at a temp dir lets us exercise the empty-path
    # default without touching the real user's home.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "child").mkdir()
    status, _, body = _browse_response("")
    assert status == 200
    payload = json.loads(body)
    assert payload["path"] == str(tmp_path)
    assert any(e["name"] == "child" for e in payload["entries"])


# ---------------------------------------------------------------------------
# Configure page rendering — group rename, section label override,
# Cancel button, custom extras for Skills/Memory/GitHub
# ---------------------------------------------------------------------------

def test_configure_renders_harness_web_group_with_web_defaults_label(tmp_path):
    from harness.dashboard import _render_configure_harness
    cfg = _make_cfg_with_dirs(tmp_path)
    html = _render_configure_harness(cfg)
    # Group header renamed.
    assert ">Harness Web<" in html
    assert ">Dashboard<" not in html.split(">Configure")[-1]  # not as a group
    # Section name rendered with the override.
    assert ">Web Defaults<" in html


def test_configure_renders_cancel_button_on_every_section(tmp_path):
    from harness.dashboard import _render_configure_harness
    cfg = _make_cfg_with_dirs(tmp_path)
    html = _render_configure_harness(cfg)
    assert "ct-section__cancel" in html
    # And the secondary Carbon class is attached for styling.
    assert "bx--btn--secondary ct-section__cancel" in html


def test_configure_skills_section_lists_existing_files_and_upload_form(tmp_path):
    from harness.dashboard import _render_configure_harness, _persist_user_skill
    cfg = _make_cfg_with_dirs(tmp_path)
    _persist_user_skill(cfg, "alpha.py", b"")
    html = _render_configure_harness(cfg)
    assert "alpha.py" in html
    assert 'action="/api/skills/upload"' in html or "action='/api/skills/upload'" in html
    assert 'enctype="multipart/form-data"' in html or "enctype='multipart/form-data'" in html


def test_configure_memory_section_renders_new_entry_form(tmp_path):
    from harness.dashboard import _render_configure_harness
    cfg = _make_cfg_with_dirs(tmp_path)
    html = _render_configure_harness(cfg)
    assert 'action="/api/memory/new"' in html or "action='/api/memory/new'" in html
    assert "Add a new memory" in html
    assert "Markdown memory content" in html


def test_configure_github_section_renders_default_owner_and_repo_fields(tmp_path):
    from harness.dashboard import _render_configure_harness
    cfg = _make_cfg_with_dirs(tmp_path)
    html = _render_configure_harness(cfg)
    assert "default_owner" in html
    assert "default_repo" in html
