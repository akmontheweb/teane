"""Unit tests for harness/arch_summary.py — the §11 jsonc extractor + preamble."""

from __future__ import annotations

import json
from pathlib import Path

from harness import arch_summary


# ---------------------------------------------------------------------------
# Fixtures — building blocks for synthetic SPEC_ARCHITECTURE.md docs
# ---------------------------------------------------------------------------

_VALID_SUMMARY: dict = {
    "schema_version": 1,
    "project_name": "demo",
    "backend_language": "python_fastapi",
    "frontend": "react",
    "db_engine": "postgres",
    "auth_strategy": "jwt",
    "change_request_mode": False,
    "agile_mode": True,
    "stack_skills": ["harness/skills/python_fastapi.md", "harness/skills/react.md"],
    "workspace_layout": {
        "backend_root": "demo-backend",
        "frontend_root": "demo-frontend",
        "contracts_dir": "contracts",
        "docs_dir": "docs",
        "tests_dir": "tests",
    },
    "backend": {
        "package_root": "app",
        "framework_version": "0.115",
        "layers": ["router", "service", "repository", "model", "schema"],
        "endpoints": [
            {
                "id": "EP-001",
                "method": "POST",
                "path": "/api/v1/auth/login",
                "request_schema": "LoginRequest",
                "response_schema": "TokenResponse",
                "auth_required": False,
                "rsd_story_ids": ["STORY-1"],
                "rsd_feature_ids": ["FEAT-1"],
                "rsd_fr_ids": [],
            },
            {
                "id": "EP-002",
                "method": "GET",
                "path": "/api/v1/users/{id}",
                "request_schema": "",
                "response_schema": "UserResponse",
                "auth_required": True,
                "rsd_story_ids": ["STORY-3"],
                "rsd_feature_ids": ["FEAT-2"],
                "rsd_fr_ids": ["FR-016"],
            },
        ],
    },
    "contract": {
        "openapi_spec_path": "contracts/openapi.json",
        "extraction_method": "fastapi_builtin",
        "extraction_command": "python -c \"from app.main import app; ...\"",
    },
    "frontend_spec": {
        "type_gen_command": "npx openapi-typescript@7 ...",
        "type_output_path": "src/types/api.ts",
        "api_client_path": "src/lib/api-client.ts",
        "components": [
            {
                "name": "LoginForm",
                "path": "pages/auth/LoginPage.tsx",
                "rsd_story_ids": ["STORY-1"],
                "rsd_feature_ids": ["FEAT-1"],
                "rsd_fr_ids": [],
                "radix_primitives": ["Form", "Label"],
            },
        ],
    },
    "adrs": [{"id": "ADR-001", "title": "JWT over session", "status": "Accepted"}],
}


def _doc_with_jsonc(summary: dict, prelude: str = "") -> str:
    """Wrap a summary dict in a minimally-realistic arch doc.

    We deliberately put an illustrative ``json`` block in §10 before
    the §11 jsonc block so tests can assert the extractor picks the
    LAST fence (the real summary), not the first.
    """
    illustrative = (
        '{ "error": { "code": "ILLUSTRATIVE", '
        '"message": "kept inline so §11 stays last" } }'
    )
    return (
        prelude
        + "# Architecture Document\n\n"
        + "## §10 Errors\n\n"
        + "```json\n" + illustrative + "\n```\n\n"
        + "## §11 Machine-readable summary\n\n"
        + "```jsonc\n" + json.dumps(summary, indent=2) + "\n```\n"
    )


def _write_arch(tmp_path: Path, content: str) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / "SPEC_ARCHITECTURE.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_arch_summary — happy path
# ---------------------------------------------------------------------------

def test_load_valid_summary_returns_dict(tmp_path):
    _write_arch(tmp_path, _doc_with_jsonc(_VALID_SUMMARY))
    result = arch_summary.load_arch_summary(str(tmp_path))
    assert result is not None
    assert result["schema_version"] == 1
    assert result["backend_language"] == "python_fastapi"
    assert len(result["backend"]["endpoints"]) == 2


def test_load_picks_last_jsonc_block_not_first(tmp_path):
    """An earlier illustrative ``json`` block (e.g. the §10 error shape)
    must not be mistaken for the §11 summary."""
    _write_arch(tmp_path, _doc_with_jsonc(_VALID_SUMMARY))
    result = arch_summary.load_arch_summary(str(tmp_path))
    assert result is not None and "backend" in result, (
        "extractor picked the §10 illustrative ``json`` block instead of §11"
    )


def test_load_tolerates_line_comments_in_jsonc(tmp_path):
    body = (
        "# Architecture Document\n\n"
        "## §11\n\n"
        "```jsonc\n"
        "{\n"
        '  "schema_version": 1,                 // gate\n'
        '  "project_name": "demo",\n'
        '  "backend_language": "python_fastapi",\n'
        '  "frontend": "none",                  // headless\n'
        '  "db_engine": "none",\n'
        '  "auth_strategy": "none",\n'
        '  "backend": {"endpoints": []},\n'
        '  "contract": {"openapi_spec_path": "contracts/openapi.json"}\n'
        "}\n"
        "```\n"
    )
    _write_arch(tmp_path, body)
    result = arch_summary.load_arch_summary(str(tmp_path))
    assert result is not None
    assert result["frontend"] == "none"


def test_load_preserves_url_with_double_slash_inside_string(tmp_path):
    """``//`` inside a quoted string must NOT be stripped as a comment."""
    summary = dict(_VALID_SUMMARY)
    summary["contract"] = dict(summary["contract"])
    summary["contract"]["extraction_command"] = "curl http://localhost:8080/v3/api-docs"
    _write_arch(tmp_path, _doc_with_jsonc(summary))
    result = arch_summary.load_arch_summary(str(tmp_path))
    assert result is not None
    assert result["contract"]["extraction_command"].startswith("curl http://")


# ---------------------------------------------------------------------------
# load_arch_summary — failure modes (every one returns None, no raise)
# ---------------------------------------------------------------------------

def test_load_returns_none_on_missing_file(tmp_path):
    assert arch_summary.load_arch_summary(str(tmp_path)) is None


def test_load_returns_none_on_empty_workspace_path():
    assert arch_summary.load_arch_summary("") is None


def test_load_returns_none_on_no_fenced_block(tmp_path):
    _write_arch(tmp_path, "# Architecture\n\nNo summary fence here.\n")
    assert arch_summary.load_arch_summary(str(tmp_path)) is None


def test_load_returns_none_on_malformed_json(tmp_path):
    bad = "# Arch\n\n```jsonc\n{ this is not valid json\n```\n"
    _write_arch(tmp_path, bad)
    assert arch_summary.load_arch_summary(str(tmp_path)) is None


def test_load_returns_none_on_schema_version_mismatch(tmp_path):
    summary = dict(_VALID_SUMMARY)
    summary["schema_version"] = 99
    _write_arch(tmp_path, _doc_with_jsonc(summary))
    assert arch_summary.load_arch_summary(str(tmp_path)) is None


def test_load_returns_none_on_missing_schema_version(tmp_path):
    summary = {k: v for k, v in _VALID_SUMMARY.items() if k != "schema_version"}
    _write_arch(tmp_path, _doc_with_jsonc(summary))
    assert arch_summary.load_arch_summary(str(tmp_path)) is None


def test_load_returns_none_when_block_is_not_an_object(tmp_path):
    _write_arch(tmp_path, "# Arch\n\n```jsonc\n[1, 2, 3]\n```\n")
    assert arch_summary.load_arch_summary(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# render_arch_preamble
# ---------------------------------------------------------------------------

def test_render_preamble_empty_on_none():
    assert arch_summary.render_arch_preamble(None) == ""


def test_render_preamble_empty_on_empty_dict():
    assert arch_summary.render_arch_preamble({}) == ""


def test_render_preamble_includes_endpoint_table():
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY)
    assert "Architecture summary" in out
    assert "EP-001" in out and "EP-002" in out
    assert "/api/v1/auth/login" in out
    assert "LoginRequest" in out
    assert "TokenResponse" in out


def test_render_preamble_includes_contract_path():
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY)
    assert "contracts/openapi.json" in out


def test_render_preamble_includes_components_when_frontend_react():
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY)
    assert "Component map" in out
    assert "LoginForm" in out
    assert "Form" in out and "Label" in out


def test_render_preamble_omits_components_when_frontend_none():
    summary = dict(_VALID_SUMMARY)
    summary["frontend"] = "none"
    out = arch_summary.render_arch_preamble(summary)
    # Endpoint map is still useful for headless backends.
    assert "EP-001" in out
    assert "Component map" not in out


def test_render_preamble_renders_rsd_ids_in_id_cell():
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY)
    # Story + feature + (when present) FR identifiers all surface.
    assert "STORY-1" in out
    assert "FEAT-2" in out
    assert "FR-016" in out


def test_render_preamble_includes_arch_gap_guidance():
    """The patcher preamble must tell the LLM what to do when a
    decision is missing — that's the whole point of the structured
    handoff."""
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY)
    assert "ARCH_GAP" in out, "patcher preamble should instruct the LLM to halt on missing decisions"


def test_render_preamble_reviewer_consumer_uses_drift_language():
    """The reviewer should be told to flag *drift* (contradiction
    between code and tables), not to halt — halting belongs to the
    patcher."""
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY, consumer="reviewer")
    assert "ARCH_GAP" not in out
    assert "finding" in out.lower()
    assert "drift" in out.lower()


def test_render_preamble_test_generator_consumer_uses_coverage_language():
    """Test-generation should treat the tables as a coverage target —
    every endpoint at least one test, every component at least one
    render test."""
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY, consumer="test_generator")
    assert "ARCH_GAP" not in out
    assert "coverage" in out.lower()
    # Still surfaces the endpoint table — same structural index.
    assert "EP-001" in out


def test_render_preamble_unknown_consumer_falls_back_to_patcher():
    """An unknown consumer string MUST NOT crash; it should fall back
    to the patcher block (the most informative default)."""
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY, consumer="nonsense")
    assert "ARCH_GAP" in out


def test_render_preamble_security_consumer_emphasises_consistency():
    """The security consumer should frame fixes as "stay within the
    resolved stack" — auth strategy, ORM, contract path stay fixed."""
    out = arch_summary.render_arch_preamble(_VALID_SUMMARY, consumer="security")
    lower = out.lower()
    assert "consistent" in lower
    assert "auth" in lower
    # Must NOT carry the patcher's NO_PROGRESS instruction — that
    # would conflict with the autofix → repair → re-scan loop the
    # security gate already manages.
    assert "ARCH_GAP" not in out
    # The endpoint map still surfaces so the LLM knows where its
    # changes will land.
    assert "EP-001" in out


def test_render_preamble_falls_back_to_legacy_frontend_object():
    """Pre-rename summaries (where ``frontend`` was the object, not the
    enum) must still render — the schema split landed mid-flight and
    older arch docs on disk won't have re-generated yet."""
    summary = {
        "schema_version": 1,
        "backend_language": "python_fastapi",
        "frontend": {                              # object, not "react"
            "components": [
                {"name": "LegacyCard", "path": "components/LegacyCard.tsx",
                 "rsd_story_ids": ["STORY-9"], "radix_primitives": ["Tooltip"]},
            ],
        },
        "backend": {"endpoints": []},
        "contract": {"openapi_spec_path": "contracts/openapi.json"},
    }
    out = arch_summary.render_arch_preamble(summary)
    assert "LegacyCard" in out
    assert "Tooltip" in out


def test_render_preamble_handles_endpoints_without_rsd_ids():
    """Bare-minimum endpoint entries should still render — the LLM may
    omit some ID arrays on first pass and we don't want a KeyError."""
    summary = dict(_VALID_SUMMARY)
    summary["backend"] = {
        "package_root": "app",
        "framework_version": "0.115",
        "layers": [],
        "endpoints": [
            {"id": "EP-999", "method": "GET", "path": "/health"},
        ],
    }
    out = arch_summary.render_arch_preamble(summary)
    assert "EP-999" in out
    assert "/health" in out


# ---------------------------------------------------------------------------
# Internal: _strip_jsonc_comments quoting safety
# ---------------------------------------------------------------------------

def test_strip_jsonc_keeps_string_with_internal_slashes():
    src = '{ "url": "http://x//y", "note": "trailing // not a comment" }'
    cleaned = arch_summary._strip_jsonc_comments(src)
    assert "http://x//y" in cleaned
    assert "trailing // not a comment" in cleaned


def test_strip_jsonc_removes_line_comment_after_value():
    src = '{\n  "a": 1, // tail\n  "b": 2\n}'
    cleaned = arch_summary._strip_jsonc_comments(src)
    assert "// tail" not in cleaned
    # Must remain valid JSON after stripping.
    parsed = json.loads(cleaned)
    assert parsed == {"a": 1, "b": 2}
