"""Tests for the change_requests/ folder convention (PR-1 scope).

PR-1 introduces:
  - `change_requests_dir` config field + name validation
  - CR-N ID assignment from filenames + archive state
  - `ingest_change_requests_node` graph node
  - `route_after_start` change-request branch
  - `_archive_consumed_change_requests` session-end helper

PR-2 will add the delta-mode discovery / write_spec / gatekeeper behavior;
the assertions here cover only the PR-1 surface.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from harness.cli import (
    _archive_consumed_change_requests,
    _list_pending_change_request_files,
    _resolve_change_requests_dir,
    _run_requires_pending_change_request,
)
from harness.deploy import (
    _build_synthesis_change_request_addendum,
    _generate_caddyfile,
    _generate_compose_file,
    _generate_dockerfile,
    generate_assets_from_blueprint,
)
from harness.graph import (
    _assign_change_request_ids,
    _build_change_request_preamble,
    _sample_workspace_for_reverse_engineer,
    _scan_archived_cr_ids,
    ingest_change_requests_node,
    reverse_engineer_architecture_node,
    route_after_start,
    write_spec_node,
)


# ---------------------------------------------------------------------------
# CR-N ID assignment
# ---------------------------------------------------------------------------

class TestScanArchivedCrIds:

    def test_empty_when_archive_missing(self, tmp_path):
        assert _scan_archived_cr_ids(str(tmp_path / "applied")) == set()

    def test_picks_up_top_level_files(self, tmp_path):
        archive = tmp_path / "applied"
        archive.mkdir()
        (archive / "CR-3-foo.txt").write_text("x")
        (archive / "CR-7-bar.txt").write_text("x")
        (archive / "not-a-cr.txt").write_text("x")
        assert _scan_archived_cr_ids(str(archive)) == {3, 7}

    def test_picks_up_one_level_subdirs(self, tmp_path):
        archive = tmp_path / "applied"
        (archive / "session-aaa").mkdir(parents=True)
        (archive / "session-aaa" / "CR-5-foo.txt").write_text("x")
        (archive / "session-bbb").mkdir()
        (archive / "session-bbb" / "CR-9-bar.txt").write_text("x")
        assert _scan_archived_cr_ids(str(archive)) == {5, 9}


class TestAssignChangeRequestIds:

    def test_empty_archive_starts_at_one(self, tmp_path):
        records = _assign_change_request_ids(
            ["alpha.txt", "beta.txt"], str(tmp_path / "applied"),
        )
        assert [r["cr_id"] for r in records] == [1, 2]
        assert [r["original_name"] for r in records] == ["alpha.txt", "beta.txt"]

    def test_continues_from_max_archived(self, tmp_path):
        archive = tmp_path / "applied"
        archive.mkdir()
        (archive / "CR-5-x.txt").write_text("x")
        (archive / "CR-9-y.txt").write_text("x")
        records = _assign_change_request_ids(["new.txt"], str(archive))
        assert records == [{"cr_id": 10, "original_name": "new.txt"}]

    def test_operator_supplied_id_is_respected(self, tmp_path):
        records = _assign_change_request_ids(
            ["alpha.txt", "CR-42-explicit.txt", "beta.txt"],
            str(tmp_path / "applied"),
        )
        by_name = {r["original_name"]: r["cr_id"] for r in records}
        assert by_name == {
            "CR-42-explicit.txt": 42,
            "alpha.txt": 1,
            "beta.txt": 2,
        }

    def test_operator_supplied_does_not_displace_sequential(self, tmp_path):
        # Sequential assignment must skip operator-supplied IDs even when
        # they fall in the middle of the natural range.
        records = _assign_change_request_ids(
            ["a.txt", "b.txt", "CR-2-pinned.txt"],
            str(tmp_path / "applied"),
        )
        # CR-2 is taken; sequential allocation skips 2 and uses 1, 3.
        by_name = {r["original_name"]: r["cr_id"] for r in records}
        assert by_name == {
            "CR-2-pinned.txt": 2,
            "a.txt": 1,
            "b.txt": 3,
        }

    def test_collision_with_archive_raises(self, tmp_path):
        archive = tmp_path / "applied"
        archive.mkdir()
        (archive / "CR-42-old.txt").write_text("x")
        with pytest.raises(ValueError, match="CR-42"):
            _assign_change_request_ids(
                ["CR-42-new.txt"], str(archive),
            )


# ---------------------------------------------------------------------------
# Folder helpers
# ---------------------------------------------------------------------------

class TestListPendingChangeRequestFiles:

    def test_empty_when_missing(self, tmp_path):
        assert _list_pending_change_request_files(str(tmp_path / "absent")) == []

    def test_lists_spec_files_sorted_skipping_applied(self, tmp_path):
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        (cr_dir / "zeta.txt").write_text("x")
        (cr_dir / "alpha.txt").write_text("x")
        (cr_dir / "notes.md").write_text("x")          # .md is allowed
        (cr_dir / "ignored.json").write_text("x")      # wrong extension
        (cr_dir / "applied").mkdir()                    # archive subdir
        (cr_dir / "applied" / "CR-1-old.txt").write_text("x")
        result = _list_pending_change_request_files(str(cr_dir))
        # .txt + .md picked up; .json + applied/ skipped; alphabetical.
        assert result == ["alpha.txt", "notes.md", "zeta.txt"]


class TestRunRequiresPendingChangeRequest:
    """Only `teane patch` consumes change requests, so it is the ONLY flow the
    CR gate can block. Every other target (build/deploy/test, and any future
    flow) is exempt unconditionally."""

    def test_patch_without_cr_is_blocked(self):
        assert _run_requires_pending_change_request("patch", False, []) is True

    def test_patch_with_cr_is_allowed(self):
        assert _run_requires_pending_change_request("patch", False, ["a.txt"]) is False

    def test_patch_with_new_build_is_exempt(self):
        # Defensive: --new-build uses product_spec_dir, not change_requests/.
        assert _run_requires_pending_change_request("patch", True, []) is False

    @pytest.mark.parametrize("flow", ["build", "deploy", "test"])
    def test_non_patch_flows_never_blocked_without_cr(self, flow):
        # Regression: `teane deploy`/`test`/`build` on a clean workspace must
        # not demand a change request.
        assert _run_requires_pending_change_request(flow, False, []) is False

    @pytest.mark.parametrize("flow", ["build", "deploy", "test"])
    def test_non_patch_flows_ignore_present_crs_too(self, flow):
        # Presence of CRs is irrelevant for non-patch flows — they never read them.
        assert _run_requires_pending_change_request(flow, False, ["a.txt"]) is False

    def test_unknown_future_flow_is_exempt(self):
        # Allowlist semantics: a new flow can't accidentally trip the CR gate.
        assert _run_requires_pending_change_request("verify", False, []) is False


class TestResolveChangeRequestsDir:

    def test_default_when_config_missing(self, tmp_path):
        result = _resolve_change_requests_dir(str(tmp_path), None)
        assert result == os.path.normpath(str(tmp_path / "change_requests"))

    def test_custom_when_config_provided(self, tmp_path):
        result = _resolve_change_requests_dir(str(tmp_path), "deltas")
        assert result == os.path.normpath(str(tmp_path / "deltas"))


# ---------------------------------------------------------------------------
# Router precedence
# ---------------------------------------------------------------------------

class TestRouteAfterStart:

    def test_change_request_mode_wins_over_skip_discovery(self):
        # Plan invariant: change_request_mode=True beats skip_discovery=True
        # even when both flags are set, so a misconfigured run still goes
        # through the gatekeeper pipeline (PR-2+) instead of bare patching.
        state = {"change_request_mode": True, "skip_discovery": True}
        assert route_after_start(state) == "ingest_change_requests_node"

    def test_skip_discovery_routes_to_patching(self):
        state = {"change_request_mode": False, "skip_discovery": True}
        assert route_after_start(state) == "patching_node"

    def test_default_routes_to_requirements_discovery(self):
        state = {"change_request_mode": False, "skip_discovery": False}
        assert route_after_start(state) == "requirements_discovery_node"

    def test_skip_discovery_plus_agile_routes_to_decomposition(self):
        """`--agile=true` MUST engage the per-batch pipeline even when
        discovery is skipped (the common case: specs synthesised in
        cmd_run before the graph starts). Without this branch
        ``route_after_start`` falls through to ``patching_node`` and
        the agile flag silently degrades to monolithic patching."""
        state = {
            "change_request_mode": False,
            "skip_discovery": True,
            "decomposition_enabled": True,
        }
        assert route_after_start(state) == "decomposition_node"

    def test_skip_discovery_no_agile_still_routes_to_patching(self):
        """Regression guard: agile mode is opt-in. Without
        ``decomposition_enabled``, ``skip_discovery=True`` keeps its
        legacy monolithic behavior so existing non-agile callers and
        tests stay byte-identical."""
        state = {
            "change_request_mode": False,
            "skip_discovery": True,
            "decomposition_enabled": False,
        }
        assert route_after_start(state) == "patching_node"


# ---------------------------------------------------------------------------
# Ingest node
# ---------------------------------------------------------------------------

def _initial_state_for_ingest(cr_dir: str) -> dict:
    return {
        "change_request_mode": True,
        "change_requests_dir_abs": cr_dir,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "placeholder seed prompt"},
        ],
    }


class TestIngestChangeRequestsNode:

    def test_consolidates_with_cr_headers_and_replaces_user_message(self, tmp_path):
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        (cr_dir / "alpha.txt").write_text("first request body")
        (cr_dir / "beta.txt").write_text("second request body")

        result = asyncio.run(
            ingest_change_requests_node(_initial_state_for_ingest(str(cr_dir)))
        )

        records = result["change_request_files"]
        assert [r["cr_id"] for r in records] == [1, 2]
        assert [r["original_name"] for r in records] == ["alpha.txt", "beta.txt"]
        for r in records:
            assert r["abs_path"].endswith(r["original_name"])

        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        body = messages[1]["content"]
        assert "CR-1: alpha.txt" in body
        assert "CR-2: beta.txt" in body
        assert "first request body" in body
        assert "second request body" in body
        # Seed prompt must be replaced (not appended).
        assert "placeholder seed prompt" not in body

    def test_ingest_consolidates_md_alongside_txt(self, tmp_path):
        # Operators can drop a Markdown change request alongside a .txt
        # one and the ingest node treats both as first-class inputs —
        # bodies are concatenated under CR-N headers and ID assignment
        # is alphabetical across all spec extensions.
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        (cr_dir / "alpha.txt").write_text("plain-text body")
        (cr_dir / "beta.md").write_text("# Markdown body\n\nWith a list.")

        result = asyncio.run(
            ingest_change_requests_node(_initial_state_for_ingest(str(cr_dir)))
        )

        records = result["change_request_files"]
        by_name = {r["original_name"]: r["cr_id"] for r in records}
        assert by_name == {"alpha.txt": 1, "beta.md": 2}

        body = result["messages"][1]["content"]
        assert "CR-1: alpha.txt" in body
        assert "CR-2: beta.md" in body
        assert "plain-text body" in body
        assert "# Markdown body" in body

    def test_respects_operator_supplied_cr_ids(self, tmp_path):
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        (cr_dir / "CR-100-pinned.txt").write_text("pinned body")
        (cr_dir / "unprefixed.txt").write_text("auto body")

        result = asyncio.run(
            ingest_change_requests_node(_initial_state_for_ingest(str(cr_dir)))
        )

        by_name = {r["original_name"]: r["cr_id"] for r in result["change_request_files"]}
        assert by_name == {"CR-100-pinned.txt": 100, "unprefixed.txt": 1}

    def test_collision_with_archive_surfaces_system_message(self, tmp_path):
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        (cr_dir / "applied").mkdir()
        (cr_dir / "applied" / "CR-42-old.txt").write_text("x")
        (cr_dir / "CR-42-new.txt").write_text("conflict")

        result = asyncio.run(
            ingest_change_requests_node(_initial_state_for_ingest(str(cr_dir)))
        )

        # Hard-fail path: exit_code surfaces non-zero and a system message
        # captured the collision so the operator sees it in the transcript.
        assert result["exit_code"] == 1
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        joined = "\n".join(m["content"] for m in system_msgs)
        assert "CR-42" in joined and "ingestion failed" in joined.lower()


# ---------------------------------------------------------------------------
# Archival helper
# ---------------------------------------------------------------------------

class TestArchiveConsumedChangeRequests:

    def test_moves_files_and_writes_manifest(self, tmp_path):
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        src1 = cr_dir / "alpha.txt"
        src1.write_text("body 1")
        src2 = cr_dir / "beta.txt"
        src2.write_text("body 2")
        archive = cr_dir / "applied" / "session-xyz"
        records = [
            {"cr_id": 1, "original_name": "alpha.txt", "abs_path": str(src1)},
            {"cr_id": 2, "original_name": "beta.txt", "abs_path": str(src2)},
        ]

        _archive_consumed_change_requests(
            records, str(archive),
            session_id="session-xyz",
            status="success",
            modified_files=["app/foo.py"],
        )

        # Sources moved.
        assert not src1.exists() and not src2.exists()
        assert (archive / "CR-1-alpha.txt").read_text() == "body 1"
        assert (archive / "CR-2-beta.txt").read_text() == "body 2"
        # Manifest written.
        manifest = json.loads((archive / "manifest.json").read_text())
        assert manifest["session_id"] == "session-xyz"
        assert manifest["status"] == "success"
        assert manifest["modified_files"] == ["app/foo.py"]
        archived_ids = sorted(c["cr_id"] for c in manifest["change_requests"])
        assert archived_ids == [1, 2]

    def test_strips_existing_cr_prefix_to_avoid_double_naming(self, tmp_path):
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        src = cr_dir / "CR-42-pinned.txt"
        src.write_text("pinned body")
        archive = cr_dir / "applied" / "session-xyz"

        _archive_consumed_change_requests(
            [{"cr_id": 42, "original_name": "CR-42-pinned.txt", "abs_path": str(src)}],
            str(archive),
            session_id="session-xyz",
            status="success",
            modified_files=[],
        )

        # CR-42-pinned.txt → CR-42-pinned.txt (not CR-42-CR-42-pinned.txt).
        assert (archive / "CR-42-pinned.txt").exists()
        assert not (archive / "CR-42-CR-42-pinned.txt").exists()

    def test_preserves_md_and_pdf_extensions(self, tmp_path):
        # The archive helper must not coerce .md / .pdf into .txt when
        # stripping an existing CR-N prefix — the operator's file shape
        # has to survive round-trip into change_requests/applied/.
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        md_src = cr_dir / "CR-7-rewrite.md"
        md_src.write_text("# markdown body")
        pdf_src = cr_dir / "design.pdf"
        pdf_src.write_bytes(b"%PDF-1.4\n%dummy\n")  # bytes preserved verbatim
        archive = cr_dir / "applied" / "session-xyz"

        _archive_consumed_change_requests(
            [
                {"cr_id": 7, "original_name": "CR-7-rewrite.md", "abs_path": str(md_src)},
                {"cr_id": 8, "original_name": "design.pdf", "abs_path": str(pdf_src)},
            ],
            str(archive),
            session_id="session-xyz",
            status="success",
            modified_files=[],
        )

        # .md keeps its extension; .pdf bytes are moved untouched.
        assert (archive / "CR-7-rewrite.md").read_text() == "# markdown body"
        assert (archive / "CR-8-design.pdf").read_bytes() == b"%PDF-1.4\n%dummy\n"

    def test_tolerates_missing_source(self, tmp_path):
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        archive = cr_dir / "applied" / "session-xyz"
        records = [
            {"cr_id": 1, "original_name": "alpha.txt",
             "abs_path": str(cr_dir / "absent.txt")},
        ]

        # Must not raise.
        _archive_consumed_change_requests(
            records, str(archive),
            session_id="session-xyz",
            status="success",
            modified_files=[],
        )

        # Manifest still written, source flagged missing.
        manifest = json.loads((archive / "manifest.json").read_text())
        assert manifest["change_requests"][0]["source_missing"] is True

    def test_noop_when_records_empty(self, tmp_path):
        archive = tmp_path / "applied" / "session-xyz"
        _archive_consumed_change_requests(
            [], str(archive),
            session_id="session-xyz",
            status="success",
            modified_files=[],
        )
        assert not archive.exists()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def _base_config(self) -> dict:
        return {
            "build_command": "make build",
            "models": {"x:y": {}},
            "model_routing": {
                "planning_primary": "x:y",
                "patching_primary": "x:y",
                "repair_primary": "x:y",
            },
            "product_spec_dir": "product_spec",
        }

    def test_change_requests_dir_with_path_separator_rejected(self):
        from harness.cli import ConfigError, validate_config_strict
        cfg = self._base_config()
        cfg["change_requests_dir"] = "nested/path"
        with pytest.raises(ConfigError) as exc_info:
            validate_config_strict(cfg, source="test-config")
        assert "change_requests_dir" in str(exc_info.value)

    def test_change_requests_dir_absent_is_ok_for_this_rule(self):
        # change_requests_dir is optional; its omission must NOT produce a
        # change_requests_dir-specific error. Other unrelated errors are
        # fine here — the assertion targets ONLY this rule.
        from harness.cli import ConfigError, validate_config_strict
        cfg = self._base_config()  # change_requests_dir omitted
        try:
            validate_config_strict(cfg, source="test-config")
        except ConfigError as exc:
            assert "change_requests_dir" not in str(exc), (
                f"change_requests_dir-related error fired despite omission: {exc}"
            )


# ===========================================================================
# PR-2: delta-mode discovery + spec-write + prompt injection + deploy markers
# ===========================================================================

# ---------------------------------------------------------------------------
# Delta-mode prompt preamble
# ---------------------------------------------------------------------------

def _state_with_active_crs(crs: list[dict]) -> dict:
    return {
        "change_request_mode": True,
        "change_request_files": crs,
    }


class TestChangeRequestPreamble:

    def test_empty_when_not_in_change_request_mode(self):
        state = {"change_request_mode": False, "change_request_files": []}
        assert _build_change_request_preamble(state, "requirements") == ""

    def test_empty_when_no_records(self):
        state = {"change_request_mode": True, "change_request_files": []}
        assert _build_change_request_preamble(state, "requirements") == ""

    def test_lists_every_active_cr_id_in_every_phase(self):
        state = _state_with_active_crs([
            {"cr_id": 7, "original_name": "rewrite-auth.txt"},
            {"cr_id": 8, "original_name": "add-rate-limit.txt"},
        ])
        for phase in ("requirements", "architecture", "deployment",
                      "patching", "tests"):
            preamble = _build_change_request_preamble(state, phase)
            assert "CR-7" in preamble, f"{phase}: missing CR-7"
            assert "CR-8" in preamble, f"{phase}: missing CR-8"
            assert "rewrite-auth.txt" in preamble
            assert "add-rate-limit.txt" in preamble

    def test_phase_specific_rules_differ_meaningfully(self):
        state = _state_with_active_crs([{"cr_id": 1, "original_name": "x.txt"}])
        req = _build_change_request_preamble(state, "requirements")
        arch = _build_change_request_preamble(state, "architecture")
        depl = _build_change_request_preamble(state, "deployment")
        patch = _build_change_request_preamble(state, "patching")
        tests = _build_change_request_preamble(state, "tests")

        # Spec phases mention BEGIN/END markers for inline tagging.
        assert "BEGIN CR-N" in req or "BEGIN/END CR-N" in req
        assert "BEGIN/END CR-N" in arch
        # Architecture and deployment must tell the LLM how to short-circuit.
        assert "modules=[]" in arch and "complete=true" in arch
        assert "modules=[]" in depl and "complete=true" in depl
        # Code phases mention the inline `# CR-N:` / `// CR-N:` comments.
        assert "CR-N:" in patch
        assert "test_cr_N" in tests


# ---------------------------------------------------------------------------
# write_spec_node delta mode
# ---------------------------------------------------------------------------

def _make_messages(prior_qa: list[tuple[str, str]] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}]
    msgs += [{"role": role, "content": content} for role, content in (prior_qa or [])]
    return msgs


class TestWriteSpecNodeDeltaMode:

    def _state(self, workspace: str, gate: str = "REQUIREMENTS",
               change_request_mode: bool = True) -> dict:
        return {
            "workspace_path": workspace,
            "current_gate": gate,
            "messages": _make_messages([
                ("user", "discovery prompt"),
                ("assistant", '{"modules": []}'),
            ]),
            "change_request_mode": change_request_mode,
            "change_request_files": [
                {"cr_id": 7, "original_name": "x.txt"},
                {"cr_id": 8, "original_name": "y.txt"},
            ],
            "session_id": "session-xyz",
        }

    def test_preserves_existing_spec_and_prepends_revision_header(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        prior_content = "# Baseline Requirements\n\nExisting content that must survive.\n"
        (docs_dir / "SPEC_REQUIREMENTS.md").write_text(prior_content)

        result = asyncio.run(write_spec_node(self._state(str(tmp_path))))

        written = (docs_dir / "SPEC_REQUIREMENTS.md").read_text()
        # Revision header is at the TOP and mentions both active CRs.
        assert written.startswith("## Revision: CR-7, CR-8 — session session-xyz")
        # Baseline content is preserved verbatim somewhere in the file.
        assert "Existing content that must survive." in written
        # The result still points the gatekeeper at the right path.
        assert result["spec_requirements_path"] == str(docs_dir / "SPEC_REQUIREMENTS.md")

    def test_overwrites_when_not_in_change_request_mode(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        prior_content = "# Baseline\nThis must be overwritten in greenfield mode.\n"
        (docs_dir / "SPEC_REQUIREMENTS.md").write_text(prior_content)

        asyncio.run(write_spec_node(self._state(str(tmp_path),
                                                change_request_mode=False)))

        written = (docs_dir / "SPEC_REQUIREMENTS.md").read_text()
        # Greenfield path is unchanged: the baseline is gone.
        assert "must be overwritten" not in written
        assert "Revision:" not in written

    def test_no_prior_file_falls_back_to_simple_write(self, tmp_path):
        # First-ever change-request session on a repo without a SPEC file
        # should still produce a valid spec — no revision header, no
        # crash, just the synthesized content.
        docs_dir = tmp_path / "docs"
        # No SPEC_REQUIREMENTS.md yet.
        asyncio.run(write_spec_node(self._state(str(tmp_path))))
        written = (docs_dir / "SPEC_REQUIREMENTS.md").read_text()
        assert "Revision:" not in written
        assert "Requirements Specification" in written


# ---------------------------------------------------------------------------
# Routing precedence (PR-2 changes the post-ingest edge target)
# ---------------------------------------------------------------------------

class TestIngestSetsDiscoveryActive:
    """PR-2 routes ingest → requirements_discovery_node. The downstream
    pipeline branches on ``skip_discovery``; ingest must clear it so the
    interview loop honours follow-ups instead of short-circuiting to the
    gatekeeper."""

    def test_ingest_returns_skip_discovery_false(self, tmp_path):
        cr_dir = tmp_path / "change_requests"
        cr_dir.mkdir()
        (cr_dir / "alpha.txt").write_text("first request")

        state = {
            "change_request_mode": True,
            "change_requests_dir_abs": str(cr_dir),
            "skip_discovery": True,   # simulates --spec-discovery false (the default)
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "seed"},
            ],
        }

        result = asyncio.run(ingest_change_requests_node(state))

        # The router downstream consults skip_discovery; if ingest left it
        # at True the spec_review would short-circuit straight to the
        # gatekeeper without the interview loop. PR-2 contract: ingest
        # forces it to False.
        assert result["skip_discovery"] is False


# ---------------------------------------------------------------------------
# deploy.py cr_attribution
# ---------------------------------------------------------------------------

class TestDeployCrAttribution:

    def _blueprint(self) -> dict:
        return {
            "services": {
                "auth": {
                    "build_context": "auth",
                    "ports": ["8080:8080"],
                },
                "redis": {
                    "base_image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                },
            },
            "volumes": {},
            "networks": {"app-net": {"driver": "bridge"}},
        }

    def test_compose_byte_identical_when_no_attribution(self):
        blueprint = self._blueprint()
        plain = _generate_compose_file(blueprint)
        explicit_none = _generate_compose_file(blueprint, cr_attribution=None)
        assert plain == explicit_none
        # And no CR markers appear in either rendering.
        assert "CR-" not in plain

    def test_compose_emits_marker_on_annotated_service(self):
        out = _generate_compose_file(
            self._blueprint(),
            cr_attribution={"redis": "CR-7: added redis service for sessions"},
        )
        # The marker comment must appear in the YAML, immediately above
        # the `redis:` service block.
        idx_marker = out.find("# CR-7: added redis service for sessions")
        idx_redis = out.find("  redis:")
        assert idx_marker != -1 and idx_redis != -1, out
        assert idx_marker < idx_redis, (
            "marker must precede the redis service block"
        )
        # Non-annotated service stays unmarked.
        assert "# CR-" not in out[:out.find("  auth:")]

    def test_caddyfile_byte_identical_when_no_attribution(self):
        blueprint = self._blueprint()
        plain = _generate_caddyfile(blueprint)
        explicit_none = _generate_caddyfile(blueprint, cr_attribution=None)
        assert plain == explicit_none
        assert "CR-" not in plain

    def test_caddyfile_emits_marker_on_annotated_stanza(self):
        out = _generate_caddyfile(
            self._blueprint(),
            cr_attribution={"auth": "CR-9: rate-limited /login endpoint"},
        )
        idx_marker = out.find("# CR-9: rate-limited /login endpoint")
        idx_auth_stanza = out.find("auth.localhost {")
        assert idx_marker != -1 and idx_auth_stanza != -1
        assert idx_marker < idx_auth_stanza

    def test_dockerfile_byte_identical_when_no_attribution(self, tmp_path):
        svc_spec = {"build_context": "auth", "ports": ["8080:8080"]}
        plain = _generate_dockerfile("auth", svc_spec, "python", str(tmp_path))
        explicit_none = _generate_dockerfile(
            "auth", svc_spec, "python", str(tmp_path), cr_attribution=None,
        )
        assert plain == explicit_none
        assert "CR-" not in plain

    def test_dockerfile_emits_marker_on_annotated_service(self, tmp_path):
        svc_spec = {"build_context": "auth", "ports": ["8080:8080"]}
        out = _generate_dockerfile(
            "auth", svc_spec, "python", str(tmp_path),
            cr_attribution={"auth": "CR-11: install postgres client libs"},
        )
        # Marker on the very first line so it survives multi-stage builds.
        assert out.startswith("# CR-11: install postgres client libs\n")


# ===========================================================================
# PR-3: reverse-engineer architecture node + cr_attribution flow-through
# ===========================================================================

# ---------------------------------------------------------------------------
# File sampler
# ---------------------------------------------------------------------------

class TestReverseEngineerSampler:

    def test_sampler_returns_empty_for_empty_workspace(self, tmp_path):
        assert _sample_workspace_for_reverse_engineer(str(tmp_path), None) == []

    def test_sampler_prioritises_entry_points(self, tmp_path):
        # main.py and pyproject.toml should rank ahead of arbitrary .py
        # files when both are present.
        (tmp_path / "main.py").write_text("print('hi')\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "obscure.py").write_text("# obscure\n")
        sampled = _sample_workspace_for_reverse_engineer(str(tmp_path), None)
        rels = [rel for rel, _ in sampled]
        assert "main.py" in rels and "pyproject.toml" in rels
        # main.py is index 0 in the priority list, pyproject.toml is later
        # but still before obscure.py.
        assert rels.index("main.py") < rels.index("lib/obscure.py")

    def test_sampler_skips_noise_directories(self, tmp_path):
        (tmp_path / "main.py").write_text("x\n")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "junk.js").write_text("noise\n")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("noise\n")
        sampled = _sample_workspace_for_reverse_engineer(str(tmp_path), None)
        rels = [rel for rel, _ in sampled]
        assert not any("node_modules" in r for r in rels)
        assert not any(r.startswith(".git") for r in rels)


# ---------------------------------------------------------------------------
# reverse_engineer_architecture_node
# ---------------------------------------------------------------------------

class _StubResponse:
    def __init__(self, content: str):
        self.content = content
        class _Usage:
            input_tokens = 50
            output_tokens = 40
            cached_tokens = 0
            cost_usd = 0.001
            model = "stub"
        self.usage = _Usage()


class _StubGateway:
    """Records dispatch calls and returns a canned architecture spec."""

    class config:
        repair_fallback = ""
        planning_fallback = ""

    def __init__(self, content: str):
        self._content = content
        self.dispatched: list[dict] = []

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
        self.dispatched.append({"messages": list(messages), "role": role})
        return _StubResponse(self._content), budget_remaining_usd - 0.10

    def aggregate_tokens(self, tracker, usage, role=None):
        out = dict(tracker or {})
        out["total_cost_usd"] = out.get("total_cost_usd", 0.0) + 0.001
        return out


@pytest.fixture
def stub_gateway():
    from harness import graph as graph_mod
    installed: list[_StubGateway] = []

    def _set(content: str) -> _StubGateway:
        gw = _StubGateway(content)
        graph_mod.set_gateway(gw)
        installed.append(gw)
        return gw

    yield _set
    from harness import graph as graph_mod  # noqa: F811
    graph_mod.set_gateway(None)


class TestReverseEngineerArchitectureNode:

    def _seeded_workspace(self, tmp_path) -> str:
        (tmp_path / "main.py").write_text("def run():\n    pass\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        return str(tmp_path)

    def _state(self, workspace: str, *, budget: float = 2.00) -> dict:
        return {
            "workspace_path": workspace,
            "change_request_mode": True,
            "change_request_files": [
                {"cr_id": 1, "original_name": "x.txt"},
            ],
            "budget_remaining_usd": budget,
            "change_requests_config": {"reverse_engineer_budget_usd": 0.50},
        }

    def test_no_op_when_not_in_change_request_mode(self, tmp_path):
        state = {
            "workspace_path": str(tmp_path),
            "change_request_mode": False,
            "budget_remaining_usd": 2.00,
        }
        # No gateway needed — the node bails before dispatching.
        result = asyncio.run(reverse_engineer_architecture_node(state))
        assert result == {}
        assert not (tmp_path / "docs" / "SPEC_ARCHITECTURE.md").exists()

    def test_skipped_when_spec_already_exists(self, tmp_path, stub_gateway):
        # Pre-seed an existing SPEC_ARCHITECTURE.md so the node short-
        # circuits to the file-stat skip without paying the LLM cost.
        ws = self._seeded_workspace(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "SPEC_ARCHITECTURE.md").write_text("# Pre-existing spec\n")
        gw = stub_gateway("# Should NOT be written")
        result = asyncio.run(reverse_engineer_architecture_node(self._state(ws)))
        # The spec path is returned but the LLM was NOT called.
        assert result["spec_architecture_path"].endswith("SPEC_ARCHITECTURE.md")
        assert gw.dispatched == []
        assert (docs / "SPEC_ARCHITECTURE.md").read_text() == "# Pre-existing spec\n"

    def test_synthesizes_spec_when_missing(self, tmp_path, stub_gateway):
        ws = self._seeded_workspace(tmp_path)
        gw = stub_gateway("# Synthesized Architecture\n\nDescribes the system.\n")
        result = asyncio.run(reverse_engineer_architecture_node(self._state(ws)))
        assert result["spec_architecture_path"] == str(tmp_path / "docs" / "SPEC_ARCHITECTURE.md")
        # Spec landed on disk with the LLM content.
        written = (tmp_path / "docs" / "SPEC_ARCHITECTURE.md").read_text()
        assert "Synthesized Architecture" in written
        # Single dispatch — the node is a one-shot, not a discovery loop.
        assert len(gw.dispatched) == 1
        # Budget reduced by the canned dispatch delta.
        assert result["budget_remaining_usd"] == pytest.approx(1.90, rel=0.01)

    def test_budget_gate_skips_when_remaining_below_cap(self, tmp_path, stub_gateway):
        ws = self._seeded_workspace(tmp_path)
        gw = stub_gateway("# Should NOT be written")
        state = self._state(ws, budget=0.10)  # below the $0.50 default
        result = asyncio.run(reverse_engineer_architecture_node(state))
        assert result == {}
        assert gw.dispatched == []
        assert not (tmp_path / "docs" / "SPEC_ARCHITECTURE.md").exists()

    def test_empty_workspace_skips(self, tmp_path, stub_gateway):
        # No source files → nothing to sample → bail before dispatching.
        gw = stub_gateway("ignored")
        result = asyncio.run(
            reverse_engineer_architecture_node(self._state(str(tmp_path)))
        )
        assert result == {}
        assert gw.dispatched == []


# ---------------------------------------------------------------------------
# generate_assets_from_blueprint flow-through
# ---------------------------------------------------------------------------

class TestGenerateAssetsCrAttributionFlow:

    def _blueprint(self) -> dict:
        return {
            "services": {
                "auth": {
                    "build_context": "auth",
                    "ports": ["8080:8080"],
                },
                "redis": {
                    "base_image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                },
            },
            "volumes": {},
            "networks": {"app-net": {"driver": "bridge"}},
            "proxy_service": "caddy",
        }

    def _telemetry(self) -> dict:
        return {"languages": ["python"], "databases_detected": [], "frameworks_detected": []}

    def test_byte_identical_when_no_attribution(self, tmp_path):
        bp = self._blueprint()
        plain = generate_assets_from_blueprint(bp, self._telemetry(), str(tmp_path))
        assert plain["success"] is True
        compose = (tmp_path / "docker-compose.yml").read_text()
        caddy = (tmp_path / "Caddyfile").read_text()
        assert "CR-" not in compose
        assert "CR-" not in caddy

    def test_attribution_propagates_via_explicit_kwarg(self, tmp_path):
        attribution = {"redis": "CR-7: added redis service for sessions"}
        generate_assets_from_blueprint(
            self._blueprint(), self._telemetry(), str(tmp_path),
            cr_attribution=attribution,
        )
        compose = (tmp_path / "docker-compose.yml").read_text()
        assert "# CR-7: added redis service for sessions" in compose

    def test_attribution_propagates_via_blueprint_field(self, tmp_path):
        # When the kwarg is omitted, the function falls back to
        # blueprint['cr_attribution'] — so the deployment synthesizer can
        # carry the attribution data inline with the blueprint.
        bp = self._blueprint()
        bp["cr_attribution"] = {"auth": "CR-9: rate-limited /login"}
        generate_assets_from_blueprint(bp, self._telemetry(), str(tmp_path))
        compose = (tmp_path / "docker-compose.yml").read_text()
        caddy = (tmp_path / "Caddyfile").read_text()
        assert "# CR-9: rate-limited /login" in compose
        assert "# CR-9: rate-limited /login" in caddy
        # Dockerfile for auth carries the marker on its first line.
        dockerfile_path = tmp_path / "Dockerfile"
        assert dockerfile_path.exists()
        assert dockerfile_path.read_text().startswith(
            "# CR-9: rate-limited /login\n"
        )

    def test_invalid_attribution_type_logged_and_ignored(self, tmp_path):
        bp = self._blueprint()
        bp["cr_attribution"] = ["not-a-dict"]   # operator error
        result = generate_assets_from_blueprint(
            bp, self._telemetry(), str(tmp_path),
        )
        assert result["success"] is True
        compose = (tmp_path / "docker-compose.yml").read_text()
        # The malformed input is ignored, not crashed on.
        assert "CR-" not in compose


# ---------------------------------------------------------------------------
# synthesize_architecture cr_attribution addendum
# ---------------------------------------------------------------------------

class TestSynthesisChangeRequestAddendum:
    """The deployment synthesizer's prompt is extended in change-request
    mode to ask the LLM to populate ``blueprint.cr_attribution``. The
    generate_assets fallback at ``blueprint.get("cr_attribution")``
    picks it up unchanged — no separate plumbing channel."""

    def test_empty_addendum_when_no_change_requests(self):
        rules, schema = _build_synthesis_change_request_addendum(None)
        assert rules == ""
        assert schema == ""

    def test_empty_addendum_when_empty_list(self):
        rules, schema = _build_synthesis_change_request_addendum([])
        assert rules == ""
        assert schema == ""

    def test_addendum_names_active_cr_ids(self):
        records = [
            {"cr_id": 7, "original_name": "rewrite-auth.txt"},
            {"cr_id": 8, "original_name": "add-rate-limit.txt"},
        ]
        rules, schema = _build_synthesis_change_request_addendum(records)
        assert "CR-7" in rules and "rewrite-auth.txt" in rules
        assert "CR-8" in rules and "add-rate-limit.txt" in rules
        # Schema fragment introduces cr_attribution so the JSON shape the
        # LLM is asked to follow includes it at the top level.
        assert '"cr_attribution"' in schema

    def test_addendum_instructs_short_circuit_when_no_significance(self):
        records = [{"cr_id": 1, "original_name": "tiny.txt"}]
        rules, _ = _build_synthesis_change_request_addendum(records)
        # The deployment delta is allowed to be empty when nothing about
        # the CR moves infra — that's the whole point of "deployment-
        # significance" gating.
        assert "deployment-significant" in rules
        # And the LLM is told explicitly how to signal "nothing changed".
        assert "empty object" in rules.lower() or "omit" in rules.lower()


class TestSynthesizeArchitectureChangeRequestMode:
    """End-to-end check that synthesize_architecture passes the CR
    addendum into the LLM prompt when change_request_files is supplied,
    and produces a byte-identical prompt when it isn't. Uses the
    _StubGateway to avoid network."""

    def _telemetry(self) -> dict:
        return {
            "languages": ["python"],
            "databases_detected": [],
            "frameworks_detected": [],
            "src_directories": ["app"],
            "web_servers_detected": [],
            "port_hints": {},
            "app_name": "myapp",
        }

    def test_prompt_does_not_mention_cr_when_no_records(self, tmp_path, stub_gateway):
        from harness.deploy import synthesize_architecture
        gw = stub_gateway('{"services": {}, "volumes": {}, "networks": {}, "proxy_service": null}')
        asyncio.run(synthesize_architecture(self._telemetry(), str(tmp_path)))
        assert len(gw.dispatched) == 1
        prompt_body = "\n".join(
            m["content"] for m in gw.dispatched[0]["messages"]
            if m["role"] == "user"
        )
        # Greenfield prompt: no CR references, no cr_attribution schema.
        assert "CR-" not in prompt_body
        assert "cr_attribution" not in prompt_body

    def test_prompt_includes_cr_addendum_when_records_supplied(
        self, tmp_path, stub_gateway,
    ):
        from harness.deploy import synthesize_architecture
        gw = stub_gateway(
            '{"services": {}, "volumes": {}, "networks": {}, '
            '"proxy_service": null, "cr_attribution": {}}'
        )
        records = [{"cr_id": 11, "original_name": "scale-up.txt"}]
        asyncio.run(synthesize_architecture(
            self._telemetry(), str(tmp_path),
            change_request_files=records,
        ))
        assert len(gw.dispatched) == 1
        prompt_body = "\n".join(
            m["content"] for m in gw.dispatched[0]["messages"]
            if m["role"] == "user"
        )
        # The active CR is named, and the schema fragment introduces
        # cr_attribution so the LLM knows where to put its mapping.
        assert "CR-11" in prompt_body
        assert "scale-up.txt" in prompt_body
        assert "cr_attribution" in prompt_body
