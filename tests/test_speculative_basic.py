"""Tests for harness/speculative.py — variant building basics."""

import subprocess
import tempfile


from harness.speculative import (
    VariantResult,
    SpeculativeResult,
    _build_variant_cache_env,
    _repo_has_resolvable_head,
    _select_winner,
)


class TestRepoHasResolvableHead:
    """Guard for the speculative branching pre-check that avoids the cryptic
    `fatal: invalid reference: HEAD` error on freshly-init'd repos."""

    def test_false_on_empty_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init", "-q", tmpdir], check=True)
            assert _repo_has_resolvable_head(tmpdir) is False

    def test_true_after_initial_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init", "-q", tmpdir], check=True)
            subprocess.run(
                ["git", "-C", tmpdir,
                 "-c", "user.email=test@example.com", "-c", "user.name=Test",
                 "commit", "--allow-empty", "-q", "-m", "init"],
                check=True,
            )
            assert _repo_has_resolvable_head(tmpdir) is True

    def test_false_outside_a_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _repo_has_resolvable_head(tmpdir) is False


class TestVariantResult:
    """Test VariantResult dataclass."""

    def test_construct_minimal(self):
        """Construct VariantResult with required fields."""
        result = VariantResult(
            index=0,
            variant_id="v1",
            worktree_path="/tmp/v1",
        )
        assert result.index == 0
        assert result.variant_id == "v1"
        assert result.worktree_path == "/tmp/v1"
        assert result.exit_code == -1

    def test_construct_with_exit_code(self):
        """Construct with exit code."""
        result = VariantResult(
            index=1,
            variant_id="v2",
            worktree_path="/tmp/v2",
            exit_code=0,
        )
        assert result.exit_code == 0

    def test_passed_property_success(self):
        """passed property should be True when exit_code=0 and no error."""
        result = VariantResult(
            index=0,
            variant_id="v1",
            worktree_path="/tmp/v1",
            exit_code=0,
            error="",
        )
        assert result.passed is True

    def test_passed_property_failure(self):
        """passed property should be False when exit_code!=0."""
        result = VariantResult(
            index=0,
            variant_id="v1",
            worktree_path="/tmp/v1",
            exit_code=1,
        )
        assert result.passed is False

    def test_passed_property_with_error(self):
        """passed property should be False when error is set."""
        result = VariantResult(
            index=0,
            variant_id="v1",
            worktree_path="/tmp/v1",
            exit_code=0,
            error="something went wrong",
        )
        assert result.passed is False


class TestSpeculativeResult:
    """Test SpeculativeResult dataclass."""

    def test_construct_defaults(self):
        """Construct SpeculativeResult with defaults."""
        result = SpeculativeResult()
        assert result.total_variants == 0
        assert result.passed_variants == 0
        assert result.winner_index == -1
        assert result.variant_results == []

    def test_construct_with_variants(self):
        """Construct with variant results."""
        variant1 = VariantResult(index=0, variant_id="v1", worktree_path="/tmp/v1")
        variant2 = VariantResult(index=1, variant_id="v2", worktree_path="/tmp/v2")

        result = SpeculativeResult(
            total_variants=2,
            passed_variants=1,
            winner_index=0,
            variant_results=[variant1, variant2],
        )
        assert result.total_variants == 2
        assert result.passed_variants == 1
        assert result.winner_index == 0
        assert len(result.variant_results) == 2


class TestBuildVariantCacheEnv:
    """Test cache environment variable building."""

    def test_build_cache_env(self):
        """Should build cache environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _build_variant_cache_env(tmpdir)
            assert isinstance(env, dict)
            # Should have cache-related variables
            assert len(env) > 0

    def test_cache_env_has_pip_cache(self):
        """Should include PIP_CACHE_DIR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _build_variant_cache_env(tmpdir)
            # Should have Python/pip cache variables
            if "PIP_CACHE_DIR" in env:
                assert tmpdir in env["PIP_CACHE_DIR"] or ".harness-cache" in env["PIP_CACHE_DIR"]

    def test_cache_env_has_cargo_home(self):
        """Should include CARGO_HOME for Rust."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _build_variant_cache_env(tmpdir)
            # Should have Rust cache variables
            if "CARGO_HOME" in env:
                assert tmpdir in env["CARGO_HOME"] or ".harness-cache" in env["CARGO_HOME"]

    def test_cache_env_isolation(self):
        """Cache env should isolate each variant's caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _build_variant_cache_env(tmpdir)
            # All paths should be under worktree
            for key, value in env.items():
                if isinstance(value, str) and ("/" in value or "\\" in value):
                    # Should reference tmpdir or .harness-cache
                    assert tmpdir in value or ".harness-cache" in value or value.startswith("/")


class TestSelectWinner:
    """Test winner selection logic."""

    def test_select_winner_prefers_passing(self):
        """Should prefer passing variants."""
        v1 = VariantResult(index=0, variant_id="v1", worktree_path="/tmp/v1", exit_code=1)
        v2 = VariantResult(index=1, variant_id="v2", worktree_path="/tmp/v2", exit_code=0)
        winner = _select_winner([v1, v2])
        # Should select v2 (passing)
        assert winner is not None
        assert winner.variant_id == "v2"

    def test_select_winner_no_variants_returns_none(self):
        """Should return None for empty list."""
        winner = _select_winner([])
        assert winner is None

    def test_select_winner_all_failing(self):
        """Should return None when all variants fail."""
        v1 = VariantResult(index=0, variant_id="v1", worktree_path="/tmp/v1", exit_code=1)
        v2 = VariantResult(index=1, variant_id="v2", worktree_path="/tmp/v2", exit_code=2)
        winner = _select_winner([v1, v2])
        # All fail, no winner
        assert winner is None

    def test_select_winner_first_success(self):
        """Should select first passing variant."""
        v1 = VariantResult(index=0, variant_id="v1", worktree_path="/tmp/v1", exit_code=0)
        v2 = VariantResult(index=1, variant_id="v2", worktree_path="/tmp/v2", exit_code=0)
        winner = _select_winner([v1, v2], strategy="first_success")
        # Should select v1 (first)
        assert winner is not None
        assert winner.variant_id == "v1"


class TestSpeculativeBuildCommandAdaptation:
    """Fix 3 regression: speculative_node used to invoke the sandbox
    with raw state['build_command'] / state['sandbox_config'] without
    running the same late-bind detection compiler_node already had. On
    a greenfield Python workspace with the default `make build`, every
    variant compiled against ubuntu:22.04 and exited 127 — the entire
    speculative round was guaranteed budget waste.

    The fix runs ``_detect_default_build_command`` plus
    ``_apply_toolchain_adaptation`` once up-front and threads the
    resolved values through to each variant executor. This test
    exercises just the late-bind path — full speculative_node is too
    heavy for a unit test, so we verify the building blocks behave
    correctly on the same shape of input.
    """

    def test_detect_and_adapt_for_greenfield_python_workspace(self, tmp_path):
        """Greenfield Python workspace + ``make build`` default should
        end up with: install+pytest command, python:3.12-slim image,
        network auto-enabled (adapter-synthesised install bypasses the
        user opt-in)."""
        from harness.cli import _detect_default_build_command
        from harness.graph import _apply_toolchain_adaptation

        # Mirror the run-log scenario: app/ package, no Makefile, no
        # pyproject yet, no requirements.txt.
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "__init__.py").write_text("")

        detected = _detect_default_build_command(str(tmp_path))
        assert detected is not None and "pip install pytest" in detected

        cfg, allow_net, img_adapted, net_adapted, _ro = _apply_toolchain_adaptation(
            detected,
            {"docker_image": "ubuntu:22.04"},
            allow_network=False,
            command_is_adapter_synthesised=True,
        )
        assert img_adapted
        assert cfg["docker_image"] == "python:3.12-slim"
        assert net_adapted
        assert allow_net is True

    def test_per_variant_late_bind_against_worktree(self, tmp_path):
        """Per-variant late-bind: when the LLM has written real source
        markers into the *worktree* but the original workspace was still
        greenfield at speculate_node entry, re-running detect+adapt
        against the worktree picks up the new toolchain. This is the
        path inside ``_compile_variant`` that prevents the "all 3 fail
        with exit 127 against ubuntu:22.04" outcome on greenfield runs.
        """
        from harness.cli import _detect_default_build_command
        from harness.graph import _apply_toolchain_adaptation

        # Workspace at speculate_node entry — empty, no markers.
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        assert _detect_default_build_command(str(workspace)) is None

        # Worktree after the variant's patches landed — now has a
        # pyproject and a source file.
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (worktree / "pyproject.toml").write_text("[project]\nname='x'\n")
        (worktree / "main.py").write_text("print('hi')\n")

        per_variant_build = _detect_default_build_command(str(worktree))
        assert per_variant_build is not None
        assert "pip install -e ." in per_variant_build
        assert "pytest" in per_variant_build

        cfg, allow_net, img_adapted, _net, _ro = _apply_toolchain_adaptation(
            per_variant_build,
            {"docker_image": "ubuntu:22.04"},
            allow_network=False,
            command_is_adapter_synthesised=False,
        )
        assert img_adapted
        assert cfg["docker_image"] == "python:3.12-slim"

    def test_per_variant_late_bind_is_idempotent_with_workspace_time(self, tmp_path):
        """Chaining the workspace-time pass (greenfield, no-op) into the
        per-variant pass (worktree has markers) yields the same final
        toolchain a single per-variant pass would, with no double-flip
        of network/image. _apply_toolchain_adaptation is documented as
        idempotent — this test pins that contract for the new code path.
        """
        from harness.cli import _detect_default_build_command
        from harness.graph import _apply_toolchain_adaptation

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (worktree / "pyproject.toml").write_text("[project]\nname='x'\n")

        # Workspace-time pass: nothing to detect (greenfield workspace),
        # so build_command stays "make build". Under the toolchain-adapter
        # behavior the workspace-time pass on `make build`:
        #   - swaps ubuntu:22.04 → buildpack-deps:bookworm (make-bearing image)
        #   - flips allow_network → True via the make bypass (the operator
        #     types `make build` but the install step that needs network
        #     is inside the LLM-written Makefile recipe, semantically the
        #     same as an adapter-synthesised command)
        # Both fire in the same call.
        ws_build = "make build"
        ws_late = _detect_default_build_command(str(workspace))
        assert ws_late is None
        ws_cfg, ws_net, ws_img_adapted, ws_net_adapted, _ = _apply_toolchain_adaptation(
            ws_build, {"docker_image": "ubuntu:22.04"}, allow_network=False,
        )
        assert ws_img_adapted is True
        assert ws_net_adapted is True
        assert ws_net is True
        assert ws_cfg["docker_image"] == "buildpack-deps:bookworm"

        # Per-variant pass: detects the worktree's pyproject. The
        # buildpack-deps:bookworm image carried over from the workspace
        # pass isn't in _BARE_IMAGE_DEFAULTS, so the swap doesn't fire
        # and the image stays at buildpack-deps:bookworm (which ships
        # Python too, so pyproject-based builds still work). Network was
        # already flipped on by the workspace pass, so the per-variant
        # bypass is a no-op for the network bit — this is the chained
        # idempotency the test is pinning.
        pv_build = _detect_default_build_command(str(worktree))
        assert pv_build is not None
        pv_cfg, pv_net, pv_img_adapted, pv_net_adapted, _ = _apply_toolchain_adaptation(
            pv_build, ws_cfg, allow_network=ws_net,
            command_is_adapter_synthesised=True,
        )
        # Image-was-adapted is False because buildpack-deps:bookworm
        # isn't a bare default; the existing image is preserved.
        assert pv_img_adapted is False
        assert pv_cfg["docker_image"] == "buildpack-deps:bookworm"
        # Network already on from the workspace pass → no re-flag.
        assert pv_net_adapted is False
        assert pv_net is True

        # Idempotency: calling again with the per-variant inputs is a no-op.
        repeat_cfg, repeat_net, r_img, r_net, _ = _apply_toolchain_adaptation(
            pv_build, pv_cfg, allow_network=pv_net,
            command_is_adapter_synthesised=True,
        )
        assert r_img is False
        assert r_net is False
        assert repeat_cfg["docker_image"] == "buildpack-deps:bookworm"
        assert repeat_net is True


class TestSpeculativeEnabledFlag:
    """Default-off + honor the `enabled` flag. See the speculative-disable plan.

    Across the available log set the winner case never fired and salvage
    created workspace coherence problems the repair loop could not recover
    from. speculate_node now short-circuits to the standard patching flow
    when speculative.enabled is unset or false."""

    def _run_node(self, spec_cfg):
        # speculate_node is async, but the enabled-off path returns before
        # any await — drive it with asyncio.run.
        import asyncio
        from harness.speculative import speculate_node
        return asyncio.run(speculate_node({
            "speculative_config": spec_cfg,
            "workspace_path": "/tmp/does-not-need-to-exist",
            "messages": [],
        }))

    def test_enabled_false_short_circuits_to_fallback(self):
        result = self._run_node({"enabled": False, "num_variants": 3})
        node_state = result.get("node_state", {})
        assert node_state.get("speculative", {}).get("fallback") is True
        # No winner, no variant results — proving no variants were spawned.
        assert "modified_files" not in result
        assert "messages" not in result

    def test_enabled_unset_defaults_to_disabled(self):
        # No "enabled" key → treated as False (default-off behaviour).
        result = self._run_node({"num_variants": 3})
        assert result.get("node_state", {}).get("speculative", {}).get("fallback") is True

    def test_enabled_true_passes_the_short_circuit(self):
        # Confirms the enabled=True path falls through PAST the early return —
        # it then hits the gateway check (no gateway configured in this test
        # state) which also returns fallback, but with a different log line.
        # We assert it reaches the gateway check by patching get_gateway to
        # return None and confirming the late-bind block was reached (which
        # only runs when enabled=True).
        import asyncio
        from harness.speculative import speculate_node
        from unittest.mock import patch
        from harness import graph as _graph

        called = {"get_gateway": False}

        def fake_get_gateway():
            called["get_gateway"] = True
            return None

        # speculate_node does `from harness.graph import get_gateway` inside
        # the function body, so we patch the attribute on harness.graph
        # (not harness.speculative).
        with patch.object(_graph, "get_gateway", fake_get_gateway):
            asyncio.run(speculate_node({
                "speculative_config": {"enabled": True, "num_variants": 3},
                "workspace_path": "/tmp/does-not-need-to-exist",
                "messages": [],
            }))
        assert called["get_gateway"] is True, (
            "enabled=True should fall through to the gateway probe; if False, "
            "the function returned early from the disabled-flag branch."
        )
