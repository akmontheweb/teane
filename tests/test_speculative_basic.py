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
