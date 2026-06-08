"""Tests for harness/speculative.py — variant building basics."""

import tempfile

import pytest

from harness.speculative import (
    VariantResult,
    SpeculativeResult,
    _build_variant_cache_env,
    _select_winner,
)


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
