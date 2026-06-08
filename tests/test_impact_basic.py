"""Tests for harness/impact.py — impact analysis basics."""

import tempfile

import pytest

from harness.impact import (
    ImpactResult,
    DependencyGraph,
    ImpactAnalyzer,
)


class TestImpactResult:
    """Test ImpactResult dataclass."""

    def test_construct_minimal(self):
        """Construct ImpactResult with required field."""
        result = ImpactResult(modified_files=["a.py"])
        assert result.modified_files == ["a.py"]
        assert result.impacted_files == []
        assert result.total_impacted == 0
        assert result.graph_incomplete is False
        assert result.files_scanned == 0

    def test_construct_with_impacted_files(self):
        """Construct with impacted files list."""
        result = ImpactResult(
            modified_files=["a.py"],
            impacted_files=["b.py", "c.py"],
            total_impacted=2,
            files_scanned=10,
        )
        assert result.modified_files == ["a.py"]
        assert result.impacted_files == ["b.py", "c.py"]
        assert result.total_impacted == 2

    def test_has_impact_with_impacted_files(self):
        """has_impact should return True when impacted files exist."""
        result = ImpactResult(
            modified_files=["a.py"],
            impacted_files=["b.py"],
            total_impacted=1,
        )
        assert result.has_impact() is True

    def test_has_impact_no_impacted_files(self):
        """has_impact should return False when no impacted files."""
        result = ImpactResult(
            modified_files=["a.py"],
            impacted_files=[],
            total_impacted=0,
        )
        assert result.has_impact() is False

    def test_incomplete_flag(self):
        """graph_incomplete flag should be set correctly."""
        result = ImpactResult(
            modified_files=["a.py"],
            impacted_files=[],
            graph_incomplete=True,
            files_scanned=500,
        )
        assert result.graph_incomplete is True
        assert result.files_scanned == 500

    def test_symbol_impact_mapping(self):
        """symbol_impact should map symbols to affected files."""
        result = ImpactResult(
            modified_files=["a.py"],
            impacted_files=["b.py"],
            symbol_impact={
                "MyClass.method": ["b.py", "c.py"],
                "helper_func": ["d.py"],
            },
        )
        assert "MyClass.method" in result.symbol_impact
        assert result.symbol_impact["MyClass.method"] == ["b.py", "c.py"]

    def test_warning_message(self):
        """warning field should store warning text."""
        warning_text = "Analysis incomplete: scanned 100 of ~1000 files"
        result = ImpactResult(
            modified_files=["a.py"],
            warning=warning_text,
        )
        assert result.warning == warning_text


class TestDependencyGraphBasics:
    """Test DependencyGraph initialization."""

    def test_graph_init_with_workspace(self):
        """Graph should initialize with workspace path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = DependencyGraph(tmpdir)
            assert graph is not None

    def test_graph_init_with_max_scan_files(self):
        """Graph should accept max_scan_files parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = DependencyGraph(tmpdir, max_scan_files=1000)
            assert graph is not None

    def test_graph_init_with_ignore_patterns(self):
        """Graph should accept ignore_patterns parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = DependencyGraph(
                tmpdir,
                ignore_patterns=["*.test.py", "__pycache__"],
            )
            assert graph is not None


class TestImpactAnalyzerBasics:
    """Test ImpactAnalyzer initialization."""

    def test_analyzer_init_with_workspace(self):
        """Analyzer should initialize with workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ImpactAnalyzer(tmpdir)
            assert analyzer is not None

    def test_analyzer_init_with_max_scan_files(self):
        """Analyzer should accept max_scan_files parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ImpactAnalyzer(tmpdir, max_scan_files=100)
            assert analyzer is not None

    def test_analyzer_analyze_returns_impact_result(self):
        """analyze() should return an ImpactResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ImpactAnalyzer(tmpdir)
            result = analyzer.analyze(modified_files=[])
            assert isinstance(result, ImpactResult)

    def test_analyzer_analyze_empty_list(self):
        """analyze() with empty modified_files should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ImpactAnalyzer(tmpdir)
            result = analyzer.analyze(modified_files=[])
            assert result.files_scanned >= 0

    def test_analyzer_analyze_with_modified_files(self):
        """analyze() with modified_files should analyze."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_py = f"{tmpdir}/test.py"
            with open(test_py, "w") as f:
                f.write("x = 1\n")

            analyzer = ImpactAnalyzer(tmpdir)
            result = analyzer.analyze(modified_files=["test.py"])
            assert isinstance(result, ImpactResult)
