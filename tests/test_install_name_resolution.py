"""Fix I — install-name resolution.

Covers the three-layer precedence used by ``_resolve_install_name``:

    1. Config dict passed by the caller (test seam).
    2. Hardcoded ``_DEP_INSTALL_NAMES`` map (import-name / pip-name quirks).
    3. Identity (symbol == pip name).

Also verifies the extended default map (dotenv → python-dotenv, jwt → PyJWT,
etc.) so a future edit that accidentally drops an entry fails loudly instead
of silently regressing the FinancialResearch session that motivated the fix.
"""

from __future__ import annotations

import pytest

from harness.autofix import (
    _resolve_install_name,
    _try_missing_dep,
)
from harness.patcher import OperationType


# ---------------------------------------------------------------------------
# Defaults — the hardcoded map
# ---------------------------------------------------------------------------

class TestDefaultInstallNames:

    @pytest.mark.parametrize(
        "symbol,expected",
        [
            # Pre-Fix I entries — unchanged, keep them regression-guarded.
            ("yaml", "PyYAML"),
            ("cv2", "opencv-python"),
            ("PIL", "Pillow"),
            ("sklearn", "scikit-learn"),
            ("skimage", "scikit-image"),
            ("bs4", "beautifulsoup4"),
            # Fix I — additions.
            ("dotenv", "python-dotenv"),
            ("jwt", "PyJWT"),
            ("magic", "python-magic"),
            ("Crypto", "pycryptodome"),
            ("dateutil", "python-dateutil"),
            ("docx", "python-docx"),
            ("pptx", "python-pptx"),
            ("dns", "dnspython"),
            ("attr", "attrs"),
            ("git", "GitPython"),
        ],
    )
    def test_symbol_maps_to_pypi_name(self, symbol, expected):
        assert _resolve_install_name(symbol) == expected

    def test_unmapped_symbol_returns_identity(self):
        # The common case: same import name and pip name.
        assert _resolve_install_name("pytest") == "pytest"
        assert _resolve_install_name("fastapi") == "fastapi"
        assert _resolve_install_name("pydantic") == "pydantic"


# ---------------------------------------------------------------------------
# Config override precedence
# ---------------------------------------------------------------------------

class TestConfigOverride:

    def test_override_wins_over_default(self):
        # An operator override for a symbol already in the hardcoded map
        # (e.g. a vendored fork of Pillow) must take precedence.
        config = {
            "dependencies": {
                "install_name_overrides": {
                    "PIL": "mycompany-pillow-hardened",
                },
            },
        }
        assert (
            _resolve_install_name("PIL", config=config)
            == "mycompany-pillow-hardened"
        )

    def test_override_covers_unmapped_symbol(self):
        # A private-registry package the harness has never seen.
        config = {
            "dependencies": {
                "install_name_overrides": {
                    "mycompany_utils": "mycompany-py-utils",
                },
            },
        }
        assert (
            _resolve_install_name("mycompany_utils", config=config)
            == "mycompany-py-utils"
        )

    def test_partial_override_leaves_other_defaults_intact(self):
        # Overriding one symbol must NOT shrink the coverage of the
        # hardcoded map — additive semantics.
        config = {
            "dependencies": {
                "install_name_overrides": {
                    "PIL": "custom-pillow",
                },
            },
        }
        # PIL override wins.
        assert _resolve_install_name("PIL", config=config) == "custom-pillow"
        # yaml still resolves via the hardcoded default.
        assert _resolve_install_name("yaml", config=config) == "PyYAML"
        # jwt still resolves via the hardcoded default.
        assert _resolve_install_name("jwt", config=config) == "PyJWT"

    def test_empty_config_is_a_no_op(self):
        assert _resolve_install_name("yaml", config={}) == "PyYAML"
        assert _resolve_install_name("yaml", config=None) == "PyYAML"

    def test_malformed_override_section_is_ignored(self):
        # Wrong types must never crash the resolver — they fall back to
        # defaults so autofix keeps working even with a broken config.
        for bad in (
            {"dependencies": "not-a-dict"},
            {"dependencies": {"install_name_overrides": "not-a-dict"}},
            {"dependencies": {"install_name_overrides": {"PIL": None}}},
            {"dependencies": {"install_name_overrides": {"PIL": ""}}},
            {"dependencies": {"install_name_overrides": {"PIL": "   "}}},
        ):
            assert _resolve_install_name("PIL", config=bad) == "Pillow"

    def test_override_value_is_stripped(self):
        config = {
            "dependencies": {
                "install_name_overrides": {
                    "PIL": "  custom-pillow  \n",
                },
            },
        }
        assert _resolve_install_name("PIL", config=config) == "custom-pillow"


# ---------------------------------------------------------------------------
# End-to-end — _try_missing_dep uses the resolver
# ---------------------------------------------------------------------------

class TestTryMissingDepUsesResolver:
    """The resolver is only useful if the autofix path actually writes
    the resolved name into ``requirements.txt``. These tests exercise
    ``_try_missing_dep`` with a real temp workspace so a regression in the
    wiring shows up here immediately."""

    def _diag(self, symbol):
        return {
            "error_code": "MISSING_DEP",
            "missing_symbol": symbol,
            "build_command": "python -m pytest",
        }

    def test_writes_pypi_name_not_import_name(self, tmp_path):
        # A ``jwt`` import miss must land ``PyJWT`` in the manifest, not
        # ``jwt`` (which is a squatted package on PyPI).
        (tmp_path / "requirements.txt").write_text("fastapi>=0.100\n")
        block = _try_missing_dep(self._diag("jwt"), str(tmp_path))
        assert block is not None
        # REPLACE_BLOCK rewrites the last line with an appended dep.
        rendered = getattr(block, "replace", None) or getattr(block, "content", "")
        assert "PyJWT" in rendered
        assert "\njwt\n" not in rendered  # the wrong name must not leak

    def test_missing_manifest_creates_with_correct_name(self, tmp_path):
        block = _try_missing_dep(self._diag("dotenv"), str(tmp_path))
        assert block is not None
        assert block.operation == OperationType.CREATE_FILE
        assert "python-dotenv" in block.content

    def test_idempotent_when_pypi_name_already_pinned(self, tmp_path):
        # PyJWT already declared — autofix must return None so the LLM
        # can investigate a deeper issue rather than duplicating the pin.
        (tmp_path / "requirements.txt").write_text("PyJWT>=2.0\n")
        block = _try_missing_dep(self._diag("jwt"), str(tmp_path))
        assert block is None
