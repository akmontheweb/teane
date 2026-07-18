"""Pytest fixtures shared across the suite.

Currently this file exists only to isolate the harness-global state.db
on a per-test basis. ``harness.story_state`` resolves the DB path via
the ``TEANE_STATE_DB`` env var first, so monkeypatching the env per
test reroutes every story_state.open_story_db() call inside that test
into its own ``tmp_path/state.db`` — the operator's real
``~/.harness/state.db`` is never read or written by the test suite.
"""

from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def isolated_state_db(tmp_path, monkeypatch):
    """Per-test override of the global state.db location.

    Autouse so every test (story-mode or not) is automatically isolated.
    Tests that don't open state.db pay no cost — the env var is set but
    nothing reads it.
    """
    db = tmp_path / "isolated-state.db"
    monkeypatch.setenv("TEANE_STATE_DB", str(db))
    yield db
    # tmp_path cleanup handles the file removal.


@pytest.fixture(autouse=True)
def _stub_moonshot_api_key(monkeypatch):
    """Provide a stub ``MOONSHOT_API_KEY`` for the whole suite.

    The shipped ``config/config.json`` routes ``patching_fallback`` /
    ``repair_fallback`` to ``moonshot:kimi-k3``, so strict validation
    (and everything that runs it — doctor, presets, the web wizard's
    "config ok?" gate, the dashboard home render) now requires the key.
    Many of those tests validate the canonical config while stubbing
    only the older provider keys they knew about; stubbing Moonshot here
    keeps them from failing merely because the runner's shell lacks a
    key the shipped routing depends on.

    Safe by construction: the only tests that assert Moonshot-key
    *absence* (``test_moonshot_provider.py``) set or clear the var with
    their own ``monkeypatch`` calls, which run after this fixture and
    win. If a future default routes to a new remote provider, add its
    key here (or generalize this to read the config's routing).
    """
    monkeypatch.setenv("MOONSHOT_API_KEY", "stub-test-key")
