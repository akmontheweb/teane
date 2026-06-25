"""
Loader for documentation-generation system prompts.

The discovery nodes in ``harness.graph`` (``requirements_discovery_node``,
``architecture_discovery_node``) and the standalone CLI doc-gen skills in
``harness.skills`` (``_DOCGEN_SYSTEM_PROMPTS``) used to ship as inline
Python triple-quoted strings. That made the prompts hard to iterate on without
touching code, and the same checklist had to be duplicated whenever a new
sector (features, abuse cases, threat model, failure modes) needed to be
captured.

This module externalizes those prompts as markdown files under
``harness/skills/docgen/`` and exposes a tiny cached loader. Each file's
*entire* body — including the canonical JSON output schema for the
discovery prompts — is the source of truth. The harness reads the file
verbatim at request time; iterating the prompt means editing the .md file.

Layout::

    harness/skills/docgen/<name>.md

Names currently shipped:
    - requirements_discovery           — first-pass requirements discovery
    - requirements_discovery_followup  — follow-up rounds
    - architecture_discovery           — first-pass architecture discovery
    - architecture_discovery_followup  — follow-up rounds
    - requirements_doc                 — standalone Markdown spec doc
    - arch_doc                         — standalone Markdown ADR

Missing files raise ``FileNotFoundError`` rather than falling back to a
truncated default — the prompts are mission-critical (the discovery JSON
parser hard-requires the canonical ``modules`` key) so a silent fallback
to a stub would produce empty interview screens.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


HARNESS_DOCGEN_DIR = os.path.join(os.path.dirname(__file__), "skills", "docgen")

_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


def load(name: str, workspace_path: Optional[str] = None) -> str:
    """Return the markdown body of the docgen prompt named ``name``.

    Resolution order:
        1. ``{workspace_path}/skills/docgen/{name}.md`` — per-project override
        2. ``harness/skills/docgen/{name}.md``         — shipped default

    The first hit wins; results are cached in-process keyed by the resolved
    absolute path so subsequent calls in the same process avoid disk I/O.

    Args:
        name: Stem of the prompt file (no extension). E.g.
            ``"requirements_discovery"``.
        workspace_path: When supplied, the per-project override directory
            is consulted first. Pass ``None`` (the default) for callers
            that have no workspace context.

    Returns:
        The full markdown body as a single string.

    Raises:
        FileNotFoundError: when no matching file exists in either tier.
        OSError: when the file exists but cannot be read (permissions,
            symlink loop, etc.). Callers should let this propagate — the
            discovery prompts are required for correct operation.
    """
    candidates: list[str] = []
    if workspace_path:
        candidates.append(os.path.join(workspace_path, "skills", "docgen", f"{name}.md"))
    candidates.append(os.path.join(HARNESS_DOCGEN_DIR, f"{name}.md"))

    for path in candidates:
        if not os.path.isfile(path):
            continue

        with _CACHE_LOCK:
            cached = _CACHE.get(path)
        if cached is not None:
            return cached

        with open(path, "r", encoding="utf-8") as fp:
            body = fp.read()

        with _CACHE_LOCK:
            _CACHE[path] = body
        logger.debug("[docgen_prompts] Loaded '%s' from %s (%d chars).",
                     name, path, len(body))
        return body

    raise FileNotFoundError(
        f"docgen prompt '{name}' not found in any of: {candidates}. "
        f"Expected shipped default at harness/skills/docgen/{name}.md."
    )


def clear_cache() -> None:
    """Drop all cached prompt bodies. Tests call this after editing files."""
    with _CACHE_LOCK:
        _CACHE.clear()
