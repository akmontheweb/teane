"""Playwright scenario generator for ``teane test``.

Phase 4 deliverable. Reads the same agile/waterfall split that
``test_data_gen`` uses and produces ``tests/e2e/<feature>.spec.ts``
files. Each scenario carries a teane-convention ``@verifies`` comment
mapping it back to either a story AC (agile) or a spec section anchor
(waterfall) so the defect emitter (Phase 2) can populate
``source_spec.md`` with a real reference.

Phase 4a (spike) caveat: the LLM-backed scenario generator is the
single biggest unknown — it must produce *runnable* Playwright with
real selectors, waits, and assertions, not narrative. Until that
spike lands, the default generator is a deterministic
:func:`fallback_scenarios` that emits a structurally valid spec with
TODO placeholders. It's intentionally *not* a substitute for the LLM
version; it exists so the rest of the pipeline (cache, file layout,
@verifies annotations, runner wiring) is testable without hitting
the gateway.

Chromium runtime: :func:`ensure_chromium_installed` invokes
``npx playwright install chromium`` idempotently. Skipped when a
prior install left a ``chromium-*`` directory under
``~/.cache/ms-playwright/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from harness.test_data_gen import (
    FLOW_AGILE,
    FLOW_WATERFALL,
    SchemaContext,
    detect_flow_kind,
    gather_schema_context,
)

logger = logging.getLogger(__name__)


_E2E_DIR = os.path.join("tests", "e2e")
_CACHE_FILE = ".cache_key"
_DEFAULT_BASE_URL = "http://localhost:3000"


# ---------------------------------------------------------------------------
# Scenario context — the input to every generator
# ---------------------------------------------------------------------------


@dataclass
class ScenarioContext:
    """Bundle of facts the generator needs.

    Reuses Phase 3's :class:`SchemaContext` for the spec/story bits and
    layers on the deploy-derived ``base_url`` (where the app is
    reachable inside the dev compose network). Phase 5 fills in
    ``base_url`` from ``docker inspect``; tests pass it explicitly.

    Per the agile/waterfall requirement:
      - ``flow_kind == "agile"`` → one scenario per acceptance criterion
        (``STORY-N.AC-M``), annotated with ``@verifies: STORY-N.AC-M``.
      - ``flow_kind == "waterfall"`` → one scenario per spec section
        anchor (e.g. ``SPEC_REQUIREMENTS.md#fr-001``), annotated with
        ``@verifies: <anchor>``.
    """

    schema: SchemaContext
    base_url: str = _DEFAULT_BASE_URL
    extra: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra context for custom generators (e.g. Phase 5's
    list of routes scraped from the deployed app)."""

    def to_normalised_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_normalised_dict(),
            "base_url": self.base_url,
            "extra": dict(sorted(self.extra.items())),
        }


def gather_scenario_context(
    workspace_path: str,
    *,
    base_url: str = _DEFAULT_BASE_URL,
    extra: Optional[dict[str, Any]] = None,
) -> ScenarioContext:
    """Build the :class:`ScenarioContext` for ``workspace_path``."""
    schema = gather_schema_context(workspace_path)
    return ScenarioContext(schema=schema, base_url=base_url, extra=dict(extra or {}))


def compute_scenario_cache_key(context: ScenarioContext) -> str:
    blob = json.dumps(context.to_normalised_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Scenario shape — internal representation generators return
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One runnable Playwright test the generator wants emitted.

    ``verifies`` is the teane-convention back-reference: ``STORY-3.AC-2``
    in agile workspaces, a spec section anchor in waterfall. The defect
    emitter (Phase 2) reads this off the .spec.ts file via the
    ``// @verifies:`` line and populates ``source_spec.md`` with the
    matching excerpt.
    """

    name: str
    verifies: str
    body: str
    """Already-rendered TypeScript body. Generators are responsible for
    producing real selectors / waits / assertions — the writer doesn't
    add any logic, only the test scaffolding."""


@dataclass
class SpecFile:
    """One Playwright .spec.ts the writer should emit."""

    filename: str
    """Relative to ``tests/e2e/`` — e.g. ``"login.spec.ts"``."""

    scenarios: list[Scenario] = field(default_factory=list)


ScenarioGenerator = Callable[[ScenarioContext], list[SpecFile]]


# ---------------------------------------------------------------------------
# Default offline generator
# ---------------------------------------------------------------------------


def fallback_scenarios(context: ScenarioContext) -> list[SpecFile]:
    """Deterministic, offline scenario generator.

    Emits one .spec.ts per agile story (or one spec for the whole
    waterfall workspace), with one scenario per AC / spec section.
    Bodies are TODO placeholders that nonetheless parse as valid
    Playwright — the runner can execute them; they'll fail until the
    LLM-backed generator replaces them with real assertions.
    """
    if context.schema.flow_kind == FLOW_AGILE:
        return _fallback_agile(context)
    return _fallback_waterfall(context)


def _fallback_agile(context: ScenarioContext) -> list[SpecFile]:
    specs: list[SpecFile] = []
    for story in context.schema.stories:
        story_key = story.get("story_key") or "STORY-UNKNOWN"
        title = story.get("title") or story_key
        slug = _slugify(title)
        ac_keys = story.get("acceptance_criteria_keys") or [f"{story_key}.AC-1"]
        scenarios = [
            Scenario(
                name=f"{story_key} {ac_key}",
                verifies=ac_key,
                body=_placeholder_body(context.base_url, ac_key),
            )
            for ac_key in ac_keys
        ]
        specs.append(SpecFile(filename=f"{slug}.spec.ts", scenarios=scenarios))
    if not specs:
        # Agile mode with no stories (shouldn't happen if detect_flow_kind
        # is agile, but be defensive) → fall back to waterfall layout.
        return _fallback_waterfall(context)
    return specs


def _fallback_waterfall(context: ScenarioContext) -> list[SpecFile]:
    anchors = _extract_spec_anchors(context.schema.spec_excerpts)
    if not anchors:
        anchors = [("SPEC_REQUIREMENTS.md#root", "Smoke check")]
    scenarios = [
        Scenario(
            name=label,
            verifies=anchor,
            body=_placeholder_body(context.base_url, anchor),
        )
        for anchor, label in anchors
    ]
    return [SpecFile(filename="smoke.spec.ts", scenarios=scenarios)]


_SECTION_HEADING = re.compile(r"^#{1,3}\s+(?P<text>.+?)\s*$", re.MULTILINE)


def _extract_spec_anchors(excerpts: dict[str, str]) -> list[tuple[str, str]]:
    """Parse markdown headings into (anchor, label) pairs.

    ``anchor`` is the file + slugified heading (e.g.
    ``SPEC_REQUIREMENTS.md#fr-001-user-login``). Phase 2's
    ``source_spec.md`` writer will read this back to look up the
    section excerpt.
    """
    out: list[tuple[str, str]] = []
    for filename, text in excerpts.items():
        base = os.path.basename(filename)
        for m in _SECTION_HEADING.finditer(text):
            label = m.group("text").strip()
            anchor = f"{base}#{_slugify(label)}"
            out.append((anchor, label))
    return out


def _placeholder_body(base_url: str, verifies: str) -> str:
    return (
        f"  await page.goto('{base_url}');\n"
        f"  // TODO Phase 4a spike replaces this with real assertions for {verifies}.\n"
        f"  await expect(page).toHaveTitle(/.+/);\n"
    )


def _slugify(text: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (cleaned or "spec")[:max_len].rstrip("-")


# ---------------------------------------------------------------------------
# Spec file writer
# ---------------------------------------------------------------------------


def generate_scenarios(
    context: ScenarioContext,
    *,
    generator: Optional[ScenarioGenerator] = None,
) -> list[SpecFile]:
    """Produce :class:`SpecFile` objects via ``generator`` (defaults to fallback)."""
    gen = generator or fallback_scenarios
    specs = gen(context)
    _validate_specs(specs)
    return specs


def _validate_specs(specs: list[SpecFile]) -> None:
    if not isinstance(specs, list):
        raise ValueError("generator must return list[SpecFile]")
    for spec in specs:
        if not isinstance(spec, SpecFile):
            raise ValueError(f"expected SpecFile, got {type(spec).__name__}")
        if not spec.filename.endswith(".spec.ts"):
            raise ValueError(f"spec filename must end in .spec.ts: {spec.filename!r}")
        for sc in spec.scenarios:
            if not isinstance(sc, Scenario):
                raise ValueError("scenarios must be Scenario instances")
            if not sc.name or not sc.verifies:
                raise ValueError("scenario.name and scenario.verifies are required")


def write_scenarios(
    workspace_path: str,
    specs: list[SpecFile],
    cache_key: str,
    *,
    e2e_dir: Optional[str] = None,
) -> list[str]:
    """Write ``specs`` to disk; return the list of written paths.

    Each file is rendered with a standard Playwright preamble and one
    ``test('name', ...)`` block per scenario. ``// @verifies: <anchor>``
    is emitted above each test so the defect emitter can recover the
    spec back-reference from the failure record.

    The cache key marker lives at ``tests/e2e/.cache_key``.
    """
    target_dir = _resolve_e2e_dir(workspace_path, e2e_dir)
    os.makedirs(target_dir, exist_ok=True)
    written: list[str] = []
    for spec in specs:
        path = os.path.join(target_dir, spec.filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_render_spec_file(spec))
        written.append(path)
    with open(os.path.join(target_dir, _CACHE_FILE), "w", encoding="utf-8") as fh:
        fh.write(cache_key)
    return written


def cached_scenarios_dir(
    workspace_path: str,
    cache_key: str,
    *,
    e2e_dir: Optional[str] = None,
) -> Optional[str]:
    """Return the e2e dir if its cache key matches, else None."""
    target_dir = _resolve_e2e_dir(workspace_path, e2e_dir)
    cache_path = os.path.join(target_dir, _CACHE_FILE)
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            stored = fh.read().strip()
    except OSError:
        return None
    return target_dir if stored == cache_key else None


def _resolve_e2e_dir(workspace_path: str, override: Optional[str]) -> str:
    if override is not None:
        return override
    return os.path.join(workspace_path, _E2E_DIR)


def _render_spec_file(spec: SpecFile) -> str:
    lines = [
        "// AUTO-GENERATED by teane test (Phase 4 scaffold).",
        "// Each test carries an `@verifies:` annotation back to the spec",
        "// artefact (acceptance criterion key for agile workspaces, spec",
        "// section anchor for waterfall) so failures emit traceable",
        "// CR-DEFECT-* records.",
        "",
        "import { test, expect } from '@playwright/test';",
        "",
    ]
    for sc in spec.scenarios:
        lines.append(f"// @verifies: {sc.verifies}")
        lines.append(f"test({json.dumps(sc.name)}, async ({{ page }}) => {{")
        lines.append(sc.body.rstrip("\n"))
        lines.append("});")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chromium runtime
# ---------------------------------------------------------------------------


def _chromium_cache_dirs() -> list[str]:
    # Playwright caches under ~/.cache/ms-playwright on Linux/macOS and
    # under %LOCALAPPDATA%\ms-playwright on Windows. Probe both so a
    # prior install on the current host is detected on either platform.
    dirs = [os.path.expanduser("~/.cache/ms-playwright")]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        dirs.append(os.path.join(local_appdata, "ms-playwright"))
    return dirs


def _chromium_cache_present() -> bool:
    for base in _chromium_cache_dirs():
        if not os.path.isdir(base):
            continue
        try:
            if any(name.startswith("chromium-") for name in os.listdir(base)):
                return True
        except OSError:
            continue
    return False


def ensure_chromium_installed(
    *,
    force: bool = False,
    runner: Optional[Callable[[list[str]], int]] = None,
) -> bool:
    """Make sure a chromium build is available for Playwright.

    Returns True on success (either already installed or installation
    completed cleanly). Logs and returns False on failure — callers
    surface that as a clear error rather than letting the Playwright
    subprocess fail later with a cryptic message.

    ``runner`` is injectable so tests can substitute the subprocess.
    """
    if not force and _chromium_cache_present():
        logger.info("[playwright_gen] chromium already installed; skipping")
        return True
    # Resolve npx so the .cmd/.exe shim is picked up on Windows —
    # subprocess.run without shell=True won't append the suffix itself.
    npx = shutil.which("npx")
    if npx is None and runner is None:
        logger.error("[playwright_gen] npx not found on PATH; install Node.js / Playwright manually")
        return False
    cmd = [npx or "npx", "playwright", "install", "chromium"]
    runner = runner or _default_runner
    try:
        rc = runner(cmd)
    except Exception as exc:  # noqa: BLE001
        logger.error("[playwright_gen] chromium install failed: %s", exc)
        return False
    if rc != 0:
        logger.error("[playwright_gen] `npx playwright install chromium` exited %s", rc)
        return False
    return True


def _default_runner(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(
            "[playwright_gen] %s failed (rc=%s): %s",
            " ".join(cmd), proc.returncode, (proc.stderr or "")[:400],
        )
    return proc.returncode


# Re-export for callers that want to plumb the same flow_kind detection
# without importing test_data_gen directly.
__all__ = [
    "FLOW_AGILE", "FLOW_WATERFALL",
    "Scenario", "SpecFile", "ScenarioContext",
    "gather_scenario_context", "compute_scenario_cache_key",
    "generate_scenarios", "fallback_scenarios",
    "write_scenarios", "cached_scenarios_dir",
    "ensure_chromium_installed",
    "detect_flow_kind",
]
