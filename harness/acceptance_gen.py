"""LLM-backed dual-altitude acceptance-scenario generator (ADR-0006, Phase 0).

Given a story and its acceptance criteria, this produces *runnable* acceptance
scenarios at two altitudes:

  * ``integration`` — an in-process HTTP test (pytest + the app's test client,
    real collaborators, seeded DB) that the build loop can run without a browser
    or a deploy. This is the altitude ADR-0006 runs *inside* the build.
  * ``e2e`` — a browser Playwright scenario for the post-deploy ``teane test``
    pass, one per AC.

Each AC is also *classified* ``backend-verifiable`` vs ``ui-only``. Only
backend-verifiable ACs get an ``integration`` scenario; every AC gets an ``e2e``
scenario so the post-deploy pass stays complete. ``ui-only`` ACs are the ones
ADR-0006 defers to the browser pass.

Design notes:
  * The generator core (:func:`generate_acceptance_scenarios`) is async and takes
    an injected ``gateway`` + budget, mirroring the structured-JSON dispatch in
    ``harness/decomposition.py``. That makes it unit-testable with a fake gateway
    and callable both standalone (the Phase-0 preview) and, later, from the async
    test pipeline / the Phase-1 ``acceptance_node``.
  * :func:`fallback_acceptance_scenarios` is the offline path (no gateway): it
    emits honest, clearly-marked TODO scaffolds — never a green-looking
    tautology. It exists so the layout/validation/rendering is exercisable
    without hitting the LLM, not as a substitute for it.
  * :func:`validate_scenarios` rejects the failure modes that make a scenario a
    rubber stamp: a missing/weak assertion, the ``toHaveTitle(/.+/)`` placeholder,
    an ``assert True``-style tautology, or a ``verifies`` that doesn't map to a
    real AC.

Nothing here is wired as a default; ADR-0006 keeps ``acceptance.*`` off until the
output is eyeballed on a real story (the Phase-0 "measure before wiring" gate).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ALTITUDE_INTEGRATION = "integration"
ALTITUDE_E2E = "e2e"
_ALTITUDES = frozenset({ALTITUDE_INTEGRATION, ALTITUDE_E2E})

CLASS_BACKEND = "backend-verifiable"
CLASS_UI = "ui-only"
_CLASSES = frozenset({CLASS_BACKEND, CLASS_UI})

# Config defaults — every cap is config-driven (see config/config.json
# ``acceptance``); these are the fall-throughs when a key is absent.
_DEFAULT_MAX_SCENARIOS_PER_STORY = 24
_DEFAULT_INTEGRATION_DIR = os.path.join("tests", "acceptance")
_DEFAULT_ALTITUDES = (ALTITUDE_INTEGRATION, ALTITUDE_E2E)
# Loopback base URL (with a port) the generated conftest hands the TestClient, so
# apps enforcing a loopback+port Host policy accept in-process requests.
_ACCEPTANCE_BASE_URL = "http://127.0.0.1:8000"

# The exact placeholder the Phase-4 scaffold emitted — a scenario whose only
# assertion is "the page has a non-empty title" is not a verification of
# anything, so we reject it outright.
_PLACEHOLDER_E2E_RE = re.compile(r"toHaveTitle\(\s*/\.\+/\s*\)")
# Tautological assertions that pass regardless of behaviour.
_TAUTOLOGY_RES = (
    re.compile(r"\bassert\s+True\b"),
    re.compile(r"\bassert\s+1\b(?!\d)"),
    re.compile(r"\bassert\s+(\w+)\s*==\s*\1\b"),  # assert x == x
    re.compile(r"expect\(\s*true\s*\)\.toBe\(\s*true\s*\)"),
)


# ---------------------------------------------------------------------------
# Context — what the generator sees for one story
# ---------------------------------------------------------------------------


@dataclass
class StoryAcceptanceContext:
    """Everything the generator needs to write real assertions for one story.

    Unlike Phase-3's ``SchemaContext`` (which carries AC *keys* only), this
    carries the AC *prose* (``text``) plus discovered routes and data-model /
    architecture excerpts — an LLM cannot assert against a criterion it can only
    see by key.
    """

    story_key: str
    title: str
    description: str
    acceptance_criteria: list[dict[str, Any]] = field(default_factory=list)
    """Each item: ``{"ac_key": "STORY-001.AC-3", "text": "Reject a future ..."}``."""
    routes: list[dict[str, str]] = field(default_factory=list)
    """Best-effort discovered HTTP routes: ``{"method": "POST", "path": "/contacts"}``."""
    data_model_excerpt: str = ""
    architecture_excerpt: str = ""
    stack: dict[str, Any] = field(default_factory=dict)
    """Backend/frontend/db hints, e.g. ``{"backend": "fastapi", "db": "sqlite"}``."""

    def ac_keys(self) -> set[str]:
        return {a.get("ac_key", "") for a in self.acceptance_criteria if a.get("ac_key")}


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class AcceptanceScenario:
    """One generated scenario at a single altitude."""

    verifies: str          # STORY-001.AC-3
    name: str
    altitude: str          # integration | e2e
    classification: str    # backend-verifiable | ui-only
    body: str              # rendered test body (pytest for integration; TS for e2e)
    rationale: str = ""    # short note: what it checks / why this classification

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifies": self.verifies,
            "name": self.name,
            "altitude": self.altitude,
            "classification": self.classification,
            "body": self.body,
            "rationale": self.rationale,
        }


@dataclass
class AcceptanceGenResult:
    """Validated scenarios for one story, plus the leftover budget and drops."""

    story_key: str
    scenarios: list[AcceptanceScenario] = field(default_factory=list)
    budget_remaining_usd: float = 0.0
    dropped: list[dict[str, str]] = field(default_factory=list)
    """Rejected candidates: ``{"verifies": ..., "altitude": ..., "reason": ...}``."""
    source: str = "llm"    # "llm" | "fallback"

    def integration(self) -> list[AcceptanceScenario]:
        return [s for s in self.scenarios if s.altitude == ALTITUDE_INTEGRATION]

    def e2e(self) -> list[AcceptanceScenario]:
        return [s for s in self.scenarios if s.altitude == ALTITUDE_E2E]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior test engineer writing ACCEPTANCE tests for one user story.

For each acceptance criterion you MUST decide a classification and emit runnable \
scenarios:

- classification "backend-verifiable": the criterion can be proven by calling the \
  HTTP API in-process (status codes, response bodies, persisted state). Emit BOTH \
  an "integration" scenario (pytest using the app's test client) AND an "e2e" \
  scenario (Playwright).
- classification "ui-only": the criterion is about rendered UI/interaction that \
  cannot be proven from the API alone (a modal opens, a toast appears, focus \
  moves). Emit ONLY an "e2e" scenario; set "integration" to null.

Hard requirements — a scenario with any of these is useless and will be rejected:
- Every scenario MUST contain at least one SUBSTANTIVE assertion tied to the \
  criterion's behaviour. Never emit `assert True`, `assert x == x`, or a bare \
  `expect(page).toHaveTitle(/.+/)`.
- integration bodies: use the injected `client` fixture (an in-process test \
  client). Assert real status codes AND response/persisted content. Do NOT import \
  the app, spin a server, or use a live URL — the harness provides `client`.
- integration bodies MUST be SELF-CONTAINED: arrange every prerequisite via the \
  API inside the test (e.g. POST a contact before you PATCH or DELETE it), and use \
  unique/generated field values so the test is robust to any shared state. NEVER \
  assume pre-seeded rows exist. For an "empty state" criterion, the harness gives \
  each test a fresh isolated database, so you may assert emptiness directly.
- e2e bodies: use `page` and real selectors/roles + `await expect(...)`. Assume \
  the app is served at the configured base URL.
- Do NOT mock internal modules (db, repositories, services). Exercise real \
  collaborators.
- Do NOT set `Host` or `Origin` request headers — the harness configures the \
  test client with an accepted loopback base URL. Overriding them yourself \
  usually breaks apps that enforce a loopback/CORS policy.
- Prefer one focused scenario per criterion; add edge cases only when the \
  criterion names them.

Return STRICT JSON, no prose, no markdown fences, exactly:
{
  "scenarios": [
    {
      "verifies": "STORY-001.AC-3",
      "classification": "backend-verifiable" | "ui-only",
      "rationale": "one sentence on what is checked / why this classification",
      "integration": { "name": "test_...", "body": "<pytest body, no def line>" } | null,
      "e2e": { "name": "human readable", "body": "<playwright body, no test() line>" }
    }
  ]
}
Emit one object per acceptance criterion, in order. `body` values are the INSIDE \
of the test function only (the harness adds the signature and the @verifies marker).\
"""


def build_user_prompt(ctx: StoryAcceptanceContext, *, max_scenarios: int) -> str:
    """Render the per-story user prompt from the context."""
    ac_lines = "\n".join(
        f"- {a.get('ac_key')}: {a.get('text', '').strip()}"
        for a in ctx.acceptance_criteria
    ) or "- (no acceptance criteria found)"
    route_lines = "\n".join(
        f"- {r.get('method', '?')} {r.get('path', '?')}" for r in ctx.routes
    ) or "- (no routes discovered — infer from the architecture excerpt)"
    stack_desc = ", ".join(f"{k}={v}" for k, v in sorted(ctx.stack.items())) or "unknown"
    parts = [
        f"# Story {ctx.story_key}: {ctx.title}",
        "",
        (ctx.description or "").strip(),
        "",
        "## Acceptance criteria (emit one scenario object per line, in order)",
        ac_lines,
        "",
        "## Known HTTP routes",
        route_lines,
        "",
        f"## Stack\n{stack_desc}",
    ]
    if ctx.data_model_excerpt.strip():
        parts += ["", "## Data model (excerpt)", ctx.data_model_excerpt.strip()]
    if ctx.architecture_excerpt.strip():
        parts += ["", "## Architecture (excerpt)", ctx.architecture_excerpt.strip()]
    parts += [
        "",
        f"Emit at most {max_scenarios} scenario objects total. Produce the JSON now.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Async LLM core
# ---------------------------------------------------------------------------


async def generate_acceptance_scenarios(
    ctx: StoryAcceptanceContext,
    *,
    gateway: Any,
    budget_remaining_usd: float,
    config: Optional[dict[str, Any]] = None,
) -> AcceptanceGenResult:
    """Generate + validate dual-altitude scenarios for one story via the LLM.

    ``gateway`` is dependency-injected (the global one in production, a fake in
    tests). Fail-soft: any dispatch/parse error returns an empty LLM result — the
    caller decides whether to fall back. Never raises on model output.
    """
    config = config or {}
    max_scenarios = int(config.get("max_scenarios_per_story", _DEFAULT_MAX_SCENARIOS_PER_STORY))
    altitudes = _resolve_altitudes(config)

    from harness.gateway import NodeRole  # lazy — avoid import cycle
    from harness.trust import strip_code_fences

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(ctx, max_scenarios=max_scenarios)},
    ]

    try:
        response, new_budget = await gateway.dispatch(
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=budget_remaining_usd,
            cache_family="acceptance:scenario_gen",
        )
    except Exception as exc:  # noqa: BLE001 — model/transport failure must not raise
        logger.warning("[acceptance_gen] dispatch failed for %s: %s", ctx.story_key, exc)
        return AcceptanceGenResult(
            story_key=ctx.story_key, budget_remaining_usd=budget_remaining_usd, source="llm",
        )

    raw = strip_code_fences(getattr(response, "content", "") or "")
    try:
        data = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[acceptance_gen] non-JSON response for %s: %s", ctx.story_key, exc)
        return AcceptanceGenResult(
            story_key=ctx.story_key, budget_remaining_usd=new_budget, source="llm",
        )

    candidates = _parse_candidates(data, altitudes)
    scenarios, dropped = validate_scenarios(candidates, ctx)
    return AcceptanceGenResult(
        story_key=ctx.story_key,
        scenarios=scenarios,
        budget_remaining_usd=new_budget,
        dropped=dropped,
        source="llm",
    )


def _resolve_altitudes(config: dict[str, Any]) -> frozenset[str]:
    raw = config.get("altitudes") or list(_DEFAULT_ALTITUDES)
    picked = frozenset(a for a in raw if a in _ALTITUDES)
    return picked or frozenset(_DEFAULT_ALTITUDES)


def _parse_candidates(
    data: dict[str, Any], altitudes: frozenset[str],
) -> list[AcceptanceScenario]:
    """Flatten the model's per-AC objects into per-altitude scenarios.

    Pure/structural — no validation of assertion quality here (that's
    :func:`validate_scenarios`); this only unpacks the JSON shape and drops
    altitudes the config didn't ask for.
    """
    out: list[AcceptanceScenario] = []
    scen = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(scen, list):
        return out
    for obj in scen:
        if not isinstance(obj, dict):
            continue
        verifies = str(obj.get("verifies", "")).strip()
        classification = str(obj.get("classification", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        for altitude in (ALTITUDE_INTEGRATION, ALTITUDE_E2E):
            if altitude not in altitudes:
                continue
            block = obj.get(altitude)
            if not isinstance(block, dict):
                continue
            body = str(block.get("body", "") or "")
            name = str(block.get("name", "") or "").strip() or f"{verifies} {altitude}"
            if not body.strip():
                continue
            out.append(AcceptanceScenario(
                verifies=verifies,
                name=name,
                altitude=altitude,
                classification=classification,
                body=body,
                rationale=rationale,
            ))
    return out


# ---------------------------------------------------------------------------
# Validation — reject rubber-stamp scenarios
# ---------------------------------------------------------------------------


def validate_scenarios(
    candidates: list[AcceptanceScenario], ctx: StoryAcceptanceContext,
) -> tuple[list[AcceptanceScenario], list[dict[str, str]]]:
    """Split candidates into (kept, dropped-with-reason).

    Drops anything that would let a scenario pass without verifying its
    criterion: bad enum, unknown ``verifies``, missing/weak/tautological
    assertion, or the Phase-4 title-check placeholder.
    """
    kept: list[AcceptanceScenario] = []
    dropped: list[dict[str, str]] = []
    known = ctx.ac_keys()

    def drop(sc: AcceptanceScenario, reason: str) -> None:
        dropped.append({"verifies": sc.verifies, "altitude": sc.altitude, "reason": reason})

    for sc in candidates:
        if sc.altitude not in _ALTITUDES:
            drop(sc, f"unknown altitude {sc.altitude!r}")
            continue
        if sc.classification not in _CLASSES:
            drop(sc, f"unknown classification {sc.classification!r}")
            continue
        if not sc.verifies:
            drop(sc, "missing verifies")
            continue
        if known and sc.verifies not in known:
            drop(sc, f"verifies {sc.verifies!r} not an AC of {ctx.story_key}")
            continue
        # ui-only criteria have no integration altitude by construction.
        if sc.classification == CLASS_UI and sc.altitude == ALTITUDE_INTEGRATION:
            drop(sc, "ui-only criterion cannot have an integration scenario")
            continue
        reason = _assertion_weakness(sc)
        if reason:
            drop(sc, reason)
            continue
        kept.append(sc)
    return kept, dropped


def _assertion_weakness(sc: AcceptanceScenario) -> Optional[str]:
    """Return a rejection reason if the body has no substantive assertion."""
    body = sc.body
    for taut in _TAUTOLOGY_RES:
        if taut.search(body):
            return "tautological assertion"
    if sc.altitude == ALTITUDE_INTEGRATION:
        if "assert" not in body:
            return "integration body has no assert"
        # A single status-code check with no content is weak but acceptable;
        # a body that never references the client/response is not a test.
        if not re.search(r"\bclient\b|\bresponse\b|\bresp\b", body):
            return "integration body never calls the client"
        # The body must form a valid function once indented — otherwise it would
        # break collection for the WHOLE file (all ACs in it), so drop just this
        # one. Compiling the rendered `def` catches model syntax slips early.
        try:
            compile(f"def _f(client):\n{_indent_body(body)}\n", "<acceptance>", "exec")
        except SyntaxError:
            return "integration body is not valid python"
        return None
    # e2e
    if _PLACEHOLDER_E2E_RE.search(body):
        return "placeholder toHaveTitle(/.+/)"
    if "expect(" not in body:
        return "e2e body has no expect() assertion"
    return None


# ---------------------------------------------------------------------------
# Offline fallback (no gateway) — honest scaffolds, never green tautologies
# ---------------------------------------------------------------------------

_UI_HINT_RE = re.compile(
    r"\b(modal|button|click|screen|page|dashboard|render|display|toast|focus|"
    r"UI|form|dropdown|close|open|refresh)\b",
    re.IGNORECASE,
)


def fallback_acceptance_scenarios(ctx: StoryAcceptanceContext) -> AcceptanceGenResult:
    """Deterministic offline generator: honest TODO scaffolds per AC.

    Classifies heuristically (UI-ish wording → ui-only) and emits bodies that
    are explicitly marked as needing real assertions and are DESIGNED to fail
    (``pytest.fail`` / ``expect(false)``) so a fallback can never masquerade as a
    passing verification.
    """
    scenarios: list[AcceptanceScenario] = []
    for a in ctx.acceptance_criteria:
        ac_key = a.get("ac_key", "")
        text = (a.get("text", "") or "").strip()
        if not ac_key:
            continue
        ui_only = bool(_UI_HINT_RE.search(text)) and not _looks_backend(text)
        classification = CLASS_UI if ui_only else CLASS_BACKEND
        if not ui_only:
            scenarios.append(AcceptanceScenario(
                verifies=ac_key,
                name=f"test_{_ident(ac_key)}",
                altitude=ALTITUDE_INTEGRATION,
                classification=classification,
                body=(
                    f"    # TODO(ADR-0006): real integration assertions for: {text}\n"
                    f"    response = client.get('/')\n"
                    f"    pytest.fail('acceptance scenario not yet generated (fallback)')"
                ),
                rationale="offline fallback scaffold",
            ))
        scenarios.append(AcceptanceScenario(
            verifies=ac_key,
            name=f"{ac_key} {text[:60]}",
            altitude=ALTITUDE_E2E,
            classification=classification,
            body=(
                f"  // TODO(ADR-0006): real e2e assertions for: {text}\n"
                f"  await page.goto('/');\n"
                f"  expect(false).toBe(true); // fallback: not yet generated"
            ),
            rationale="offline fallback scaffold",
        ))
    return AcceptanceGenResult(
        story_key=ctx.story_key, scenarios=scenarios, source="fallback",
    )


def _looks_backend(text: str) -> bool:
    return bool(re.search(r"\b(API|status|422|400|413|201|204|endpoint|payload|JSON|"
                          r"reject|returns?)\b", text, re.IGNORECASE))


def _ident(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "ac"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def integration_function_name(sc: AcceptanceScenario) -> str:
    """Deterministic pytest function name for an integration scenario.

    Shared by the renderer (which emits ``def <name>(client):``) and the runner
    (which maps a pytest nodeid back to the scenario's ``verifies`` AC), so the
    two never disagree.
    """
    return sc.name if sc.name.startswith("test_") else f"test_{_ident(sc.name)}"


def _indent_body(body: str, indent: str = "    ") -> str:
    """Normalise a generated test body to a single indent level under ``def``.

    Models return function bodies inconsistently — sometimes at column 0,
    sometimes pre-indented — which breaks Python's block syntax if pasted
    verbatim under ``def test(...):``. Dedent to the common prefix, then re-indent
    every non-blank line by one level; relative nesting inside the body is
    preserved. A blank body becomes a ``pass`` so the function still parses.
    """
    import textwrap

    dedented = textwrap.dedent(body).strip("\n")
    if not dedented.strip():
        return f"{indent}pass"
    out = []
    for line in dedented.splitlines():
        out.append(indent + line if line.strip() else "")
    return "\n".join(out)


def render_integration_file(story_key: str, scenarios: list[AcceptanceScenario]) -> str:
    """Render integration scenarios into one pytest module.

    Assumes a ``client`` fixture provided by the workspace's conftest (an
    in-process test client). Each test carries a ``# @verifies:`` marker so the
    traceability layer can link it to its AC.
    """
    integ = [s for s in scenarios if s.altitude == ALTITUDE_INTEGRATION]
    lines = [
        f'"""Acceptance tests for {story_key} (ADR-0006 integration altitude).',
        "",
        "AUTO-GENERATED. Each test asserts one acceptance criterion in-process via",
        "the `client` fixture (real collaborators, seeded DB). The `@verifies:`",
        "marker links the test to its criterion for traceability.",
        '"""',
        "",
        "import pytest",
        "",
    ]
    for sc in integ:
        fn = integration_function_name(sc)
        lines.append(f"# @verifies: {sc.verifies}")
        lines.append(f"def {fn}(client):")
        lines.append(_indent_body(sc.body))
        lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_e2e_spec(story_key: str, scenarios: list[AcceptanceScenario]) -> str:
    """Render e2e scenarios into one Playwright ``.spec.ts`` string.

    Reuses the ``@verifies`` convention the Phase-2 defect emitter reads.
    """
    e2e = [s for s in scenarios if s.altitude == ALTITUDE_E2E]
    lines = [
        f"// Acceptance E2E for {story_key} (ADR-0006 e2e altitude). AUTO-GENERATED.",
        "import { test, expect } from '@playwright/test';",
        "",
    ]
    for sc in e2e:
        lines.append(f"// @verifies: {sc.verifies}")
        lines.append(f"test({json.dumps(sc.name)}, async ({{ page }}) => {{")
        lines.append(sc.body.rstrip("\n"))
        lines.append("});")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

_ROUTE_DECORATOR_RE = re.compile(
    r"@(?:app|router)\.(get|post|put|patch|delete)\(\s*[\"']([^\"']*)[\"']",
    re.IGNORECASE,
)
_ROUTER_PREFIX_RE = re.compile(r"APIRouter\([^)]*prefix\s*=\s*[\"']([^\"']+)[\"']")


def discover_routes(workspace_path: str, *, max_files: int = 200) -> list[dict[str, str]]:
    """Best-effort HTTP route discovery by scanning Python for FastAPI decorators.

    Resolves a file's routes against the first ``APIRouter(prefix=...)`` declared
    in that same file. Router-relative when no prefix is found — passed to the LLM
    as a hint, not a contract.
    """
    routes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    scanned = 0
    for root, _dirs, files in os.walk(workspace_path):
        if any(part in root for part in (os.sep + "node_modules", os.sep + ".venv",
                                         os.sep + ".git", os.sep + "__pycache__")):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            if scanned >= max_files:
                return routes
            scanned += 1
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            pm = _ROUTER_PREFIX_RE.search(text)
            prefix = pm.group(1) if pm else ""
            for method, sub in _ROUTE_DECORATOR_RE.findall(text):
                full = (prefix + sub) or "/"
                key = (method.upper(), full)
                if key in seen:
                    continue
                seen.add(key)
                routes.append({"method": method.upper(), "path": full})
    return routes


def _read_excerpt(workspace_path: str, rel: str, *, max_chars: int) -> str:
    path = os.path.join(workspace_path, rel)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_chars)
    except OSError:
        return ""


_APP_FACTORY_RE = re.compile(r"^\s*def\s+create_app\s*\(", re.MULTILINE)
_APP_SINGLETON_RE = re.compile(r"^\s*app\s*(?::\s*FastAPI\s*)?=\s*FastAPI\s*\(", re.MULTILINE)


def discover_app_factory(workspace_path: str, *, max_files: int = 300) -> Optional[dict[str, str]]:
    """Best-effort discovery of the FastAPI app entrypoint for the conftest.

    Returns ``{"module": "server.app.main", "symbol": "create_app", "kind":
    "factory"|"singleton"}`` or None. The dotted module path is derived from the
    file's path relative to the workspace root. None → the node cannot build a
    ``client`` fixture and every integration AC safely defers (dependency-blocked).
    """
    factory: Optional[dict[str, str]] = None
    singleton: Optional[dict[str, str]] = None
    _skip = {"node_modules", ".venv", ".git", "__pycache__", "tests", "test"}
    scanned = 0
    for root, _dirs, files in os.walk(workspace_path):
        rel = os.path.relpath(root, workspace_path)
        if rel != "." and _skip.intersection(rel.split(os.sep)):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            if scanned >= max_files:
                break
            scanned += 1
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            module = os.path.relpath(path, workspace_path).replace(os.sep, ".")[:-3]
            if factory is None and _APP_FACTORY_RE.search(text):
                factory = {"module": module, "symbol": "create_app", "kind": "factory"}
            if singleton is None and _APP_SINGLETON_RE.search(text):
                singleton = {"module": module, "symbol": "app", "kind": "singleton"}
    return factory or singleton


# pydantic-settings config classes declare an env prefix; a DB-path field under
# that prefix is controllable via ``{PREFIX}{FIELD_UPPER}``. Discovering it lets
# the conftest give each test an isolated database.
_ENV_PREFIX_RE = re.compile(r"env_prefix\s*=\s*[\"']([A-Za-z0-9_]+)[\"']")
_DB_FIELD_RE = re.compile(r"^\s*(db_path|database_path|db_file|sqlite_path)\s*[:=]", re.MULTILINE)


def discover_db_env_var(workspace_path: str, *, max_files: int = 300) -> Optional[str]:
    """Best-effort discovery of the env var that overrides the app's DB path.

    Looks for a pydantic-settings class with an ``env_prefix`` and a DB-path
    field (``db_path`` etc.), and returns ``{PREFIX}{FIELD_UPPER}`` (e.g.
    ``LUMINA_DB_PATH``). None when nothing matches — the conftest then skips
    isolation and relies on self-contained scenarios.
    """
    _skip = {"node_modules", ".venv", ".git", "__pycache__", "tests", "test"}
    scanned = 0
    for root, _dirs, files in os.walk(workspace_path):
        rel = os.path.relpath(root, workspace_path)
        if rel != "." and _skip.intersection(rel.split(os.sep)):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            if scanned >= max_files:
                return None
            scanned += 1
            try:
                with open(os.path.join(root, name), "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            pm = _ENV_PREFIX_RE.search(text)
            fm = _DB_FIELD_RE.search(text)
            if pm and fm:
                return f"{pm.group(1)}{fm.group(1).upper()}"
    return None


# Self-contained (stdlib-only) seed applier inlined into the generated conftest.
# It CANNOT import teane's harness package — the conftest runs in the workspace
# sandbox, which has none of teane installed. Inserts into the already-migrated
# schema; unknown columns / bad rows are skipped per-row so a schema drift can
# never break the fixture (the run engine would defer, never false-fail).
_SEED_APPLIER_SRC = '''\
def _apply_seed(db_path):
    import json, os, sqlite3
    _p = os.path.join(os.path.dirname(__file__), "seed.json")
    if not os.path.isfile(_p):
        return
    try:
        with open(_p, encoding="utf-8") as _fh:
            _tables = (json.load(_fh) or {}).get("tables", {})
    except Exception:
        return
    _conn = sqlite3.connect(db_path)
    try:
        for _t, _rows in _tables.items():
            for _row in (_rows or []):
                _cols = [k for k in _row if not k.startswith("_")]
                if not _cols:
                    continue
                _vals = [
                    json.dumps(_row[c]) if isinstance(_row[c], (dict, list)) else _row[c]
                    for c in _cols
                ]
                _ph = ",".join("?" for _ in _cols)
                try:
                    _conn.execute(
                        "INSERT INTO %s (%s) VALUES (%s)" % (_t, ",".join(_cols), _ph),
                        _vals,
                    )
                except Exception:
                    continue
        _conn.commit()
    finally:
        _conn.close()
'''


def render_acceptance_conftest(
    discovery: dict[str, str],
    *,
    db_env_var: Optional[str] = None,
    seed: bool = False,
) -> str:
    """Render a pytest ``conftest.py`` providing the ``client`` fixture.

    Builds a FastAPI ``TestClient`` from the discovered app (factory or
    singleton). When ``db_env_var`` is known, each test gets a FRESH isolated
    SQLite database (the env var is pointed at a per-test temp file before the app
    is built, so the app migrates a clean DB) — this is what lets "empty state"
    and mutation criteria verify deterministically, and it is also the only mode
    that can apply a seed (a known DB path). When ``seed`` is set, a sibling
    ``seed.json`` is inserted into the migrated schema after app startup. Without
    isolation the fixture still yields a client and self-contained scenarios carry
    verification. Any build/seed failure surfaces as a fixture error, which the run
    engine defers — never a false acceptance failure.
    """
    module = discovery["module"]
    symbol = discovery["symbol"]
    kind = discovery.get("kind", "factory")
    build = f"{symbol}()" if kind == "factory" else symbol
    seed = seed and bool(db_env_var)  # seeding needs a known (isolated) DB path

    header = (
        '"""AUTO-GENERATED acceptance conftest (ADR-0006). Provides the in-process\n'
        '`client` fixture the integration acceptance tests use."""\n\n'
        "import os\n"
        "import shutil\n"
        "import tempfile\n"
        "import pytest\n"
        "from fastapi.testclient import TestClient\n\n"
        f"from {module} import {symbol}\n\n"
        # A loopback base URL WITH a port so apps that harden the bind/Host (e.g.
        # a "loopback + port required" NFR) accept the in-process test requests.
        f"_ACCEPTANCE_BASE_URL = {_ACCEPTANCE_BASE_URL!r}\n\n\n"
        # Drop caller-supplied Host/Origin so the loopback base_url always wins —
        # a generated test that sets its own Host must not defeat a loopback/CORS
        # policy. Deterministic; does not depend on the model omitting headers.
        "class _Client(TestClient):\n"
        "    def request(self, method, url, **kwargs):\n"
        "        _h = kwargs.get('headers')\n"
        "        if _h:\n"
        "            kwargs['headers'] = {k: v for k, v in dict(_h).items()\n"
        "                                 if k.lower() not in ('host', 'origin')}\n"
        "        return super().request(method, url, **kwargs)\n\n\n"
    )
    if seed:
        header += _SEED_APPLIER_SRC + "\n\n"

    if db_env_var:
        seed_call = "            _apply_seed(_db)\n" if seed else ""
        # A fresh temp DIRECTORY (not just a temp file): apps commonly harden the
        # DB's parent dir (chmod 0700) on startup, which fails against the shared
        # system temp root. A per-test dir gives them their own directory to lock.
        return header + (
            "@pytest.fixture\n"
            "def client():\n"
            "    # Isolate each test with a fresh temp database so mutation and\n"
            "    # empty-state criteria are deterministic.\n"
            "    _dir = tempfile.mkdtemp(prefix='acc_')\n"
            "    _db = os.path.join(_dir, 'acceptance.db')\n"
            f"    _prev = os.environ.get({db_env_var!r})\n"
            f"    os.environ[{db_env_var!r}] = _db\n"
            "    try:\n"
            f"        app = {build}\n"
            "        with _Client(app, base_url=_ACCEPTANCE_BASE_URL, raise_server_exceptions=False) as c:\n"
            f"{seed_call}"
            "            yield c\n"
            "    finally:\n"
            "        if _prev is None:\n"
            f"            os.environ.pop({db_env_var!r}, None)\n"
            "        else:\n"
            f"            os.environ[{db_env_var!r}] = _prev\n"
            "        shutil.rmtree(_dir, ignore_errors=True)\n"
        )

    return header + (
        "@pytest.fixture\n"
        "def client():\n"
        f"    app = {build}\n"
        "    with _Client(app, base_url=_ACCEPTANCE_BASE_URL, raise_server_exceptions=False) as c:\n"
        "        yield c\n"
    )


def gather_story_acceptance_context(
    workspace_path: str,
    story_key: str,
    *,
    stack: Optional[dict[str, Any]] = None,
    excerpt_chars: int = 6000,
) -> Optional[StoryAcceptanceContext]:
    """Assemble the :class:`StoryAcceptanceContext` for one story.

    Pulls the story + AC prose from the shared story DB, discovers routes, and
    reads data-model / architecture excerpts. Returns None when the story can't
    be located (caller decides how to surface that).
    """
    from harness import story_state  # lazy — DB layer

    try:
        app = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db(workspace_path=workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[acceptance_gen] could not open story DB: %s", exc)
        return None
    try:
        story = None
        for s in story_state.list_stories(conn, app):
            if s.get("story_key") == story_key:
                story = s
                break
        if story is None:
            logger.warning("[acceptance_gen] story %s not found in %s", story_key, app)
            return None
        acs = story_state.list_acceptance_criteria(conn, app, story["id"])
        ac_rows = [{"ac_key": a.get("ac_key"), "text": a.get("text", "")}
                   for a in acs if a.get("ac_key")]
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    return StoryAcceptanceContext(
        story_key=story_key,
        title=story.get("title", story_key),
        description=story.get("description", "") or "",
        acceptance_criteria=ac_rows,
        routes=discover_routes(workspace_path),
        data_model_excerpt=_read_excerpt(workspace_path, "docs/SPEC_DATA_MODEL.md", max_chars=excerpt_chars),
        architecture_excerpt=_read_excerpt(workspace_path, "docs/SPEC_ARCHITECTURE.md", max_chars=excerpt_chars),
        stack=dict(stack or {}),
    )


# ---------------------------------------------------------------------------
# LLM seed-data generator (ADR-0006 Phase 0)
# ---------------------------------------------------------------------------
#
# The Phase-3 ``test_data_gen.fallback_seed`` emits a single ``_teane_test_meta``
# stub row — useless for exercising an acceptance criterion. This produces
# schema-typed rows grounded in the workspace's ACTUAL ``CREATE TABLE`` DDL (not
# a spec doc, which may be absent — lumina has no SPEC_DATA_MODEL.md), tagged
# with ``_verifies`` so a scenario can find the row that supports its criterion.
# Output is the same ``{"tables": {...}}`` shape ``apply_seed_to_sqlite`` expects.

_DEFAULT_MAX_SEED_ROWS_PER_TABLE = 20

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"[\"'`]?([A-Za-z_]\w*)[\"'`]?\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class SeedContext:
    """Grounding for the seed generator: real schema + the ACs to exercise."""

    flow_kind: str = ""
    table_schemas: list[dict[str, str]] = field(default_factory=list)
    """Each item: ``{"table": "contacts", "ddl": "CREATE TABLE contacts (...);"}``."""
    data_model_excerpt: str = ""
    stories: list[dict[str, Any]] = field(default_factory=list)
    """Each: ``{"story_key", "title", "acceptance_criteria": [{"ac_key","text"}]}``."""

    def table_names(self) -> set[str]:
        return {t.get("table", "") for t in self.table_schemas if t.get("table")}

    def ac_keys(self) -> set[str]:
        out: set[str] = set()
        for s in self.stories:
            for a in s.get("acceptance_criteria", []):
                if a.get("ac_key"):
                    out.add(a["ac_key"])
        return out


@dataclass
class SeedGenResult:
    seed: dict[str, Any] = field(default_factory=lambda: {"tables": {}})
    budget_remaining_usd: float = 0.0
    dropped: list[dict[str, str]] = field(default_factory=list)
    source: str = "llm"

    def row_count(self) -> int:
        return sum(len(rows) for rows in self.seed.get("tables", {}).values())


_SEED_SYSTEM_PROMPT = """\
You generate realistic SEED DATA for acceptance tests of a web app.

You are given the app's real CREATE TABLE schema and the acceptance criteria the
seed must help exercise. Produce rows that make those criteria checkable — e.g. a
contact whose birthday is a few days away for a "countdown" criterion, a row to
edit, a row to delete.

Rules:
- Emit ONLY tables and columns that exist in the given schema. Match column types
  and honour NOT NULL / constraints (dates as ISO 'YYYY-MM-DD', etc.).
- Omit auto-increment primary keys unless a criterion needs a specific id.
- Tag a row with "_verifies": "STORY-N.AC-M" when it exists to support that
  criterion. "_verifies" is stripped before INSERT — never a real column.
- Realistic, varied values. Enough rows to exercise the criteria; do not pad.
- No SQL, no prose, no markdown fences.

Return STRICT JSON exactly:
{ "tables": { "<table>": [ { "<col>": <value>, "_verifies": "STORY-N.AC-M" }, ... ] } }\
"""


def build_seed_prompt(ctx: SeedContext, *, max_rows: int) -> str:
    ddl = "\n\n".join(t.get("ddl", "") for t in ctx.table_schemas) or "(no CREATE TABLE found)"
    ac_lines = []
    for s in ctx.stories:
        for a in s.get("acceptance_criteria", []):
            if a.get("ac_key"):
                ac_lines.append(f"- {a['ac_key']}: {a.get('text', '').strip()}")
    acs = "\n".join(ac_lines) or "- (no acceptance criteria found)"
    parts = [
        "## Schema (authoritative — emit only these tables/columns)",
        ddl,
        "",
        "## Acceptance criteria the seed should help exercise",
        acs,
    ]
    if ctx.data_model_excerpt.strip():
        parts += ["", "## Data model notes", ctx.data_model_excerpt.strip()]
    parts += ["", f"At most {max_rows} rows per table. Produce the JSON now."]
    return "\n".join(parts)


async def generate_seed_data_llm(
    ctx: SeedContext,
    *,
    gateway: Any,
    budget_remaining_usd: float,
    config: Optional[dict[str, Any]] = None,
) -> SeedGenResult:
    """Generate + validate schema-typed seed rows via the LLM. Fail-soft."""
    config = config or {}
    max_rows = int(config.get("max_seed_rows_per_table", _DEFAULT_MAX_SEED_ROWS_PER_TABLE))

    from harness.gateway import NodeRole  # lazy
    from harness.trust import strip_code_fences

    messages = [
        {"role": "system", "content": _SEED_SYSTEM_PROMPT},
        {"role": "user", "content": build_seed_prompt(ctx, max_rows=max_rows)},
    ]
    try:
        response, new_budget = await gateway.dispatch(
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=budget_remaining_usd,
            cache_family="acceptance:seed_gen",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[acceptance_gen] seed dispatch failed: %s", exc)
        return SeedGenResult(budget_remaining_usd=budget_remaining_usd, source="llm")

    raw = strip_code_fences(getattr(response, "content", "") or "")
    try:
        data = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[acceptance_gen] seed non-JSON response: %s", exc)
        return SeedGenResult(budget_remaining_usd=new_budget, source="llm")

    seed, dropped = validate_seed(data, ctx, max_rows=max_rows)
    return SeedGenResult(seed=seed, budget_remaining_usd=new_budget, dropped=dropped, source="llm")


def validate_seed(
    data: Any, ctx: SeedContext, *, max_rows: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Sanitise LLM seed output against the real schema.

    Drops unknown tables, empty tables, and rows with no real (non-``_``) column;
    caps rows per table; keeps only ``{"tables": {...}}``. Never raises.
    """
    from harness.test_data_gen import _validate_seed_shape

    dropped: list[dict[str, str]] = []
    tables_out: dict[str, list[dict[str, Any]]] = {}
    known = ctx.table_names()

    raw_tables = data.get("tables") if isinstance(data, dict) else None
    if not isinstance(raw_tables, dict):
        return {"tables": {}}, [{"table": "*", "reason": "no tables object"}]

    for name, rows in raw_tables.items():
        if not isinstance(name, str) or not name:
            continue
        if known and name not in known:
            dropped.append({"table": name, "reason": "not in schema"})
            continue
        if not isinstance(rows, list):
            dropped.append({"table": name, "reason": "rows not a list"})
            continue
        kept_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            real_cols = [k for k in row if not k.startswith("_")]
            if not real_cols:
                dropped.append({"table": name, "reason": "row has no real column"})
                continue
            kept_rows.append(row)
            if len(kept_rows) >= max_rows:
                break
        if kept_rows:
            tables_out[name] = kept_rows
        else:
            dropped.append({"table": name, "reason": "no valid rows"})

    seed = {"tables": tables_out}
    try:
        _validate_seed_shape(seed)
    except ValueError:
        return {"tables": {}}, dropped + [{"table": "*", "reason": "failed shape validation"}]
    return seed, dropped


def discover_table_schemas(
    workspace_path: str, *, max_files: int = 300,
) -> list[dict[str, str]]:
    """Best-effort ``CREATE TABLE`` extraction from .py/.sql across the workspace.

    Returns ``[{"table": name, "ddl": "<full CREATE TABLE ...;>"}]``. The DDL is
    authoritative schema grounding — better than a spec doc, which may be absent.
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    scanned = 0
    for root, _dirs, files in os.walk(workspace_path):
        if any(part in root for part in (os.sep + "node_modules", os.sep + ".venv",
                                         os.sep + ".git", os.sep + "__pycache__")):
            continue
        for name in files:
            if not name.endswith((".py", ".sql")):
                continue
            if scanned >= max_files:
                return out
            scanned += 1
            try:
                with open(os.path.join(root, name), "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            for m in _CREATE_TABLE_RE.finditer(text):
                table = m.group(1)
                if table in seen:
                    continue
                seen.add(table)
                out.append({"table": table, "ddl": m.group(0).strip()})
    return out


def gather_seed_context(
    workspace_path: str, *, excerpt_chars: int = 6000,
) -> SeedContext:
    """Assemble the workspace-level :class:`SeedContext` (schema + all stories' ACs)."""
    from harness import story_state  # lazy

    stories: list[dict[str, Any]] = []
    flow_kind = ""
    try:
        from harness.test_data_gen import detect_flow_kind
        flow_kind = detect_flow_kind(workspace_path)
        app = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db(workspace_path=workspace_path)
        try:
            for s in story_state.list_stories(conn, app):
                acs = story_state.list_acceptance_criteria(conn, app, s["id"])
                stories.append({
                    "story_key": s.get("story_key"),
                    "title": s.get("title"),
                    "acceptance_criteria": [
                        {"ac_key": a.get("ac_key"), "text": a.get("text", "")}
                        for a in acs if a.get("ac_key")
                    ],
                })
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("[acceptance_gen] seed context story gather failed: %s", exc)

    return SeedContext(
        flow_kind=flow_kind,
        table_schemas=discover_table_schemas(workspace_path),
        data_model_excerpt=_read_excerpt(workspace_path, "docs/SPEC_DATA_MODEL.md", max_chars=excerpt_chars),
        stories=stories,
    )


__all__ = [
    "ALTITUDE_INTEGRATION", "ALTITUDE_E2E", "CLASS_BACKEND", "CLASS_UI",
    "StoryAcceptanceContext", "AcceptanceScenario", "AcceptanceGenResult",
    "generate_acceptance_scenarios", "fallback_acceptance_scenarios",
    "validate_scenarios", "build_user_prompt",
    "render_integration_file", "render_e2e_spec", "integration_function_name",
    "discover_routes", "gather_story_acceptance_context",
    "discover_app_factory", "render_acceptance_conftest", "discover_db_env_var",
    # Seed generator
    "SeedContext", "SeedGenResult", "generate_seed_data_llm", "validate_seed",
    "build_seed_prompt", "discover_table_schemas", "gather_seed_context",
]
