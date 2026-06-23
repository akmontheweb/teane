"""
LangGraph StateGraph topology, typed state schema, and all node implementations.

This module defines the complete graph execution pipeline:
    planning_node → patching_node → compiler_node
                                       │
                             ┌─────────┼──────────┐
                             │ exit 0  │ exit ≠ 0 │
                             ▼         ▼           │
                            END    loop < 3?       │
                                   │      │        │
                                yes      no        │
                                   │      │        │
                                   ▼      ▼        │
                              repair_node  human_intervention_node
                                   │              │
                                   └──────┬───────┘
                                          │
                                    compiler_node (re-validation)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable, Literal, Optional, cast

from typing_extensions import TypedDict

from harness import _platform
from harness.sandbox import BUILDER_IMAGE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Strongly-Typed State Schema (TypedDict + Pydantic)
# ---------------------------------------------------------------------------

class TokenTrackerDict(TypedDict, total=False):
    """Cumulative token cost tracking aggregated across all LLM calls."""
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    total_cost_usd: float
    per_model: dict[str, dict[str, int | float]]


class DiagnosticObjectDict(TypedDict):
    """Structured compiler diagnostic parsed from build output."""
    file: str
    line: int
    column: int
    severity: Literal["error", "warning"]
    error_code: str
    message: str
    semantic_context: str


class MessageDict(TypedDict, total=False):
    """A single conversation turn in the messages array.

    ``content`` may be a plain string (the common case) or a list of
    Anthropic-style structured content blocks (text / tool_use /
    tool_result) for messages that span multiple block types.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: Any
    name: Optional[str]
    tool_calls: Optional[list[dict[str, Any]]]
    tool_call_id: Optional[str]


class AgentState(TypedDict, total=False):
    """
    The complete LangGraph agent state.

    Fields:
        workspace_path: Absolute path to the target repository root.
        messages: The conversation history array (system prompt anchored at index 0).
        modified_files: List of file paths edited during the active session.
        compiler_errors: Structured diagnostic objects from the last compiler run.
        token_tracker: Cumulative token usage and cost across all LLM calls.
        loop_counter: Tracks how many times each file/node has been visited
                      to prevent endless repair loops. Keys: 'patching', 'repair', 'compiler'.
        allow_network: Whether the sandbox namespace permits outbound network traffic.
        build_command: The resolved build command string (e.g., 'make build').
        budget_remaining_usd: Remaining dollar budget for LLM calls.
        session_id: Unique identifier for this graph invocation.
        exit_code: Last compiler exit code (0 = success, non-zero = failure).
        node_state: Internal routing metadata set by nodes.
    """
    workspace_path: str
    messages: list[MessageDict]
    modified_files: list[str]
    compiler_errors: list[DiagnosticObjectDict]
    token_tracker: TokenTrackerDict
    # Mostly int counters per node ("planning", "patching", "repair", ...),
    # but a few sentinel-string entries piggy-back on the dict too
    # (e.g. "missing_dep_last_symbol", "replace_block_misses_per_file").
    loop_counter: dict[str, Any]
    allow_network: bool
    build_command: str
    budget_remaining_usd: float
    session_id: str
    exit_code: int
    node_state: dict[str, Any]
    current_gate: str  # "REQUIREMENTS" | "ARCHITECTURE" | "DEPLOYMENT" | ""
    spec_requirements_path: str
    spec_architecture_path: str
    deployment_blueprint_path: str
    skip_discovery: bool
    # Paths of test files the test_generation_node has written this session.
    # Telemetry / status display only; the patcher continues to use
    # `modified_files` as the source of truth for git staging.
    generated_tests: list[str]
    # Config carried in state so test_generation_node can read it without
    # round-tripping through cli.py's config loader. Loaded from the
    # "test_generation" section of config/config.json.
    test_generation_config: dict[str, Any]
    # Speculative-execution branching parameters. Loaded from the
    # "speculative" section of config/config.json. Keys: num_variants
    # (default 3), temperature (default 0.3), selection_strategy
    # (default "first_success"; "fewest_changes" / "all_pass" supported).
    speculative_config: dict[str, Any]
    # Reviewer LLM artifacts. Each is independently populated; either may be
    # absent if the corresponding *_reviewer_primary slot is unset.
    reviewer_comments_requirements: str
    reviewer_comments_code: str
    # Discovery-shaped follow-up questions the doc reviewer wants the user to
    # answer in a second pass of the interview loop.
    reviewer_followups: list[dict[str, Any]]
    # The latest discovery JSON emitted by requirements/architecture/deployment
    # discovery nodes. Shape: {"modules": [{"name": str, "questions": [...]}],
    # "complete": bool, "summary": str}. MUST be declared here — without it,
    # LangGraph's state channel layer silently drops the key on the
    # discovery_node → discovery_interview_loop hop, leaving the loop with
    # no modules to render even though the LLM returned a full set.
    discovery_questions: dict[str, Any]
    # Per-node config sections plumbed through state so the graph nodes can
    # read them without round-tripping through cli.py's config loader. Each
    # is loaded from the corresponding section of config/config.json and
    # written into initial_state by run_graph. MUST be declared here — any
    # key the run_graph initializer sets that isn't in this TypedDict is
    # silently dropped by LangGraph's state channel layer, falling back to
    # in-node defaults (e.g. compiler_node defaulting docker_image to
    # ubuntu:22.04 and "adapting" the configured python:3.12-slim away).
    sandbox_config: dict[str, Any]
    lintgate_config: dict[str, Any]
    deployment_config: dict[str, Any]
    # Per-call LLM dispatch parameters loaded from the ``llm_dispatch``
    # section of config.json. Nodes read
    # ``llm_dispatch_config["continue_on_length"][<role>]`` to decide
    # whether to re-prompt the model when the previous dispatch
    # returned ``finish_reason == "length"``. See the
    # ``_llm_dispatch_comment`` block in config/config.json for the
    # per-role risk profile. Empty dict = inherit defaults (only
    # patching continues on length).
    llm_dispatch_config: dict[str, Any]
    # Per-repo memory config (#7). dict shape mirrors the ``memory``
    # section of config.json: {enabled, dir, max_bytes, inject_max_bytes}.
    # planning_node reads it to decide whether to inject prior-session
    # notes. Empty dict = use defaults (enabled, ~/.harness/memory).
    repo_memory_config: dict[str, Any]
    # Semantic-retrieval index config (#6). dict shape mirrors the
    # ``repo_index`` section of config.json. planning_node queries
    # top-K chunks when ``enabled`` and injects them as a system
    # message. Empty dict = disabled (no retrieval, no injection).
    repo_index_config: dict[str, Any]
    # Operator-controlled toggle for the entire deployment phase (discovery
    # → DEPLOYMENT_BLUEPRINT → gatekeeper → docker-compose up). Set by the
    # `--deploy-dev` CLI flag on `teane run`; default False. When False,
    # route_after_security_scan short-circuits to END after a clean scan
    # instead of routing into deployment_discovery_node. Distinct from
    # `deployment_config["enabled"]`, which only gates the docker step
    # inside deployment_node once the phase is already running.
    dev_deployment: bool
    # Container-deployment discovery toggle. Only meaningful when
    # dev_deployment is True. When True, deployment_discovery_node runs
    # and synthesises DEPLOYMENT_BLUEPRINT.md from the codebase. When
    # False, the deployment step skips the LLM-driven discovery and
    # synthesises the blueprint from workspace telemetry alone. Set by
    # the `--cd-discovery` CLI flag on `teane run`; default False
    # (matches the new operator-autonomous baseline).
    cd_discovery: bool
    # End-of-run installation-doc synthesis toggle. When True,
    # installation_doc_node fires at the terminal success edges (Flutter
    # short-circuit, --deploy-dev=false clean security scan, and after
    # deployment_node health-check success) and writes
    # docs/INSTALLATION.md from workspace telemetry + manifests +
    # SPEC_ARCHITECTURE.md §7 (+ the deployment blueprint when present).
    # Set by --install-doc on `teane run`; defaults to the value of
    # --new-build so greenfield generations document themselves while
    # change-request runs stay quiet.
    install_doc: bool
    # Absolute path the installation_doc_node wrote on a successful run.
    # Empty string when install_doc=False or the node failed; consumers
    # (tests, log aggregators) read it as telemetry only.
    installation_doc_path: str
    # Optional org-wide deployment policy loaded from the
    # ``deployment_defaults`` section of config.json. When populated,
    # deployment_discovery_node injects it into the planning LLM's prompt
    # so already-resolved fields don't produce questions. Empty dict =
    # no section present = current full-questionnaire behaviour. See
    # load_deployment_defaults() in cli.py for the schema.
    deployment_defaults: dict[str, Any]
    run_prod_import_smoke_check: bool
    # Change-request-mode fields. When change_request_mode is True, the
    # graph routes through ingest_change_requests_node instead of the bare
    # patching path; the ingest node populates the remaining fields with
    # the resolved folder, the per-file CR assignments, and the archive
    # target. All inert (False / empty) on greenfield + bare existing-project
    # runs — those paths execute byte-identical to pre-change behaviour.
    change_request_mode: bool
    change_requests_dir_abs: str
    # List of {cr_id: int, original_name: str, abs_path: str} records,
    # sorted by cr_id. Populated by ingest_change_requests_node; consumed
    # by the archival helper at session end and by the patching/repair
    # prompts (PR-2+) for CR-N marker injection.
    change_request_files: list[dict[str, Any]]
    archive_target_dir: str
    # Loaded from the "change_requests" section of config.json by run_graph.
    # Read by reverse_engineer_architecture_node for the budget cap.
    change_requests_config: dict[str, Any]
    # Audit #18 — files changed since the most recent green compile. Cleared
    # when ``compiler_node`` returns ``exit_code == 0``; appended-to by any
    # node that mutates source after that point (currently only
    # ``code_review_node`` on a successful repatch). The router consults
    # this set before terminal exit so a post-green mutation can't slip out
    # un-verified. Empty list = no drift = safe to exit.
    pending_mutations: list[str]
    # When True (config ``pre_exit_verify=true``), the router forces one
    # extra compile before terminal exit whenever ``pending_mutations`` is
    # non-empty. Default False — the existing per-node re-routes already
    # cover the known cases; this is defence-in-depth for future nodes that
    # forget to set their own re-verify flag.
    pre_exit_verify: bool

# ---------------------------------------------------------------------------
# 2. Default State Factory
# ---------------------------------------------------------------------------

def create_initial_state(
    *,
    workspace_path: str,
    initial_prompt: str,
    build_command: str,
    allow_network: bool = False,
    budget_usd: float = 2.00,
    session_id: str = "",
    spec_override: Optional[str] = None,
    skip_discovery: bool = False,
    change_request_mode: bool = False,
    change_requests_dir_abs: str = "",
    archive_target_dir: str = "",
    change_requests_config: Optional[dict[str, Any]] = None,
    dev_deployment: bool = False,
    cd_discovery: bool = False,
    install_doc: bool = False,
) -> AgentState:
    """
    Construct the initial graph state with anchored system prompt at messages[0]
    for maximum downstream prefix-caching discounts.

    If spec_override is provided (from the product_spec_dir requirement
    refinement that ran in cmd_run before graph build), the
    SPEC_REQUIREMENTS.md + SPEC_ARCHITECTURE.md content is **prepended**
    to the default patch-DSL system prompt. Without this concatenation
    the carefully-tuned patcher contract (DSL syntax, Edit Invariants,
    CREATE_FILE-vs-REPLACE_BLOCK rules, imports placement, Makefile
    requirements) would be silently replaced by the project spec, leaving
    the patching LLM with no instructions on how to emit edits. Empirical
    grep across pre-fix debug dumps for "Edit Invariants" / "EXACT-BYTE
    matching" returned zero matches — the section was never reaching the
    LLM in any product_spec/-driven run.
    """
    if spec_override:
        # Spec first (project context), then the patcher's operating
        # contract (DSL + Edit Invariants + workspace conventions). The
        # spec anchors prefix-cache and the LLM's framing of WHAT we're
        # building; the patcher contract governs HOW to express edits.
        system_prompt = (
            spec_override
            + "\n\n---\n\n"
            + _build_and_emit_system_prompt(workspace_path, build_command)
        )
        # When a user-approved spec already exists (from the pre-flight
        # product_spec_dir refinement), skip the graph's discovery
        # pipeline completely. Otherwise write_spec_node would overwrite
        # the approved SPEC_REQUIREMENTS.md with a minimal
        # conversation-history compilation.
        skip_discovery = True
    else:
        system_prompt = _build_and_emit_system_prompt(workspace_path, build_command)
    return AgentState(
        workspace_path=workspace_path,
        messages=[
            MessageDict(role="system", content=system_prompt),
            MessageDict(role="user", content=initial_prompt),
        ],
        modified_files=[],
        compiler_errors=[],
        token_tracker=TokenTrackerDict(
            total_input_tokens=0,
            total_output_tokens=0,
            total_cached_tokens=0,
            total_cost_usd=0.0,
            per_model={},
        ),
        loop_counter={
            "patching": 0, "repair": 0, "compiler": 0, "total_repairs": 0,
            "review_spec": 0, "review_code": 0,
        },
        allow_network=allow_network,
        build_command=build_command,
        budget_remaining_usd=budget_usd,
        session_id=session_id,
        exit_code=-1,
        node_state={},
        skip_discovery=skip_discovery,
        change_request_mode=change_request_mode,
        change_requests_dir_abs=change_requests_dir_abs,
        change_request_files=[],
        archive_target_dir=archive_target_dir,
        change_requests_config=dict(change_requests_config or {}),
        dev_deployment=bool(dev_deployment),
        cd_discovery=bool(cd_discovery),
        install_doc=bool(install_doc),
        installation_doc_path="",
        pending_mutations=[],
        pre_exit_verify=False,
    )


# Files that conventionally live at workspace root and are exempt from the
# "all source under <root>/" enforcement when a source root is detected.
# patching_node / repair_node / test_generation_node compose the patcher
# allowlist as [<source_root>/, tests/, test/, __tests__/, *_ROOT_ALLOWLIST_FILES].
_ROOT_ALLOWLIST_FILES: frozenset[str] = frozenset({
    "setup.py", "setup.cfg", "pyproject.toml",
    "conftest.py", "manage.py", "__init__.py",
    "wsgi.py", "asgi.py", "main.py",
    "tox.ini", "pytest.ini", "MANIFEST.in", ".gitignore",
    # Dependency manifests — must be in the static set so the LLM can
    # CREATE them (e.g. add pytest after env_misconfig HITL), not just
    # amend existing ones via the requirements*.txt scan below.
    "requirements.txt",
    # Build orchestration — the per-stack makefile_*.md skills instruct
    # the LLM to emit a Makefile so `make build` runs against a real
    # target instead of the noisy late-bind adaptation in speculative.py
    # / compiler_node. GNU make recognises all three casings.
    "Makefile", "makefile", "GNUmakefile",
    # Node / TypeScript root manifests — the kitchen-sink builder image
    # supports JS stacks (vendor/Dockerfile.builder), and the React/Vue/
    # Angular/Node skills expect these at the workspace root. Without
    # them in the static set, every LLM patch to package.json or
    # tsconfig.json was rejected before the build could repair.
    "package.json", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json",
    "tsconfig.json", "tsconfig.base.json",
    ".npmrc", ".nvmrc", ".node-version",
    # Container deployment — Dockerfile and docker-compose files must be
    # in the allowlist for deployment discovery and synthesis to work.
    # The deployment phase may generate or modify these, and repair nodes
    # may need to adjust them for build fixes.
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Caddyfile",
})


# Runtime-scanned Node/JS root configs — too many proliferating variants
# (jest.config.cjs vs .js vs .ts; .eslintrc vs .eslintrc.json vs .eslintrc.cjs)
# to enumerate statically. _build_patcher_allowlist picks up any matching
# entry actually present at the workspace root, the same way it picks up
# requirements*.txt.
_NODE_CONFIG_SUFFIXES: tuple[str, ...] = (
    ".config.js", ".config.cjs", ".config.mjs", ".config.ts", ".config.json",
)
_NODE_CONFIG_PREFIXES: tuple[str, ...] = (
    ".eslintrc", ".prettierrc", ".babelrc",
)


def _is_node_config_file(name: str) -> bool:
    """True when ``name`` is a Node/JS tool config worth allowing at root.

    Catches the open-ended families that don't have one canonical filename:
      - ``*.config.{js,cjs,mjs,ts,json}`` — jest, vite, next, tailwind,
        postcss, playwright, rollup, webpack, etc.
      - ``.eslintrc*`` / ``.prettierrc*`` / ``.babelrc*`` — each ships in
        bare, ``.json``, ``.js``, ``.cjs``, and ``.yaml`` forms.
    """
    if any(name.endswith(suffix) for suffix in _NODE_CONFIG_SUFFIXES):
        return True
    if any(name.startswith(prefix) for prefix in _NODE_CONFIG_PREFIXES):
        return True
    return False


SPEC_ARCHITECTURE_REL_PATH = os.path.join("docs", "SPEC_ARCHITECTURE.md")


def _read_spec_layout(workspace_path: str) -> Any:
    """Read ``<workspace>/docs/SPEC_ARCHITECTURE.md`` and parse its
    ``workspace_layout`` block. Returns a :class:`LayoutParseResult`
    or ``None`` when the spec file is absent / unreadable.

    The patcher allowlist and the system prompt both call this — kept as
    a single read-and-parse helper so they see byte-identical layout
    data within one node invocation."""
    spec_path = os.path.join(workspace_path, SPEC_ARCHITECTURE_REL_PATH)
    if not os.path.isfile(spec_path):
        return None
    try:
        with open(spec_path, "r", encoding="utf-8", errors="replace") as f:
            spec_md = f.read()
    except OSError:
        return None
    from harness.architecture_inventory import parse_layout
    return parse_layout(spec_md)


def _build_patcher_allowlist(workspace_path: str) -> Optional[list[str]]:
    """Return the patcher allowed_paths list for ``workspace_path``.

    Tiered decision:

      1. **Spec-driven (tier 1)** — when
         ``<workspace>/docs/SPEC_ARCHITECTURE.md`` exists and either its
         ``workspace_layout`` block parses or it carries a parseable
         ``files`` inventory we can derive roots from, the allowlist is
         built directly from the spec's declared roots, test directory
         convention, and root manifest list. This is the authoritative
         path — SPEC_ARCHITECTURE.md is the layout contract the
         architecture phase wrote and the human gatekeeper approved.

      2. **Filesystem fallback (tier 2)** — when the spec is absent
         (reverse-engineer mode, change-request mode against a legacy
         repo, greenfield iteration 1) or fails to yield any roots, fall
         back to :func:`harness.impact._detect_source_roots` and the
         greenfield / permissive fallback already in place. This keeps
         the harness working in modes where there is no spec yet.

    Returns ``None`` only for the greenfield iteration-1 case (no spec,
    no source files), preserving the LLM's freedom to pick the layout.
    Mirrors the language used in the system prompt's "Workspace Layout"
    section, so the LLM sees the same rules as the patcher applies.
    """
    layout = _read_spec_layout(workspace_path)
    if layout is not None and layout.has_layout:
        allowlist = _spec_driven_allowlist(workspace_path, layout)
        logger.info(
            "[allowlist] Spec-driven, roots=%s%s%s.",
            [r.path for r in layout.roots],
            (" (derived from file inventory)"
             if layout.derived_from_inventory else ""),
            (f" test_placement={layout.test_placement}"
             if layout.test_placement else ""),
        )
        return allowlist

    return _filesystem_allowlist(workspace_path)


def _spec_driven_allowlist(workspace_path: str, layout: Any) -> list[str]:
    """Build the tier-1 allowlist from a parsed workspace_layout block.

    The spec's ``roots`` list IS the prefix set. The standard test trees
    stay in the allowlist regardless of the spec's ``test_placement`` —
    a project may host its top-level integration tests in ``tests/``
    even when unit tests are co-located. The spec's ``root_files`` list
    is layered on top of the static :data:`_ROOT_ALLOWLIST_FILES`, then
    the runtime root scan (``requirements*.txt``, node configs) adds
    whatever is actually on disk.

    Spec/disk divergence is surfaced as an observability event and log
    line every call. When the operator opted into the layout-divergence
    HITL gate (``--hitl-layout-divergence true``) and the drift is
    major, an interactive prompt offers to extend the in-memory
    allowlist with the drifted directories. The choice is cached per
    workspace so the prompt fires once, not on every node invocation.
    """
    allowlist: list[str] = [
        *[f"{r.path}/" for r in layout.roots],
        "tests/", "test/", "__tests__/",
        *_ROOT_ALLOWLIST_FILES,
        *[f for f in layout.root_files if f not in _ROOT_ALLOWLIST_FILES],
    ]
    extra = _resolve_layout_divergence(workspace_path, layout)
    for entry in extra:
        if entry not in allowlist:
            allowlist.append(entry)
    _append_runtime_root_entries(workspace_path, allowlist)
    return allowlist


# Caches the operator's per-workspace decision when the layout-divergence
# HITL gate fires, so subsequent calls in the same session reuse the
# answer instead of re-prompting. Keyed by workspace_path. Values:
#   - []         → "trust spec" (or no drift / minor drift / HITL off)
#   - [paths]    → "trust disk" — append these dir prefixes to allowlist
_LAYOUT_DIVERGENCE_CACHE: dict[str, list[str]] = {}


def _resolve_layout_divergence(workspace_path: str, layout: Any) -> list[str]:
    """Detect spec/disk divergence; emit telemetry; consult the operator
    when the HITL gate is enabled. Returns extra allowlist prefixes to
    add to the spec-driven base.
    """
    if workspace_path in _LAYOUT_DIVERGENCE_CACHE:
        return _LAYOUT_DIVERGENCE_CACHE[workspace_path]

    diagnostic = _check_layout_disk_divergence(workspace_path, layout)
    if diagnostic is None:
        _LAYOUT_DIVERGENCE_CACHE[workspace_path] = []
        return []

    # Always emit the observability event so divergence is visible even
    # when the HITL gate is off (the default).
    try:
        from harness.observability import emit_event
        emit_event(
            "spec_layout_divergence",
            severity=diagnostic["severity"],
            spec_roots=diagnostic["spec_roots"],
            drifted_dirs=diagnostic["drifted_dirs"],
            total_workspace_source=diagnostic["total_workspace_source"],
        )
    except Exception:  # noqa: BLE001 — telemetry must never block the build
        pass

    drifted_summary = ", ".join(
        f"{d['path']}/ ({d['source_count']} files)"
        for d in diagnostic["drifted_dirs"]
    )

    if diagnostic["severity"] == "minor":
        logger.info(
            "[allowlist] Minor layout drift: %s. Spec roots: %s. "
            "Spec wins — drifted dirs stay off the allowlist.",
            drifted_summary, diagnostic["spec_roots"],
        )
        _LAYOUT_DIVERGENCE_CACHE[workspace_path] = []
        return []

    logger.warning(
        "[allowlist] Major layout drift. Spec declares roots %s but "
        "workspace also has substantial source under: %s. Patches "
        "targeting drifted dirs will be rejected unless the spec is "
        "updated.",
        diagnostic["spec_roots"], drifted_summary,
    )

    try:
        from harness.cli import _hitl_gate_enabled
        gate_on = _hitl_gate_enabled("layout_divergence")
    except Exception:  # noqa: BLE001
        gate_on = False
    if not gate_on:
        _LAYOUT_DIVERGENCE_CACHE[workspace_path] = []
        return []

    extra = _prompt_layout_divergence(diagnostic)
    _LAYOUT_DIVERGENCE_CACHE[workspace_path] = extra
    return extra


def _prompt_layout_divergence(diagnostic: dict[str, Any]) -> list[str]:
    """Interactive prompt for major layout drift. Returns the list of
    dir prefixes the operator authorized for this session.
    """
    drifted_paths = [d["path"] for d in diagnostic["drifted_dirs"]]
    options = [
        ("s", "Trust SPEC — proceed; drifted-dir patches will reject"),
        ("d", f"Trust DISK — append {drifted_paths} to allowlist this run"),
    ]
    print()
    drifted_render = ", ".join(
        f"{d['path']}/ ({d['source_count']} files)"
        for d in diagnostic["drifted_dirs"]
    )
    print("=" * 80)
    print("[HITL] Spec / disk layout divergence")
    print(f"  Spec roots: {diagnostic['spec_roots']}")
    print(f"  Drifted:    {drifted_render}")
    print(f"  Total workspace source files: {diagnostic['total_workspace_source']}")
    print("=" * 80)
    print("Options:")
    for key, label in options:
        print(f"  [{key}] {label}")
    print()
    try:
        from harness.hitl import get_channel
        choice = get_channel().prompt(
            "[HITL] Layout divergence — choose action",
            [k for k, _ in options],
            default="s",
            option_labels={k: lbl for k, lbl in options},
        )
    except Exception:  # noqa: BLE001 — non-interactive channel etc.
        choice = "s"
    if choice == "d":
        appended = [f"{p}/" for p in drifted_paths]
        logger.warning(
            "[allowlist] Operator chose 'trust disk' — extending allowlist "
            "with %s for the remainder of this run.", appended,
        )
        return appended
    logger.info("[allowlist] Operator chose 'trust spec' — allowlist unchanged.")
    return []


def _filesystem_allowlist(workspace_path: str) -> Optional[list[str]]:
    """Tier-2 fallback: derive the allowlist from a filesystem scan.

    Same logic the harness has used since multi-root detection landed —
    runs when the spec is unavailable. Returns ``None`` for true
    greenfield workspaces so the LLM can pick its own layout on
    iteration 1.
    """
    from harness.impact import (
        _detect_source_roots,
        _is_greenfield_workspace,
        _workspace_basename_variants,
    )
    roots = _detect_source_roots(workspace_path)

    if roots:
        # Each detected root contributes its own directory prefix. Tests
        # co-located inside a root (e.g. `client/src/Foo.test.jsx` next
        # to `client/src/Foo.jsx`, the Jest/RTL convention) pass via the
        # `<root>/` prefix automatically — they don't need an extra
        # allowlist entry. Top-level `tests/` / `test/` / `__tests__/`
        # are kept for integration-test suites that live at workspace
        # root regardless of the source layout.
        allowlist: list[str] = [
            *[f"{r}/" for r in roots],
            "tests/", "test/", "__tests__/",
            *_ROOT_ALLOWLIST_FILES,
        ]
        if len(roots) > 1:
            logger.info(
                "[allowlist] Multi-root workspace %s — allowing patches "
                "under %s. Each root carries its own co-located tests.",
                workspace_path, [f"{r}/" for r in roots],
            )
    elif _is_greenfield_workspace(workspace_path):
        # Greenfield project (no source files yet) — the LLM IS the one
        # defining the layout, so constraining its package directory name
        # to "must match the workspace basename" is the wrong rule. The
        # LLM often picks a descriptive name (e.g. `job_queue/` for a
        # workspace called `TaskDispatcher`) — a reasonable engineering
        # choice that the basename-only allowlist used to reject.
        #
        # Return ``None`` so the patcher skips the prefix check entirely.
        # The patcher's :func:`harness.trust.safe_resolve` still blocks
        # path traversal (``../``) and absolute paths, which is the only
        # real safety concern for a fresh workspace. Subsequent runs will
        # detect the source root the LLM picked and lock the allowlist
        # to it (regular ``if root:`` branch above).
        logger.info(
            "[allowlist] Greenfield workspace %s — no allowlist; the LLM "
            "defines the layout. Path-traversal still blocked by the "
            "patcher's safe_resolve guard.",
            workspace_path,
        )
        return None
    else:
        # Conservative fallback when no source root is detected but the
        # workspace also isn't truly greenfield (e.g. leftover files from a
        # previous abandoned session). Still include the workspace's
        # basename variants — without them, the LLM's natural choice of a
        # package directory matching the project name (e.g. task_dispatcher/
        # for a workspace called TaskDispatcher) gets rejected on every
        # patch attempt and the repair loop burns through to HITL.
        basename_dirs = [
            f"{name}/" for name in _workspace_basename_variants(workspace_path)
        ]
        logger.warning(
            "[allowlist] No source root detected for %s — falling back to "
            "permissive allowlist (src/, lib/, app/, pkg/, cmd/, tests/ + "
            "basename variants %s). Workspace not greenfield, so this "
            "likely contains stale files; the LLM still needs a place to "
            "put new code matching the project name.",
            workspace_path, basename_dirs,
        )
        allowlist = [
            "src/", "lib/", "app/", "pkg/", "cmd/", "internal/",
            "tests/", "test/", "__tests__/",
            *basename_dirs,
            *_ROOT_ALLOWLIST_FILES,
        ]

    _append_runtime_root_entries(workspace_path, allowlist)
    return allowlist


def _check_layout_disk_divergence(workspace_path: str, layout: Any) -> Optional[dict[str, Any]]:
    """Compare the spec's declared roots against the on-disk top-level
    directories. Detect cases where the workspace has drifted away from
    SPEC_ARCHITECTURE.md.

    Returns ``None`` when no drift is detected. Otherwise returns a
    diagnostic dict with shape::

        {
            "severity": "minor" | "major",
            "spec_roots": ["client", "server"],
            "drifted_dirs": [{"path": "web", "source_count": 12}, ...],
            "total_workspace_source": 47,
            "test_placement": "co-located",  # from the spec, if present
        }

    Severity classification:
      * **minor**: every drifted dir holds ≤2 source files. Treated as
        scratch / leftover content; allowlist-driven rejection is enough
        to keep the LLM honest.
      * **major**: at least one drifted dir holds ≥3 source files OR ≥15%
        of the workspace's total source. Signals real layout drift — the
        operator should reconcile spec vs. disk before the run continues.

    The observability event and (when ``--hitl-layout-divergence true``)
    the HITL gate are raised by the calling node, not here — this
    function is a pure inspector so it stays testable.
    """
    if not layout or not layout.has_layout:
        return None
    try:
        entries = os.listdir(workspace_path)
    except OSError:
        return None

    from harness.impact import (
        _NEVER_SOURCE_DIRS,
        _SOURCE_FILE_EXTENSIONS,
        _MAX_FILES_PER_SCAN,
    )

    spec_root_set = {r.path for r in layout.roots}
    drifted: list[tuple[str, int]] = []
    total_source = 0
    files_scanned = 0
    spec_root_total = 0

    for entry in entries:
        full = os.path.join(workspace_path, entry)
        if not os.path.isdir(full):
            if os.path.isfile(full):
                ext = os.path.splitext(entry)[1].lower()
                if ext in _SOURCE_FILE_EXTENSIONS:
                    total_source += 1
            continue
        if entry.startswith(".") or entry in _NEVER_SOURCE_DIRS:
            continue

        dir_count = 0
        for sub_root, sub_dirs, sub_files in os.walk(full):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
            ]
            for fname in sub_files:
                if os.path.splitext(fname)[1].lower() in _SOURCE_FILE_EXTENSIONS:
                    dir_count += 1
                    files_scanned += 1
                    if files_scanned >= _MAX_FILES_PER_SCAN:
                        break
            if files_scanned >= _MAX_FILES_PER_SCAN:
                break
        total_source += dir_count
        if entry in spec_root_set:
            spec_root_total += dir_count
        elif dir_count > 0:
            drifted.append((entry, dir_count))
        if files_scanned >= _MAX_FILES_PER_SCAN:
            break

    if not drifted:
        return None

    # Threshold: major when any drifted dir has ≥3 files OR ≥15% of total.
    threshold_count = 3
    threshold_pct = 0.15
    is_major = any(
        cnt >= threshold_count or (total_source > 0 and cnt >= threshold_pct * total_source)
        for _, cnt in drifted
    )

    return {
        "severity": "major" if is_major else "minor",
        "spec_roots": sorted(spec_root_set),
        "drifted_dirs": [
            {"path": p, "source_count": c}
            for p, c in sorted(drifted, key=lambda kv: kv[1], reverse=True)
        ],
        "total_workspace_source": total_source,
        "spec_root_source_count": spec_root_total,
        "test_placement": getattr(layout, "test_placement", "") or "",
    }


def _append_runtime_root_entries(workspace_path: str, allowlist: list[str]) -> None:
    """Append workspace-root files actually present that the LLM may need
    to amend: ``requirements*.txt`` and the open-ended Node/JS tool
    config families. Mutates ``allowlist`` in place.

    Used by both the spec-driven tier-1 path and the filesystem
    fallback. We allow only what's actually on disk so the LLM can amend
    existing manifests without opening up arbitrary root writes.
    """
    try:
        for entry in os.listdir(workspace_path):
            if entry.startswith("requirements") and entry.endswith(".txt"):
                if entry not in allowlist:
                    allowlist.append(entry)
            elif _is_node_config_file(entry):
                if entry not in allowlist:
                    allowlist.append(entry)
    except OSError:
        pass


def _format_spec_layout_block(spec_layout: Any) -> str:
    """Render the spec's ``workspace_layout`` block as the system-prompt
    Workspace Layout section.

    The spec gives us purpose strings per root and a test_placement hint,
    so the guidance to the LLM can be specific ("React frontend SPA"
    rather than "the client-side root"). Used by ``_build_system_prompt``
    when the spec is available; the multi-root filesystem-derived block
    is the fallback.
    """
    roots = spec_layout.roots
    if len(roots) == 1:
        r = roots[0]
        purpose_tail = (
            f" — {r.purpose}" if r.purpose
            else (f" ({r.stack})" if r.stack else "")
        )
        return (
            f"## Workspace Layout (mandatory — per SPEC_ARCHITECTURE.md)\n"
            f"The workspace declares one source root: `{r.path}/`"
            f"{purpose_tail}. **All new source files MUST be created under "
            f"`{r.path}/`.** Do NOT place new modules at workspace root or "
            f"in any other top-level directory.\n\n"
            f"{_format_test_placement_guidance(spec_layout.test_placement, [r.path])}"
            f"{_format_root_files_guidance(spec_layout.root_files)}"
        )

    bullets = []
    for r in roots:
        parts = [f"`{r.path}/`"]
        if r.purpose:
            parts.append(f"— {r.purpose}")
        if r.stack:
            parts.append(f"({r.stack})")
        bullets.append("  * " + " ".join(parts))
    roots_listing = "\n".join(bullets)
    primary = roots[0]
    return (
        "## Workspace Layout (mandatory — per SPEC_ARCHITECTURE.md)\n"
        f"This workspace is a multi-root monorepo. The architecture spec "
        f"declares the following source roots:\n{roots_listing}\n\n"
        f"**All new source files MUST be created under one of these "
        f"roots.** Do NOT place new modules at workspace root or in any "
        f"other top-level directory. When you create a file, choose the "
        f"root whose `purpose` is the closest semantic match — the "
        f"purpose strings above were written by the architect for "
        f"exactly this routing decision. When the choice is genuinely "
        f"ambiguous, prefer `{primary.path}/` (the first declared root).\n\n"
        f"{_format_test_placement_guidance(spec_layout.test_placement, [r.path for r in roots])}"
        f"{_format_root_files_guidance(spec_layout.root_files)}"
    )


def _format_test_placement_guidance(test_placement: str, root_paths: list[str]) -> str:
    """Render the test_placement line of the Workspace Layout block."""
    if test_placement == "co-located":
        return (
            f"Tests are **co-located** with source per the architecture "
            f"spec. JS/TS tests sit next to the file they test "
            f"(`Foo.test.jsx` next to `Foo.jsx`); Python tests sit in a "
            f"sibling `tests/` subdirectory inside their root "
            f"(e.g. `{root_paths[0]}/tests/`). A top-level `tests/` is "
            f"also accepted for cross-root integration tests.\n\n"
        )
    if test_placement == "centralized":
        return (
            "Tests are **centralized** per the architecture spec: every "
            "test file lives under a top-level `tests/`, `test/`, or "
            "`__tests__/` directory. Do NOT co-locate tests next to "
            "their source files.\n\n"
        )
    if test_placement == "mixed":
        return (
            "Test placement is **mixed** per the architecture spec — "
            "either co-located next to source or in a top-level `tests/` "
            "directory is acceptable. Follow whichever convention an "
            "existing test in the same root already uses.\n\n"
        )
    return (
        "Test files may live next to their source (co-located) or in a "
        "top-level `tests/` / `test/` / `__tests__/` directory.\n\n"
    )


def _format_root_files_guidance(root_files: list[str]) -> str:
    """Render the root_files line of the Workspace Layout block."""
    if not root_files:
        return (
            "Workspace-root files are limited to the standard set: "
            "`setup.py`, `setup.cfg`, `pyproject.toml`, `package.json`, "
            "`conftest.py`, `manage.py`, `requirements*.txt`, `tox.ini`, "
            "`pytest.ini`, `Makefile`, `.gitignore`, plus the standard "
            "JS tool configs (`*.config.{js,cjs,mjs,ts,json}`, `.eslintrc*`).\n"
        )
    listing = ", ".join(f"`{f}`" for f in root_files)
    return (
        f"The architecture spec lists these files as workspace-root "
        f"residents: {listing}. Other root-level files are limited to "
        f"the standard set (`setup.py`, `pyproject.toml`, `package.json`, "
        f"`Makefile`, `.gitignore`, etc.).\n"
    )


def _build_system_prompt(workspace_path: str, build_command: str) -> str:
    """
    Construct the static, immutable system prompt anchored at messages[0].
    This prompt is never mutated or truncated — it maximizes prefix caching
    across all downstream LLM calls because its position and content are fixed.

    Emits a ``system_prompt_built`` observability event with ``chars`` and
    ``lines`` so the harness can track prompt bloat over time (audit #8).
    """
    tree = _snapshot_directory_tree(workspace_path)

    # Redact secrets from the directory tree snapshot if redactor is active
    try:
        from harness.redactor import redact_text
        tree = redact_text(tree)
    except ImportError:
        pass

    # --- Two-Tier Skills System (language-aware filtering) ---
    # Detect the workspace's stack so we only inject skills the LLM will
    # actually use. A pure FastAPI project doesn't need the Angular skill
    # in its system prompt; trimming saves ~2-3 KB per call and reduces
    # noise in the prompt.
    from harness.impact import _detect_workspace_stack, _detect_source_roots
    workspace_tags = _detect_workspace_stack(workspace_path)
    source_roots = _detect_source_roots(workspace_path)

    # Tier 1: Harness skills (harness/skills/*.md) — agent standards +
    # stack-specific skills filtered by `applies_to:` frontmatter.
    harness_skills = _load_skills_markdown(
        os.path.join(os.path.dirname(__file__), "skills"),
        max_file_chars=4000,
        workspace_tags=workspace_tags,
    )
    if harness_skills:
        harness_skills = f"## Agent Skills & Standards\n{harness_skills}\n"

    # Tier 2: Project skills ({workspace_path}/skills/*.md) — per-project
    # conventions. Same filter applies so user-supplied skills can also
    # opt-in via frontmatter; skills without frontmatter always load.
    project_skills = _load_skills_markdown(
        os.path.join(workspace_path, "skills"),
        max_file_chars=3000,
        workspace_tags=workspace_tags,
    )
    if project_skills:
        project_skills = f"## Project Skills & Conventions\n{project_skills}\n"

    # --- Technology-Specific Style Guides ---
    # Two-tier like skills: shipped defaults under harness/style_guides/
    # + per-project overrides under {workspace}/style_guides/. Both are
    # filtered by `applies_to:` frontmatter against the detected stack
    # so e.g. a pure Python project never sees React style content.
    from harness.style_guides import load_style_guides
    style_guides = load_style_guides(workspace_path, workspace_tags=workspace_tags)
    if style_guides:
        style_guides = f"## Coding Style Guides\n{style_guides}\n"

    # --- Workspace Layout Constraint ---
    # When the workspace has a clear source root (e.g. `app/`, `src/`, `lib/`),
    # tell the LLM that new source files MUST land there. Paired with the
    # `allowed_paths` enforcement in patching_node / repair_node — files
    # outside the allowlist are rejected with a clear error so the LLM
    # tries again with the constraint in mind.
    #
    # Preference order for the layout block:
    #   1. Spec-driven: SPEC_ARCHITECTURE.md's workspace_layout block.
    #      Uses purpose/stack/test_placement to give precise guidance.
    #   2. Filesystem-driven multi-root: derived from _detect_source_roots.
    #      Generic heuristic guidance (UI → client side, handlers → server).
    #   3. Single-root: legacy single-root message.
    layout_block = ""
    spec_layout = _read_spec_layout(workspace_path)
    if spec_layout is not None and spec_layout.has_layout:
        layout_block = _format_spec_layout_block(spec_layout)
    elif len(source_roots) == 1:
        root = source_roots[0]
        layout_block = (
            f"## Workspace Layout (mandatory)\n"
            f"The workspace organizes its source under `{root}/`. "
            f"**All new source files MUST be created under `{root}/`.** "
            f"Do NOT place new modules at workspace root.\n\n"
            f"The only files that may live at workspace root are: "
            f"`setup.py`, `setup.cfg`, `pyproject.toml`, `conftest.py`, "
            f"`manage.py`, `__init__.py`, `wsgi.py`, `asgi.py`, `main.py`, "
            f"`requirements*.txt`, `tox.ini`, `pytest.ini`, `MANIFEST.in`, "
            f"`.gitignore`, `Makefile`. Test files live under `tests/`, "
            f"`test/`, or `__tests__/` per the language convention. "
            f"CREATE_FILE blocks that target other root paths will be "
            f"rejected by the patcher.\n"
        )
    elif len(source_roots) > 1:
        roots_inline = ", ".join(f"`{r}/`" for r in source_roots)
        primary = source_roots[0]
        layout_block = (
            f"## Workspace Layout (mandatory — multi-root monorepo)\n"
            f"This workspace is a multi-root monorepo. Source lives under: "
            f"{roots_inline}. **All new source files MUST be created under "
            f"one of these roots** — do NOT place new modules at workspace "
            f"root or in any other top-level directory.\n\n"
            f"When you create a file, choose the root whose existing code is "
            f"the closest match. As rough guidance: front-end / UI code "
            f"(React, Vue, Angular, browser bundles) belongs in the "
            f"client-side root; HTTP handlers, models, and background "
            f"workers belong in the server-side root. When the choice is "
            f"ambiguous, prefer `{primary}/` (the largest existing root).\n\n"
            f"Tests follow the convention of the root they exercise. JS / TS "
            f"tests are typically co-located next to source as "
            f"`*.test.{{js,jsx,ts,tsx}}` — that is allowed under any "
            f"detected root. Python tests typically live in a sibling "
            f"`tests/` subdirectory (e.g. `server/tests/`) or in a top-level "
            f"`tests/`. The patcher rejects any CREATE_FILE / "
            f"REPLACE_BLOCK that targets a top-level directory not in this "
            f"list.\n\n"
            f"The only files that may live at workspace root are: "
            f"`setup.py`, `setup.cfg`, `pyproject.toml`, `package.json`, "
            f"`conftest.py`, `manage.py`, `__init__.py`, `wsgi.py`, "
            f"`asgi.py`, `main.py`, `requirements*.txt`, `tox.ini`, "
            f"`pytest.ini`, `MANIFEST.in`, `.gitignore`, `Makefile`, plus "
            f"the standard JS tool configs (`jest.config.*`, `vite.config.*`, "
            f"`tsconfig*.json`, `.eslintrc*`, `.prettierrc*`, `.babelrc*`).\n"
        )

    # Web-app file manifest contract: only injected when this is a web
    # workspace. Tells the LLM to emit a structured JSON manifest the
    # harness can cross-check against the architecture inventory.
    inventory_block = ""
    if workspace_tags and ("html" in workspace_tags):
        from harness.architecture_inventory import PLANNING_INVENTORY_INSTRUCTION
        inventory_block = (
            f"## File Manifest Contract (web apps)\n"
            f"{PLANNING_INVENTORY_INSTRUCTION}\n"
        )

    return f"""You are an expert software engineer with deep knowledge of the codebase below.

## Repository Root
{workspace_path}

## Directory Structure (snapshot at invocation)
{tree}
{layout_block}{inventory_block}{harness_skills if harness_skills else ""}{project_skills if project_skills else ""}{style_guides if style_guides else ""}
## Build Command
{build_command}

## Dependency Manifest Coherence (mandatory)
The build command above is exactly what the sandbox will execute. Every CLI
the build invokes (test runner, linter, type checker, formatter) MUST be
declared in the workspace's dependency manifest — otherwise the install
step succeeds but the next step fails with "No module named X" /
"command not found", and the run wastes a repair iteration.

Audit the build command before writing any manifest:
  - `pip install -r requirements.txt && pytest`  → `requirements.txt`
    must list `pytest` (and `pytest-asyncio`, `ruff`, `mypy`, etc. if the
    command invokes them).
  - `pip install -e '.[dev]' && pytest`  → `pytest` must live under
    `[project.optional-dependencies].dev` in `pyproject.toml`.
  - `npm install && npm test`  → the test runner referenced by the `test`
    script must be in `package.json` `devDependencies`.
  - `cargo build && cargo test`  → cargo bundles the runner; no extra dep.
  - `go build ./... && go test ./...`  → go bundles the runner; no extra dep.

When you generate or amend a manifest, include every tool the build needs.
A clean separation between runtime and dev dependencies is fine, but BOTH
must be installable by the build command.

## Your Role
You are an autonomous coding agent. You will receive tasks and must:
1. Plan the implementation strategy before writing code.
2. Generate precise code patches using a strict SEARCH/REPLACE syntax.
3. Only modify files that need changes — never touch unrelated code.

## Edit Invariants — read before writing any REPLACE_BLOCK

These rules are not style guidance — they are the contract the patcher
enforces. Violating any of them produces a patch that will not apply.

1. **EXACT-BYTE matching.** A REPLACE_BLOCK / DELETE_BLOCK `search:` block
   must be a verbatim substring of the on-disk file. Whitespace,
   punctuation, quote style, trailing newlines, and Unicode characters
   all matter. Do not "clean up" the search text — copy it byte-for-byte.

2. **Strip the line-number prefix.** When file content is shown to you
   with a `  N| ` prefix (look for `## Current Content of Files You
   Need to Edit` or any patcher "Closest match" window), the prefix is
   navigation only. The actual file content starts AFTER `  N| `. Never
   include `  N| ` in a `search:` block — that string is not in the file.

3. **Indentation is part of the search.** Preserve every leading space
   and tab exactly as it appears in the line-numbered view. Patches that
   re-indent or normalize whitespace will miss.

4. **Read before you edit.** If the conversation has not shown you a
   file's current bytes — either in `## Current Content of Files You
   Need to Edit` or in a patcher "Closest match" window — you are
   guessing. Do not guess. Either limit your edit to a file you have
   been shown, or emit a `READ_FILE` block (see below) and stop. The
   harness will resolve it and re-dispatch with the content present
   before counting an iteration.

5. **Use `count:` instead of pasting larger context.** When the same
   search text would match multiple places, do NOT bloat your search to
   force uniqueness — set `count: all` to replace every occurrence or
   `count: first` to replace only the first. (See REPLACE_BLOCK below.)

## Patch Syntax
When applying patches, use these exact formats:

### REPLACE_BLOCK
```
<<<REPLACE_BLOCK>>>
file: path/to/file.ext
count: unique          # optional; one of: unique (default) | all | first
search:
<exact lines to find>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>
```

`count:` is OPTIONAL. Values:
- `unique` (default) — fail when `search` matches more than once. Matches
  historical strict behaviour.
- `all` — apply the replacement to every occurrence.
- `first` — apply only to the first occurrence.

Use `all` / `first` when you need to fix multiple identical lines and
adding context to make `search` unique would be redundant. Same field
works on `DELETE_BLOCK`.

### READ_FILE
```
<<<READ_FILE>>>
file: path/to/foo.py
range: 1-200          # optional; default = whole file (capped)
<<<END_READ_FILE>>>
```

Use this when you do not know — or are not sure of — a file's current
bytes. The harness will:

1. Read the file from disk in the same sandbox the build runs in.
2. Append the line-numbered current content to the conversation as a
   user message.
3. Re-dispatch you in the same iteration, so READ_FILE does NOT consume
   a repair-loop slot.

You may emit READ_FILE blocks alongside patch blocks in the same
response; the patches still apply. If you only need to read, emit ONLY
the READ_FILE block(s) and no patches. Cap: at most two READ_FILE
resolution rounds per iteration — after that the harness ignores
further READ_FILE blocks and applies whatever patches you emitted.

When to use READ_FILE:
- Your REPLACE_BLOCK keeps missing and the diagnostic does not include a
  "Closest match" window with the file content.
- The diagnostic points at a file you have not been shown in the
  conversation.
- You are about to write a patch but your mental model of the file is
  stale (e.g. several iterations have happened since you last saw it).

### CREATE_FILE
```
<<<CREATE_FILE>>>
file: path/to/new/file.ext
content:
<complete file contents>
<<<END_CREATE_FILE>>>
```

**CREATE_FILE vs REPLACE_BLOCK — read this carefully.**
- `CREATE_FILE` is for files that **do not yet exist**.
- If a file already exists with the **same** content, `CREATE_FILE` is a safe no-op (used for resumes).
- If a file already exists with **different** content, `CREATE_FILE` will be **REJECTED** by the patcher with the error "File already exists with different content". The patcher will NOT overwrite.
- To change an existing file, use **`REPLACE_BLOCK`** (find the exact lines you want to change, replace them with the new lines). If you need to rewrite the whole file, emit one `REPLACE_BLOCK` whose `search:` matches the current file contents.
- **If you created a file in a previous turn**, that file now exists — any subsequent edit to it must use `REPLACE_BLOCK`, not `CREATE_FILE`. The "## Files currently in workspace" section in your repair prompt (when present) is authoritative on what exists.
- **Never emit two `CREATE_FILE` blocks for the same path in a single response.** Each path gets exactly one block per response. The patcher applies blocks in order, so the SECOND `CREATE_FILE` for the same path will be REJECTED ("File already exists with different content") even though it was you who just created the file moments earlier. If you need to refine the content as you draft, edit your own response before submitting; if you need two variants, pick the one you want and emit just that. Symptom: an initial patching round logs `Applied N-2/N patches` with the rejected file appearing in the "Failed" list paired with itself.

**Imports — do NOT duplicate.** When you need a symbol that requires a new import:
- First scan the existing imports in the file for the same symbol (e.g. `AsyncGenerator`, `Optional`, `Path`).
- If an import for that symbol already exists — even from a different source module — **replace it** with `REPLACE_BLOCK`. Do NOT add a second `from ... import <same_symbol>` line; Python will shadow the first import silently and lint will flag F811 ("redefinition of unused"), but the underlying problem is that future REPLACE_BLOCK searches against the import region keep missing because the file has drifted from your mental model.
- Concretely: if the file has `from typing import AsyncGenerator` and you want the `collections.abc` version, replace the typing line — don't append the collections.abc line.

**Imports — placement.** Every `import` / `from ... import ...` statement MUST appear at the **top of the file**, above any non-import code (class, def, module-level expression). This applies to NEW files (CREATE_FILE) AND to existing files (REPLACE_BLOCK / INSERT_AT_BLOCK). Concretely:
- When CREATE_FILE'ing a new module, put every import at the top, before any class / def / assignment.
- When REPLACE_BLOCK'ing to add a new import, place it next to the existing import block at the top — NEVER at the bottom of the file, NEVER inside a function/class body, NEVER between `def` and `class` definitions further down.
- If a symbol is only used inside one function and you're tempted to "import locally" — still put it at the top unless you have a documented reason (circular-import workaround, optional-dependency guard).
- Bottom-of-file imports cause `NameError` at module import time when the bottom-imported symbol is referenced anywhere above its line. Lint will catch it as F811 or F821, but more importantly the test collector will fail before the repair LLM ever sees the actual root cause.

**Makefile — MUST declare a `build:` target.** If you CREATE_FILE a `Makefile` (or `makefile` / `GNUmakefile`) at workspace root, it MUST declare a `build:` target. The harness invokes `make build` by default — a Makefile with `install:`, `test:`, `run:`, etc. but no `build:` target crashes the sandbox immediately:
- The shell can't find `build` → `make` reports "No rule to make target 'build'"; before that, the base `ubuntu:22.04` image doesn't even ship `make` itself → exit 127 in under a second.
- Zero diagnostics get extracted, the repair LLM has nothing to act on, the loop spins for 5 iterations and routes to HITL.
- Minimum acceptable shape for a Python project:
  ```
  .PHONY: build test
  build:
  	python3 -m pip install -r requirements.txt
  test:
  	python3 -m pytest -q
  ```
  Use a TAB to indent recipe lines (not spaces — `make` rejects spaces with `*** missing separator. Stop.`).
- The `build:` target should perform the install / compile step a CI would need to run before tests. Tests live under `test:` (or a separate target wired into `test:`).
- If you're not sure whether to emit a Makefile at all, DON'T — the harness picks the right install + test command from `pyproject.toml` / `requirements.txt` / `package.json` / etc. A Makefile is only useful when YOU genuinely want operators to run `make build` themselves.

### DELETE_BLOCK
```
<<<DELETE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to delete>
<<<END_DELETE_BLOCK>>>
```

### INSERT_AT_BLOCK
```
<<<INSERT_AT_BLOCK>>>
file: path/to/file.ext
anchor: <function or class name to insert relative to>
placement: before|after
content:
<lines to insert>
<<<END_INSERT_AT_BLOCK>>>
```

## Code Quality Standards
- Write modular, self-contained functions/classes with single responsibility.
- Include proper error handling: try/except, input validation, graceful fallbacks.
- Add type hints (Python), type annotations (TypeScript), or equivalent in the target language.
- Use meaningful variable/function names; include docstrings and inline comments.
- Follow the principle of least surprise — never modify unrelated code or files.
- Handle edge cases: empty inputs, None/null values, network failures, timeouts.
- Write production-ready code: no debug print statements, no hardcoded secrets or credentials.
- Prefer composition over inheritance; keep coupling loose and interfaces clean.

## Rules
- Never remove or alter existing comments unless instructed.
- Preserve existing indentation and code style.
- If you are unsure about a change, ask for clarification rather than guessing.
"""


def _build_and_emit_system_prompt(workspace_path: str, build_command: str) -> str:
    """Wrap :func:`_build_system_prompt` so callers get the same prompt but
    a ``system_prompt_built`` observability event also lands in the log.

    Kept separate from ``_build_system_prompt`` so unit tests that only care
    about prompt content (the majority) stay synchronous and don't have to
    stub out observability.
    """
    prompt = _build_system_prompt(workspace_path, build_command)
    try:
        from harness.observability import emit_event
        tree_lines = prompt.split("## Directory Structure", 1)
        emit_event(
            "system_prompt_built",
            chars=len(prompt),
            lines=prompt.count("\n") + 1,
            tree_lines=(
                tree_lines[1].split("\n## ", 1)[0].count("\n")
                if len(tree_lines) > 1 else 0
            ),
        )
    except Exception:  # noqa: BLE001 — telemetry must never block
        pass
    return prompt


_TREE_NOISE_DIRS = frozenset({
    "node_modules", "__pycache__", "target", "build", "dist", ".git",
    # Audit (#8/#9): keep the tree small enough that the system prompt
    # stays under the 60-line aspirational target without hiding any
    # real source. These directories carry vendored artefacts, caches,
    # or generated output — useful to operators, noise for the LLM.
    ".venv", "venv", "env", ".tox", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".idea", ".vscode", ".gradle", ".next", ".nuxt",
    "coverage", "htmlcov", ".cache", "vendor",
})


def _snapshot_directory_tree(
    path: str, max_depth: int = 3, max_files_per_dir: int = 20,
) -> str:
    """
    Generate a lightweight directory tree snapshot for the system prompt.
    Limits depth and file count to avoid bloating the prompt.

    Audit (#8): defaults tightened from (depth=4, files=50) to (3, 20).
    The previous defaults could emit 500+ tree lines on a midsize repo;
    the LLM rarely needs more than the top-2 layers + a representative
    file list to orient. Files past the cap collapse into a single
    ``... (N more files)`` marker so the cardinality stays visible.
    """
    lines: list[str] = []
    try:
        for root, dirs, files in os.walk(path):
            depth = root[len(path):].count(os.sep)
            if depth > max_depth:
                dirs.clear()
                continue
            # Skip hidden and common noise directories.
            dirs[:] = [
                d
                for d in sorted(dirs)
                if not d.startswith(".") and d not in _TREE_NOISE_DIRS
            ]
            indent = "  " * (depth + 1)
            rel = os.path.relpath(root, path)
            if rel == ".":
                lines.append(f"{os.path.basename(path)}/")
            else:
                lines.append(f"{indent[:-2]}{os.path.basename(root)}/")
            shown = sorted(files)[:max_files_per_dir]
            for f in shown:
                lines.append(f"{indent}{f}")
            if len(files) > max_files_per_dir:
                lines.append(f"{indent}... ({len(files) - max_files_per_dir} more files)")
    except (OSError, PermissionError) as exc:
        # Surface as a WARNING so operators see the failure in logs — the
        # previous silent return injected the error string straight into the
        # LLM system prompt with no other signal, and the LLM would then
        # hallucinate file paths against a workspace it cannot see.
        logger.warning(
            "[graph] Could not snapshot directory tree at %s: %s", path, exc,
        )
        lines.append(f"[Error reading directory: {exc}]")
    if not lines:
        return (
            f"[Unable to read directory structure at {path!s}. "
            "The workspace appears empty or inaccessible — patches generated "
            "against this snapshot will likely target non-existent files.]"
        )
    return "\n".join(lines)


_APPLIES_TO_RE = re.compile(
    r'^---\s*\n\s*applies_to\s*:\s*\[([^\]]*)\]\s*\n---\s*\n',
    re.MULTILINE,
)


def _parse_skill_frontmatter(content: str) -> tuple[Optional[set[str]], str]:
    """Extract the ``applies_to:`` tag list from a skill file's frontmatter.

    Recognises the minimal form::

        ---
        applies_to: [tag1, tag2]
        ---

        ... rest of the skill ...

    Returns ``(tags, body)`` where ``tags`` is ``None`` when no frontmatter
    is present (skill loads unconditionally) or a set of tag strings when
    the frontmatter declares them. ``body`` is the markdown content with
    the frontmatter stripped.

    This is a deliberately tiny hand-rolled parser — we don't want a YAML
    dependency for a one-field schema.
    """
    m = _APPLIES_TO_RE.match(content)
    if not m:
        return None, content
    tag_blob = m.group(1)
    tags = {t.strip() for t in tag_blob.split(",") if t.strip()}
    body = content[m.end():]
    return tags, body


def _load_skills_markdown(
    skills_dir: str,
    max_file_chars: int = 4000,
    workspace_tags: Optional[set[str]] = None,
) -> str:
    """
    Scan a skills/ directory for .md files and return their concatenated content.

    Each file's content is truncated to ``max_file_chars`` to prevent system
    prompt bloat. Files are sorted alphabetically for deterministic ordering.

    When ``workspace_tags`` is provided, skills with an ``applies_to:``
    frontmatter are loaded only if at least one of their declared tags
    appears in ``workspace_tags``. Skills with no frontmatter (e.g. the
    universal ``agent-standards.md``) always load.

    Args:
        skills_dir: Absolute path to the skills directory.
        max_file_chars: Maximum characters to read per skill file.
        workspace_tags: Tags returned by ``impact._detect_workspace_stack``.
                        ``None`` disables filtering (legacy behavior).

    Returns:
        Concatenated markdown content (frontmatter stripped), or empty
        string if no skills match.
    """
    if not os.path.isdir(skills_dir):
        return ""
    parts: list[str] = []
    try:
        for fname in sorted(os.listdir(skills_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(skills_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as sf:
                    content = sf.read(max_file_chars)
            except OSError:
                logger.warning("[graph] Could not read skills file: %s", fpath)
                continue

            applies_to, body = _parse_skill_frontmatter(content)

            # Filter by workspace tags when both sides have something to say.
            # Skills with no frontmatter (applies_to is None) load
            # unconditionally — that's the "universal skill" pattern used by
            # agent-standards.md and any user-supplied project skill.
            if workspace_tags is not None and applies_to is not None:
                if not (applies_to & workspace_tags):
                    logger.debug(
                        "[graph] Skipping skill %s (applies_to=%s, workspace=%s)",
                        fname, sorted(applies_to), sorted(workspace_tags),
                    )
                    continue

            if body.strip():
                parts.append(body)
    except OSError:
        return ""
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 2b. Gateway Injection Mechanism
# ---------------------------------------------------------------------------

_gateway: Optional[Any] = None
_gateway_config: Optional[Any] = None


def set_gateway(gateway: Any) -> None:
    """
    Inject the LLM Gateway instance for use by graph nodes.
    Called by the CLI layer before running the graph.

    Also stashes ``gateway.config`` via :func:`set_gateway_config` so the
    reviewer nodes (``spec_review_node``, ``code_review_node``) and the
    pre-flight reviewer block in ``cmd_run`` see a non-None config. Prior
    to this, ``set_gateway_config`` had no caller anywhere — every
    ``get_gateway_config()`` consumer read None and silently skipped,
    which made every configured reviewer LLM effectively dead code.
    """
    global _gateway
    _gateway = gateway
    config = getattr(gateway, "config", None)
    if config is not None:
        set_gateway_config(config)
    logger.info("[graph] Gateway instance injected.")


def get_gateway() -> Optional[Any]:
    """Retrieve the injected Gateway instance, or None if not set."""
    return _gateway


def set_gateway_config(config: Any) -> None:
    """Inject GatewayConfig for node model selection."""
    global _gateway_config
    _gateway_config = config


def get_gateway_config() -> Optional[Any]:
    """Retrieve the injected GatewayConfig."""
    return _gateway_config


# ---------------------------------------------------------------------------
# 3. Memory Cleanse Utility (Module 4)
# ---------------------------------------------------------------------------

def _emit_per_stage_spend_summary(token_tracker: dict[str, Any]) -> None:
    """Log per-stage cumulative spend + soft warnings when a stage exceeds
    its configured share of the total budget. Observability-only (C4 scaffold);
    hard enforcement is a follow-up.

    Reads the optional ``token_budget.stages`` dict from gateway.config —
    a map of role name → target fraction. Logs warnings when:
      - A stage has spent more than its target_fraction × hard_cap_usd.
      - Speculative/repair combined exceed a hard ratio (caps the historic
        "speculative ate the repair budget" failure mode).
    """
    per_stage = token_tracker.get("per_stage") or {}
    if not per_stage:
        return
    gw = get_gateway()
    if gw is None:
        return
    stages_cfg = getattr(gw.config, "stages", None)
    hard_cap = float(getattr(gw.config, "hard_cap_usd", 0.0) or 0.0)
    parts = []
    for role_key, info in sorted(per_stage.items()):
        cost = float(info.get("cost_usd", 0.0))
        calls = int(info.get("calls", 0))
        parts.append(f"{role_key}=${cost:.4f}({calls}c)")
        if stages_cfg and hard_cap > 0 and isinstance(stages_cfg, dict):
            target_frac = stages_cfg.get(role_key)
            if isinstance(target_frac, (int, float)) and target_frac > 0:
                target_usd = target_frac * hard_cap
                if cost > target_usd:
                    logger.warning(
                        "[budget] Stage %r spent $%.4f, exceeding its soft "
                        "target of $%.4f (%.0f%% of hard_cap $%.2f). "
                        "Observability-only today; consider adjusting "
                        "token_budget.stages or model_routing.",
                        role_key, cost, target_usd, target_frac * 100, hard_cap,
                    )
    if parts:
        logger.info("[budget] Per-stage spend: %s", ", ".join(parts))


def apply_repair_iteration_cleanse(state: AgentState) -> dict[str, Any]:
    """Trim mid-loop debugging chatter before a repair iteration ≥ 2.

    Unlike :func:`apply_memory_cleanse` (which fires only on success / HITL
    and compresses the whole history into a single summary), this trims
    *between* iterations so iteration N's LLM call doesn't carry the bloat
    of iterations 1..N-1. We keep:

      - ``messages[0]`` (the anchored system prompt — never trimmed)
      - the first user message (the original planning prompt — defines the
        goal the LLM is repairing toward)
      - the most recent assistant message (the LAST repair turn's patch
        attempt — the LLM needs to remember what it just tried so its next
        response is a delta, not a re-do)

    Everything between is dropped. The repair_node will then append a fresh
    error_summary (diagnostics + patch-failure feedback + workspace inventory
    + allowlist + new failures), so the LLM sees full structured context
    *for this turn* without N turns of stale history.

    Returns an empty dict (no-op) when there's nothing to trim — fewer than
    4 messages or no prior assistant turn yet.
    """
    messages: list[MessageDict] = state.get("messages", [])
    if len(messages) < 4:
        return {}

    system_prompt = messages[0] if messages else None
    planning_message: Optional[MessageDict] = None
    for m in messages[1:]:
        if m.get("role") == "user":
            planning_message = m
            break

    last_assistant: Optional[MessageDict] = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m
            break

    # Without a prior assistant turn (first repair-iteration cleanse would
    # find none), there's nothing useful to compress — leave as-is.
    if last_assistant is None:
        return {}

    cleansed: list[MessageDict] = []
    if system_prompt is not None:
        cleansed.append(system_prompt)
    if planning_message is not None and planning_message is not system_prompt:
        cleansed.append(planning_message)
    cleansed.append(last_assistant)

    if len(cleansed) >= len(messages):
        return {}  # No-op — nothing trimmed.

    logger.info(
        "[memory_cleanse] Trimmed mid-loop messages: %d → %d (kept system + "
        "planning + last assistant; dropped %d intermediate turns).",
        len(messages), len(cleansed), len(messages) - len(cleansed),
    )
    return {"messages": cleansed}


def apply_memory_cleanse(state: AgentState, resolution_kind: str = "compiler_success") -> dict[str, Any]:
    """
    Purge verbose intermediate repair-loop messages from the conversation history
    and compress the debugging session into a single structured summary line.

    Triggered on:
        - Compiler exit code 0 (resolution_kind = "compiler_success")
        - Human intervention resolution (resolution_kind = "human_intervention")

    Retains:
        - messages[0] (system prompt — never truncated)
        - The original planning/user message
        - The final successful patching message (last assistant message before the cleanse)
        - The compression summary injected as a system-tagged message

    Returns:
        A state update dict with cleansed messages that should be merged into the state.
    """
    messages: list[MessageDict] = state.get("messages", [])
    if len(messages) < 3:
        return {}  # Nothing to cleanse

    # Locate the original planning exchange: first user message after system prompt
    system_prompt = messages[0] if messages else None
    planning_message: Optional[MessageDict] = None
    for m in messages[1:]:
        if m.get("role") == "user":
            planning_message = m
            break

    # The final assistant message (the patch that succeeded)
    final_patch: Optional[MessageDict] = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            final_patch = m
            break

    # Collect metrics for the compression summary
    repair_iterations = state.get("loop_counter", {}).get("total_repairs", 0)
    token_tracker: TokenTrackerDict = state.get("token_tracker", {})
    debug_cost = token_tracker.get("total_cost_usd", 0.0)
    debug_input = token_tracker.get("total_input_tokens", 0)
    debug_output = token_tracker.get("total_output_tokens", 0)

    # Determine which file was fixed
    target_file = "unknown"
    modified: list[str] = state.get("modified_files", [])
    if modified:
        target_file = modified[-1]
    elif state.get("compiler_errors"):
        errs: list[DiagnosticObjectDict] = state.get("compiler_errors", [])
        if errs:
            target_file = errs[0].get("file", "unknown")

    # Build the compression summary
    if resolution_kind == "human_intervention":
        summary = (
            f"[System]: Resolved via manual human intervention.\n"
            f"  Target File: {target_file}\n"
            f"  Iterations Before Intervention: {repair_iterations}\n"
            f"  Debug Token Cost: ${debug_cost:.6f} (In: {debug_input}, Out: {debug_output})"
        )
    else:
        summary = (
            f"[System]: Successfully resolved compilation fault in file {target_file}. "
            f"({repair_iterations} repair iteration(s), "
            f"${debug_cost:.6f} debug token cost, "
            f"In: {debug_input}, Out: {debug_output})"
        )

    # Reconstruct the cleansed message list
    cleansed: list[MessageDict] = []
    if system_prompt is not None:
        cleansed.append(system_prompt)
    if planning_message is not None:
        cleansed.append(planning_message)
    if final_patch is not None:
        cleansed.append(final_patch)
    cleansed.append(MessageDict(role="system", content=summary))

    logger.info(
        "[memory_cleanse] Compressed %d messages → %d messages. File: %s, Iterations: %d, Cost: $%.6f",
        len(messages), len(cleansed), target_file, repair_iterations, debug_cost,
    )

    return {"messages": cleansed}


# ---------------------------------------------------------------------------
# 4. Node Implementations (Gateway-Integrated)
# ---------------------------------------------------------------------------

def _web_tool_cap_from_state(_state: "AgentState") -> int:
    """Resolve the per-dispatch tool-call cap from the registered web
    tools skill, falling back to 3 when the skills aren't registered.

    Reads it from the SkillRegistry rather than threading config through
    AgentState so existing checkpoints (no ``web_tools_config`` field)
    keep working unchanged. Cheap (one dict lookup); no I/O.
    """
    try:
        from harness.skills import SkillRegistry
        from harness.web_tools import WebFetchSkill
        skill = SkillRegistry().get("web_fetch")
        if isinstance(skill, WebFetchSkill):
            return int(getattr(skill, "_cfg").tool_call_cap_per_dispatch)
    except Exception:  # noqa: BLE001
        pass
    return 3


# Cap on read_file rounds inside one patching turn. Stops a chatty model
# from reading the whole repo before emitting a single edit. The cap
# applies to the number of *re-dispatches*; the model can ask for
# multiple files in one round and each counts as a single round.
_PATCHING_READ_FILE_CAP = 6

# Cap on continuation cycles for nodes that opt into
# llm_dispatch.continue_on_length.<role>. See the comment on that
# section in config/config.json for the per-role risk profile.
_MAX_CONTINUATION_CYCLES = 3

# Per-role default for ``continue_on_length`` when the operator's
# config.json omits the section or the role entry. Patching's default
# is True because full-stack blueprints regularly exceed its 16384
# token cap; every other node defaults to False because the failure
# mode is more nuanced (see _llm_dispatch_comment in config.json).
_CONTINUE_ON_LENGTH_DEFAULTS: dict[str, bool] = {
    "planning": False,
    "patching": True,
    "repair": False,
    "doc_reviewer": False,
    "code_reviewer": False,
}


def _resolve_continue_on_length(
    state: "AgentState", role: str,
) -> bool:
    """Resolve the per-role ``continue_on_length`` flag.

    Reads ``state.llm_dispatch_config.continue_on_length.<role>``;
    falls back to :data:`_CONTINUE_ON_LENGTH_DEFAULTS` when the
    operator's config omits the section or the role entry. A missing /
    None config produces the documented default behaviour (only
    patching continues on length).
    """
    cfg = state.get("llm_dispatch_config", {}) or {}
    role_map = cfg.get("continue_on_length") or {}
    default = _CONTINUE_ON_LENGTH_DEFAULTS.get(role, False)
    val = role_map.get(role, default)
    return bool(val) if val is not None else default


async def _continue_on_length(
    *,
    initial_response: Any,
    initial_budget: float,
    messages: list["MessageDict"],
    dispatch: Any,
    continue_prompt: str,
    enabled: bool,
    role_label: str,
) -> tuple[Any, float, list[str]]:
    """Run the finish_reason==length continuation loop for a node.

    The patching node has used this pattern since session
    web-6d5ef9b18f6a — when the LLM hits its output-token cap mid-
    response, append the truncated assistant turn + a role-specific
    'continue' user prompt and re-dispatch, up to
    :data:`_MAX_CONTINUATION_CYCLES`. Other nodes opt into the same
    pattern via ``llm_dispatch.continue_on_length.<role>`` in
    config.json.

    Parameters
    ----------
    initial_response:
        The response from the node's first dispatch.
    initial_budget:
        The post-first-dispatch budget; later cycles thread through
        ``dispatch`` and the final value is returned.
    messages:
        The conversation list mutated in place — the helper appends
        ``assistant`` + ``user`` turns per cycle.
    dispatch:
        Async callable ``(messages, budget) -> (response, new_budget)``.
        The caller captures any extra per-cycle state (tool results,
        files-seen maps) via the closure.
    continue_prompt:
        Role-specific user-turn text appended on each cycle. Patching
        keeps its existing DSL-flavoured prompt; planning / repair /
        reviewer nodes supply their own (see callers).
    enabled:
        Pass the resolved config flag. When False, returns immediately
        with the initial response — no continuation, no extra cost.
    role_label:
        Used for the per-cycle info log line and the "still truncated"
        warning.

    Returns
    -------
    ``(final_response, final_budget, accumulated_text_chunks)``. The
    caller decides what to do with the chunks — patching concatenates
    them through ``process_llm_patch_output``; planning / reviewer
    nodes use them as-is.
    """
    accumulated_chunks: list[str] = [initial_response.content or ""]
    if not enabled:
        return initial_response, initial_budget, accumulated_chunks
    # Skip continuation when the model returned tool_calls — tool-use
    # mode already multi-turns inside its own loop (e.g. patching's
    # _patching_tool_loop) and a second continuation here would
    # double-charge for content the model already finalised.
    if getattr(initial_response, "tool_calls", None):
        return initial_response, initial_budget, accumulated_chunks

    response = initial_response
    budget = initial_budget
    continuation_cycles = 0
    # ``getattr`` with a "stop" default keeps stub responses in tests
    # (which historically omit finish_reason) on the non-continuation
    # path — only real gateway responses with an explicit "length"
    # trigger the loop.
    while (
        getattr(response, "finish_reason", "stop") == "length"
        and continuation_cycles < _MAX_CONTINUATION_CYCLES
    ):
        continuation_cycles += 1
        logger.info(
            "[%s] hit output token cap (cycle %d/%d) — requesting "
            "continuation.",
            role_label, continuation_cycles, _MAX_CONTINUATION_CYCLES,
        )
        messages.append(MessageDict(
            role="assistant", content=response.content or "",
        ))
        messages.append(MessageDict(
            role="user", content=continue_prompt,
        ))
        response, budget = await dispatch(messages, budget)
        accumulated_chunks.append(response.content or "")
    if (
        continuation_cycles >= _MAX_CONTINUATION_CYCLES
        and getattr(response, "finish_reason", "stop") == "length"
    ):
        logger.warning(
            "[%s] LLM still truncated after %d continuation cycle(s); "
            "accepting what landed and moving on.",
            role_label, _MAX_CONTINUATION_CYCLES,
        )
    return response, budget, accumulated_chunks
# Cap on bytes returned per read_file tool_result so a single edit
# attempt against a multi-megabyte file can't blow the context window.
# Aligns with the harness's existing READ_FILE text-DSL cap.
_PATCHING_READ_FILE_MAX_BYTES = 200_000


def _build_assistant_tool_turn(response: Any) -> "MessageDict":
    """Reconstruct the assistant turn from an LLM response that included
    tool calls. Returns a ``MessageDict`` whose ``content`` is the
    canonical Anthropic-style typed-block list ([{type: text, text:
    ...}, {type: tool_use, id, name, input}, ...]).

    The OpenAI-shape providers consume this through
    :func:`gateway._normalize_messages_for_openai_tools`, so we keep
    one in-memory representation.
    """
    blocks: list[dict[str, Any]] = []
    if response.content:
        blocks.append({"type": "text", "text": response.content})
    for call in getattr(response, "tool_calls", None) or []:
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or "",
            "name": call.get("name") or "",
            "input": call.get("input") or {},
        })
    return MessageDict(role="assistant", content=blocks)


def _resolve_read_file_call(
    call: dict[str, Any], workspace_root: str,
) -> str:
    """Resolve a single ``read_file`` tool call against ``workspace_root``.

    Honours optional ``start_line`` / ``end_line`` inputs (1-indexed,
    inclusive). Mirrors the byte cap of the existing text-DSL
    READ_FILE resolver so a single read can't dominate the context
    window. Returns the bytes (possibly truncated, with a marker) or
    an error string — never raises.
    """
    args = call.get("input") or {}
    rel_path = str(args.get("file_path") or "").strip()
    if not rel_path:
        return "Error: read_file requires file_path."
    # Refuse absolute / traversal paths up front. The patcher would
    # reject them on the apply side anyway; doing it here means the
    # model gets a clear error in the same turn.
    # On Windows, ``split_path_components`` normalises backslashes so
    # paths like ``..\\foo`` or ``C:\\Windows`` are caught by the same
    # POSIX-shaped traversal/absolute check. POSIX behaviour is
    # byte-identical because backslash is a legal filename character there.
    if (
        rel_path.startswith("/")
        or os.path.isabs(rel_path)
        or ".." in _platform.split_path_components(rel_path)
    ):
        return f"Error: refused absolute / traversal path {rel_path!r}."
    abs_path = os.path.join(workspace_root, rel_path)
    if not os.path.isfile(abs_path):
        return f"Error: file not found: {rel_path}"
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as exc:
        return f"Error reading {rel_path}: {exc}"
    start = args.get("start_line")
    end = args.get("end_line")
    if start is not None or end is not None:
        try:
            lines = content.splitlines()
            s_idx = max(0, int(start) - 1) if start is not None else 0
            e_idx = (
                min(len(lines), int(end))
                if end is not None else len(lines)
            )
            if e_idx < s_idx:
                e_idx = s_idx
            content = "\n".join(lines[s_idx:e_idx])
        except (TypeError, ValueError):
            return "Error: start_line / end_line must be integers."
    encoded = content.encode("utf-8")
    if len(encoded) > _PATCHING_READ_FILE_MAX_BYTES:
        truncated = encoded[:_PATCHING_READ_FILE_MAX_BYTES].decode(
            "utf-8", errors="replace",
        )
        return (
            truncated
            + f"\n\n[... truncated at "
            f"{_PATCHING_READ_FILE_MAX_BYTES} bytes; full file is "
            f"{len(encoded)} bytes. Re-read with start_line/end_line "
            f"for a specific window.]"
        )
    return content


async def _patching_tool_loop(
    *,
    gateway: Any,
    messages: list["MessageDict"],
    budget: float,
    workspace: str,
    use_tools: bool,
    tools: list[dict[str, Any]],
    state: "AgentState",
) -> tuple[Any, float, list["MessageDict"], dict[str, str]]:
    """Drive a patching/repair dispatch that may use native tool-use.

    Single round when ``use_tools=False`` (legacy text-DSL behaviour).
    When ``use_tools=True``, dispatches with ``tools=PATCH_TOOLS``; if
    the model responds with ``read_file`` tool calls, resolves them
    against the workspace and re-dispatches with ``tool_result`` blocks
    appended, up to :data:`_PATCHING_READ_FILE_CAP` rounds.

    The loop terminates as soon as the model returns a response that
    contains *no* ``read_file`` calls — that means either it has the
    patch operations it needs (other tool_calls) or it fell back to a
    pure-text response.

    Returns ``(final_response, budget_remaining, updated_messages,
    files_seen)``. ``files_seen`` is the ``{rel_path: sha256}`` map of
    every file the loop showed the model — the caller threads it into
    :func:`apply_patch_blocks` so B5 drift detection / read-before-edit
    sees the bytes the model just consumed.
    """
    from harness.gateway import NodeRole

    dispatch_kwargs: dict[str, Any] = {}
    if use_tools:
        dispatch_kwargs["tools"] = tools

    response, budget = await gateway.dispatch(
        messages=list(messages),
        role=NodeRole.PATCHING,
        budget_remaining_usd=budget,
        **dispatch_kwargs,
    )

    # Seed from any existing node_state.files_seen_by_llm so a resumed
    # session doesn't lose prior reads.
    files_seen: dict[str, str] = dict(
        (state.get("node_state", {}) or {}).get("files_seen_by_llm") or {}
    )

    if not use_tools or not getattr(response, "tool_calls", None):
        return response, budget, messages, files_seen

    rounds = 0
    while rounds < _PATCHING_READ_FILE_CAP:
        tool_calls_list = getattr(response, "tool_calls", None) or []
        read_calls = [
            c for c in tool_calls_list if c.get("name") == "read_file"
        ]
        if not read_calls:
            return response, budget, messages, files_seen
        rounds += 1
        # Persist the assistant turn — typed blocks so the next round's
        # tool_result references match against the right tool_use ids.
        messages.append(_build_assistant_tool_turn(response))
        # Resolve each read_file and assemble the tool_result user turn.
        tool_results: list[dict[str, Any]] = []
        for call in read_calls:
            result_text = _resolve_read_file_call(call, workspace)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.get("id", ""),
                "content": result_text,
            })
            # Record the sha256 of the bytes we just showed so B5 drift
            # detection / read-before-edit can use them on the next
            # dispatch.
            args = call.get("input") or {}
            rel = str(args.get("file_path") or "").strip()
            is_error = result_text.startswith("Error:")
            if rel and not is_error:
                try:
                    import hashlib as _hl
                    files_seen[rel] = _hl.sha256(
                        result_text.encode("utf-8")
                    ).hexdigest()
                except Exception:  # noqa: BLE001 — best-effort
                    pass
            try:
                from harness.observability import emit_event, log_failure
                if is_error:
                    log_failure(
                        "tool_call_failed",
                        tool_name="read_file",
                        reason=result_text[:200],
                    )
                else:
                    emit_event("tool_call_succeeded", tool_name="read_file")
            except Exception:  # noqa: BLE001 — telemetry must never block
                pass
        messages.append(MessageDict(role="user", content=tool_results))
        # Re-dispatch.
        try:
            response, budget = await gateway.dispatch(
                messages=list(messages),
                role=NodeRole.PATCHING,
                budget_remaining_usd=budget,
                **dispatch_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — surface and stop the loop
            logger.warning(
                "[patching_tool_loop] re-dispatch failed at round %d: %s",
                rounds, exc,
            )
            break

    if rounds >= _PATCHING_READ_FILE_CAP:
        logger.info(
            "[patching_tool_loop] hit cap of %d read_file rounds; "
            "patching proceeds with what the LLM has so far.",
            _PATCHING_READ_FILE_CAP,
        )
        # Audit §6.3: tell the model the read-cap is hit and ask it
        # to emit patches now. Without this synthetic note, the final
        # response still carries unfulfilled ``read_file`` tool_calls
        # which downstream code treats as "the LLM is asking for more
        # files" — the LLM never learns to stop and produce patches.
        try:
            # Best-effort one more dispatch with a guiding nudge.
            messages.append(MessageDict(
                role="user",
                content=(
                    f"[System]: You've used the read_file tool "
                    f"{_PATCHING_READ_FILE_CAP} times — the cap is reached. "
                    f"Emit SEARCH/REPLACE patches now using the file content "
                    f"you've already seen. Do NOT call read_file again."
                ),
            ))
            response, budget = await gateway.dispatch(
                messages=list(messages),
                role=NodeRole.PATCHING,
                budget_remaining_usd=budget,
                **dispatch_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, caller has the cap response
            logger.debug("[patching_tool_loop] post-cap dispatch failed: %s", exc)
    return response, budget, messages, files_seen


async def _run_tool_loop(
    *,
    initial_response_content: str,
    messages: list["MessageDict"],
    gateway: Any,
    role: Any,
    budget: float,
    cap: int = 3,
    token_tracker: Optional[dict[str, Any]] = None,
) -> tuple[str, list["MessageDict"], float, int]:
    """Iteratively resolve ``<<<WEB_FETCH ...>>>`` / ``<<<WEB_SEARCH ...>>>``
    blocks in the LLM response.

    Each round:
      1. Parse tool blocks from the latest response content.
      2. If none → return the content unchanged.
      3. Otherwise, dispatch each block via :class:`SkillRegistry`,
         append a ``user`` message containing the tool results,
         re-dispatch the same role, append the new assistant reply,
         and repeat.

    Stops at ``cap`` rounds. The final content is the response from the
    last dispatch with all tool blocks **stripped** (so downstream
    consumers — patcher, planning blueprint storage — never see them).

    Returns ``(final_content, messages, budget_remaining_usd, rounds_run)``.

    Best-effort: if the skill registry isn't initialised, or the
    ``web_tools`` skills aren't registered (operator hasn't opted in),
    the function quietly returns the original content unchanged and the
    tool blocks remain in the text — the LLM will see them on a future
    dispatch and learn that tools aren't available.
    """
    try:
        from harness.web_tools import parse_tool_blocks, strip_tool_blocks
        from harness.skills import SkillRegistry
    except Exception as exc:  # noqa: BLE001 — web tools are optional
        logger.debug("[tool_loop] web tools unavailable: %s", exc)
        return initial_response_content, messages, budget, 0

    # MCP blocks are parsed by a separate module (the parser ships with
    # mcp_client.py to keep the MCP surface self-contained). The import
    # is best-effort: if MCP isn't enabled / installed, the variable
    # stays None and only web tool blocks are intercepted.
    try:
        from harness.mcp_client import parse_mcp_blocks, strip_mcp_blocks
    except Exception:  # noqa: BLE001 — MCP is optional
        parse_mcp_blocks = None  # type: ignore[assignment]
        strip_mcp_blocks = None  # type: ignore[assignment]

    def _all_blocks(text: str) -> list[Any]:
        out = list(parse_tool_blocks(text))
        if parse_mcp_blocks is not None:
            out.extend(parse_mcp_blocks(text))
        return out

    def _strip_all_blocks(text: str) -> str:
        text = strip_tool_blocks(text)
        if strip_mcp_blocks is not None:
            text = strip_mcp_blocks(text)
        return text

    current_content = initial_response_content
    rounds = 0
    while rounds < cap:
        blocks = _all_blocks(current_content)
        if not blocks:
            break
        rounds += 1
        registry = SkillRegistry()
        tool_results: list[str] = []
        for block in blocks:
            skill = registry.get(block.skill_name)
            if skill is None:
                try:
                    from harness.observability import log_failure
                    log_failure(
                        "tool_call_failed",
                        tool_name=block.skill_name,
                        reason="not_registered",
                    )
                except Exception:  # noqa: BLE001 — telemetry must never block
                    pass
                tool_results.append(
                    f"[tool {block.skill_name}] not registered "
                    f"(set web_tools.enabled=true in config to enable)."
                )
                continue
            failure_reason: Optional[str] = None
            try:
                result = await skill.execute(**block.kwargs)
            except Exception as exc:  # noqa: BLE001 — never let a skill blow up the graph
                logger.exception("[tool_loop] skill %s failed", block.skill_name)
                failure_reason = f"unexpected error: {exc}"
                result = {"error": failure_reason}
            if failure_reason is None and isinstance(result, dict) and result.get("error"):
                failure_reason = str(result.get("error"))
            try:
                from harness.observability import emit_event, log_failure
                if failure_reason is not None:
                    log_failure(
                        "tool_call_failed",
                        tool_name=block.skill_name,
                        reason=failure_reason[:200],
                    )
                else:
                    emit_event("tool_call_succeeded", tool_name=block.skill_name)
            except Exception:  # noqa: BLE001 — telemetry must never block
                pass
            tool_results.append(
                f"[tool {block.skill_name}({block.kwargs})] -> {result!r}"
            )

        # Strip the tool blocks (web + MCP) from the assistant text we
        # just got so the patcher / blueprint store never sees them, and
        # so the LLM doesn't see them re-quoted in the next round.
        clean_assistant = _strip_all_blocks(current_content).strip()
        if clean_assistant:
            messages.append(MessageDict(role="assistant", content=clean_assistant))

        # Append tool results as a user message — same shape as the rest of
        # the harness's tool DSL (READ_FILE results come back this way too).
        messages.append(
            MessageDict(
                role="user",
                content=(
                    "Tool execution results (round %d/%d):\n%s\n\n"
                    "Continue your previous task. Emit more tool blocks if "
                    "needed, or proceed to the final answer."
                ) % (rounds, cap, "\n".join(tool_results)),
            )
        )

        # Re-dispatch
        try:
            response, budget = await gateway.dispatch(
                messages=list(messages),
                role=role,
                budget_remaining_usd=budget,
            )
        except Exception as exc:  # noqa: BLE001 — surface and stop the loop
            logger.warning("[tool_loop] re-dispatch failed: %s", exc)
            break
        current_content = response.content or ""
        # Account for the re-dispatch tokens in the caller's tracker. If
        # the caller passed token_tracker=None they've opted out and the
        # tokens won't show up in their per-stage breakdown — but the
        # gateway has already deducted the cost from budget either way.
        if token_tracker is not None:
            try:
                gateway.aggregate_tokens(token_tracker, response.usage, role=role)
            except Exception as exc:  # noqa: BLE001 — telemetry must not block
                logger.debug("[tool_loop] token aggregate skipped: %s", exc)

    # Strip any residual tool blocks (web + MCP) from the final content
    # before returning so downstream code paths (blueprint storage,
    # repair grep, …) never see a half-parsed <<<WEB_FETCH...>>> /
    # <<<MCP_CALL...>>> token.
    try:
        final_content = _strip_all_blocks(current_content)
    except Exception:  # noqa: BLE001
        final_content = current_content
    if rounds >= cap:
        logger.info("[tool_loop] hit cap of %d rounds; forcing LLM to proceed.", cap)
    return final_content, messages, budget, rounds


async def planning_node(state: AgentState) -> dict[str, Any]:
    """
    Node 1: The Architect.

    Uses the configured reasoning model (planning_primary in .harness_config.json)
    via the model-agnostic gateway to generate an implementation blueprint.
    The blueprint is a structured plan that the patching_node will execute.

    This node reads the user's task from the last user message and produces
    a detailed step-by-step implementation plan.
    """
    logger.info("[planning_node] Generating implementation blueprint...")

    messages: list[MessageDict] = list(state.get("messages", []))
    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["planning"] = loop_counter.get("planning", 0) + 1

    gateway = get_gateway()
    if gateway is None:
        logger.error("[planning_node] No gateway configured. Cannot call LLM.")
        return {
            "node_state": {"current_node": "planning", "error": "No gateway configured"},
            "loop_counter": loop_counter,
        }

    try:
        from harness.gateway import NodeRole

        # Inject per-repo memory (prior-session notes) into the planner
        # context when memory is enabled and the file exists. The note
        # goes in as a SECOND system message right after the canonical
        # planner system prompt, so the cache_control marker on
        # messages[0] still bites. Best-effort: any failure logs and
        # falls back to the unmodified messages list.
        try:
            workspace_path = state.get("workspace_path", "")
            if workspace_path:
                from harness.repo_memory import RepoMemoryConfig, read_repo_memory
                cfg = state.get("repo_memory_config") or {}
                mem_cfg = RepoMemoryConfig(
                    enabled=bool(cfg.get("enabled", True)),
                    dir=str(cfg.get("dir", "~/.harness/memory")),
                    max_bytes=int(cfg.get("max_bytes", 100_000)),
                    inject_max_bytes=int(cfg.get("inject_max_bytes", 8_000)),
                )
                memory_text = read_repo_memory(workspace_path, mem_cfg)
                if memory_text:
                    # Find the index just after the last system message
                    # so the prior-session notes ride alongside the
                    # main system prompt rather than landing as a user
                    # message.
                    insert_at = 0
                    for i, m in enumerate(messages):
                        if m.get("role") == "system":
                            insert_at = i + 1
                        else:
                            break
                    messages.insert(
                        insert_at,
                        MessageDict(
                            role="system",
                            content=(
                                "### Prior session memory for this repository\n\n"
                                "Use this only as background context — do not "
                                "echo it back. Each entry summarises a past "
                                "run on the same workspace.\n\n"
                                + memory_text
                            ),
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[planning_node] memory injection skipped: %s", exc)

        # Inject semantic retrieval results — top-K relevant chunks for
        # the user's prompt. Independent of the memory injection above
        # and gated on its own ``repo_index.enabled`` flag.
        try:
            workspace_path = state.get("workspace_path", "")
            idx_cfg_dict = state.get("repo_index_config") or {}
            idx_enabled = bool(idx_cfg_dict.get("enabled", False))
            if workspace_path and idx_enabled:
                from harness.repo_index import (
                    RepoIndexConfig,
                    async_query_top_chunks,
                    render_results_for_injection,
                )
                idx_cfg = RepoIndexConfig(
                    enabled=True,
                    backend=str(idx_cfg_dict.get("backend", "tfidf")),
                    top_k=int(idx_cfg_dict.get("top_k", 5)),
                    chunk_lines=int(idx_cfg_dict.get("chunk_lines", 200)),
                    chunk_overlap=int(idx_cfg_dict.get("chunk_overlap", 20)),
                    inject_max_bytes=int(idx_cfg_dict.get("inject_max_bytes", 4000)),
                    index_dir=str(idx_cfg_dict.get("index_dir", "~/.harness/repo_index")),
                )
                # Use the last user message as the retrieval query —
                # that's what represents "what the operator asked for".
                query = ""
                for m in reversed(messages):
                    if m.get("role") == "user":
                        query = m.get("content", "") or ""
                        break
                if query:
                    results = await async_query_top_chunks(
                        workspace_path, query, cfg=idx_cfg,
                    )
                    block = render_results_for_injection(
                        results, max_bytes=idx_cfg.inject_max_bytes,
                    )
                    if block:
                        insert_at = 0
                        for i, m in enumerate(messages):
                            if m.get("role") == "system":
                                insert_at = i + 1
                            else:
                                break
                        messages.insert(
                            insert_at,
                            MessageDict(
                                role="system",
                                content=(
                                    "### Repository context (semantic retrieval)\n\n"
                                    "These are the top-scoring code chunks for the "
                                    "current request. Use them to ground your plan "
                                    "in the existing codebase. Lower scores = less "
                                    "relevant; ignore if not useful.\n\n"
                                    + block
                                ),
                            ),
                        )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[planning_node] repo_index injection skipped: %s", exc)

        budget = state.get("budget_remaining_usd", 2.00)
        response, new_budget = await gateway.dispatch(
            messages=list(messages),
            role=NodeRole.PLANNING,
            budget_remaining_usd=budget,
        )

        # Continuation on finish_reason=="length" — opt-in via
        # llm_dispatch.continue_on_length.planning. See the
        # _llm_dispatch_comment block in config/config.json for the
        # risk profile (continuation may introduce duplicated headings
        # / second summary paragraphs; reach for it only when planning
        # truncation is observed in logs).
        async def _planning_dispatch(msgs, budget_remaining):
            return await gateway.dispatch(
                messages=list(msgs),
                role=NodeRole.PLANNING,
                budget_remaining_usd=budget_remaining,
            )

        response, new_budget, _planning_chunks = await _continue_on_length(
            initial_response=response,
            initial_budget=new_budget,
            messages=messages,
            dispatch=_planning_dispatch,
            continue_prompt=(
                "You hit the output token cap mid-plan. Continue from "
                "where you stopped — emit only the remaining plan "
                "sections (e.g. additional modules, deployment notes, "
                "follow-ups). Do not restate sections you already "
                "produced."
            ),
            enabled=_resolve_continue_on_length(state, "planning"),
            role_label="planning_node",
        )
        _planning_combined_content = "\n".join(
            chunk for chunk in _planning_chunks if chunk
        )

        # Update token tracker (per-stage attribution: planning)
        token_tracker = state.get("token_tracker", {})
        token_tracker = gateway.aggregate_tokens(
            token_tracker, response.usage, role=NodeRole.PLANNING,
        )

        # Resolve any <<<WEB_FETCH ...>>> / <<<WEB_SEARCH ...>>> blocks
        # the planner emitted. When web_tools.enabled is false the
        # blocks fall through with a "tool not registered" notice and
        # the LLM proceeds without them — no graph rewiring needed.
        final_content, messages, new_budget, tool_rounds = await _run_tool_loop(
            initial_response_content=_planning_combined_content,
            messages=messages,
            gateway=gateway,
            role=NodeRole.PLANNING,
            budget=new_budget,
            cap=_web_tool_cap_from_state(state),
            token_tracker=token_tracker,
        )

        # Append the planning response (with tool blocks stripped) to messages
        messages.append(MessageDict(role="assistant", content=final_content))

        # Cross-check the plan's file inventory against the architecture
        # spec's. Best-effort: when SPEC_ARCHITECTURE.md doesn't exist
        # yet, lacks an inventory block, or the planner produced no
        # inventory of its own, the check no-ops. When discrepancies are
        # found we surface them as a system message the patcher (or a
        # follow-up planner turn) will see — that turns silent drift
        # into a concrete instruction to reconcile.
        inventory_diag_count = 0
        try:
            from harness.architecture_inventory import (
                cross_check_inventories,
                parse_inventory,
            )
            workspace_path = state.get("workspace_path", "")
            arch_path = (
                os.path.join(workspace_path, "docs", "SPEC_ARCHITECTURE.md")
                if workspace_path else ""
            )
            arch_files: list[Any] = []
            if arch_path and os.path.isfile(arch_path):
                with open(arch_path, "r", encoding="utf-8") as _f:
                    arch_md = _f.read()
                arch_files = parse_inventory(arch_md).files
            plan_files = parse_inventory(final_content).files
            if arch_files and plan_files:
                diags = cross_check_inventories(arch_files, plan_files)
                inventory_diag_count = len(diags)
                if diags:
                    blocking = [d for d in diags if d.is_error]
                    advisory = [d for d in diags if not d.is_error]
                    lines: list[str] = [
                        "### Plan / architecture inventory mismatch",
                        "",
                        (
                            "The plan you just produced disagrees with "
                            "docs/SPEC_ARCHITECTURE.md on the file layout. "
                            "Reconcile before patching — adjust the plan's "
                            "file list (or the architecture spec) so they "
                            "match. Diagnostics:"
                        ),
                        "",
                    ]
                    for d in blocking + advisory:
                        lines.append(f"- {d.format_compiler_style()}")
                    messages.append(MessageDict(
                        role="system",
                        content="\n".join(lines),
                    ))
                    logger.warning(
                        "[planning_node] inventory cross-check found %d "
                        "diagnostic(s) (%d blocking, %d advisory).",
                        len(diags), len(blocking), len(advisory),
                    )
        except Exception as exc:  # noqa: BLE001 — cross-check is advisory
            logger.debug("[planning_node] inventory cross-check skipped: %s", exc)

        logger.info(
            "[planning_node] Blueprint generated. tokens_in=%d tokens_out=%d cost=$%.6f budget_left=$%.4f tool_rounds=%d inv_diags=%d",
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost_usd,
            new_budget,
            tool_rounds,
            inventory_diag_count,
        )

        return {
            "messages": messages,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "planning",
                "plan_complete": True,
                "inventory_diag_count": inventory_diag_count,
            },
        }
    except RuntimeError as exc:
        # Budget exhausted or other guardrail triggered
        logger.warning("[planning_node] Gateway refused: %s", exc)
        return {
            "node_state": {"current_node": "planning", "error": str(exc), "budget_exhausted": True},
            "loop_counter": loop_counter,
        }
    except Exception as exc:
        logger.exception("[planning_node] Unexpected error during LLM call.")
        return {
            "node_state": {"current_node": "planning", "error": str(exc)},
            "loop_counter": loop_counter,
        }


async def patching_node(state: AgentState) -> dict[str, Any]:
    """
    Node 2: The Builder.

    Uses the configured fast code model (patching_primary in .harness_config.json)
    via the gateway to parse the planning blueprint and produce precise
    SEARCH/REPLACE blocks that are written to disk by the hybrid patcher engine.

    Output: Updates `modified_files` with paths of files that were changed.
    """
    logger.info("[patching_node] Generating and applying code patches...")

    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["patching"] = loop_counter.get("patching", 0) + 1

    gateway = get_gateway()
    if gateway is None:
        logger.error("[patching_node] No gateway configured. Cannot call LLM.")
        return {
            "node_state": {"current_node": "patching", "error": "No gateway configured"},
            "loop_counter": loop_counter,
        }

    try:
        from harness.gateway import NodeRole
        from harness.patcher import apply_patch_blocks, process_llm_patch_output

        messages = list(state.get("messages", []))
        budget = state.get("budget_remaining_usd", 2.00)
        # B6 — decide once per node call whether to use native tool-use.
        # Gated on (a) the operator's config switch, (b) the patching role's
        # routed model supporting native tools (the gateway will silently
        # drop tools= for models that can't take them, but checking here
        # lets us also skip the text-DSL format reminder when tool-use is
        # in play — tool definitions carry the same semantic content).
        use_tools = bool(getattr(gateway.config, "use_structured_tools", False))

        # Compute the allowlist UP FRONT so we can both (a) surface it to the
        # LLM in the format reminder below and (b) feed the same list into
        # process_llm_patch_output. Without (a), the first patching pass
        # routinely burns 10+ patches on paths the patcher then rejects
        # (e.g. `api/__init__.py`, `config.py` at workspace root when the
        # allowlist is scoped to `task_dispatcher/`). Telling the LLM the
        # allowed roots BEFORE it writes any patches turns those rejections
        # into right-first-time placements.
        workspace = state.get("workspace_path", os.getcwd())
        allowed_paths = _build_patcher_allowlist(workspace)
        # Two-phase generation: this node is PHASE 1 — production code
        # only. A separate test_generation_node fires later (after the
        # prod-import smoke check verifies production imports cleanly)
        # and writes tests against the now-known-good prod modules. The
        # split is a hard-won lesson from sessions 19b28eff, 0a5c6fe8,
        # etc. — when the LLM emits prod + tests in one shot, test code
        # contains stale assumptions about prod signatures and the
        # repair loop wastes iterations triaging "is this a prod bug
        # cascading through tests, or a test bug?". Phase 1's narrower
        # scope eliminates that ambiguity at the source.
        _PHASE1_PRODUCTION_ONLY_NOTE = (
            "[PHASE 1: PRODUCTION CODE ONLY]\n"
            "This is the PRODUCTION-CODE generation phase. Do NOT emit "
            "any test files or test-only fixtures in this pass. A "
            "dedicated test-generation node runs AFTER your production "
            "code has been verified to import cleanly — it will write "
            "all `tests/`, `test/`, `__tests__/`, `conftest.py`, and "
            "`pytest.ini` files against your finalised prod APIs.\n"
            "Rules for THIS turn:\n"
            "  - Do NOT CREATE_FILE under `tests/`, `test/`, "
            "    `__tests__/`. Patches targeting those paths will be "
            "    DROPPED by the harness before they reach the patcher.\n"
            "  - Do NOT CREATE_FILE `conftest.py` or `pytest.ini` at "
            "    workspace root.\n"
            "  - DO write a real production implementation: handlers, "
            "    services, models, config, db, main entry point, "
            "    requirements.txt, etc.\n"
            "  - DO write a Makefile / pyproject.toml as needed for "
            "    the production build to succeed.\n"
            "Save your test ideas for the dedicated test-generation "
            "phase that runs next.\n\n"
        )
        if allowed_paths:
            allowlist_preamble = (
                _PHASE1_PRODUCTION_ONLY_NOTE
                + "[ALLOWED PATHS]\n"
                "Every file path in your CREATE_FILE / REPLACE_BLOCK / "
                "DELETE_BLOCK / INSERT_AT_BLOCK blocks must start with one "
                "of these prefixes (or match one of these exact files). "
                "Paths outside this list will be REJECTED by the patcher "
                "and NOT land on disk. Place new modules under the source "
                "root prefix (the first directory entry below); test paths "
                "shown below are reserved for the next phase and must NOT "
                "receive CREATE_FILE blocks in this turn.\n"
                + "\n".join(f"- {p}" for p in allowed_paths)
                + "\n\n"
            )
        else:
            allowlist_preamble = (
                _PHASE1_PRODUCTION_ONLY_NOTE
                + "[ALLOWED PATHS]\n"
                "No layout constraint this round; place files under whatever "
                "package directory matches your project structure. The "
                "patcher will still block path traversal (../ or absolute "
                "paths). Production code only — tests are deferred to the "
                "next phase.\n\n"
            )

        # Change-request mode: prepend a CR-N attribution block so the
        # patching LLM tags each modified function / class / new file with
        # a single language-appropriate `CR-N` comment. No-op (empty
        # string) when change_request_mode is False — greenfield runs see
        # the format reminder unchanged.
        cr_preamble = _build_change_request_preamble(state, "patching")

        # Inject a format reminder to ensure the LLM outputs patch blocks
        _FORMAT_REMINDER = allowlist_preamble + cr_preamble + """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the patch block syntax below. Do NOT include any explanations,
markdown code fences, or text outside the blocks. Your entire response must be parseable
as one or more patch blocks.

Valid blocks:
<<<CREATE_FILE>>>
file: path/to/file.ext
content:
<complete file contents>
<<<END_CREATE_FILE>>>

<<<REPLACE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to find>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>

<<<DELETE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to delete>
<<<END_DELETE_BLOCK>>>

<<<INSERT_AT_BLOCK>>>
file: path/to/file.ext
anchor: <function or class name>
placement: before|after
content:
<lines to insert>
<<<END_INSERT_AT_BLOCK>>>

Quality: Write modular, production-ready code with proper error handling, type hints, and docstrings. Handle edge cases.
Generate your patches NOW. Only the blocks above. No other text."""
        _TOOL_USE_REMINDER = (
            allowlist_preamble
            + cr_preamble
            + "[PATCHING — NATIVE TOOL-USE MODE]\n"
            "The harness has exposed the patch operations as native tools "
            "(`read_file`, `edit_file`, `create_file`, `delete_block`, "
            "`insert_at_block`). Call them directly via the provider's "
            "tool-use API instead of emitting the text DSL. Lead with "
            "`read_file` for any file you intend to edit but have not yet "
            "been shown this turn — the harness resolves it inline and "
            "re-dispatches you in the same iteration with the bytes. "
            "Production code only this turn; test files are handled by "
            "a later phase.\n"
            "Quality: modular, production-ready, with proper error "
            "handling, type hints, and docstrings.\n"
        )
        messages.append({
            "role": "user",
            "content": _TOOL_USE_REMINDER if use_tools else _FORMAT_REMINDER,
        })

        from harness.tool_schemas import PATCH_TOOLS, tool_calls_to_patch_blocks

        response, new_budget, messages, tool_files_seen = await _patching_tool_loop(
            gateway=gateway,
            messages=messages,
            budget=budget,
            workspace=workspace,
            use_tools=use_tools,
            tools=PATCH_TOOLS,
            state=state,
        )

        # Continuation on finish_reason=="length": the patching role's
        # output cap is enough for a few hundred lines of patches, but
        # a full-stack blueprint that wants both backend AND frontend
        # modules routinely runs past it. Without continuation the
        # harness accepts the truncated DSL, moves into repair, and
        # never re-introduces the missing files — session
        # web-6d5ef9b18f6a is the canonical example (backend only, no
        # React tree). Patching's default is ``True`` (see
        # _CONTINUE_ON_LENGTH_DEFAULTS); operators can override per role
        # via llm_dispatch.continue_on_length in config.json. The shared
        # helper handles the cycle accounting + cap warning and
        # already skips when the response carries tool_calls (tool-use
        # mode multi-turns inside _patching_tool_loop).

        async def _patching_dispatch(msgs, budget_remaining):
            nonlocal messages, tool_files_seen
            resp, new_b, msgs_out, files_out = await _patching_tool_loop(
                gateway=gateway,
                messages=msgs,
                budget=budget_remaining,
                workspace=workspace,
                use_tools=use_tools,
                tools=PATCH_TOOLS,
                state=state,
            )
            messages = msgs_out
            tool_files_seen = files_out
            return resp, new_b

        response, new_budget, accumulated_text_chunks = await _continue_on_length(
            initial_response=response,
            initial_budget=new_budget,
            messages=messages,
            dispatch=_patching_dispatch,
            continue_prompt=(
                "You hit the output token cap mid-patch. Continue with "
                "ADDITIONAL CREATE_FILE / REPLACE_BLOCK / DELETE_BLOCK / "
                "INSERT_AT_BLOCK blocks for the remaining files in the "
                "architecture inventory (e.g. the frontend / client tier "
                "when the backend was emitted first). Do NOT repeat "
                "blocks you've already emitted; emit only the missing "
                "ones. Same DSL rules as before — block syntax only, "
                "no prose outside the blocks."
            ),
            enabled=_resolve_continue_on_length(state, "patching"),
            role_label="patching_node",
        )

        # Update token tracker (per-stage attribution: patching).
        # Note: aggregate_tokens here only sees the FINAL response's
        # usage — the intermediate continuation dispatches were
        # already individually emitted by gateway.dispatch's
        # per-call event log, so cost accounting at the session
        # level is intact; the token-tracker just reflects the last
        # slice. Refining the aggregator to sum across cycles is a
        # future cleanup if cost-per-role attribution drifts.
        token_tracker = state.get("token_tracker", {})
        token_tracker = gateway.aggregate_tokens(
            token_tracker, response.usage, role=NodeRole.PATCHING,
        )

        existing_modified = list(state.get("modified_files", []))

        # Pre-join the text-DSL chunks once; the text-DSL branch
        # below feeds this concatenation through patch parsing so
        # continuation patches land on disk alongside the initial
        # round's blocks. Tool-use mode ignores it (tool calls were
        # already executed inside the loop).
        combined_response_text = "\n".join(accumulated_text_chunks)

        _resp_tool_calls = getattr(response, "tool_calls", None) or []
        if _resp_tool_calls:
            # Native tool-use path. Convert the LLM's tool calls into
            # PatchBlock objects and apply via the structured pipeline.
            # The read_file calls have already been resolved inside
            # _patching_tool_loop — any tool_calls returned here are
            # patch operations only.
            blocks, _reads = tool_calls_to_patch_blocks(_resp_tool_calls)
            # Phase 1 filter: drop any blocks targeting test paths even
            # in tool-use mode. The LLM gets the same "production only"
            # instruction; this is a belt-and-suspenders guard against
            # an over-eager call.
            kept_blocks, dropped_test_files = (
                _filter_test_patch_blocks(blocks)
            )
            if dropped_test_files:
                logger.info(
                    "[patching_node:phase1] Dropped %d test-targeting "
                    "tool_call(s): %s",
                    len(dropped_test_files),
                    ", ".join(dropped_test_files[:10]),
                )
            patch_results, modified_files = await apply_patch_blocks(
                kept_blocks,
                workspace,
                existing_modified,
                allowed_paths=allowed_paths,
                files_seen_by_llm=tool_files_seen,
                enforce_read_before_edit=bool(
                    getattr(
                        gateway.config, "enforce_read_before_edit", True,
                    )
                ),
            )
            filtered_response = response.content or ""
            dropped_test_blocks = dropped_test_files
        else:
            # Text-DSL path (legacy / fallback when provider didn't
            # support native tools or returned a pure-text response).
            # combined_response_text is the concatenation of the
            # initial response and any continuation cycles above —
            # without it the patcher would only ever see the first
            # truncated slice when finish_reason was "length".
            filtered_response, dropped_test_blocks = (
                _filter_test_blocks_from_patch_response(combined_response_text)
            )
            if dropped_test_blocks:
                logger.info(
                    "[patching_node:phase1] Dropped %d test-targeting block(s) "
                    "from this round — test generation happens in the next "
                    "phase. Dropped: %s",
                    len(dropped_test_blocks),
                    ", ".join(dropped_test_blocks[:10]),
                )
            # Apply patches to disk using the same allowlist we surfaced to the
            # LLM above — keeps the LLM's expectation and the patcher's enforcement
            # in lockstep.
            patch_results, modified_files = await process_llm_patch_output(
                filtered_response,
                workspace,
                existing_modified,
                allowed_paths=allowed_paths,
            )

        # Append the LLM response to messages (the original, unfiltered,
        # so the LLM's own history reflects what it actually emitted).
        # In tool-use mode the assistant turn(s) were already appended
        # inside _patching_tool_loop; only the text-DSL path needs to
        # add a synthetic assistant message here. We append only the
        # FINAL chunk — earlier chunks were appended inside the
        # continuation loop as it advanced, so re-appending them now
        # would double them in the conversation history.
        if not _resp_tool_calls:
            messages.append(MessageDict(role="assistant", content=response.content))

        # Report patch application results
        success_count = sum(1 for r in patch_results if r.success)
        fail_count = len(patch_results) - success_count
        # Carve out allowlist rejections from generic failures so the next
        # repair iteration sees the exact paths and reason — without this,
        # the LLM keeps re-proposing the same blocked paths.
        allowlist_rejections = [
            {"file": r.file, "operation": r.operation, "reason": r.error}
            for r in patch_results
            if not r.success and isinstance(r.error, str)
            and "not in skill allowlist" in r.error
        ]
        # Capture the remaining (non-allowlist) failures so repair_node can
        # tell the LLM *why* its patches didn't land. Without this the LLM
        # only sees "Failed: foo.txt" and re-proposes the same bad search
        # block on the next round.
        patch_failures = [
            {
                "file": r.file,
                "operation": (
                    r.operation.value
                    if hasattr(r.operation, "value") else str(r.operation)
                ),
                "error": _store_patch_failure_error(r.error),
            }
            for r in patch_results
            if not r.success and isinstance(r.error, str)
            and "not in skill allowlist" not in r.error
        ][:5]
        if success_count > 0:
            status_msg = f"[System]: Applied {success_count}/{len(patch_results)} patches successfully."
            if fail_count > 0:
                failed_files = [r.file for r in patch_results if not r.success]
                status_msg += f" Failed on: {', '.join(failed_files)}."
        else:
            status_msg = f"[System]: Failed to apply {fail_count} patch(es)."
        if allowlist_rejections:
            rejected_paths = ", ".join(sorted({str(r["file"]) for r in allowlist_rejections}))
            status_msg += (
                f"\n[Allowlist] Rejected paths outside the configured layout: "
                f"{rejected_paths}. Allowed roots: {allowed_paths}."
            )
        messages.append(MessageDict(role="system", content=status_msg))

        # Audit §6.4: surface a cross-file impact warning so the LLM
        # sees downstream files that reference modified symbols and
        # can update them in the same turn. The module ships an
        # ImpactAnalyzer; previously nothing called it. Best-effort:
        # any analyzer crash is swallowed so a malformed dependency
        # graph never blocks a successful patch.
        if modified_files:
            try:
                from harness.impact import ImpactAnalyzer
                _impact_raw = state.get("impact_config") if isinstance(state, dict) else None
                impact_cfg: dict[str, Any] = _impact_raw if isinstance(_impact_raw, dict) else {}
                _impact = ImpactAnalyzer(
                    workspace_path=workspace,
                    max_scan_files=int(impact_cfg.get("max_scan_files", 500)),
                    enabled=bool(impact_cfg.get("enabled", True)),
                )
                _impact.analyze_and_warn(list(modified_files), messages)  # type: ignore[arg-type]
            except Exception as _impact_exc:  # noqa: BLE001
                logger.debug("[impact] analysis failed (%s); skipping warning.", _impact_exc)

            # Audit §6.5: invalidate / refresh the repo index for the
            # files we just changed so the planner's "top relevant
            # chunks" injection doesn't ship pre-patch content on
            # subsequent dispatches. Best-effort: index churn must
            # never fail the patching node.
            try:
                from harness.repo_index import (
                    update_index_for_files, RepoIndexConfig,
                )
                idx_cfg_dict = (state.get("repo_index_config") or {}) if isinstance(state, dict) else {}
                if idx_cfg_dict:
                    _idx_cfg = RepoIndexConfig(**{
                        k: v for k, v in idx_cfg_dict.items()
                        if k in RepoIndexConfig.__dataclass_fields__
                    })
                else:
                    _idx_cfg = RepoIndexConfig()
                update_index_for_files(workspace, list(modified_files), _idx_cfg)
            except Exception as _idx_exc:  # noqa: BLE001
                logger.debug("[repo_index] incremental update failed: %s", _idx_exc)

        logger.info(
            "[patching_node] Patches applied. tokens_in=%d tokens_out=%d cost=$%.6f budget_left=$%.4f "
            "patches=%d succeed=%d fail=%d",
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost_usd,
            new_budget,
            len(patch_results), success_count, fail_count,
        )

        return {
            "messages": messages,
            "modified_files": modified_files,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "patching",
                "patch_complete": True,
                "patch_success": success_count,
                "patch_fail": fail_count,
                "allowlist_rejections": allowlist_rejections,
                "patch_failures": patch_failures,
                "allowed_paths": allowed_paths,
            },
        }
    except RuntimeError as exc:
        logger.warning("[patching_node] Gateway refused: %s", exc)
        return {
            "node_state": {"current_node": "patching", "error": str(exc), "budget_exhausted": True},
            "loop_counter": loop_counter,
        }
    except Exception as exc:
        logger.exception("[patching_node] Unexpected error during patching.")
        return {
            "node_state": {"current_node": "patching", "error": str(exc)},
            "loop_counter": loop_counter,
        }


def _toolchain_image_for(build_command: str) -> Optional[str]:
    """Return the harness builder image for any build command.

    The harness now ships a single kitchen-sink image
    (``harness/vendor/Dockerfile.builder``) that bakes Python + pip,
    Java JDK + Maven + Gradle, Node + npm/yarn/pnpm, SQLite, Playwright +
    Chromium, and the make/gcc/git glue all into one container. So there
    is no longer any per-command image dispatch — every supported stack's
    toolchain is already present.

    Kept as a function (returning the constant) so callers that branch
    on ``desired_image`` to set ``sandbox_config["docker_image"]`` when
    it's unset still work without changes.
    """
    del build_command  # No longer dispatches on command shape.
    return BUILDER_IMAGE


def _command_is_make(build_command: str) -> bool:
    """True when ``build_command`` invokes ``make`` (``make``, ``make build``,
    ``make test``, ...). Prefix-match on the stripped command, NOT substring
    search, so we don't false-match ``cmake build``, ``make-something``,
    ``makefile``, or ``echo make build``. Mirrors the safer form used by
    :func:`_toolchain_image_for`.
    """
    stripped = build_command.strip().lower()
    return stripped == "make" or stripped.startswith("make ")


def _build_command_needs_network(build_command: str) -> bool:
    """True when the build command performs a package install that needs
    to reach a registry (pip/npm/yarn/pnpm/cargo/go) — OR invokes ``make``.

    The ``make`` clause exists because the LLM-generated ``Makefile`` (per
    ``harness/skills/makefile_python.md``) conventionally puts
    ``pip install -r requirements.txt`` (or ``npm install``, etc.) INSIDE
    the target's recipe lines. The command string the harness invokes is
    just ``"make build"`` — the ``pip install`` substring lives in the
    Makefile, not in the outer command, so the install-token heuristic
    can't see it. Parsing recipe text is the alternative, but it's brittle
    against sub-makes, includes, and ``$(MAKE) -C subdir`` patterns. The
    cleaner rule: any ``make <target>`` is treated as install-needing.
    """
    cmd = build_command.lower()
    if _command_is_make(build_command):
        return True
    return any(token in cmd for token in (
        "pip install", "pip3 install", "npm install", "yarn install",
        "pnpm install", "cargo build", "cargo test", "go mod",
        "go get", "poetry install",
    ))


# Patterns that mean the build container is missing a required runtime
# (interpreter, test framework, package manager) — NOT a code bug. When
# one of these fires we route to HITL immediately instead of burning the
# whole 3-iteration repair budget on a problem the LLM cannot fix from
# inside the sandbox.
#
# Split by source kind because the repairability heuristic differs:
#   - Python ModuleNotFoundError → ALWAYS repairable. The fix is "add the
#     missing distribution to requirements.txt / pyproject deps", which
#     the autofix / repair LLM can handle for any single-segment module
#     name (httpx, fastapi, pydantic, sqlalchemy — every regular pip
#     package). The earlier whitelist-only policy meant any framework
#     dep outside the test/lint tool set short-circuited to HITL.
#   - Shell `command not found` → repairable only when the command is a
#     known pip-installable Python tool (pytest, ruff, mypy, ...). For
#     anything else (npm, cargo, go, docker) the container needs a
#     different base image, which only the operator can change.
_PYTHON_MODULE_MISS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Python: "/usr/local/bin/python3: No module named pytest"
    # Dotted names (e.g. 'api.database') are excluded — those signal a
    # local-import bug in the user's code, which the repair loop can fix.
    re.compile(r"(?m)^/[^:\s]+/python3?: No module named (?P<sym>[^.\s]+)\s*$"),
    # Python: "ModuleNotFoundError: No module named 'pytest'"
    # Same dotted-name exclusion as above.
    re.compile(r"ModuleNotFoundError: No module named ['\"](?P<sym>[^'\".]+)['\"]"),
)

_SHELL_COMMAND_MISS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Shell-style "<cmd>: command not found" (covers npm, cargo, go, etc.)
    re.compile(r"(?m)^(?:/bin/sh: \d+: )?(?P<sym>[\w.\-+]+): command not found\s*$"),
    re.compile(r"(?m)^(?:bash: )?(?P<sym>[\w.\-+]+): command not found\s*$"),
    # Dash / busybox shells say "X: not found" (no "command"). The
    # prefix may be "/bin/sh: 1: " (most debian-based) or "sh: 1: " (some
    # slim images that don't symlink /bin/sh into PATH the same way).
    # Session 51ecb569 hit the latter form with `sh: 1: make: not found`
    # in python:3.12-slim and the original /bin/-only pattern missed it,
    # so compiler_node never tagged env_misconfig and the repair loop
    # burned 5 LLM iterations on a problem no patch could fix.
    re.compile(r"(?m)^(?:/bin/)?sh: \d+: (?P<sym>\S+): not found\s*$"),
    # Docker entrypoint missing (exec format / OCI runtime error)
    re.compile(r"executable file not found in \$?PATH: (?P<sym>\S+)"),
    re.compile(r'exec: "(?P<sym>[^"]+)": executable file not found'),
    # Node: "node: not found" or "npm: not found"
    re.compile(r"(?m)^(?P<sym>node|npm|yarn|pnpm): not found\s*$"),
)

# Composite — preserved for callers that don't care about the source kind.
_ENV_MISCONFIG_PATTERNS: tuple[re.Pattern[str], ...] = (
    *_PYTHON_MODULE_MISS_PATTERNS,
    *_SHELL_COMMAND_MISS_PATTERNS,
)


def _is_env_misconfig(
    raw_output: str,
    workspace_path: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Return ``(symbol, kind)`` where ``kind`` is ``"python"`` for a
    Python ``ModuleNotFoundError`` or ``"shell"`` for a shell
    ``command not found`` — or ``None`` if the build output doesn't
    look like an environment miss.

    The caller uses ``kind`` to decide repairability:
      - ``"python"`` → always repairable as MISSING_DEP. Any single-segment
        Python module name is a pip-installable distribution; autofix R4
        (``_try_missing_dep``) lands the requirements.txt edit without an
        LLM call. The earlier whitelist-only policy meant common app
        deps like ``httpx`` / ``fastapi`` / ``pydantic`` short-circuited
        to HITL for no good reason — they're as fixable as ``pytest``.
      - ``"shell"`` → repairable only when the symbol matches
        ``_PIP_INSTALLABLE_SYMBOLS`` (pytest, ruff, ...). Everything else
        (npm, cargo, go, docker) needs a different base image and the
        repair LLM can't fix it from inside the sandbox.

    When ``workspace_path`` is supplied and the matched symbol corresponds
    to a directory or top-level Python module in the workspace (e.g.
    ``task_dispatcher`` is the project's own package under ``src/``), we
    return ``None`` — the import error is a layout / sys.path bug in the
    user's own code, which the repair LLM can fix differently
    (conftest.py, pyproject src-layout, etc.).
    """
    if not raw_output:
        return None
    # Scan only the tail — these errors land at the end of the log when
    # the missing-binary line is the last thing the container prints.
    tail = raw_output[-4000:]
    for pattern in _PYTHON_MODULE_MISS_PATTERNS:
        match = pattern.search(tail)
        if match:
            sym = match.group("sym").strip("'\"")
            if workspace_path and _symbol_exists_in_workspace(sym, workspace_path):
                logger.info(
                    "[env_misconfig] '%s' looks like a missing module but a "
                    "directory or file by that name exists in %s — treating "
                    "as a user-code import bug, not env misconfig.",
                    sym, workspace_path,
                )
                return None
            return (sym, "python")
    for pattern in _SHELL_COMMAND_MISS_PATTERNS:
        match = pattern.search(tail)
        if match:
            sym = match.group("sym").strip("'\"")
            if workspace_path and _symbol_exists_in_workspace(sym, workspace_path):
                logger.info(
                    "[env_misconfig] '%s' looks like a missing command but a "
                    "directory or file by that name exists in %s — treating "
                    "as a user-code import bug, not env misconfig.",
                    sym, workspace_path,
                )
                return None
            return (sym, "shell")
    return None


def _symbol_exists_in_workspace(symbol: str, workspace_path: str) -> bool:
    """True when ``symbol`` matches a directory or Python module file in the
    workspace tree (anywhere under root, excluding never-source dirs).

    Used by :func:`_is_env_misconfig` to distinguish a missing-system-binary
    failure (``pytest`` not installed → genuine env misconfig) from a
    missing-own-package failure (``task_dispatcher`` not on sys.path → fixable
    user-code bug).
    """
    if not symbol or not workspace_path or not os.path.isdir(workspace_path):
        return False
    from harness.impact import _NEVER_SOURCE_DIRS
    sym = symbol.strip().strip("'\"")
    if not sym or "/" in sym or os.sep in sym:
        return False
    try:
        for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
            ]
            if sym in sub_dirs:
                return True
            if f"{sym}.py" in sub_files:
                return True
    except OSError:
        return False
    return False


# pytest's exit code 5 means "no tests collected" — distinct from a genuine
# compile/test failure. Confirm via the literal "no tests ran" line so we
# don't misclassify a config error that happens to produce exit 5.
_NO_TESTS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?m)^=+\s*no tests ran in [\d.]+s\s*=*$"),
    re.compile(r"(?m)^no tests ran in [\d.]+s\s*$"),
)


# pip's resolver emits one of these lines when version pins in the build's
# requirements file (or its transitive deps) can't be satisfied together.
# The error message rarely names BOTH sides of the conflict, so the repair
# LLM is forced to guess — autofix strips the pins entirely instead.
# pip's resolver emits one of these lines when version pins in the build's
# requirements file (or its transitive deps) can't be satisfied together.
# The error message rarely names BOTH sides of the conflict, so the repair
# LLM is forced to guess — autofix strips the pins entirely instead.
#
# Audit §6.10: bounded ``[^\n]{1,500}`` instead of ``.+`` to avoid
# catastrophic backtracking on pathological single-line pip logs.
_PIP_RESOLUTION_CONFLICT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?m)^ERROR: ResolutionImpossible\b"),
    re.compile(
        r"(?m)^ERROR: Cannot install [^\n]{1,500} because these package versions have "
        r"conflicting dependencies"
    ),
    re.compile(
        r"(?m)^ERROR: pip's dependency resolver does not currently take into "
        r"account all the packages that are installed"
    ),
)


def _is_pip_resolution_conflict(raw_output: str, build_command: str) -> bool:
    """True when the build failed because pip's resolver couldn't satisfy
    the requested version pins together.

    Used by compiler_node to emit a distinct ``DEP_RESOLUTION_CONFLICT``
    diagnostic. Autofix R5 then drops every version specifier from
    ``requirements.txt`` — for greenfield runs where every pin is a
    fresh LLM guess, that resolves the conflict 95% of the time without
    spending a repair iteration.
    """
    if not raw_output:
        return False
    if "pip" not in build_command.lower():
        return False
    tail = raw_output[-4000:]
    return any(p.search(tail) for p in _PIP_RESOLUTION_CONFLICT_PATTERNS)


def _is_no_tests_collected(exit_code: int, raw_output: str, build_command: str) -> bool:
    """True when the build's failure is pytest's exit-5 'no tests collected'
    rather than an actual test/compile failure.

    Detected by matching the literal 'no tests ran' line in the output AND
    requiring the build to actually exercise pytest. Treating this as a
    generic build failure burns repair iterations on a problem the repair
    LLM cannot fix (there's nothing to repair — the test runner found
    nothing). The router uses this to route to test_generation_node or
    HITL with a precise message instead.
    """
    if exit_code != 5 or not raw_output:
        return False
    if "pytest" not in build_command.lower():
        return False
    tail = raw_output[-4000:]
    return any(p.search(tail) for p in _NO_TESTS_PATTERNS)


# Lines matching these patterns are stripped from build output before the
# repair LLM sees them. They're informational warnings the toolchain emits
# regardless of whether the build succeeded, and feeding them to the LLM
# wastes context AND tempts the model into "fixing" warnings that don't
# actually block the build. Real errors (build failures, assertion fails,
# stack traces) are unaffected — those don't match.
_BUILD_OUTPUT_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Python deprecation / future / pending warnings emitted as a single
    # warning summary line OR the multi-line stack-style emission.
    re.compile(r"^.*\b(?:Deprecation|PendingDeprecation|Future|ResourceWarning)Warning\b.*$"),
    # pytest's terminal "Warnings summary" header — followed by per-test
    # warning blocks. Drop the header so the model doesn't infer a warnings
    # section was important.
    re.compile(r"^=+\s*warnings summary\s*=+\s*$", re.IGNORECASE),
    # pip's chatty noise: "[notice] A new release of pip is available…",
    # the root-user warning, build wheel progress bars, and so on.
    re.compile(r"^\s*\[notice\]\s+.*$"),
    re.compile(r"^WARNING: Running pip as the 'root' user.*$"),
    re.compile(r"^WARNING: .*?Skipping.*?already satisfied.*$"),
    # Setuptools / pkg_resources transition warnings that have spammed
    # every Python build for two years and aren't actionable.
    re.compile(r"^.*\bpkg_resources is deprecated\b.*$"),
    re.compile(r"^.*\bSetuptoolsDeprecationWarning\b.*$"),
)


def _strip_build_output_noise(raw_output: str) -> str:
    """Drop deprecation warnings and other non-actionable noise from
    ``raw_output`` before it reaches the repair LLM.

    Real errors are untouched (none of the patterns match). The filter
    runs line-by-line so a stack trace mixed with a deprecation line
    only loses the deprecation line itself, not the surrounding context.
    """
    if not raw_output:
        return raw_output
    kept: list[str] = []
    # Audit §6.9: a *Warning header (e.g. "foo.py:42: DeprecationWarning:
    # use X instead") is typically followed by the offending source line
    # and an empty separator. Drop the header AND the indented follow-up
    # lines until the first blank/separator so the LLM sees neither the
    # warning nor its stack-style continuation.
    skip_indented_until_blank = False
    for line in raw_output.splitlines():
        if skip_indented_until_blank:
            stripped = line.strip()
            if not stripped:
                skip_indented_until_blank = False
                continue
            # Continuation lines are typically indented OR start with
            # an identifier-like token (the source-line snippet pytest /
            # python prints). Drop indented lines; stop on non-indented
            # non-blank (a new diagnostic).
            if line[:1] in (" ", "\t"):
                continue
            skip_indented_until_blank = False
        if any(p.search(line) for p in _BUILD_OUTPUT_NOISE_PATTERNS):
            # If this looked like a Warning header line, also skip the
            # following indented lines (the source-snippet that follows
            # the ``...py:NN: SomeWarning: message`` header).
            if "Warning" in line:
                skip_indented_until_blank = True
            continue
        kept.append(line)
    # Preserve trailing-newline shape so the slicer's char-budget math
    # downstream isn't off by one for files that ended in \n.
    out = "\n".join(kept)
    if raw_output.endswith("\n"):
        out += "\n"
    return out


def _slice_build_output_for_repair(
    raw_output: str,
    head_chars: int = 1500,
    tail_chars: int = 2000,
) -> str:
    """Return a head+tail slice of ``raw_output`` for the repair LLM.

    Long build logs (C++ template explosions, Java dependency stacks, Cargo
    compilation walls) tend to put the root-cause error near the START and
    cascading downstream errors near the END. A pure tail slice (the old
    behaviour) shows the repair LLM only the cascade, hiding the underlying
    cause. Slicing both ends gives the model both the original failure and
    the final state, separated by an explicit truncation marker so it knows
    chars were dropped.

    Also strips deprecation / pip-notice noise (see
    :func:`_strip_build_output_noise`) so the LLM doesn't waste a repair
    iteration trying to "fix" a DeprecationWarning that doesn't block the
    build.

    For outputs shorter than ``head_chars + tail_chars + 200`` we return
    the whole thing unchanged — the split adds noise without saving space.
    """
    if not raw_output:
        return ""
    cleaned = _strip_build_output_noise(raw_output)
    total = len(cleaned)
    if total <= head_chars + tail_chars + 200:
        return cleaned
    head = cleaned[:head_chars]
    tail = cleaned[-tail_chars:]
    dropped = total - head_chars - tail_chars
    return (
        f"{head}\n"
        f"... [truncated {dropped} chars from the middle of {total}-char build log] ...\n"
        f"{tail}"
    )


_DEP_MANIFEST_CANDIDATES: tuple[str, ...] = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements/base.txt",
    "requirements/dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Pipfile",
)


def _collect_manifest_snippets_for_repair(
    errors: list[Any], workspace_path: str,
) -> str:
    """Build a Markdown block embedding the current contents of any dep
    manifest the build command references.

    Fires only when at least one diagnostic is shaped as a MISSING_DEP.
    Without the actual file contents in the repair prompt, the LLM resorts
    to ``CREATE_FILE`` (fails when the file exists) or ``REPLACE_BLOCK``
    with a search string it guessed (fails when the guess doesn't match).
    Returns the empty string when no MISSING_DEP diagnostic is present or
    no manifest is found, so callers can string-append unconditionally.
    """
    if not errors or not workspace_path:
        return ""
    missing_deps = [
        e for e in errors
        if str(e.get("error_code", "")) in ("MISSING_DEP", "DEP_RESOLUTION_CONFLICT")
    ]
    if not missing_deps:
        return ""
    if not os.path.isdir(workspace_path):
        return ""

    # Pick the manifests the build command actually references when we can
    # tell — otherwise fall back to whatever exists at workspace root.
    build_cmd = str(missing_deps[0].get("build_command", "") or "").lower()
    preferred: list[str] = []
    if "pyproject" in build_cmd or "pip install -e" in build_cmd:
        preferred = ["pyproject.toml", "setup.py", "setup.cfg"]
    elif "requirements" in build_cmd:
        preferred = ["requirements.txt", "requirements-dev.txt"]
    elif "npm install" in build_cmd or "yarn" in build_cmd or "pnpm" in build_cmd:
        preferred = ["package.json"]
    candidates = preferred or list(_DEP_MANIFEST_CANDIDATES)

    found: list[tuple[str, str]] = []
    for rel in candidates:
        abs_path = os.path.join(workspace_path, rel)
        if not os.path.isfile(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                # Cap at 8KB per manifest — anything bigger is almost
                # certainly the wrong file.
                content = f.read(8192)
        except OSError:
            continue
        found.append((rel, content))
        if len(found) >= 3:
            break

    if not found:
        return ""
    missing = sorted({
        str(e.get("missing_symbol", "") or "")
        for e in missing_deps if e.get("missing_symbol")
    })
    lines = [
        "\n## Dependency Manifests (current workspace contents)",
        (
            f"The build is missing {missing!r}. Use REPLACE_BLOCK / "
            f"INSERT_AT_BLOCK against the exact bytes below — do NOT "
            f"emit CREATE_FILE for a path that already exists, and do "
            f"NOT invent search strings that aren't in the file."
        ),
    ]
    for rel, content in found:
        lang = "toml" if rel.endswith(".toml") else (
            "json" if rel.endswith(".json") else "text"
        )
        lines.append(f"\n### `{rel}`\n```{lang}\n{content.rstrip()}\n```")
    return "\n".join(lines) + "\n"


# Marker that the patcher emits when a REPLACE_BLOCK search misses and we
# attach the line-numbered window of current file content around the
# closest match (see _find_closest_match in harness/patcher.py). When an
# error contains this marker we MUST NOT truncate it on the way into
# node_state["patch_failures"] — the wider window can be up to ~2000
# chars and is the single most useful signal the LLM has for correcting
# its next search block. Truncating slices the window mid-line and
# defeats the purpose, which is what hit session 2f2d48cc-...
_PATCH_ERROR_WIDER_CONTEXT_MARKER = "Current file content (around closest match):"


def _store_patch_failure_error(error_text: str) -> str:
    """Prepare a patcher error message for storage in node_state.

    When the error includes the wider-context window (marker above), return
    the full text. Otherwise cap at 3000 chars — generous enough for any
    "regular" error message but bounded so a runaway log line can't blow
    the state. The previous 800-char cap was tight enough to slice the
    wider-context window mid-line in iteration N+1's repair prompt.
    """
    err = error_text or ""
    if _PATCH_ERROR_WIDER_CONTEXT_MARKER in err:
        return err
    return err[:3000]


_TEST_DIR_PREFIXES = ("tests/", "test/", "__tests__/")
# Workspace-root files that are test-infrastructure even though they
# don't live under a test/ prefix. Used by phase-1 patching to drop
# any LLM blocks that target them — test infrastructure is generated
# in phase 2 (test_generation_node) after prod imports cleanly.
_TEST_INFRA_ROOT_FILES = frozenset({"conftest.py", "pytest.ini"})


def _is_test_path(path: str) -> bool:
    """True when ``path`` is somewhere the test-generation phase owns,
    not the phase-1 production-patching phase."""
    if not path:
        return False
    p = path.strip().lstrip("./")
    if any(p.startswith(prefix) for prefix in _TEST_DIR_PREFIXES):
        return True
    return p in _TEST_INFRA_ROOT_FILES


def _filter_test_patch_blocks(blocks: list[Any]) -> tuple[list[Any], list[str]]:
    """Sister of :func:`_filter_test_blocks_from_patch_response` that
    operates on pre-parsed :class:`PatchBlock` instances.

    Used by the B6 native tool-use path in :func:`patching_node` —
    `tool_calls_to_patch_blocks` returns ``PatchBlock`` objects directly,
    so there is no text body to scan with the regex helper. Returns
    ``(kept, dropped_files)``.
    """
    kept: list[Any] = []
    dropped: list[str] = []
    for block in blocks:
        file_path = getattr(block, "file", "") or ""
        if _is_test_path(file_path):
            op = getattr(block, "operation", "")
            op_name = getattr(op, "value", str(op))
            dropped.append(f"{op_name.lower()}:{file_path}")
            continue
        kept.append(block)
    return kept, dropped


def _filter_test_blocks_from_patch_response(
    response_content: str,
) -> tuple[str, list[str]]:
    """Strip every patch block targeting a test path from the LLM's
    response. Returns ``(filtered_content, dropped_paths)``.

    Phase 1's job is production code; the harness drops any block the
    LLM emits against ``tests/``, ``test/``, ``__tests__/``,
    ``conftest.py``, or ``pytest.ini`` — those are handled by the
    test-generation phase that runs AFTER prod is verified.

    All four block types (CREATE_FILE, REPLACE_BLOCK, DELETE_BLOCK,
    INSERT_AT_BLOCK) are handled. The first line inside each block is
    always ``file: <path>``.
    """
    import re as _re
    block_pattern = _re.compile(
        r"<<<(CREATE_FILE|REPLACE_BLOCK|DELETE_BLOCK|INSERT_AT_BLOCK)>>>"
        r"\s*\nfile:\s*([^\n]+)\n"
        r".*?"
        r"<<<END_\1>>>\n?",
        _re.DOTALL,
    )
    dropped: list[str] = []

    def _maybe_drop(match: "_re.Match[str]") -> str:
        path = match.group(2).strip()
        if _is_test_path(path):
            dropped.append(f"{match.group(1).lower()}:{path}")
            return ""
        return match.group(0)

    filtered = block_pattern.sub(_maybe_drop, response_content)
    return filtered, dropped
# Directories the prod-import smoke check should NOT walk into when
# enumerating production modules. Mix of VCS / cache / build / virtualenv /
# test-tree names.
_SMOKE_CHECK_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "node_modules", "build", "dist", ".venv", "venv", "env",
    "tests", "test", "__tests__",
})


def _walk_prod_python_modules(workspace_path: str) -> list[str]:
    """Walk ``workspace_path`` and return dotted module names for every
    production ``*.py`` file. Skips tests, build dirs, and dotfiles.

    Used by the prod-import smoke check (fix #6 / user's two-phase
    request): produces the import list the sandbox should ``import ...``
    before pytest runs. Empty list when no prod sources exist yet.
    """
    if not os.path.isdir(workspace_path):
        return []
    modules: list[str] = []
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [
            d for d in dirs
            if d not in _SMOKE_CHECK_SKIP_DIRS
            and not d.startswith(".")
            and not d.endswith(".egg-info")
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, workspace_path)
            parts = rel.split(os.sep)
            if parts[-1] == "__init__.py":
                if len(parts) == 1:
                    continue  # root-level __init__.py is degenerate
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]  # strip .py
            if not parts:
                continue
            # Skip standalone setup.py / conftest.py / manage.py — they're
            # scripts, not import targets, and trying to import them
            # often runs side-effecting top-level code.
            if len(parts) == 1 and parts[0] in (
                "setup", "conftest", "manage", "asgi", "wsgi",
            ):
                continue
            modules.append(".".join(parts))
    return sorted(set(modules))


async def _run_prod_import_smoke_check(
    workspace_path: str,
    sandbox_config: dict[str, Any],
    allow_network: bool,
    install_step: str,
    session_id: str,
) -> list[Any]:
    """Run ``python -c 'import a; import b; ...'`` inside the sandbox
    against every production module. Returns a list of diagnostic dicts
    (tagged ``PROD_IMPORT_SMOKE``) for any imports that failed; empty
    list when every prod module imports cleanly.

    Caller is responsible for routing those diagnostics into
    ``compiler_errors``. By surfacing prod-import failures BEFORE running
    the actual build (pytest), the repair LLM sees production-side errors
    in isolation, without the cascade amplification that happens when
    pytest tries to collect tests that import broken prod modules.

    Requires ``install_step`` (e.g. ``python3 -m pip install -r
    requirements.txt``) so the imports have dependencies available.
    """
    modules = _walk_prod_python_modules(workspace_path)
    if not modules:
        return []
    # Build a Python script that imports each module under try/except so
    # one failure doesn't abort the rest. Failures are printed in a
    # parseable format the caller can grep.
    py_lines = [
        "import sys",
        f"_mods = {modules!r}",
        "_fails = []",
        "for _m in _mods:",
        "    try:",
        "        __import__(_m)",
        "    except BaseException as _e:",
        "        _fails.append((_m, type(_e).__name__, str(_e)))",
        "if _fails:",
        "    print('=== PROD_IMPORT_SMOKE_FAILURES ===')",
        "    for _m, _et, _msg in _fails:",
        "        print(f'FAIL: {_m}: {_et}: {_msg}')",
        "    sys.exit(1)",
        "print('=== PROD_IMPORT_SMOKE_OK ===')",
    ]
    py_script = "\n".join(py_lines)
    # Use python3 -c with the script. Escape double quotes minimally;
    # we use single-quoted f-strings inside.
    cmd = f"{install_step} && python3 -c \"{py_script}\""

    from harness.sandbox import SandboxExecutor
    executor = SandboxExecutor(
        workspace_path=workspace_path,
        allow_network=allow_network,
        sandbox_config=sandbox_config,
        session_id=session_id,
    )
    logger.info(
        "[prod-smoke] Running prod-import smoke check across %d module(s).",
        len(modules),
    )
    result = await executor.run(cmd)
    if result.exit_code == 0:
        logger.info(
            "[prod-smoke] All %d production module(s) imported cleanly.",
            len(modules),
        )
        return []
    # Parse FAIL: lines from the output. The smoke script prints them
    # one per line right after the FAILURES header.
    diagnostics: list[dict[str, Any]] = []
    seen = False
    for line in (result.raw_output or "").splitlines():
        if "PROD_IMPORT_SMOKE_FAILURES" in line:
            seen = True
            continue
        if not seen:
            continue
        if not line.startswith("FAIL:"):
            continue
        # Parse "FAIL: <module>: <ExceptionType>: <message>"
        body = line[len("FAIL:"):].strip()
        # Split into at most 3 parts
        try:
            module, exc_type, message = [p.strip() for p in body.split(":", 2)]
        except ValueError:
            module, exc_type, message = body, "ImportError", body
        diagnostics.append({
            "error_code": f"PROD_IMPORT_SMOKE:{exc_type}",
            "message": (
                f"Production module `{module}` failed to import "
                f"({exc_type}): {message}"
            ),
            "file": module.replace(".", "/") + ".py",
            "line": 0,
            "column": 0,
            "severity": "error",
            "semantic_context": "",
        })
    if not diagnostics:
        # Fall back to a single coarse diagnostic carrying the tail of
        # the output so the LLM has something to work with.
        diagnostics.append({
            "error_code": "PROD_IMPORT_SMOKE",
            "message": (result.raw_output or "")[-1500:],
            "file": "<prod-import-smoke>",
            "line": 0,
            "column": 0,
            "severity": "error",
            "semantic_context": "",
        })
    logger.warning(
        "[prod-smoke] %d production module(s) failed to import. Surfacing "
        "as diagnostics; the actual build (pytest) will NOT run this "
        "round — repair must fix prod imports first.",
        len(diagnostics),
    )
    return diagnostics
# Error codes that almost certainly mean "test fails to import / collect
# because something in production is wrong." These point the cascade at
# production code, not the test file itself.
_PROD_CASCADE_ERROR_CODES = (
    "ImportError", "ModuleNotFoundError", "NameError", "AttributeError",
    "SyntaxError", "TypeError",
    "F821", "F401", "E0401", "E0602", "E0001",
    "TEST_FAILURE:IMPORTERROR", "TEST_FAILURE:MODULENOTFOUND",
)


def _corresponding_prod_paths_for_test(
    test_path: str, workspace_path: str,
) -> list[str]:
    """Best-effort mapping from a test-file path to candidate production
    file paths in the workspace.

    Heuristic: strip ``test_`` from the basename, drop the leading
    ``tests/`` (or ``test/`` / ``__tests__/``), then look for matches in
    common source layouts (``<stem>.py`` at root, ``<src>/<stem>.py``,
    ``<package>/<stem>.py``). Returns paths that actually exist on disk.

    Used by the test-cascade reframe (fix #5) to attach the relevant
    production module's content to the repair prompt so the LLM can see
    whether the symbol the test imports actually exists.
    """
    if not test_path:
        return []
    # Strip the test-dir prefix.
    rel = test_path
    for prefix in _TEST_DIR_PREFIXES:
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    # Strip "test_" from the basename, then keep the parent path.
    parts = rel.split("/")
    if not parts:
        return []
    basename = parts[-1]
    if basename.startswith("test_"):
        basename = basename[len("test_"):]
    elif basename.endswith("_test.py"):
        basename = basename[:-len("_test.py")] + ".py"
    parts[-1] = basename
    stem_rel = "/".join(parts)
    # Common source-root prefixes to probe.
    candidates: list[str] = [stem_rel]
    for src in ("src", "app", "lib"):
        candidates.append(f"{src}/{stem_rel}")
    # Also probe at workspace root with just the basename (for flat layouts).
    candidates.append(basename)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    # Filter to those that actually exist.
    found: list[str] = []
    for c in deduped:
        full = os.path.join(workspace_path, c)
        if os.path.isfile(full):
            found.append(c)
    return found


def _format_test_collection_cascade_section(
    errors: list[Any], workspace_path: str,
) -> str:
    """Detect pytest test-collection cascade errors and emit a section
    that (a) reframes the situation for the LLM and (b) attaches the
    current content of the most likely corresponding production files.

    Trigger: ≥ 1 diagnostic whose file path lives under ``tests/`` AND
    whose error_code matches a production-cascade pattern (ImportError,
    ModuleNotFoundError, NameError, AttributeError, F821, F401, ...).

    Returns empty string when no trigger fires.
    """
    if not errors:
        return ""
    candidates: list[tuple[str, str]] = []  # (test_path, code)
    for e in errors:
        path = str(e.get("file", "") or "")
        if not any(path.startswith(p) for p in _TEST_DIR_PREFIXES):
            continue
        code = str(e.get("error_code", "") or "")
        upper = code.upper()
        if not any(upper.startswith(p.upper()) for p in _PROD_CASCADE_ERROR_CODES):
            continue
        candidates.append((path, code))
    if not candidates:
        return ""

    # Map each unique test-file path to the candidate prod files.
    test_paths = sorted({p for p, _ in candidates})
    prod_attachments: dict[str, str] = {}  # prod_rel -> content
    for tp in test_paths:
        for prod_rel in _corresponding_prod_paths_for_test(tp, workspace_path):
            if prod_rel in prod_attachments:
                continue
            full = os.path.join(workspace_path, prod_rel)
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue
            # Cap each attachment so the prompt doesn't blow up.
            if len(content) > 6000:
                content = content[:6000] + "\n# ...(truncated)\n"
            prod_attachments[prod_rel] = content

    lines = [
        "\n## Test-collection cascade hint",
        (
            "Some of the diagnostics below point at TEST files but the "
            "shape (ImportError / ModuleNotFoundError / NameError / "
            "F821 / F401) suggests the underlying bug is in PRODUCTION "
            "code that the test is trying to import. Pytest reports the "
            "frame where the import raised — that's the test file — "
            "but the FIX usually belongs in the production module.\n"
            "Order of triage: (1) check whether the imported symbol "
            "actually exists in the production source below, (2) if "
            "not, add / rename it in PRODUCTION, (3) only edit the "
            "test file if the test's own import path is genuinely wrong."
        ),
    ]
    if prod_attachments:
        lines.append(
            "\nProduction source files that correspond to the failing test(s) "
            "(use these to verify the imported symbols actually exist):"
        )
        for rel, content in prod_attachments.items():
            lang = "python" if rel.endswith(".py") else "text"
            lines.append(f"\n### `{rel}`\n```{lang}\n{content.rstrip()}\n```")
    else:
        lines.append(
            "\n(Could not auto-locate the corresponding production source — "
            "search the workspace inventory below for the module the test "
            "tried to import.)"
        )
    return "\n".join(lines) + "\n"


def _format_replace_block_miss_directive(rb_misses: dict[str, int]) -> str:
    """When a file has ≥ 2 consecutive REPLACE_BLOCK misses, emit a
    directive telling the LLM to break the pattern by using a different
    operation on its next attempt at that file.

    Sessions 19b28eff and 0a5c6fe8 stuck in HITL because the LLM kept
    emitting REPLACE_BLOCK for the same file across iterations even
    though the searches never matched. Forcing the LLM out of that
    pattern with an explicit "use a different operation" instruction
    breaks the loop deterministically.

    Returns empty string when no file is in the danger zone.
    """
    if not rb_misses:
        return ""
    stuck = sorted(f for f, n in rb_misses.items() if n >= 2)
    if not stuck:
        return ""
    lines = [
        "\n## REPLACE_BLOCK pattern-repetition trap",
        (
            "You have failed REPLACE_BLOCK on the following file(s) TWO "
            "OR MORE times in a row. Your next attempt MUST NOT use "
            "REPLACE_BLOCK for any of these files. Instead use either:\n"
            "  (a) `DELETE_BLOCK` to remove the offending lines + "
            "`INSERT_AT_BLOCK` to put the corrected lines back, OR\n"
            "  (b) `DELETE_BLOCK` on the entire current content of the "
            "file + `CREATE_FILE` with the new full content.\n"
            "Either path forces a different patcher code path that won't "
            "miss the way your REPLACE_BLOCK search has. Affected files:"
        ),
    ]
    for f in stuck:
        lines.append(f"- `{f}` (consecutive REPLACE_BLOCK misses: {rb_misses[f]})")
    return "\n".join(lines) + "\n"


def _extract_wider_context_from_failure(failure: Any) -> tuple[str, Optional[str]]:
    """Split a patch_failure error into ``(text_without_wider_context,
    wider_context_block)``.

    The patcher emits errors of shape:

        Search block not found in foo.py. ...explanation...
        Current file content (around closest match):
         1| line 1
         2| line 2
         ...

    The portion after the marker is the most useful signal for the next
    repair iteration. Splitting lets us promote that portion to a
    top-of-prompt "Current Content of Files You Need to Edit" section
    while keeping the bare error message in the per-failure block.
    Returns ``(original_error, None)`` when no marker is present.
    """
    err = (failure.get("error", "") if isinstance(failure, dict) else "") or ""
    marker = _PATCH_ERROR_WIDER_CONTEXT_MARKER
    idx = err.find(marker)
    if idx < 0:
        return err, None
    prefix = err[:idx].rstrip()
    content_start = idx + len(marker)
    if content_start < len(err) and err[content_start] == "\n":
        content_start += 1
    wider = err[content_start:].rstrip()
    return prefix, wider or None


_PREFLIGHT_FILE_LINE_CAP = 300
_PREFLIGHT_FILE_CHAR_CAP = 6000
_PREFLIGHT_SECTION_CHAR_CAP = 24000


def _render_file_with_line_numbers(
    path: str,
    *,
    max_lines: int = _PREFLIGHT_FILE_LINE_CAP,
    max_chars: int = _PREFLIGHT_FILE_CHAR_CAP,
) -> Optional[str]:
    """Read ``path`` and return its content with ``  N|`` line-number
    prefixes, bounded by ``max_lines`` and ``max_chars``. Returns ``None``
    when the file is missing, unreadable, or empty — best-effort.

    Mirrors the formatting of ``patcher._find_closest_match`` so the
    pre-flight ``## Current Content of Files You Need to Edit`` section
    looks identical to the patcher's post-miss closest-match window. The
    LLM sees one consistent layout regardless of whether the content
    arrived proactively or after a failed REPLACE_BLOCK.
    """
    rendered, _ = _render_file_with_line_numbers_and_hash(
        path, max_lines=max_lines, max_chars=max_chars,
    )
    return rendered


def _render_file_with_line_numbers_and_hash(
    path: str,
    *,
    max_lines: int = _PREFLIGHT_FILE_LINE_CAP,
    max_chars: int = _PREFLIGHT_FILE_CHAR_CAP,
) -> tuple[Optional[str], Optional[str]]:
    """Same as ``_render_file_with_line_numbers`` but also returns the
    sha256 hex digest of the file's *full* bytes — even when the rendered
    view is truncated. The hash always reflects what is on disk so the
    B5 drift detector compares like-for-like.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None, None
    if not text:
        return None, None
    import hashlib
    # Hash the raw on-disk bytes (re-read as binary for fidelity — the text
    # read above used errors="replace" and may have substituted characters).
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        file_hash: Optional[str] = h.hexdigest()
    except OSError:
        file_hash = None
    text_lines = text.splitlines()
    if not text_lines:
        return None, file_hash
    if len(text_lines) > max_lines:
        text_lines = text_lines[:max_lines]
        truncated_lines = True
    else:
        truncated_lines = False
    width = max(2, len(str(max(1, len(text_lines)))))
    rendered = "\n".join(
        f"{(i + 1):>{width}}| {text_lines[i]}" for i in range(len(text_lines))
    )
    if truncated_lines:
        rendered += "\n...(truncated to first %d lines)" % max_lines
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "\n...(truncated to first %d chars)" % max_chars
    return rendered, file_hash


def _format_current_file_content(failures: list[Any]) -> str:
    """Collect every wider-context file-content block from prior patch
    failures into a single top-of-prompt section.

    Promotes the line-numbered file content from `node_state["patch_failures"]`
    to its own top-level Markdown section so the LLM sees it BEFORE the
    diagnostics, patch-failure summaries, allowlist, and inventory. The
    previous structure buried the file content inside the patch-failures
    block; sessions 19b28eff, 0a5c6fe8 still hit HITL because the LLM was
    anchoring on its own prior bad search rather than the actual file
    content. Lifting this content to a prominent slot is the structural
    fix for that drift.

    Dedupes by filepath — if multiple failures on the same file carry
    the same wider context, we show it once.
    """
    if not failures:
        return ""
    seen: dict[str, str] = {}
    for f in failures:
        if not isinstance(f, dict):
            continue
        file_ref = f.get("file", "?")
        _, wider = _extract_wider_context_from_failure(f)
        if wider and file_ref not in seen:
            seen[file_ref] = wider
    if not seen:
        return ""
    lines = [
        "\n## Current Content of Files You Need to Edit",
        (
            "Your last response's REPLACE_BLOCK search blocks did not "
            "match the on-disk content. The line-numbered views below are "
            "the **actual current content** of each file. Build your next "
            "REPLACE_BLOCK search by copying lines from here verbatim "
            "(WITHOUT the `  N| ` line-number prefix). Do NOT invent "
            "search strings that aren't in the file."
        ),
    ]
    for file_ref, wider in seen.items():
        lines.append(f"\n### `{file_ref}`\n```\n{wider}\n```")
    return "\n".join(lines) + "\n"


def _format_preflight_file_content(
    files: list[tuple[str, str]],
    *,
    intro: Optional[str] = None,
) -> str:
    """Front-load a ``## Current Content of Files You Need to Edit`` section
    from a list of (rel_path, line_numbered_content) pairs.

    Used by ``patching_node``, ``test_generation_node``, and ``repair_node``
    iter-1 to give the LLM the actual file bytes BEFORE its first patch
    attempt. Without this section the LLM hallucinates search strings from
    its mental model of files it has never seen (root cause behind the
    100-run failure streak on session 2d0164f0).

    Dedupes by filepath (first wins) and stops adding files once the
    rendered total crosses ``_PREFLIGHT_SECTION_CHAR_CAP`` so big projects
    don't blow the prompt budget.
    """
    if not files:
        return ""
    seen: dict[str, str] = {}
    for rel_path, content in files:
        if not content or rel_path in seen:
            continue
        seen[rel_path] = content
    if not seen:
        return ""
    if intro is None:
        intro = (
            "The line-numbered views below are the **actual current "
            "content** of files you may need to edit. Build any "
            "REPLACE_BLOCK / DELETE_BLOCK search by copying lines from "
            "here verbatim (WITHOUT the `  N| ` line-number prefix). Do "
            "NOT guess at what these files contain — use exactly what "
            "you see below."
        )
    lines = [
        "\n## Current Content of Files You Need to Edit",
        intro,
    ]
    accumulated = sum(len(s) for s in lines)
    truncated_at: Optional[str] = None
    for rel_path, content in seen.items():
        block = f"\n### `{rel_path}`\n```\n{content}\n```"
        if accumulated + len(block) > _PREFLIGHT_SECTION_CHAR_CAP:
            truncated_at = rel_path
            break
        lines.append(block)
        accumulated += len(block)
    if truncated_at is not None:
        lines.append(
            f"\n(omitted further files starting at `{truncated_at}` to "
            f"keep this section under {_PREFLIGHT_SECTION_CHAR_CAP // 1000}k "
            f"chars — emit a READ_FILE block for any other file you need.)"
        )
    return "\n".join(lines) + "\n"


def _resolve_read_blocks(
    read_blocks: list[tuple[str, Optional[tuple[int, int]]]],
    workspace_path: str,
    *,
    record_hashes_into: Optional[dict[str, str]] = None,
) -> str:
    """Resolve a list of ``parse_read_blocks`` outputs into a user-message
    payload containing line-numbered current content for each file.

    Returns an empty string when nothing could be resolved (no files exist
    or all reads errored), so callers can append unconditionally.

    When ``record_hashes_into`` is supplied, every successfully-read file's
    sha256 is recorded so the patcher's B5 drift detector knows what the
    LLM has actually been shown.
    """
    if not read_blocks:
        return ""
    sections: list[str] = []
    for rel_path, rng in read_blocks:
        abs_path = os.path.join(workspace_path, rel_path)
        if rng is None:
            rendered, file_hash = _render_file_with_line_numbers_and_hash(abs_path)
            if record_hashes_into is not None and file_hash is not None:
                record_hashes_into[rel_path] = file_hash
        else:
            # Window mode: render a sub-range. Re-use the renderer's
            # bounded reader by reading the whole file, then slicing.
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                sections.append(
                    f"\n### `{rel_path}` (range {rng[0]}-{rng[1]})\n"
                    f"```\n(file not found or unreadable)\n```"
                )
                continue
            all_lines = text.splitlines()
            lo = max(1, rng[0])
            hi = min(len(all_lines), rng[1])
            if lo > hi:
                rendered = "(empty range)"
            else:
                width = max(2, len(str(hi)))
                rendered = "\n".join(
                    f"{i:>{width}}| {all_lines[i - 1]}" for i in range(lo, hi + 1)
                )
            # Drift detection still uses the FULL on-disk hash even when the
            # rendered view is a range — the patcher checks the whole file
            # has not changed, regardless of which slice the LLM was shown.
            if record_hashes_into is not None:
                import hashlib
                try:
                    h = hashlib.sha256()
                    with open(abs_path, "rb") as bf:
                        for chunk in iter(lambda: bf.read(65536), b""):
                            h.update(chunk)
                    record_hashes_into[rel_path] = h.hexdigest()
                except OSError:
                    pass
        if rendered is None:
            sections.append(
                f"\n### `{rel_path}`\n```\n(file not found or unreadable)\n```"
            )
        else:
            header = (
                f"\n### `{rel_path}` (range {rng[0]}-{rng[1]})"
                if rng is not None else f"\n### `{rel_path}`"
            )
            sections.append(f"{header}\n```\n{rendered}\n```")
    intro = (
        "## READ_FILE results\n"
        "You requested the current content of the following file(s). Use "
        "these line-numbered views — WITHOUT the `  N| ` prefix — as the "
        "source of truth for any REPLACE_BLOCK / DELETE_BLOCK search you "
        "now write. Do NOT emit another READ_FILE for these same files in "
        "your next response; just write the patches."
    )
    return intro + "\n" + "\n".join(sections) + "\n"


def _collect_workspace_file_content(
    workspace_path: str,
    rel_paths: Iterable[str],
    *,
    max_files: int = 12,
    record_hashes_into: Optional[dict[str, str]] = None,
) -> list[tuple[str, str]]:
    """Read each ``rel_paths`` entry under ``workspace_path`` and return
    ``(rel_path, line_numbered_content)`` pairs. Silently skips files that
    don't exist or can't be rendered. Caps at ``max_files`` so large
    inventories don't blow token budgets — the LLM can READ_FILE
    explicitly for anything else.

    When ``record_hashes_into`` is supplied, the sha256 of each
    rendered file's on-disk bytes is stored at ``record_hashes_into[rel]``.
    The caller persists this dict into ``node_state.files_seen_by_llm`` so
    the patcher's B5 drift detector can later confirm the file hasn't
    changed since the LLM was shown its content.
    """
    out: list[tuple[str, str]] = []
    for rel in rel_paths:
        if len(out) >= max_files:
            break
        if not isinstance(rel, str) or not rel.strip():
            continue
        abs_path = os.path.join(workspace_path, rel)
        rendered, file_hash = _render_file_with_line_numbers_and_hash(abs_path)
        if rendered is None:
            continue
        out.append((rel, rendered))
        if record_hashes_into is not None and file_hash is not None:
            record_hashes_into[rel] = file_hash
    return out


def _format_prior_patch_failures(failures: list[Any]) -> str:
    """Format prior-attempt patch failures into a Markdown block for the
    repair LLM prompt. Empty string when there are no failures so callers
    can string-append unconditionally.

    The wider-context file-content portion of each failure is extracted
    and emitted SEPARATELY by :func:`_format_current_file_content` so it
    can live in a more prominent top-of-prompt section; this block keeps
    only the patcher's diagnostic prose. A short pointer is added when
    file content for the same path lives in the top section.
    """
    if not failures:
        return ""
    lines = [
        "\n## Patch Failures (PREVIOUS attempt)",
        (
            "Your last attempt produced patches that the patcher could not "
            "apply. The exact reasons are below — read them carefully. Do "
            "NOT re-emit the same SEARCH block verbatim: either match the "
            "actual file bytes shown in the `## Current Content of Files "
            "You Need to Edit` section above, add more context lines if "
            "your search was ambiguous, or switch operations (e.g. "
            "INSERT_AT_BLOCK instead of REPLACE_BLOCK when the target "
            "line doesn't yet exist)."
        ),
    ]
    for f in failures:
        file_ref = f.get("file", "?") if isinstance(f, dict) else "?"
        op_ref = f.get("operation", "replace_block") if isinstance(f, dict) else "replace_block"
        prefix, wider = _extract_wider_context_from_failure(f)
        err_ref = prefix.strip()
        if not err_ref:
            err_ref = f"({op_ref} on `{file_ref}` failed)"
        if wider:
            err_ref += (
                f"\n(see the `## Current Content of Files You Need "
                f"to Edit` section for the actual content of "
                f"`{file_ref}`)"
            )
        lines.append(
            f"\n### `{file_ref}` ({op_ref})\n```\n{err_ref}\n```"
        )
    return "\n".join(lines) + "\n"


def _workspace_has_source_files(workspace_path: str) -> bool:
    """True when the workspace contains at least one source file under a
    non-ignored directory. Used by the router to decide whether 'no tests
    collected' should route to test_generation_node (source exists, needs
    tests) or HITL (truly empty workspace — the LLM failed to scaffold)."""
    from harness.impact import (
        _SOURCE_FILE_EXTENSIONS,
        _NEVER_SOURCE_DIRS,
    )
    if not workspace_path or not os.path.isdir(workspace_path):
        return False
    try:
        for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
                and d not in {"tests", "test", "__tests__"}
            ]
            for fname in sub_files:
                if os.path.splitext(fname)[1].lower() in _SOURCE_FILE_EXTENSIONS:
                    return True
    except OSError:
        return False
    return False


# Pip-installable test / lint tools. When `_is_env_misconfig` flags one
# of these as missing, the LLM repair loop CAN fix it by appending the
# package to the workspace's dependency manifest — no image swap needed.
# Contrast with non-installable symbols (npm, node, cargo, go, rustc,
# docker) where the base image itself is wrong and only a config change
# can unblock the run; those still short-circuit to HITL.
_PIP_INSTALLABLE_SYMBOLS: frozenset[str] = frozenset({
    "pytest", "pytest-asyncio", "pytest-cov", "pytest-mock", "pytest-xdist",
    "ruff", "mypy", "black", "isort", "flake8", "pylint",
    "coverage", "tox", "nox", "poetry",
})


def _repairable_dep_hint(symbol: str, build_command: str) -> str:
    """Repair-friendly diagnostic for a missing pip-installable test / lint
    tool. Reaches the repair LLM via compiler_errors and points it at the
    smallest possible patch (add the dep to requirements.txt or pyproject
    dev extras). Distinct from :func:`_env_misconfig_hint`, which is sent
    to HITL because no in-container patch can fix it.
    """
    return (
        f"Build failed: '{symbol}' is required by the build command but is "
        f"not declared as a dependency. The sandbox runs "
        f"`{build_command.strip()}`, which invokes `{symbol}` after the "
        f"install step — but '{symbol}' isn't in the workspace's "
        f"dependency manifest, so pip never installs it.\n\n"
        f"Fix in ONE place:\n"
        f"  - If the install step is `pip install -r requirements.txt`: "
        f"add `{symbol}` to `requirements.txt`. If the file does not "
        f"exist yet, CREATE it with one dependency per line "
        f"(including `{symbol}`).\n"
        f"  - If the install step is `pip install -e '.[dev]'`: add "
        f"`{symbol}` to `[project.optional-dependencies].dev` in "
        f"`pyproject.toml`. If the section does not exist, CREATE it.\n"
        f"Do not change the build_command or docker_image — the package "
        f"is pip-installable and the current image is correct."
    )


def _env_misconfig_hint(symbol: str, build_command: str) -> str:
    """Build the actionable HITL message for an env-misconfig hit."""
    # Pick the most likely installer for the missing symbol.
    py_symbols = {"pytest", "ruff", "mypy", "black", "poetry", "tox", "nox"}
    if symbol.lower() in py_symbols or "python" in symbol.lower():
        installer = f"pip install {symbol}"
    elif symbol in {"npm", "node", "yarn", "pnpm"}:
        installer = (
            f"use a node-bearing docker_image (e.g. node:20-slim) — "
            f"'{symbol}' is not installable from inside the container"
        )
    elif symbol in {"cargo", "rustc"}:
        installer = (
            f"use a rust-bearing docker_image (e.g. rust:1.79-slim) — "
            f"'{symbol}' is not installable from inside the container"
        )
    elif symbol in {"go"}:
        installer = (
            f"use a go-bearing docker_image (e.g. golang:1.22) — "
            f"'{symbol}' is not installable from inside the container"
        )
    else:
        installer = f"install {symbol} before running the build"
    return (
        f"Build container is missing '{symbol}'. The repair LLM cannot fix this — "
        f"it's a sandbox/dependency setup issue. "
        f"Fix: update build_command to prepend the install step "
        f"(e.g. '{installer} && {build_command.strip()}') "
        f"or switch sandbox.docker_image to one that ships '{symbol}'."
    )


def _apply_toolchain_adaptation(
    build_command: str,
    sandbox_config: Optional[dict[str, Any]],
    allow_network: bool,
    *,
    command_is_adapter_synthesised: bool = False,
) -> tuple[dict[str, Any], bool, bool, bool, bool]:
    """Idempotently adapt sandbox_config/allow_network to match the build
    command's implied toolchain.

    Returns
    -------
    (new_sandbox_config, allow_network, image_was_adapted,
     network_was_adapted, ro_root_was_adapted).

    Calling twice with the same inputs returns *_was_adapted=False on
    the second call — each conditional only fires when its precondition is
    still true (allow_network still False, read_only_root unset / still
    True). ``image_was_adapted`` is permanently False — the harness's
    one-image-fits-all builder is selected by DockerBackend's default and
    needs no per-command swap.

    P1.3: the network auto-enable on a pip/npm install heuristic is gated
    by ``sandbox.auto_enable_network_for_install`` (default ``false``).
    When the heuristic would fire on a user-typed build command but the
    opt-in is off, the function declines to flip ``allow_network`` and
    logs a warning so the operator sees the divergence.

    ``command_is_adapter_synthesised`` (default False) marks the case
    where the install step was introduced by the harness's own late-bind
    detection (``_detect_default_build_command`` fell through to a
    bootstrapping branch), NOT by the operator. The user's opt-in
    governs auto-flips on commands the OPERATOR wrote; it doesn't make
    sense to apply it to commands the harness invented to make a
    greenfield build work at all. When ``True``, network is auto-enabled
    for the install step regardless of the opt-in (the operator can
    still hard-pin ``allow_network=False`` at the workspace config level
    if they really want it).

    ``make <target>`` commands receive the same bypass. The operator
    types ``make build`` (or accepts the CLI default), but the recipe
    that actually runs — including any ``pip install -r requirements.txt``
    — is authored by the LLM per the Makefile skill
    (``harness/skills/makefile_python.md``). Treating it like an operator-
    typed install would silently route to the warn-and-fail branch.
    Bypassing the opt-in keeps the deterministic-build promise; the
    workspace-config ``allow_network`` hard-pin still wins for genuine
    airgap operators.
    """
    cfg = dict(sandbox_config or {})
    image_was_adapted = False
    network_was_adapted = False
    ro_root_was_adapted = False

    # The harness ships a single kitchen-sink builder image
    # (``harness/vendor/Dockerfile.builder``, exported as
    # ``harness.sandbox.BUILDER_IMAGE``) that bakes every supported
    # toolchain. DockerBackend already defaults its ``image`` argument to
    # ``BUILDER_IMAGE``, so when ``cfg["docker_image"]`` is unset the
    # sandbox uses the right image without anything happening here.
    # Explicit operator pins (custom corporate registries, locked-down CI
    # bases) are respected by passing through whatever cfg already has.
    # ``image_was_adapted`` therefore always stays ``False``; it remains
    # in the return tuple for caller compatibility.

    new_allow_network = allow_network
    needs_install = _build_command_needs_network(build_command)
    if not allow_network and needs_install:
        # When the harness itself produced the install step, the opt-in
        # doesn't apply — the operator never typed `pip install`, the
        # adapter inserted it to bootstrap a greenfield workspace. The
        # user's network policy still applies at the workspace config
        # level (sandbox config) but the opt-in flag is about user-typed
        # commands, not adapter-synthesised ones.
        #
        # ``make <target>`` rides the same bypass: the operator may have
        # typed `make build`, but the install step that needs network is
        # inside the LLM-written Makefile recipe — semantically the same
        # situation as an adapter-synthesised command, just expressed via
        # a different file. See the docstring for the full rationale.
        if command_is_adapter_synthesised or _command_is_make(build_command):
            new_allow_network = True
            network_was_adapted = True
        elif cfg.get("auto_enable_network_for_install", False):
            new_allow_network = True
            network_was_adapted = True
        else:
            logger.warning(
                "[sandbox] Build command requires network access for install "
                "but sandbox.auto_enable_network_for_install is false. "
                "Build will run offline and likely fail; either pre-install "
                "deps in the sandbox image, run `teane run --allow-network`, "
                "or set sandbox.auto_enable_network_for_install=true in "
                ".harness_config.json. Build command: %s",
                build_command,
            )

    # Install commands (pip install -e ., npm install -g, cargo install)
    # write to system locations the --read-only root FS makes unreachable.
    # Pip's `--user` fallback also fails because /root sits on the RO root.
    # Auto-flip read_only_root → False so the install can land, unless the
    # user has explicitly set it to True (respect explicit opt-in to hard
    # isolation even if it breaks the build — they'll need a baked image).
    if needs_install and "read_only_root" not in cfg:
        cfg["read_only_root"] = False
        ro_root_was_adapted = True

    return (
        cfg,
        new_allow_network,
        image_was_adapted,
        network_was_adapted,
        ro_root_was_adapted,
    )


async def compiler_node(state: AgentState) -> dict[str, Any]:
    """
    Node 3: The Verifier.

    A purely deterministic local node. Executes the project's build command
    inside an isolated sandbox (Linux namespace jail) via asyncio subprocess.

    Reads `build_command` and `workspace_path` from state.
    Writes `exit_code` and structured `compiler_errors` back to state.

    If exit_code == 0, triggers the memory cleanse utility to compress
    verbose debugging turns before the next graph transition.
    """
    logger.info("[compiler_node] Running build command in sandbox...")

    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["compiler"] = loop_counter.get("compiler", 0) + 1
    # Audit #18 — when the router sent us back here purely to re-verify
    # post-green mutations (no failure to repair, just defence-in-depth),
    # record that this final-verify slot has now been spent. The router
    # checks this counter so a single extra re-compile can't expand into
    # a thrash loop.
    is_pre_exit_verify = bool(state.get("pending_mutations"))
    if is_pre_exit_verify:
        loop_counter["final_verify"] = loop_counter.get("final_verify", 0) + 1

    workspace = state.get("workspace_path", os.getcwd())
    build_cmd = state.get("build_command", "make build")
    allow_network = state.get("allow_network", False)
    sandbox_cfg = dict(state.get("sandbox_config", {}) or {})

    # Late-bound adaptive detection: if the resolved build_cmd is the
    # historical default and the workspace genuinely has no Makefile,
    # re-sniff now that codegen has likely populated the tree. This
    # rescues greenfield runs where workspace detection at cmd_run start
    # had nothing to go on but the spec file.
    adapted_build_cmd: Optional[str] = None
    if build_cmd.strip() == "make build" and not any(
        os.path.exists(os.path.join(workspace, name))
        for name in ("Makefile", "makefile", "GNUmakefile")
    ):
        from harness.cli import _detect_default_build_command
        late = _detect_default_build_command(workspace)
        if late and late != "make build":
            logger.info(
                "[compiler_node] Workspace has no Makefile; adapting build command "
                "from default 'make build' to detected: %s", late,
            )
            adapted_build_cmd = late
            build_cmd = late

    # Mid-session upgrade: when the build_cmd is the bare "pip install <tool>
    # && pytest -q" fallback (chosen earlier because the workspace had no
    # manifest yet) AND a manifest has since appeared on disk — typically
    # because autofix R4 just wrote requirements.txt to install a missing
    # pip-installable dep — re-detect now so the next compile uses
    # `pip install -r requirements.txt`. Without this upgrade the bare
    # pytest install never sees the new dep and the same MISSING_DEP loops
    # forever.
    if (
        adapted_build_cmd is None
        and "pip install" in build_cmd
        and "-r" not in build_cmd
        and (
            os.path.isfile(os.path.join(workspace, "requirements.txt"))
            or os.path.isfile(os.path.join(workspace, "pyproject.toml"))
        )
    ):
        from harness.cli import _detect_default_build_command
        re_detected = _detect_default_build_command(workspace)
        if re_detected and re_detected != build_cmd and re_detected != "make build":
            logger.info(
                "[compiler_node] Workspace gained a dependency manifest mid-session; "
                "upgrading build command from %r to detected %r so installed deps "
                "are honored.",
                build_cmd, re_detected,
            )
            adapted_build_cmd = re_detected
            build_cmd = re_detected

    # Late-bound sandbox image / network adaptation. With the pre-flight
    # adaptation in run_graph this is now a safety net — it only fires when
    # the build_command was just adapted above (greenfield rescue), or on
    # resume from a pre-fix checkpoint whose sandbox_config wasn't yet
    # adapted. The helper is idempotent: if the image already matches the
    # toolchain, image_was_adapted is False and no extra log line appears.
    # When the build_command was just adapter-synthesised, tell the
    # toolchain adapter so it can bypass the user-opt-in network gate —
    # the operator never typed this command, the harness invented it.
    prev_image = sandbox_cfg.get("docker_image", BUILDER_IMAGE)
    (
        sandbox_cfg,
        allow_network,
        image_was_adapted,
        network_was_adapted,
        ro_root_was_adapted,
    ) = _apply_toolchain_adaptation(
        build_cmd,
        sandbox_cfg,
        allow_network,
        command_is_adapter_synthesised=adapted_build_cmd is not None,
    )
    if image_was_adapted:
        logger.info(
            "[compiler_node] Adapting sandbox docker_image from %r to %r "
            "to match build toolchain implied by command: %s",
            prev_image, sandbox_cfg["docker_image"], build_cmd,
        )
    if network_was_adapted:
        logger.info(
            "[compiler_node] Adapting allow_network from False to True "
            "because build command requires package install: %s",
            build_cmd,
        )
    if ro_root_was_adapted:
        logger.info(
            "[compiler_node] Adapting sandbox.read_only_root from True to False "
            "because build command installs packages (pip/npm/cargo/go) into "
            "system locations the read-only root FS would block: %s",
            build_cmd,
        )

    # Fix #6 / two-phase: prod-import smoke check BEFORE running the
    # actual build. Verifies every production module imports cleanly so
    # the LLM never has to disambiguate "is this a prod bug cascading
    # through tests, or a test bug?" — prod errors surface first, on
    # their own, with a [PROD_IMPORT_SMOKE] error_code tag the repair
    # node + cascade hints recognise.
    gw_for_cfg = get_gateway()
    smoke_enabled = True
    if gw_for_cfg is not None:
        # The flag lives under config.compiler.run_prod_import_smoke_check
        # but the gateway config doesn't carry an arbitrary nested
        # section, so we re-read it from the raw config dict via
        # discover_config-stashed state. Default true (matches the
        # documented config behaviour).
        smoke_enabled = bool(state.get("run_prod_import_smoke_check", True))
    if (
        smoke_enabled
        and "pip install" in build_cmd
        and "pytest" in build_cmd
    ):
        # Reuse the install step from the build_cmd so we don't double-
        # install. Take everything up to the first `&&` as the install
        # phase, the rest (pytest) is the actual build we'd run otherwise.
        install_step = build_cmd.split("&&")[0].strip()
        smoke_errors = await _run_prod_import_smoke_check(
            workspace_path=workspace,
            sandbox_config=sandbox_cfg,
            allow_network=allow_network,
            install_step=install_step,
            session_id=state.get("session_id", "unknown"),
        )
        if smoke_errors:
            # Short-circuit: skip the actual build until prod imports
            # are clean. compiler_errors carries the smoke failures
            # alone — no test cascade for the LLM to wade through.
            # Merge into the existing node_state so cross-iteration signals
            # (patch_failures, allowlist_rejections, allowed_paths) survive
            # for the next repair_node — see graph.py:2755 fix.
            short_circuit_state = dict(state.get("node_state", {}) or {})
            short_circuit_state["current_node"] = "compiler"
            short_circuit_state["prod_smoke_failed"] = True
            short_circuit_state["last_build_output"] = (
                "Prod-import smoke check failed. The actual build "
                "(pytest) was not run because production modules "
                "could not be imported cleanly. Fix the import "
                "errors above before pytest is attempted."
            )
            return {
                "exit_code": 1,
                "compiler_errors": smoke_errors,
                "node_state": short_circuit_state,
                "loop_counter": loop_counter,
            }

    # Delegate to the sandbox module for actual execution.
    from harness.sandbox import SandboxExecutor

    executor = SandboxExecutor(
        workspace_path=workspace,
        allow_network=allow_network,
        sandbox_config=sandbox_cfg,
    )
    result = await executor.run(build_cmd)

    exit_code: int = result.exit_code
    compiler_errors: list[Any] = [d.to_dict() for d in result.diagnostics]
    raw_log: str = result.raw_output

    # Audit §6.8: allow operators to declare non-zero exit codes as
    # ``advisory`` for this build_command — useful for tools like
    # ``terraform validate`` that emit non-zero on benign drift. Listed
    # codes get folded to 0 so the repair loop doesn't fire on noise.
    _compiler_cfg_raw = state.get("compiler_config")
    compiler_cfg: dict[str, Any] = _compiler_cfg_raw if isinstance(_compiler_cfg_raw, dict) else {}
    advisory_codes = compiler_cfg.get("advisory_exit_codes") or []
    if exit_code != 0 and isinstance(advisory_codes, list):
        try:
            advisory_set = {int(c) for c in advisory_codes}
        except (TypeError, ValueError):
            advisory_set = set()
        if exit_code in advisory_set:
            logger.warning(
                "[compiler_node] Build exited with advisory code %d (per "
                "compiler.advisory_exit_codes config); treating as success.",
                exit_code,
            )
            exit_code = 0

    logger.info(
        "[compiler_node] Build finished with exit code %d. %d diagnostic(s) extracted.",
        exit_code,
        len(compiler_errors),
    )

    # Detect "sandbox is missing a required runtime" failures and split into
    # two routes:
    #   - Pip-installable test/lint tools (pytest, ruff, mypy, ...): the
    #     repair LLM CAN fix it by amending the workspace's dep manifest.
    #     Emit a MISSING_DEP diagnostic and let normal routing take over so
    #     repair_node gets the diagnostic as context.
    #   - Everything else (npm/node/cargo/go/docker, single-segment local
    #     modules): the image itself is wrong / the LLM cannot help from
    #     inside the sandbox. Short-circuit to HITL as before.
    # pip ResolutionImpossible — emit a distinct diagnostic so autofix R5
    # (`_try_dep_resolution_conflict`) can strip the version pins from
    # requirements.txt instead of forcing the repair LLM to guess which
    # side of the conflict to relax. Without this we burn the entire
    # 3-iteration repair budget on conflicts the LLM doesn't have enough
    # information to resolve (pip's error doesn't name both sides).
    if (
        exit_code != 0
        and not compiler_errors
        and _is_pip_resolution_conflict(raw_log, build_cmd)
    ):
        logger.warning(
            "[compiler_node] pip resolution conflict detected — routing "
            "through repair loop so autofix can strip pins from requirements."
        )
        compiler_errors = [{
            "file": "<sandbox>",
            "line": 0,
            "column": 0,
            "severity": "error",
            "error_code": "DEP_RESOLUTION_CONFLICT",
            "message": (
                "pip's resolver couldn't satisfy the version pins in the "
                "dependency manifest together. The repair-loop fix is to "
                "loosen or drop the version specifiers (>=, ==, ~=, <, !=) "
                "from the manifest so pip can pick a self-consistent set."
            ),
            "semantic_context": f"Build command: {build_cmd}.",
            "missing_symbol": "",
            "build_command": build_cmd,
            "miss_kind": "resolution",
        }]

    env_misconfig_symbol: Optional[str] = None
    env_misconfig_is_repairable: bool = False
    if exit_code != 0 and not compiler_errors:
        env_match = _is_env_misconfig(raw_log, workspace)
        if env_match:
            env_misconfig_symbol, miss_kind = env_match
            # Python ModuleNotFoundError is ALWAYS repairable — any
            # single-segment module name is a pip-installable distribution,
            # and the autofix R4 (`_try_missing_dep`) + the repair LLM can
            # land the requirements.txt edit. Shell `command not found` is
            # only repairable when the symbol is a pip-installable Python
            # tool listed in `_PIP_INSTALLABLE_SYMBOLS`; everything else
            # (npm, cargo, go, docker) needs an operator-side image swap.
            env_misconfig_is_repairable = (
                miss_kind == "python"
                or env_misconfig_symbol.lower() in _PIP_INSTALLABLE_SYMBOLS
            )
            if env_misconfig_is_repairable:
                logger.info(
                    "[compiler_node] Missing '%s' (kind=%s) is pip-installable. "
                    "Routing through repair loop so autofix / LLM can amend the "
                    "dep manifest.",
                    env_misconfig_symbol, miss_kind,
                )
                msg = _repairable_dep_hint(env_misconfig_symbol, build_cmd)
                code = "MISSING_DEP"
            else:
                logger.warning(
                    "[compiler_node] Environment misconfig detected: missing '%s' "
                    "(kind=%s). Short-circuiting repair loop — this needs a config "
                    "fix, not an LLM patch.",
                    env_misconfig_symbol, miss_kind,
                )
                msg = _env_misconfig_hint(env_misconfig_symbol, build_cmd)
                code = "ENV_MISCONFIG"
            compiler_errors = [{
                "file": "<sandbox>",
                "line": 0,
                "column": 0,
                "severity": "error",
                "error_code": code,
                "message": msg,
                "semantic_context": (
                    f"Missing runtime: {env_misconfig_symbol}. "
                    f"Build command: {build_cmd}. "
                    f"docker_image: {sandbox_cfg.get('docker_image', BUILDER_IMAGE)}."
                ),
                # Structured fields the autofix + repair-prompt builders read
                # without re-parsing the human-readable message.
                "missing_symbol": env_misconfig_symbol,
                "build_command": build_cmd,
                "miss_kind": miss_kind,
            }]

    # Track consecutive repeats of the same MISSING_DEP symbol across
    # iterations. The router's "deterministic autofixable" bypass at the
    # repair limit (see route_after_compiler) keeps re-entering repair_node
    # forever when the missing symbol is something the manifest fix cannot
    # actually install — e.g. `pip` itself missing from a docker image like
    # buildpack-deps. Each LLM patch ostensibly "lands" but the very next
    # build emits the same MISSING_DEP. Without a tripwire here the session
    # spins past the configured max_iterations indefinitely (observed in
    # session 083770ac: 21+ attempts on missing 'pip', $0.02 of budget
    # burnt before being killed externally). The router consumes this
    # counter and escalates to HITL with a "fix the image, not the
    # manifest" message when it crosses the threshold.
    primary_missing_dep_symbol = ""
    for err in compiler_errors:
        if str(err.get("error_code", "")).upper() == "MISSING_DEP":
            sym = str(err.get("missing_symbol", "") or "").strip().lower()
            if sym:
                primary_missing_dep_symbol = sym
                break
    prior_symbol = str(loop_counter.get("missing_dep_last_symbol", "") or "")
    if primary_missing_dep_symbol:
        if primary_missing_dep_symbol == prior_symbol:
            loop_counter["missing_dep_consecutive_same"] = (
                int(loop_counter.get("missing_dep_consecutive_same", 0) or 0) + 1
            )
        else:
            loop_counter["missing_dep_consecutive_same"] = 1
            loop_counter["missing_dep_last_symbol"] = primary_missing_dep_symbol
    else:
        # No MISSING_DEP this round (either build succeeded or different
        # error shape) — reset so a future MISSING_DEP starts fresh.
        loop_counter["missing_dep_consecutive_same"] = 0
        loop_counter["missing_dep_last_symbol"] = ""

    # Build the return dictionary by MERGING into the existing node_state,
    # not replacing it. Cross-iteration signals like patch_failures (the
    # patcher's "Current file content around closest match" window),
    # allowlist_rejections, and allowed_paths live on node_state and must
    # survive compiler_node between repair iterations — otherwise the next
    # repair_node has no view of the actual file bytes and the LLM
    # hallucinates SEARCH strings (sessions 19b28eff, 0a5c6fe8, 2d0164f0).
    node_state: dict[str, Any] = dict(state.get("node_state", {}) or {})
    node_state["current_node"] = "compiler"
    node_state["last_build_output"] = raw_log
    # Only set the short-circuit flag for symbols the LLM truly can't fix.
    # Repairable symbols carry their diagnostic into repair_node normally.
    if env_misconfig_symbol and not env_misconfig_is_repairable:
        node_state["env_misconfig"] = True
        node_state["env_misconfig_symbol"] = env_misconfig_symbol

    # Pytest exit-5 (no tests collected) is NOT a build failure — the test
    # runner just had nothing to run. Surface as a distinct condition so the
    # router can fan out to test_generation (when source exists) or HITL
    # (when the workspace is empty), instead of burning the repair budget
    # on a problem the repair LLM cannot fix from inside the loop.
    #
    # Deliberately do NOT populate `compiler_errors` here. If we did, the
    # downstream `route_after_test_generation` would see a non-empty errors
    # list (even after test_generation_node skips or finishes cleanly) and
    # spin into a repair loop trying to "fix" a non-error.
    if exit_code != 0 and not compiler_errors and _is_no_tests_collected(
        exit_code, raw_log, build_cmd,
    ):
        has_source = _workspace_has_source_files(workspace)
        node_state["no_tests_collected"] = True
        node_state["no_tests_has_source"] = has_source
        logger.warning(
            "[compiler_node] pytest exit=5: no tests collected. "
            "Source files present: %s. Router will %s.",
            has_source,
            "route to test_generation" if has_source else "route to HITL",
        )

    return_dict: dict[str, Any] = {
        "exit_code": exit_code,
        "compiler_errors": compiler_errors,
        "loop_counter": loop_counter,
        "node_state": node_state,
    }

    # Persist the adapted command so repair_node / patching_node prompts
    # and subsequent compiler invocations see the updated value.
    if adapted_build_cmd is not None:
        return_dict["build_command"] = adapted_build_cmd
    if image_was_adapted:
        return_dict["sandbox_config"] = sandbox_cfg
    if network_was_adapted:
        return_dict["allow_network"] = allow_network

    # Trigger memory cleanse on successful build
    if exit_code == 0:
        logger.info("[compiler_node] Build succeeded. Applying memory cleanse.")
        cleanse_update = apply_memory_cleanse(state, resolution_kind="compiler_success")
        return_dict.update(cleanse_update)
        # Audit #18 — green compile clears the post-green-mutation tracker.
        # Anything appended after this point reflects a real drift between
        # what tests verified and what's on disk.
        return_dict["pending_mutations"] = []

    return return_dict


def _repair_budget_warning(total_repairs: int, cap: int) -> Optional[str]:
    """Return a soft system-message warning when repair iterations are
    running out (audit #19).

    The harness already enforces ``cap`` as a hard ceiling — past it the
    router moves to HITL. The warning fires at the last two iterations
    (``remaining <= 2``) so the LLM has a chance to adjust strategy
    *before* it gets cut off. A model that knows it's near the wall
    favours small surgical edits over rewrites, which is exactly the
    right move late in a run.

    Returns ``None`` when there's still slack. The 2-step ramp (medium
    warning at remaining==2, hard warning at remaining==1) gives the
    model a chance to register the signal before the final attempt.
    """
    if cap <= 0:
        return None
    remaining = cap - total_repairs
    if remaining <= 0 or remaining > 2:
        return None
    if remaining == 1:
        return (
            "[System budget warning] This is the LAST repair iteration "
            "before the harness routes to human intervention. Emit the "
            "smallest possible patch that fixes the failing diagnostic — "
            "no refactors, no rewrites, no speculative cleanups. If the "
            "fix is uncertain, prefer a narrow change over a broad one."
        )
    return (
        "[System budget warning] 2 repair iterations remain before the "
        "harness escalates. Favour focused, surgical fixes over broad "
        "changes. If the same error keeps recurring across attempts, "
        "narrow the scope rather than widening it."
    )


async def repair_node(state: AgentState) -> dict[str, Any]:
    """
    Node 4: The Fixer.

    Invoked when the compiler_node returns a non-zero exit code and the
    loop_counter is under the throttle limit (3).

    Uses the configured repair model (repair_primary in .harness_config.json)
    with thinking mode enabled via the gateway to analyze structured compiler
    diagnostics and produce a targeted fix patch. Applies the resulting patches
    to disk immediately.

    Increments the repair loop counter. If the counter hits the threshold,
    the conditional edge will route to human_intervention_node instead.
    """
    logger.info("[repair_node] Analyzing compiler errors and preparing fix...")

    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["repair"] = loop_counter.get("repair", 0) + 1
    loop_counter["total_repairs"] = loop_counter.get("total_repairs", 0) + 1

    # Failure-path memory cleanse: from the SECOND repair iteration onward,
    # trim the message list so iteration N's LLM call doesn't carry the bloat
    # of 1..N-1. apply_memory_cleanse fires only on success/HITL; this
    # complement runs between failed iterations. The fresh error_summary
    # (diagnostics + patch-failures + workspace inventory + allowlist) is
    # appended below as a new user message regardless, so the LLM always
    # sees full structured context for the current turn.
    if loop_counter["total_repairs"] >= 2:
        cleanse_update = apply_repair_iteration_cleanse(state)
        if cleanse_update:
            # Mutate the state's messages view in-place so the rest of this
            # node (which reads state.get("messages")) sees the trimmed list.
            state = cast(AgentState, dict(state))
            state["messages"] = cleanse_update["messages"]

    # Build a concise repair prompt from structured diagnostics. Drop any
    # diagnostic the parsers flagged as warning-severity — DeprecationWarning,
    # PendingDeprecationWarning, ruff style nits, pip notices — none of these
    # block the build, and surfacing them to the repair LLM tempts it to
    # spend an iteration "fixing" something that wasn't broken. The unfiltered
    # list stays in state for observability; only the LLM input is trimmed.
    raw_errors: list[DiagnosticObjectDict] = state.get("compiler_errors", [])
    errors: list[DiagnosticObjectDict] = [
        e for e in raw_errors
        if str(e.get("severity", "error")).lower() != "warning"
    ]
    if len(errors) < len(raw_errors):
        logger.info(
            "[repair_node] Filtered %d warning-severity diagnostic(s); "
            "%d error(s) passed through to repair.",
            len(raw_errors) - len(errors), len(errors),
        )

    # --- Deterministic autofix pass (R1+R2+R3+R4+R5+R6) ---
    # Try to resolve diagnostics with compiler-suggested fixes,
    # missing-import insertion, known-safe security autofixes, or
    # web-asset reference rewrites BEFORE spending an LLM call. Anything
    # still unhandled falls through to the LLM exactly as before.
    from harness.autofix import (
        apply_autofixes,
        autofix_system_message,
        web_asset_diagnostics_to_standard,
    )
    workspace_path = state.get("workspace_path", os.getcwd())

    # Bridge: lintgate writes web_asset_errors as dicts in node_state. R6
    # consumes them through the same standardized diagnostic shape as every
    # other dispatcher. Convert and merge here so the autofix pass sees the
    # full pool in one loop.
    lintgate_state = state.get("node_state", {}).get("lintgate", {}) or {}
    web_asset_diags = web_asset_diagnostics_to_standard(
        lintgate_state.get("web_asset_errors", [])
    )
    combined_input: list[dict[str, Any]] = [dict(e) for e in errors] + list(web_asset_diags)
    unhandled, applied_fixes = await apply_autofixes(combined_input, workspace_path)
    # Strip any web-asset diagnostics that the LLM should NOT see in the
    # compiler-errors framing — they're already surfaced via the lint_errors
    # channel further up. Only compiler errors should fall through to the
    # LLM's compiler-style repair prompt.
    unhandled = [d for d in unhandled
                 if d.get("error_code") != "WEB_ASSET_REF"]
    autofix_modified_files = list(state.get("modified_files", []))
    autofix_messages = list(state.get("messages", []))
    if applied_fixes:
        for afr in applied_fixes:
            if afr.file not in autofix_modified_files:
                autofix_modified_files.append(afr.file)
        sys_msg = autofix_system_message(applied_fixes)
        if sys_msg:
            autofix_messages.append({"role": "system", "content": sys_msg})
        logger.info(
            "[repair_node] autofix resolved %d of %d diagnostic(s) without LLM.",
            len(applied_fixes), len(errors),
        )

    # Short-circuit: every diagnostic was resolved deterministically →
    # skip the LLM call entirely. Hand the routing back to compiler_node
    # for re-verification with the modified files in state.
    if applied_fixes and not unhandled:
        return {
            "messages": autofix_messages,
            "modified_files": autofix_modified_files,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "repair",
                "repair_context": "all diagnostics resolved by autofix",
                "repair_success": len(applied_fixes),
                "repair_fail": 0,
                "autofix": {
                    "applied": len(applied_fixes),
                    "fix_kinds": sorted({r.fix_kind for r in applied_fixes}),
                },
            },
        }

    # The LLM only sees the unhandled tail.
    errors = cast("list[DiagnosticObjectDict]", unhandled)

    # Promote the wider-context file-content snippets from prior patch
    # failures to a top-of-prompt section BEFORE the diagnostic block.
    # The LLM's anchoring behaviour benefits from seeing the actual
    # current file content first; sessions 19b28eff and 0a5c6fe8 stuck
    # in HITL because the LLM was anchoring on its own previous bad
    # search instead of the file content (which used to be buried inside
    # the patch-failures block at the bottom of the summary).
    error_summary = _format_current_file_content(
        state.get("node_state", {}).get("patch_failures") or []
    )
    # files_seen_by_llm carries (rel_path → sha256) for every file whose
    # bytes the LLM has been shown this turn (pre-flight inject + READ_FILE
    # resolves). Used by the patcher's B5 drift detector to reject patches
    # against a file that changed under the LLM's mental model. Seeded from
    # the prior node_state so multi-iteration sessions accumulate the record.
    files_seen_by_llm: dict[str, str] = dict(
        state.get("node_state", {}).get("files_seen_by_llm") or {}
    )
    # B1 pre-flight: on iter 1 (or any time there are no prior
    # patch_failures to source the closest-match window from), proactively
    # include line-numbered content for every file the diagnostics point
    # at. Claude Code gets this for free via Read-before-Edit; we have to
    # inject it explicitly. Without this the LLM hallucinates SEARCH
    # strings from its mental model of files it has never been shown —
    # root cause behind the 100-run failure streak on session 2d0164f0.
    if not error_summary:
        diag_files: list[str] = []
        for e in errors or []:
            f = (e.get("file") or "").strip() if isinstance(e, dict) else ""
            if not f or f == "<sandbox>" or f == "<test_runner>":
                continue
            if f not in diag_files:
                diag_files.append(f)
        if diag_files:
            preflight_pairs = _collect_workspace_file_content(
                workspace_path, diag_files,
                record_hashes_into=files_seen_by_llm,
            )
            error_summary += _format_preflight_file_content(
                preflight_pairs,
                intro=(
                    "These are the **actual current bytes** of the files "
                    "your diagnostics point at. The LLM has not patched "
                    "these yet (or the patcher applied them cleanly with "
                    "no miss), so there is no closest-match window from a "
                    "prior failure to anchor on. Use these line-numbered "
                    "views — WITHOUT the `  N| ` prefix — as the source of "
                    "truth for any REPLACE_BLOCK / DELETE_BLOCK search "
                    "you write."
                ),
            )
    # Fix #3: if any file has accumulated ≥ 2 consecutive REPLACE_BLOCK
    # misses, force the LLM out of the pattern with an explicit "use a
    # different operation" directive before the diagnostics.
    _rb_per_file_raw = loop_counter.get("replace_block_misses_per_file")
    error_summary += _format_replace_block_miss_directive(
        _rb_per_file_raw if isinstance(_rb_per_file_raw, dict) else {}
    )
    # Fix #5: when diagnostics point at test files but the error shape
    # looks like a production-cascade (ImportError / NameError / F821 /
    # ...), prepend a reframe + attach the most likely corresponding
    # production file content. Steers the LLM at root cause instead of
    # symptom-patching the test.
    error_summary += _format_test_collection_cascade_section(
        errors, workspace_path,
    )
    error_summary += _format_diagnostics_for_repair(errors)

    gateway = get_gateway()
    if gateway is None:
        logger.error("[repair_node] No gateway configured. Cannot call LLM.")
        return {
            "node_state": {"current_node": "repair", "error": "No gateway configured"},
            "loop_counter": loop_counter,
        }

    # Detect security-finding repair: every diagnostic carries an
    # error_code prefixed by the scanner name (BANDIT:, SEMGREP:, TRIVY:,
    # GITLEAKS:, GITLEAKS-FALLBACK:) because _findings_to_diagnostics in
    # harness/security.py builds them that way. When ALL diagnostics
    # carry such a prefix we know the repair is being driven by the
    # security gate, not a compile failure, and we swap in framing that
    # tells the LLM to fix the vulnerability root cause rather than
    # patch over a build error.
    _SECURITY_PREFIXES = ("BANDIT:", "SEMGREP:", "TRIVY:", "GITLEAKS:", "GITLEAKS-FALLBACK:")
    is_security_repair = bool(errors) and all(
        str(e.get("error_code", "")).upper().startswith(_SECURITY_PREFIXES) for e in errors
    )

    # Detect repair driven by harness-generated test failures. The
    # test_generation_node tags each diagnostic with an error_code starting
    # with "TEST_FAILURE" so we can swap in framing that tells the LLM these
    # are unit-test failures (not compile errors) and that fixing the
    # implementation is preferred over weakening the test assertion.
    is_test_failure_repair = bool(errors) and all(
        str(e.get("error_code", "")).upper().startswith("TEST_FAILURE") for e in errors
    )

    # Include lintgate errors from the previous compiler run
    lintgate_state = state.get("node_state", {}).get("lintgate", {})
    lint_errors_list: list[str] = lintgate_state.get("lint_errors", [])
    if lint_errors_list:
        lint_summary = "\n## Lint Gate Errors\n" + "\n".join(f"  - {e}" for e in lint_errors_list)
        error_summary += lint_summary

    # If the previous patching/repair attempt had any allowlist rejections,
    # surface them so the LLM stops re-proposing the same blocked paths. The
    # repair LLM otherwise has no way to know its patches keep being thrown
    # away by the path filter.
    prior_rejections = state.get("node_state", {}).get("allowlist_rejections") or []
    if prior_rejections:
        prior_allowed = state.get("node_state", {}).get("allowed_paths") or []
        prior_rejected_paths: list[str] = sorted({str(r.get("file", "")) for r in prior_rejections if r.get("file")})
        rejection_block = (
            "\n## Allowlist Rejections (PREVIOUS attempt)\n"
            "Your last attempt produced patches targeting paths the patcher's "
            "skill allowlist rejected. These patches did NOT land on disk. "
            "Do NOT re-propose the same paths verbatim — relocate the file or "
            "use one of the allowed roots.\n"
            f"Rejected: {prior_rejected_paths}\n"
            f"Allowed roots: {prior_allowed}\n"
        )
        error_summary += rejection_block

    # Surface non-allowlist patch failures from the previous attempt — the
    # patcher's "Search block not found ... Closest match: <bytes>" suggestion
    # is the single most useful signal the LLM has to correct a bad patch.
    # Without this the loop just retries the same broken search block.
    error_summary += _format_prior_patch_failures(
        state.get("node_state", {}).get("patch_failures") or []
    )

    # Workspace inventory: the single biggest cause of stuck repair loops is
    # the LLM CREATE_FILE-ing files that already exist (from the initial
    # patching pass or from a salvaged speculative variant). The patcher
    # rejects those with "File already exists with different content" and
    # the LLM, having no idea what's on disk, keeps emitting the same
    # CREATE_FILE next round. Surfacing the current modified_files list
    # alongside the system-prompt rule about CREATE_FILE-vs-REPLACE_BLOCK
    # kills the guessing.
    inventory_files = sorted({p for p in (state.get("modified_files") or []) if p})
    if inventory_files:
        SNAPSHOT_CAP = 50
        shown = inventory_files[:SNAPSHOT_CAP]
        extra = max(0, len(inventory_files) - SNAPSHOT_CAP)
        error_summary += (
            "\n## Files currently in workspace\n"
            "These files have been created or modified earlier in this "
            "session and now exist on disk. **CREATE_FILE on any of them "
            "will be REJECTED** — use REPLACE_BLOCK to modify them:\n"
            + "\n".join(f"- {p}" for p in shown)
        )
        if extra:
            error_summary += f"\n- (+ {extra} more not shown)"
        error_summary += "\n"

    # Current allowlist snapshot. The system prompt is anchored at messages[0]
    # and never refreshed, so on greenfield starts the LLM saw "no layout
    # constraint" but by iteration 2 the workspace has a materialised source
    # root the patcher will now enforce. Re-detecting the allowlist here and
    # surfacing it unconditionally (not only when there are rejections) lets
    # the LLM see the same rules the patcher applies *this round*.
    try:
        current_allowed = _build_patcher_allowlist(workspace_path)
    except Exception:  # noqa: BLE001 — diagnostics, never let this fail repair
        current_allowed = None
    if current_allowed:
        error_summary += (
            "\n## Allowed roots (current)\n"
            "The patcher will reject any CREATE_FILE / REPLACE_BLOCK targeting "
            "paths outside these roots:\n"
            + "\n".join(f"- {p}" for p in current_allowed)
            + "\n"
        )
    else:
        error_summary += (
            "\n## Allowed roots (current)\n"
            "Workspace layout is unconstrained this round; the patcher only "
            "enforces path-traversal safety. Place new modules under whatever "
            "package directory matches the existing files in the workspace "
            "inventory above.\n"
        )

    # If no structured diagnostics, include the raw build output so the LLM can see the actual error
    if not errors:
        raw_output = state.get("node_state", {}).get("last_build_output", "")
        if raw_output:
            error_summary += f"\n## Raw Build Output\n```\n{_slice_build_output_for_repair(raw_output)}\n```"

    # When any diagnostic is MISSING_DEP, attach the current contents of the
    # dependency manifest the build command references. Without this the LLM
    # tries CREATE_FILE (which fails — "already exists with different content")
    # then REPLACE_BLOCK with a guessed search string that doesn't match.
    # Autofix R4 (`_try_missing_dep`) handles requirements.txt automatically;
    # this block covers the pyproject / package.json fallback path where
    # autofix deliberately defers to the LLM.
    workspace_for_manifest = state.get("workspace_path", os.getcwd())
    manifest_attachments = _collect_manifest_snippets_for_repair(
        errors, workspace_for_manifest,
    )
    if manifest_attachments:
        error_summary += manifest_attachments

    try:
        from harness.gateway import NodeRole
        from harness.patcher import process_llm_patch_output

        # Use the autofix-augmented messages list so the LLM sees the
        # "we already fixed X" system message and doesn't re-fix.
        messages = list(autofix_messages)
        budget = state.get("budget_remaining_usd", 2.00)

        # --- Cross-Model Speculative Execution ---
        # Use the cheap model (repair_primary) for the first N-1 attempts;
        # escalate to the heavy reasoning model (repair_fallback) for the
        # LAST attempt before HITL. This saves tokens by only spending the
        # expensive model when the cheap one has had every chance.
        # N comes from the operator's config (node_throttle.max_patch_repair_iterations).
        total_repairs = loop_counter["total_repairs"]
        max_repair_attempts = int(getattr(gateway.config, "max_patch_repair_iterations", 5))
        use_escalation = total_repairs >= max(1, max_repair_attempts - 1)

        if use_escalation:
            escalation_model = gateway.config.repair_fallback or gateway.config.planning_fallback
            primary_model = gateway.config.repair_primary
            if escalation_model and escalation_model != primary_model:
                logger.warning(
                    "[repair_node] Cheap model failed %d time(s). Escalating to reasoning model: %s",
                    total_repairs - 1,
                    escalation_model,
                )
                # Escalated repair will use NodeRole.REPAIR with thinking mode enabled
            elif escalation_model:
                # Operator configured the same model for primary AND fallback
                # — "escalating" would just re-run the same model. Skip the
                # model swap so the misleading "Escalating to <same model>"
                # banner never fires; thinking mode is already enabled for
                # the repair role via repair_mode config.
                logger.info(
                    "[repair_node] %d previous attempt(s) on the cheap model; "
                    "repair_fallback==repair_primary (%s), so re-running same "
                    "model with role=repair (thinking mode honors repair_mode).",
                    total_repairs - 1, primary_model,
                )
                use_escalation = False
            else:
                logger.warning(
                    "[repair_node] Cheap model failed %d time(s), but no escalation model configured. "
                    "Continuing with primary repair model.",
                    total_repairs - 1,
                )
                use_escalation = False
        else:
            escalation_model = None

        # Inject the error summary as a user message for context. The
        # framing sentence differs across three cases:
        #   1. Security gate flagged vulnerabilities → tell the LLM these
        #      are post-build security findings (build passes!) so it
        #      doesn't try to "fix a broken build" by removing tests or
        #      catching exceptions broadly. Emphasise minimum diff and
        #      not weakening other controls.
        #   2. Build failed and the cheap model has already missed twice
        #      → escalate to the reasoning model with reasoning framing.
        #   3. Build failed, first or second attempt → standard framing.
        if is_security_repair:
            repair_prompt = (
                "The deterministic security gate flagged the following vulnerabilities "
                "in code that has already passed the build. Generate precise SEARCH/REPLACE "
                "patches that REMOVE the root cause without regressing existing tests. "
                "Prefer the minimum diff: do not refactor unrelated code, do not weaken "
                "the security control elsewhere, and if a finding requires a dependency "
                "upgrade, write the new version into the manifest rather than vendoring "
                f"a patched copy.\n\n{error_summary}"
            )
        elif is_test_failure_repair:
            repair_prompt = (
                "The harness-generated unit tests just failed when executed in the "
                "sandbox. These are NOT compile errors — the code builds. For each "
                "failure, decide whether the implementation is wrong or the test "
                "expectation is wrong. Default to fixing the implementation when the "
                "behaviour was specified in the requirements; only adjust the test "
                "when the expectation itself contradicts the spec. Do NOT add mocks "
                "to make a test pass — if a test cannot be exercised without external "
                "dependencies, rewrite it to use the test runner's built-in fakes "
                "(monkeypatch / tmp_path / httptest / @TempDir) instead."
                f"\n\n{error_summary}"
            )
        elif use_escalation:
            repair_prompt = (
                f"The build has failed {total_repairs} time(s) despite previous fix attempts. "
                f"The simpler model could not resolve these errors. You are a senior reasoning model. "
                f"Carefully analyze the errors and produce a definitive fix.\n\n{error_summary}"
            )
        else:
            repair_prompt = (
                f"The build failed with the following errors. Generate precise SEARCH/REPLACE "
                f"patches to fix them.\n\n{error_summary}"
            )
        # Soft turn-budget warning (audit #19). Injected as a system
        # message so the LLM treats it with authority. Fires only on the
        # last two repair iterations; quiet otherwise.
        budget_warning = _repair_budget_warning(total_repairs, max_repair_attempts)
        if budget_warning is not None:
            messages.append(MessageDict(role="system", content=budget_warning))
            try:
                from harness.observability import emit_event
                emit_event(
                    "repair_budget_warning",
                    total_repairs=total_repairs,
                    cap=max_repair_attempts,
                    remaining=max_repair_attempts - total_repairs,
                )
            except Exception:  # noqa: BLE001 — telemetry must not block
                pass

        # Append the repair prompt first
        messages.append(MessageDict(role="user", content=repair_prompt))
        # Then append the strict format reminder (same as patching_node).
        # In change-request mode prepend the CR-N attribution rules so
        # repair patches also carry the marker comments.
        _REPAIR_FORMAT_REMINDER = _build_change_request_preamble(
            state, "patching"
        ) + """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the patch block syntax below. Do NOT include any explanations,
markdown code fences, or text outside the blocks. Your entire response must be parseable
as one or more patch blocks.

<<<REPLACE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to find>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>

<<<CREATE_FILE>>>
file: path/to/file.ext
content:
<complete file contents>
<<<END_CREATE_FILE>>>

Quality: Write modular, production-ready code with proper error handling, type hints, and docstrings. Handle edge cases.
Generate your fix patches NOW. Only the blocks above. No other text."""
        messages.append({"role": "user", "content": _REPAIR_FORMAT_REMINDER})

        # Universal LLM-call dump (config.debug.dump_llm_calls) writes the
        # input messages + response to ~/.harness/debug after the dispatch
        # returns. Handled centrally in Gateway.dispatch — no longer the
        # repair-node's responsibility. Filenames now look like
        # <sid>_<seqno>_<role>_<model>.txt and cover ALL roles.

        # Use the non-mutating model_override path so concurrent dispatches
        # don't see each other's transient config mutations and exceptions
        # don't leave gateway.config in an inconsistent state.
        async def _dispatch_repair(
            cur_messages: list[MessageDict], cur_budget: float,
        ) -> tuple[Any, float]:
            if use_escalation and escalation_model:
                return await gateway.dispatch(
                    messages=list(cur_messages),
                    role=NodeRole.REPAIR,
                    budget_remaining_usd=cur_budget,
                    model_override=escalation_model,
                )
            return await gateway.dispatch(
                messages=list(cur_messages),
                role=NodeRole.REPAIR,
                budget_remaining_usd=cur_budget,
            )

        workspace = state.get("workspace_path", os.getcwd())
        response, new_budget = await _dispatch_repair(messages, budget)

        # Continuation on finish_reason=="length" — opt-in via
        # llm_dispatch.continue_on_length.repair. Cost-aware: repair
        # already dispatches once per repair iteration plus an inner
        # READ_FILE round, so enabling this multiplies dispatches by
        # up to 3×. See _llm_dispatch_comment in config/config.json.
        response, new_budget, _repair_chunks = await _continue_on_length(
            initial_response=response,
            initial_budget=new_budget,
            messages=messages,
            dispatch=_dispatch_repair,
            continue_prompt=(
                "You hit the output token cap mid-repair. Continue "
                "with ADDITIONAL CREATE_FILE / REPLACE_BLOCK / "
                "DELETE_BLOCK / INSERT_AT_BLOCK blocks for the "
                "remaining files. Do NOT repeat blocks you've already "
                "emitted. Same DSL rules as before — block syntax "
                "only, no prose outside blocks."
            ),
            enabled=_resolve_continue_on_length(state, "repair"),
            role_label="repair_node",
        )
        # Splice the accumulated chunks into the final response so
        # downstream READ_FILE parsing + the patcher see every block
        # across continuation cycles. LLMResponse is a mutable
        # dataclass — see harness/gateway.py:LLMResponse.
        if len(_repair_chunks) > 1:
            response.content = "\n".join(c for c in _repair_chunks if c)

        # READ_FILE inline resolve (B3): if the LLM emitted READ_FILE blocks
        # instead of (or alongside) patch blocks, resolve them here without
        # consuming a repair iteration. The model fans out a single
        # "show me the file → write the patch" round in one logical turn,
        # matching how Claude Code's Read-before-Edit feels. Capped at
        # READ_FILE_MAX_RESOLVES so the LLM can't loop forever asking for
        # files instead of patching.
        from harness.patcher import (
            parse_read_blocks as _parse_read_blocks,
            strip_read_blocks as _strip_read_blocks,
        )
        READ_FILE_MAX_RESOLVES = 2
        for _resolve_round in range(READ_FILE_MAX_RESOLVES):
            read_reqs = _parse_read_blocks(response.content)
            if not read_reqs:
                break
            # Persist the LLM's READ_FILE request as an assistant turn so the
            # next dispatch sees what was asked, and inject our resolution as
            # a follow-up user message.
            messages.append(MessageDict(
                role="assistant", content=response.content,
            ))
            resolution = _resolve_read_blocks(
                read_reqs, workspace,
                record_hashes_into=files_seen_by_llm,
            )
            messages.append(MessageDict(role="user", content=resolution))
            logger.info(
                "[repair_node] READ_FILE resolved for %d file(s): %s. "
                "Re-dispatching without consuming an iteration.",
                len(read_reqs), [r[0] for r in read_reqs],
            )
            response, new_budget = await _dispatch_repair(messages, new_budget)
        # If the LLM is still emitting READ_FILE after the cap, ignore them
        # (strip below) and let the rest of the response apply. The cap log
        # above is the operator's signal that the loop got chatty.

        # Update token tracker (per-stage attribution: repair). Reflects
        # the FINAL response after any READ_FILE resolutions.
        token_tracker = state.get("token_tracker", {})
        token_tracker = gateway.aggregate_tokens(
            token_tracker, response.usage, role=NodeRole.REPAIR,
        )

        # Apply the fix patches to disk. Seed the modified-files list with
        # files the autofix pass already touched so they survive the LLM
        # round-trip into state. Same source-root allowlist as patching_node
        # so the repair LLM can't widen the surface area by writing new
        # modules outside the configured layout.
        existing_modified = list(autofix_modified_files)
        allowed_paths = _build_patcher_allowlist(workspace)
        # Strip any residual READ_FILE blocks so the patcher doesn't try to
        # parse them as patches and so commit messages stay clean.
        patch_payload = _strip_read_blocks(response.content)
        # B5 enforce flag: opt-in via gateway config. False by default so we
        # don't break callers that haven't been updated to emit READ_FILE.
        gw_cfg = getattr(gateway, "config", None)
        enforce_read = bool(
            getattr(gw_cfg, "enforce_read_before_edit", False)
        )
        patch_results, modified_files = await process_llm_patch_output(
            patch_payload,
            workspace,
            existing_modified,
            allowed_paths=allowed_paths,
            files_seen_by_llm=files_seen_by_llm,
            enforce_read_before_edit=enforce_read,
        )

        # Append the LLM response to messages
        messages.append(MessageDict(role="assistant", content=response.content))

        # Report results
        success_count = sum(1 for r in patch_results if r.success)
        fail_count = len(patch_results) - success_count
        # Real successes exclude resume-safe idempotency no-ops (file already
        # at target state). The repair loop's consecutive-zero tripwire and
        # the per-iteration commit must look at real progress, not at the LLM
        # re-emitting already-applied patches. See harness/patcher.py for the
        # five no-op return sites (CREATE_FILE/REPLACE_BLOCK/DELETE_BLOCK and
        # INSERT_AT_BLOCK BEFORE/AFTER).
        no_op_count = sum(
            1 for r in patch_results if r.success and getattr(r, "no_op", False)
        )
        real_success_count = success_count - no_op_count
        # Track allowlist rejections so the *next* repair iteration sees the
        # exact paths and reason and stops re-proposing them. Without this,
        # the LLM has no signal that its patches keep vanishing.
        allowlist_rejections = [
            {"file": r.file, "operation": r.operation, "reason": r.error}
            for r in patch_results
            if not r.success and isinstance(r.error, str)
            and "not in skill allowlist" in r.error
        ]
        # Capture the remaining failures (search-block-not-found,
        # file-already-exists, etc.) so the *next* repair iteration sees the
        # full error including the patcher's closest-match suggestion. The
        # LLM otherwise only sees "Failed: foo.txt" and proposes the same
        # bad patch again — see the requirements.txt loop in the issue logs.
        # _store_patch_failure_error keeps wider-context messages whole
        # (the file-content window the LLM needs to write a correct
        # SEARCH block) and caps everything else at 3000 chars.
        patch_failures = [
            {
                "file": r.file,
                "operation": (
                    r.operation.value
                    if hasattr(r.operation, "value") else str(r.operation)
                ),
                "error": _store_patch_failure_error(r.error),
            }
            for r in patch_results
            if not r.success and isinstance(r.error, str)
            and "not in skill allowlist" not in r.error
        ][:5]
        status_msg = f"[System]: Repair attempt {loop_counter['total_repairs']}: applied {success_count}/{len(patch_results)} patches."
        if no_op_count > 0:
            status_msg += (
                f" {no_op_count} were idempotency no-ops (target file already "
                f"at expected state — no actual change made)."
            )
        if fail_count > 0:
            failed_files = [r.file for r in patch_results if not r.success]
            status_msg += f" Failed: {', '.join(failed_files)}."
        if allowlist_rejections:
            rejected_paths = ", ".join(sorted({str(r["file"]) for r in allowlist_rejections}))
            status_msg += (
                f"\n[Allowlist] Rejected paths outside the configured layout: "
                f"{rejected_paths}. Allowed roots: {allowed_paths}."
            )
        messages.append(MessageDict(role="system", content=status_msg))

        # Consecutive-zero-patch tripwire: track how many repair rounds in
        # a row landed zero patches. Two in a row means the loop is stuck
        # (LLM keeps emitting bad blocks, patcher keeps rejecting them) and
        # further iterations just burn budget. route_after_compiler short-
        # circuits to HITL when this counter hits the threshold.
        #
        # Use real_success_count (success minus no-ops) so a DELETE_BLOCK on
        # an already-deleted file — or any other idempotency no-op — does NOT
        # reset the tripwire. Without this guard the LLM can mask a stuck
        # loop by re-emitting patches that have nothing to apply.
        if real_success_count == 0:
            loop_counter["consecutive_zero_patch_rounds"] = (
                loop_counter.get("consecutive_zero_patch_rounds", 0) + 1
            )
        else:
            loop_counter["consecutive_zero_patch_rounds"] = 0

        # Per-file REPLACE_BLOCK miss tracker (fix #3). Bump for each
        # failed REPLACE_BLOCK, clear for any file with a successful
        # operation this round. The next iteration's prompt will direct
        # the LLM to use a different operation on any file at ≥ 2
        # consecutive misses — see _format_replace_block_miss_directive.
        _rb_raw = loop_counter.get("replace_block_misses_per_file", {})
        rb_misses: dict[str, int] = dict(_rb_raw) if isinstance(_rb_raw, dict) else {}
        for r in patch_results:
            op_str = (
                r.operation.value if hasattr(r.operation, "value") else str(r.operation)
            )
            if op_str != "replace_block":
                continue
            if r.success:
                rb_misses.pop(r.file, None)
            elif (
                isinstance(r.error, str)
                and "not in skill allowlist" not in r.error
            ):
                rb_misses[r.file] = rb_misses.get(r.file, 0) + 1
        loop_counter["replace_block_misses_per_file"] = rb_misses

        # Audit §6.4: surface cross-file impact warnings after repair
        # patches too. Best-effort.
        if modified_files:
            try:
                from harness.impact import ImpactAnalyzer as _ImpactAnalyzer
                _impact_raw = state.get("impact_config") if isinstance(state, dict) else None
                impact_cfg: dict[str, Any] = _impact_raw if isinstance(_impact_raw, dict) else {}
                _impact = _ImpactAnalyzer(
                    workspace_path=workspace,
                    max_scan_files=int(impact_cfg.get("max_scan_files", 500)),
                    enabled=bool(impact_cfg.get("enabled", True)),
                )
                _impact.analyze_and_warn(list(modified_files), messages)  # type: ignore[arg-type]
            except Exception as _impact_exc:  # noqa: BLE001
                logger.debug("[impact] analysis failed (%s); skipping warning.", _impact_exc)

            # Audit §6.5: refresh the repo index for changed files.
            try:
                from harness.repo_index import (
                    update_index_for_files as _update_idx,
                    RepoIndexConfig as _RIConfig,
                )
                idx_cfg_dict = (state.get("repo_index_config") or {}) if isinstance(state, dict) else {}
                if idx_cfg_dict:
                    _idx_cfg = _RIConfig(**{
                        k: v for k, v in idx_cfg_dict.items()
                        if k in _RIConfig.__dataclass_fields__
                    })
                else:
                    _idx_cfg = _RIConfig()
                _update_idx(workspace, list(modified_files), _idx_cfg)
            except Exception as _idx_exc:  # noqa: BLE001
                logger.debug("[repo_index] incremental update failed: %s", _idx_exc)

        # Per-iteration commit (C3): when this round landed any patches,
        # commit the working tree with a structured per-iteration message
        # so the operator can `git log` / `git bisect` between iterations
        # and individual rounds are easy to revert. Best-effort: failures
        # are logged + swallowed inside commit_repair_iteration, which is
        # also a no-op when we're not on a harness patch branch.
        # Gate on real_success_count to skip empty commits when this round
        # only landed idempotency no-ops on already-correct files.
        if real_success_count > 0:
            try:
                from harness.security import GitGuardian as _GitGuardian
                session_id = state.get("session_id", "unknown")
                _GitGuardian(workspace).commit_repair_iteration(
                    session_id=session_id,
                    iteration=loop_counter["total_repairs"],
                    modified_files=list(modified_files),
                    success_count=success_count,
                    fail_count=fail_count,
                    exit_code=int(state.get("exit_code", -1)),
                )
            except Exception as exc:  # noqa: BLE001 — never let commit break the loop
                logger.debug("[repair_node] Per-iteration commit skipped: %s", exc)

        logger.info(
            "[repair_node] Repair #%d complete. tokens_in=%d tokens_out=%d cost=$%.6f budget_left=$%.4f "
            "patches=%d succeed=%d no_op=%d fail=%d consecutive_zero=%d",
            loop_counter["total_repairs"],
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost_usd,
            new_budget,
            len(patch_results), real_success_count, no_op_count, fail_count,
            loop_counter["consecutive_zero_patch_rounds"],
        )
        _emit_per_stage_spend_summary(token_tracker)

        return {
            "messages": messages,
            "modified_files": modified_files,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                **(state.get("node_state") or {}),
                "current_node": "repair",
                "repair_context": error_summary,
                "repair_success": success_count,
                "repair_fail": fail_count,
                "allowlist_rejections": allowlist_rejections,
                "patch_failures": patch_failures,
                "allowed_paths": allowed_paths,
                # B5: persist what the LLM has been shown so the next
                # repair iteration's drift detector has the full record.
                "files_seen_by_llm": files_seen_by_llm,
            },
        }
    except RuntimeError as exc:
        # Distinguish empty-response (P1.5) and budget-too-low (P1.4) from
        # the generic budget-exhausted case so the HITL router can surface
        # a precise message instead of pretending budget was depleted.
        from harness.gateway import EmptyLLMResponseError, BudgetTooLowError
        if isinstance(exc, EmptyLLMResponseError):
            logger.warning("[repair_node] LLM returned empty content: %s", exc)
            return {
                "node_state": {
                    "current_node": "repair",
                    "error": str(exc),
                    "llm_silent": True,
                    "hitl_trigger": "llm_silent",
                },
                "loop_counter": loop_counter,
            }
        if isinstance(exc, BudgetTooLowError):
            logger.warning("[repair_node] Pre-flight budget refusal: %s", exc)
            return {
                "node_state": {
                    "current_node": "repair",
                    "error": str(exc),
                    "budget_exhausted": True,
                    "hitl_trigger": "budget_preflight",
                },
                "loop_counter": loop_counter,
            }
        logger.warning("[repair_node] Gateway refused during repair: %s", exc)
        return {
            "node_state": {"current_node": "repair", "error": str(exc), "budget_exhausted": True},
            "loop_counter": loop_counter,
        }
    except Exception as exc:
        logger.exception("[repair_node] Unexpected error during repair.")
        return {
            "node_state": {"current_node": "repair", "error": str(exc)},
            "loop_counter": loop_counter,
        }


async def human_intervention_node(state: AgentState) -> dict[str, Any]:
    """
    Node 5: The Breakpoint.

    Terminal HITL node invoked when:
        - Budget is exhausted ($2.00 cap breached)
        - Repair loop counter hits the throttle limit (3 iterations)
        - Any other guardrail violation

    Presents an interactive stdin menu to the developer:
        [v] View diffs
        [r] Resume (re-run compiler)
        [e] Inject hint for the repair node
        [m] Pause for manual IDE edits
        [b] Increase budget (+$2.00)
        [q] Abandon and git rollback

    Delegates to the CLI layer's hitl_menu_loop for actual user interaction.
    The menu blocks until the developer makes a choice, then returns an
    updated state dict with routing signals.
    """
    logger.info("[human_intervention_node] Triggering HITL breakpoint...")

    # Determine why we were invoked
    loop_counter = state.get("loop_counter", {})
    budget_remaining = state.get("budget_remaining_usd", 0.0)

    # Same source of truth as route_after_compiler — the operator's
    # node_throttle.max_patch_repair_iterations in config.json. Keeping
    # the two reads identical means a config change can never produce a
    # mismatch where the router decided "you hit the limit" but the HITL
    # banner says something else.
    gw = get_gateway()
    max_repair = (
        int(getattr(gw.config, "max_patch_repair_iterations", 5))
        if gw is not None else 3
    )
    trigger_reason = "unknown"
    if state.get("node_state", {}).get("env_misconfig"):
        sym = state.get("node_state", {}).get("env_misconfig_symbol", "")
        trigger_reason = f"env_misconfig:{sym}" if sym else "env_misconfig"
    elif budget_remaining <= 0.0:
        trigger_reason = "budget_exhausted"
    elif loop_counter.get("total_repairs", 0) >= max_repair:
        trigger_reason = "repair_loop_limit"
    elif state.get("exit_code", -1) != 0:
        trigger_reason = "persistent_build_failure"

    # Inject trigger reason into state so the menu can display it
    state_dict: dict[str, Any] = dict(state)
    state_dict["node_state"] = dict(state_dict.get("node_state") or {})
    state_dict["node_state"]["current_node"] = "human_intervention"
    state_dict["node_state"]["hitl_trigger"] = trigger_reason
    state_dict["node_state"]["hitl_active"] = True
    state_dict["node_state"]["hitl_awaiting_input"] = True

    # Delegate to the CLI layer's interactive menu loop.
    # This blocks on stdin until the developer makes a choice.
    from harness.cli import hitl_menu_loop

    updated_state = hitl_menu_loop(state_dict)

    # Extract the node_state back — hitl_menu_loop returns a full state dict
    return updated_state


# ---------------------------------------------------------------------------
# 5. Helper Utilities
# ---------------------------------------------------------------------------

def _format_diagnostics_for_repair(errors: list[DiagnosticObjectDict]) -> str:
    """Format structured diagnostics into a concise, **grouped** repair prompt.

    Without grouping, 16 occurrences of "F821 Undefined name 'pytest'" across
    16 test files render as 16 separate Error N: blocks — the LLM sees noise
    instead of a pattern. We group by (error_code, message) so the LLM sees
    the diagnostic shape once and a count of the files it affects. The first
    occurrence's semantic_context is shown (the others would mostly repeat).
    Groups are emitted in original error order so the LLM still gets a
    stable "fix this one first" cue, but with explicit counts.
    """
    if not errors:
        return "No structured diagnostics available. Check raw build output."

    # Group preserving first-seen order. Key is (error_code, message); the
    # group's first error keeps its full context for the LLM.
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for err in errors:
        code = str(err.get("error_code", "UNKNOWN"))
        msg = str(err.get("message", "No message"))
        key = (code, msg)
        if key not in groups:
            groups[key] = {
                "code": code,
                "message": msg,
                "severity": str(err.get("severity", "error")),
                "first": err,
                "locations": [],
            }
        groups[key]["locations"].append(
            f"{err.get('file', '?')}:{err.get('line', 0)}:{err.get('column', 0)}"
        )

    # Rank groups by likely cascade impact so the top-N the LLM sees are the
    # ones whose fix is most likely to make other errors disappear.
    # Heuristic order:
    #   1. Severity error > warning (warnings don't break the build).
    #   2. "Upstream" kinds first — undefined names, missing imports, missing
    #      deps. F821/F401 are pyflakes-shaped; ImportError /
    #      ModuleNotFoundError surface from pytest. Fix one of these and
    #      multiple downstream diagnostics often vanish at once.
    #   3. Original first-seen position breaks ties so the LLM still gets a
    #      stable order across iterations.
    _UPSTREAM_PREFIXES = (
        "F821", "F401", "E0001", "E0401", "E0602", "E1101",
        "MISSING_DEP", "MISSING_IMPORT", "IMPORTERROR", "MODULENOTFOUND",
        "SYNTAXERR", "TEST_FAILURE:IMPORTERROR",
    )

    def _severity_rank(g: dict[str, Any]) -> int:
        return 0 if str(g.get("severity", "error")).lower() == "error" else 1

    def _kind_rank(g: dict[str, Any]) -> int:
        code = str(g.get("code", "")).upper()
        return 0 if any(code.startswith(p) for p in _UPSTREAM_PREFIXES) else 1

    # Fix #4: cascade detection. When a group has ≥ 3 occurrences and
    # every location lives under tests/ (or test/ / __tests__/), the
    # group is almost certainly a single production-code bug rippling
    # through every test file that imports it. Tag the group so the
    # render loop can emit a "cascade hint" pointing the LLM at
    # production code first. Same for groups whose message references
    # one identifier (e.g. `Undefined name 'Job'` × 5) — fix the symbol
    # in production code and every diagnostic vanishes at once.
    import re as _re
    _SYMBOL_PATTERN = _re.compile(r"['`]([A-Za-z_][A-Za-z0-9_.]*)['`]")

    def _all_in_tests(group: dict[str, Any]) -> bool:
        locs = group.get("locations", [])
        if not locs:
            return False
        return all(
            any(str(loc).startswith(p) for p in _TEST_DIR_PREFIXES)
            for loc in locs
        )

    def _shared_symbol(group: dict[str, Any]) -> Optional[str]:
        msg = str(group.get("message", ""))
        m = _SYMBOL_PATTERN.search(msg)
        return m.group(1) if m else None

    for group in groups.values():
        count = len(group.get("locations", []))
        if count < 3:
            continue
        sym = _shared_symbol(group)
        all_tests = _all_in_tests(group)
        if all_tests and sym:
            group["cascade_hint"] = (
                f"This pattern hits {count} test files and references the "
                f"symbol `{sym}`. Root cause is almost certainly in the "
                f"production module that should export `{sym}` — fix it "
                f"there first instead of editing the test files. Every "
                f"diagnostic in this group will resolve at once when the "
                f"production-side definition is correct."
            )
        elif all_tests:
            group["cascade_hint"] = (
                f"This pattern hits {count} test files only. Root cause is "
                f"likely a single production-side error that cascades through "
                f"every test that imports the affected module. Check the "
                f"production source before editing any of these test files."
            )
        elif sym and count >= 3:
            group["cascade_hint"] = (
                f"This diagnostic references symbol `{sym}` in {count} places. "
                f"If the symbol is supposed to come from one module, fixing "
                f"that module will collapse all {count} diagnostics. Check "
                f"that the symbol's source-of-truth definition is correct first."
            )

    group_list = list(groups.values())
    ranked = sorted(
        enumerate(group_list),
        key=lambda pair: (_severity_rank(pair[1]), _kind_rank(pair[1]), pair[0]),
    )
    TOP_N = 3
    shown = [g for _, g in ranked[:TOP_N]]
    hidden = [g for _, g in ranked[TOP_N:]]

    lines: list[str] = [
        f"## Compiler Diagnostics ({len(errors)} total, "
        f"{len(groups)} distinct shape{'s' if len(groups) != 1 else ''})\n"
    ]
    if hidden:
        lines.append(
            f"_Showing the top {len(shown)} of {len(groups)} groups ranked by "
            f"likely cascade impact (fixing an upstream undefined-name / "
            f"missing-import often resolves multiple downstream errors). "
            f"Address these first; the {len(hidden)} other group(s) may "
            f"resolve on their own._\n"
        )
    for i, group in enumerate(shown, 1):
        locs = group["locations"]
        count = len(locs)
        # Show up to 4 locations per group so the LLM sees the spread without
        # a 16-line dump. The leading location is always shown verbatim.
        if count == 1:
            loc_display = f"`{locs[0]}`"
        else:
            head = ", ".join(f"`{loc}`" for loc in locs[:4])
            tail = "" if count <= 4 else f" (+ {count - 4} more)"
            loc_display = head + tail
        lines.append(
            f"**Error {i}:** `{group['code']}` × {count} "
            f"[{group['severity']}]"
        )
        lines.append(f"  Message: {group['message']}")
        lines.append(f"  Locations: {loc_display}")
        cascade = group.get("cascade_hint")
        if cascade:
            lines.append(f"  **Cascade hint:** {cascade}")
        context = (group["first"].get("semantic_context") or "").strip()
        if context:
            lines.append(f"  Context (first occurrence):\n```\n{context}\n```")
    if hidden:
        # One-line summary of the deferred groups so the LLM knows they exist
        # without bloating the prompt with full context blocks. Each entry
        # lists code + a short message excerpt + count.
        tail_lines = ["", f"### Deferred ({len(hidden)} group(s) — fix the top {len(shown)} first):"]
        for g in hidden:
            msg = g["message"]
            if len(msg) > 80:
                msg = msg[:77] + "..."
            tail_lines.append(
                f"- `{g['code']}` × {len(g['locations'])}: {msg}"
            )
        lines.extend(tail_lines)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Conditional Edge: Compiler → Next Node
# ---------------------------------------------------------------------------

def route_after_compiler(state: AgentState) -> Literal["repair_node", "human_intervention_node", "security_scan_node", "test_generation_node"]:
    """
    Conditional edge router executed after compiler_node completes.

    Decision matrix:
        exit_code == 0                     → END (memory cleanse already applied in compiler_node)
        exit_code != 0 AND repairs < 3     → repair_node
        exit_code != 0 AND repairs >= 3    → human_intervention_node
        budget_remaining <= 0              → human_intervention_node
        no_tests_collected AND has_source  → test_generation_node
        no_tests_collected AND empty repo  → human_intervention_node
    """
    exit_code: int = state.get("exit_code", -1)
    loop_counter: dict[str, Any] = state.get("loop_counter", {})
    budget_remaining: float = state.get("budget_remaining_usd", 0.0)
    total_repairs: int = loop_counter.get("total_repairs", 0)
    # Read the repair-loop limit from the operator's config (single source:
    # node_throttle.max_patch_repair_iterations). Falls back to the historical
    # default of 3 when the gateway hasn't been initialised — only happens in
    # narrow test scenarios that construct the router in isolation.
    gw = get_gateway()
    max_iterations: int = (
        int(getattr(gw.config, "max_patch_repair_iterations", 5))
        if gw is not None else 3
    )

    _Dest = Literal["repair_node", "human_intervention_node", "security_scan_node", "test_generation_node"]

    def _transition(dest: _Dest) -> _Dest:
        try:
            from harness.observability import emit_event
            emit_event("node_transition",
                       from_node="compiler_node", to_node=dest,
                       exit_code=exit_code, total_repairs=total_repairs,
                       budget_remaining_usd=budget_remaining)
        except Exception:  # noqa: BLE001
            pass
        return dest

    # Check budget first — financial guardrail takes priority
    if budget_remaining <= 0.0:
        logger.warning("[router] Budget exhausted ($%.4f remaining). Routing to HITL.", budget_remaining)
        return _transition("human_intervention_node")

    if exit_code == 0:
        logger.info("[router] Build succeeded (exit 0). Routing to security scan.")
        return _transition("security_scan_node")

    # Environment misconfig (missing pytest, npm, etc.) — LLM repair cannot
    # fix this. Skip the budget burn and surface the actionable error.
    if state.get("node_state", {}).get("env_misconfig"):
        symbol = state.get("node_state", {}).get("env_misconfig_symbol", "")
        logger.warning(
            "[router] Sandbox env misconfig (missing '%s'). Skipping repair loop, routing to HITL.",
            symbol,
        )
        return _transition("human_intervention_node")

    # P1.5: empty LLM response detected. Three rounds of an empty repair LLM
    # would burn three compile cycles with no chance of success; short-circuit
    # to HITL with the precise trigger so the operator sees "llm_silent"
    # instead of the generic repair-limit message.
    if state.get("node_state", {}).get("llm_silent"):
        logger.warning("[router] LLM returned empty content. Routing to HITL immediately.")
        return _transition("human_intervention_node")

    # Pytest exit=5 (no tests collected) is not a build failure.
    #   * Source files exist → route to test_generation_node, which will
    #     generate unit tests targeting them. Don't spend a repair on a
    #     non-error.
    #   * No source files either → the prior patching pass produced nothing
    #     usable (e.g. allowlist rejected every patch). Route to HITL with
    #     the precise diagnostic so the operator can fix the layout/config
    #     instead of letting the repair loop spin.
    if state.get("node_state", {}).get("no_tests_collected"):
        if state.get("node_state", {}).get("no_tests_has_source"):
            logger.info(
                "[router] No tests collected but source files exist. "
                "Routing to test_generation_node."
            )
            return _transition("test_generation_node")
        logger.warning(
            "[router] No tests collected and no source files in workspace. "
            "Routing to HITL — the patching pass did not produce code."
        )
        return _transition("human_intervention_node")

    # Autofixable diagnostics bypass the repair-limit gate. R4
    # (_try_missing_dep, MISSING_DEP) appends to requirements.txt with
    # ZERO LLM calls; R5 (_try_dep_resolution_conflict,
    # DEP_RESOLUTION_CONFLICT) strips version pins the same way. Refusing
    # to enter repair_node when the limit is reached AND the diagnostic
    # is deterministic-autofixable strands the operator in HITL on a
    # problem they can't fix from inside the loop (compiler_node's
    # MISSING_DEP detection was logging "Routing through repair loop so
    # autofix / LLM can amend the dep manifest" right before the router
    # then refused to do so — observed in session d880f762).
    compiler_errors: list[Any] = state.get("compiler_errors", []) or []
    autofixable_codes: frozenset[str] = frozenset({
        "MISSING_DEP", "DEP_RESOLUTION_CONFLICT",
    })
    has_autofixable = bool(compiler_errors) and all(
        str(err.get("error_code", "")).upper() in autofixable_codes
        for err in compiler_errors
    )

    # No-patches-landed tripwire (A6). Two consecutive repair rounds where
    # the patcher applied zero patches means the loop is stuck (LLM keeps
    # emitting blocks the patcher rejects, build state never changes). Don't
    # let it burn through the rest of max_iterations on a loop that
    # demonstrably isn't making progress — go straight to HITL with a
    # specific reason so the operator knows what to fix manually.
    consecutive_zero = int(loop_counter.get("consecutive_zero_patch_rounds", 0))
    if consecutive_zero >= 2 and not has_autofixable:
        logger.warning(
            "[router] %d consecutive repair iteration(s) landed zero patches. "
            "Loop is stuck; routing to HITL early (saves %d remaining iteration(s)).",
            consecutive_zero, max(0, max_iterations - total_repairs),
        )
        return _transition("human_intervention_node")

    # Audit §6.1: generic no-progress tripwire that is NOT gated on
    # has_autofixable. The earlier guard exempted any iteration that
    # had an autofixable diagnostic — but an alternating cycle (pip
    # missing → autofix → wheel missing → autofix → pip missing again)
    # resets the per-symbol counter and slips past the same-symbol
    # ceiling. After 5 consecutive zero-patch rounds we escalate even
    # when autofix is in play; that's enough cycles to give a real
    # autofix progression a fair chance while still bounding loops
    # that aren't actually making the build move forward.
    _GENERIC_NO_PROGRESS_LIMIT = 5
    if consecutive_zero >= _GENERIC_NO_PROGRESS_LIMIT:
        logger.warning(
            "[router] %d consecutive repair iteration(s) landed zero real "
            "patches even with autofix in play. Loop is not advancing "
            "(likely alternating MISSING_DEP cycle); routing to HITL.",
            consecutive_zero,
        )
        return _transition("human_intervention_node")

    if total_repairs >= max_iterations and not has_autofixable:
        logger.warning(
            "[router] Repair limit reached (%d/%d). Routing to HITL.",
            total_repairs,
            max_iterations,
        )
        return _transition("human_intervention_node")

    # Same-MISSING_DEP-symbol tripwire (bug #1 from the latest log
    # review). When the SAME missing_symbol recurs N times consecutively,
    # the deterministic autofix has demonstrably failed to resolve it —
    # typically because the missing tool is something the manifest cannot
    # install (`pip` itself, `make`, a system package). Continuing to
    # bypass the iteration limit just burns budget on patches that can't
    # land the fix; escalate to HITL with a message that points at the
    # sandbox image, not the manifest. Session 083770ac demonstrates the
    # unguarded loop: 21+ attempts on missing 'pip' against
    # buildpack-deps:bookworm before being killed externally.
    SAME_MISSING_DEP_LIMIT = 3
    consecutive_same_dep = int(
        loop_counter.get("missing_dep_consecutive_same", 0) or 0
    )
    if (
        has_autofixable
        and consecutive_same_dep >= SAME_MISSING_DEP_LIMIT
    ):
        last_symbol = str(
            loop_counter.get("missing_dep_last_symbol", "") or "?"
        )
        logger.warning(
            "[router] Missing dependency '%s' has recurred %d times in a "
            "row despite landed patches. The deterministic-autofix bypass "
            "cannot resolve it from inside the loop — the sandbox image "
            "almost certainly does not ship the bootstrap tool (`%s`). "
            "Routing to HITL: fix the docker_image (sandbox.docker_image "
            "in config.json or your project's build_command), then resume.",
            last_symbol, consecutive_same_dep, last_symbol,
        )
        return _transition("human_intervention_node")

    if total_repairs >= max_iterations and has_autofixable:
        logger.info(
            "[router] Repair limit (%d/%d) reached but all %d diagnostic(s) "
            "are deterministically autofixable (codes=%s). Routing to "
            "repair_node so autofix can land the fix without an LLM call.",
            total_repairs, max_iterations, len(compiler_errors),
            sorted({str(e.get("error_code", "")) for e in compiler_errors}),
        )

    logger.info("[router] Build failed (exit %d). Repair attempt %d/%d.", exit_code, total_repairs + 1, max_iterations)
    return _transition("repair_node")


# ---------------------------------------------------------------------------
# 6b'. Change-Request Ingestion (existing-project delta entry point)
# ---------------------------------------------------------------------------

# Filename pattern recognising operator-supplied CR-N prefixes (e.g.
# "CR-42-rewrite-auth.txt"). The harness respects pre-assigned IDs so
# external trackers (Jira ticket IDs, etc.) can map 1:1; collisions with
# already-archived CR IDs cause the ingest node to abort with a clear
# error so the operator can rename and retry.
_CR_FILENAME_PREFIX = re.compile(r"^CR-(\d+)(?:[-_].*)?\.txt$", re.IGNORECASE)


def _scan_archived_cr_ids(archive_root: str) -> set[int]:
    """Return the set of CR-N IDs already present under
    ``<change_requests_dir>/applied/``. The scan is non-recursive at the
    top level then walks each per-session subdirectory one level deep —
    that's the only layout the archive helper writes — so a corrupted
    deeper tree can't cause an infinite walk. Missing archive → empty set.
    """
    used: set[int] = set()
    if not os.path.isdir(archive_root):
        return used
    try:
        for entry in os.listdir(archive_root):
            full = os.path.join(archive_root, entry)
            m = _CR_FILENAME_PREFIX.match(entry)
            if m and os.path.isfile(full):
                used.add(int(m.group(1)))
            elif os.path.isdir(full):
                try:
                    for inner in os.listdir(full):
                        m_inner = _CR_FILENAME_PREFIX.match(inner)
                        if m_inner:
                            used.add(int(m_inner.group(1)))
                except OSError:
                    continue
    except OSError:
        return used
    return used


def _assign_change_request_ids(
    pending_filenames: list[str],
    archive_root: str,
) -> list[dict[str, Any]]:
    """Assign CR-N IDs to ``pending_filenames`` in sorted order.

    - A filename matching ``CR-<N>-*`` or ``CR-<N>.txt`` keeps its
      operator-supplied ``N``.
    - Otherwise, the next sequential ID is allocated from
      ``max(used) + 1`` (used = archived IDs ∪ already-assigned IDs in
      this batch). First-ever assignment starts at 1.
    - Collisions between an operator-supplied ID and an existing
      archived ID raise ``ValueError`` so the caller can abort early.

    Returns a list of ``{cr_id, original_name}`` records, sorted by
    ``cr_id``. Pure function — no I/O beyond the archive scan that the
    caller performed.
    """
    archived = _scan_archived_cr_ids(archive_root)
    used: set[int] = set(archived)
    records: list[dict[str, Any]] = []
    # First pass: lock in operator-supplied IDs and detect collisions
    # against the archive. A pending-vs-pending collision (two files both
    # declaring CR-N) also raises — the second one to register hits the
    # same `used`-set check.
    for name in pending_filenames:
        m = _CR_FILENAME_PREFIX.match(name)
        if m is None:
            continue
        n = int(m.group(1))
        if n in used:
            raise ValueError(
                f"change-request file {name!r} declares CR-{n}, but that "
                f"ID is already used in {archive_root} or another pending "
                "file. Rename one of them and re-run."
            )
        records.append({"cr_id": n, "original_name": name})
        used.add(n)
    # Second pass: assign sequential IDs to unprefixed filenames.
    # The starting point is `max(archive) + 1` when the archive holds
    # prior CRs, else 1 — that preserves monotone history across sessions.
    # We then skip any ID an operator pinned in this batch so sequential
    # allocation stays dense (CR-2 pinned + a, b pending → a=1, b=3).
    next_id = (max(archived) + 1) if archived else 1
    for name in pending_filenames:
        if _CR_FILENAME_PREFIX.match(name):
            continue
        while next_id in used:
            next_id += 1
        records.append({"cr_id": next_id, "original_name": name})
        used.add(next_id)
        next_id += 1
    records.sort(key=lambda r: r["cr_id"])
    return records


_REVERSE_ENGINEER_DEFAULT_BUDGET_USD: float = 0.50
_REVERSE_ENGINEER_MAX_FILES: int = 30
_REVERSE_ENGINEER_MAX_BYTES: int = 100_000

# Priority order when sampling files for the reverse-engineer walk. Lower
# index = higher priority. Files outside this list are still considered
# but sorted last (alphabetically). The list captures "entry points first"
# (main.py, app.py, index.ts) → "framework configs" (pyproject.toml,
# package.json) → "module roots" — exactly what an architect would skim
# to map a codebase in 5 minutes.
_REVERSE_ENGINEER_PRIORITY_BASENAMES: tuple[str, ...] = (
    "main.py", "app.py", "wsgi.py", "asgi.py", "manage.py",
    "index.ts", "index.js", "server.ts", "server.js",
    "main.go", "main.rs", "Main.java", "lib.rs",
    "pyproject.toml", "package.json", "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
    "Makefile", "README.md",
)
_REVERSE_ENGINEER_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".dart", ".cs", ".cpp", ".c", ".h", ".hpp", ".swift",
    ".sql", ".proto", ".yaml", ".yml", ".toml",
})


def _sample_workspace_for_reverse_engineer(
    workspace_path: str,
    source_root: Optional[str],
) -> list[tuple[str, str]]:
    """Return up to ``_REVERSE_ENGINEER_MAX_FILES`` files from the
    workspace, capped at ``_REVERSE_ENGINEER_MAX_BYTES`` cumulative bytes,
    biased toward priority entry-point filenames. Each tuple is
    ``(workspace-relative path, content)``. Pure I/O — no LLM call.

    Walk skips dot-directories, ``node_modules``, ``__pycache__``,
    ``.git``, ``venv``, etc. so we don't hand the LLM lockfiles or
    vendored dependencies.
    """
    skip_dirs = {
        ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv",
        "venv", "env", "dist", "build", ".pytest_cache", ".tox",
        "target", ".idea", ".vscode", ".mypy_cache", ".ruff_cache",
        "applied",  # change_requests/applied/
    }

    candidates: list[tuple[int, str]] = []
    root_for_walk = (
        os.path.join(workspace_path, source_root) if source_root else workspace_path
    )
    if not os.path.isdir(root_for_walk):
        root_for_walk = workspace_path

    for dirpath, dirnames, filenames in os.walk(root_for_walk):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if (
                fname not in _REVERSE_ENGINEER_PRIORITY_BASENAMES
                and ext not in _REVERSE_ENGINEER_SOURCE_EXTENSIONS
            ):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fname), workspace_path)
            if fname in _REVERSE_ENGINEER_PRIORITY_BASENAMES:
                priority = _REVERSE_ENGINEER_PRIORITY_BASENAMES.index(fname)
            else:
                priority = len(_REVERSE_ENGINEER_PRIORITY_BASENAMES) + 1
            candidates.append((priority, rel))

    candidates.sort(key=lambda kv: (kv[0], kv[1]))

    sampled: list[tuple[str, str]] = []
    total_bytes = 0
    for _priority, rel in candidates:
        if len(sampled) >= _REVERSE_ENGINEER_MAX_FILES:
            break
        abs_path = os.path.join(workspace_path, rel)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(_REVERSE_ENGINEER_MAX_BYTES)
        except OSError:
            continue
        if total_bytes + len(content) > _REVERSE_ENGINEER_MAX_BYTES:
            # Truncate the last file to fit under the cap rather than
            # silently dropping it — the LLM gets *some* signal from
            # whatever fits, which is usually the file's imports + module
            # docstring + first class/function, exactly what architecture
            # synthesis needs.
            remaining = max(0, _REVERSE_ENGINEER_MAX_BYTES - total_bytes)
            if remaining < 200:
                break
            content = content[:remaining] + "\n... (truncated)\n"
        sampled.append((rel, content))
        total_bytes += len(content)
    return sampled


async def reverse_engineer_architecture_node(state: AgentState) -> dict[str, Any]:
    """Synthesize ``SPEC_ARCHITECTURE.md`` for an existing codebase that
    lacks one, on first contact in change-request mode.

    Pre-conditions: ``change_request_mode=True`` AND no
    ``SPEC_ARCHITECTURE.md`` exists at the conventional output path
    (``<workspace>/docs/``). On a workspace that already has the spec
    this node is a fast no-op (file-stat only). Subsequent change-request
    sessions on the same repo therefore never re-pay the LLM cost.

    Budget-gated by ``change_requests.reverse_engineer_budget_usd``
    (defaults to ``$0.50``): when the remaining session budget is below
    the cap the node logs and short-circuits — the architecture review
    cycle that follows will still run, just without the synthesized
    baseline, which mirrors how the discovery pipeline handles a missing
    SPEC_ARCHITECTURE.md elsewhere.
    """
    if not state.get("change_request_mode", False):
        return {}

    workspace = state.get("workspace_path", os.getcwd())
    output_dir = os.path.join(workspace, "docs")
    arch_path = os.path.join(output_dir, "SPEC_ARCHITECTURE.md")

    if os.path.isfile(arch_path):
        logger.info(
            "[reverse_engineer] %s already exists — skipping the one-shot "
            "LLM walk. Subsequent CR sessions reuse this baseline.",
            arch_path,
        )
        return {"spec_architecture_path": arch_path}

    # Budget gate. Read the change_requests config section if it's
    # plumbed through state (CR-3 PR ships it in cmd_run); otherwise use
    # the conservative default.
    cr_cfg = state.get("change_requests_config", {}) or {}
    budget_cap = float(
        cr_cfg.get(
            "reverse_engineer_budget_usd",
            _REVERSE_ENGINEER_DEFAULT_BUDGET_USD,
        )
    )
    current_budget = float(state.get("budget_remaining_usd", 0.0))
    if current_budget < budget_cap:
        logger.warning(
            "[reverse_engineer] Remaining budget $%.4f is below the "
            "reverse_engineer cap $%.2f. Skipping the one-shot synthesis "
            "and falling through to the discovery pipeline without a "
            "baseline architecture spec.",
            current_budget, budget_cap,
        )
        return {}

    gateway = get_gateway()
    if gateway is None:
        logger.error(
            "[reverse_engineer] No gateway configured — cannot synthesize "
            "SPEC_ARCHITECTURE.md. Falling through to discovery."
        )
        return {}

    # Cheap structural context: stack tags + source root. The discovery
    # pipeline already uses these helpers for its own prompts.
    from harness.impact import (
        _detect_source_root,
        _detect_workspace_stack,
    )
    stack_tags = sorted(_detect_workspace_stack(workspace))
    source_root = _detect_source_root(workspace)

    sampled = _sample_workspace_for_reverse_engineer(workspace, source_root)
    if not sampled:
        logger.warning(
            "[reverse_engineer] No representative source files found in %s "
            "— skipping the one-shot synthesis.", workspace,
        )
        return {}

    cr_records = state.get("change_request_files", []) or []
    cr_summary_lines = "\n".join(
        f"  - CR-{r['cr_id']}: {r.get('original_name', '?')}" for r in cr_records
    ) or "  (no active CRs)"

    sample_blocks: list[str] = []
    for rel, content in sampled:
        sample_blocks.append(f"### `{rel}`\n```\n{content}\n```")
    sample_body = "\n\n".join(sample_blocks)

    system_prompt = (
        "You are a Principal Software Architect performing a one-shot "
        "reverse-engineering pass on an existing codebase. Your job is to "
        "produce SPEC_ARCHITECTURE.md — a concise, accurate snapshot of the "
        "system AS IT EXISTS TODAY. Subsequent sessions will amend this "
        "document with `## Revision: CR-N — …` headers when change requests "
        "modify the architecture. Be specific. Cite filenames you saw. "
        "Do NOT invent components that aren't in the sampled files."
    )
    from harness.architecture_inventory import ARCHITECTURE_INVENTORY_INSTRUCTION
    user_prompt = (
        f"# Workspace fingerprint\n\n"
        f"- Workspace path: `{workspace}`\n"
        f"- Detected stack tags: {', '.join(stack_tags) or '(none detected)'}\n"
        f"- Detected source root: `{source_root or '(flat layout)'}`\n"
        f"- Pending change requests driving this session:\n"
        f"{cr_summary_lines}\n\n"
        f"# Representative source files ({len(sampled)} sampled)\n\n"
        f"{sample_body}\n\n"
        "---\n\n"
        "# Task\n\n"
        "Produce SPEC_ARCHITECTURE.md describing this system. Required sections:\n\n"
        "1. **Module map** — top-level components and what each is responsible "
        "for. Cite the directory and 1-2 representative files.\n"
        "2. **Data model** — entities, stores, schemas. State unknowns explicitly.\n"
        "3. **Integration surface** — external services, APIs called or exposed.\n"
        "4. **Build & runtime** — toolchain, entry point, deploy unit.\n"
        "5. **Known unknowns** — areas where the sample didn't give you enough "
        "signal, framed as concrete questions for the operator.\n\n"
        f"{ARCHITECTURE_INVENTORY_INSTRUCTION}\n"
        "Output ONLY the markdown document — no preamble, no fences. The "
        "fenced ```json inventory block IS part of the markdown body and "
        "must be included verbatim."
    )

    from harness.gateway import NodeRole
    try:
        response, new_budget = await gateway.dispatch(
            messages=[
                MessageDict(role="system", content=system_prompt),
                MessageDict(role="user", content=user_prompt),
            ],
            role=NodeRole.PLANNING,
            budget_remaining_usd=current_budget,
        )
    except Exception as exc:
        logger.exception(
            "[reverse_engineer] LLM dispatch failed: %s — continuing "
            "without baseline architecture.", exc,
        )
        return {}

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(arch_path, "w", encoding="utf-8") as f:
            f.write(response.content)
    except OSError as exc:
        logger.error(
            "[reverse_engineer] Could not write %s: %s. Continuing without "
            "baseline architecture.", arch_path, exc,
        )
        return {"budget_remaining_usd": new_budget}

    logger.info(
        "[reverse_engineer] Synthesized %s (%d chars) from %d source files. "
        "Subsequent CR sessions on this repo will skip this step. "
        "Budget: $%.4f → $%.4f.",
        arch_path, len(response.content), len(sampled),
        current_budget, new_budget,
    )
    return {
        "spec_architecture_path": arch_path,
        "budget_remaining_usd": new_budget,
    }


async def ingest_change_requests_node(state: AgentState) -> dict[str, Any]:
    """Consume the change_requests/ folder and inject the requests as the
    LLM's task description.

    For PR-1 the node short-circuits to ``patching_node`` (see
    ``route_after_start``). The consolidated change-request text becomes
    the user message that drives patching, mirroring how the bare
    ``-p`` prompt drives existing-project runs today. PR-2+ will route
    through ``requirements_discovery_node`` and the gatekeeper instead.

    Side effects on returned state:
      - ``change_request_files``: list of ``{cr_id, original_name, abs_path}``
        records, sorted by ``cr_id``. Source of truth for the archival
        helper at session end.
      - ``messages``: a new user message replacing the seed prompt (the
        seed prompt was the bare CLI ``-p``; when in change-request
        mode the CLI either dropped it or it was empty). The
        replacement uses ``# === CR-7: <relative-path> ===`` headers so
        the LLM can attribute each request by CR ID.
    """
    cr_dir = state.get("change_requests_dir_abs", "")
    if not cr_dir or not os.path.isdir(cr_dir):
        logger.error(
            "[change_requests] ingest_change_requests_node reached without a "
            "valid change_requests_dir_abs in state — this should be caught "
            "earlier in cmd_run. Falling through with no changes."
        )
        return {}

    archive_root = os.path.join(cr_dir, "applied")
    try:
        entries = sorted(os.listdir(cr_dir))
    except OSError as exc:
        logger.error("[change_requests] Could not list %s: %s", cr_dir, exc)
        return {}

    pending = [
        e for e in entries
        if e != "applied"
        and e.endswith(".txt")
        and os.path.isfile(os.path.join(cr_dir, e))
    ]

    if not pending:
        logger.error(
            "[change_requests] No pending .txt files under %s — cmd_run "
            "should have rejected this earlier.", cr_dir,
        )
        return {}

    try:
        records = _assign_change_request_ids(pending, archive_root)
    except ValueError as exc:
        logger.error("[change_requests] %s", exc)
        # Surface as a system message so the session terminates cleanly
        # rather than crashing the graph runtime.
        return {
            "messages": list(state.get("messages", [])) + [
                MessageDict(
                    role="system",
                    content=(
                        f"Change-request ingestion failed: {exc}\n"
                        "Resolve the collision and re-run."
                    ),
                )
            ],
            "exit_code": 1,
        }

    # Attach absolute paths now that IDs are assigned.
    for rec in records:
        rec["abs_path"] = os.path.join(cr_dir, rec["original_name"])

    sections: list[str] = [
        f"# Change requests ({len(records)} pending)",
        "",
        "Each request below is a self-contained ask. Each carries a CR-N "
        "identifier so downstream artifacts (spec revisions, code, tests, "
        "infra) can be traced back to the originating request via `grep CR-N`.",
        "",
    ]
    for rec in records:
        try:
            with open(rec["abs_path"], "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as exc:
            logger.warning(
                "[change_requests] Could not read %s: %s — skipping.",
                rec["abs_path"], exc,
            )
            continue
        sections.append(f"# === CR-{rec['cr_id']}: {rec['original_name']} ===")
        sections.append("")
        sections.append(content.rstrip())
        sections.append("")

    consolidated = "\n".join(sections)
    cr_ids = ", ".join(f"CR-{r['cr_id']}" for r in records)
    logger.info(
        "[change_requests] Ingested %d request(s): %s. Source dir: %s",
        len(records), cr_ids, cr_dir,
    )

    # Replace the seed user message (messages[1]) with the consolidated
    # change requests. messages[0] stays as the anchored system prompt for
    # prefix-cache reuse.
    new_messages = list(state.get("messages", []))
    if len(new_messages) >= 2 and new_messages[1].get("role") == "user":
        new_messages[1] = MessageDict(role="user", content=consolidated)
    else:
        new_messages.append(MessageDict(role="user", content=consolidated))

    return {
        "messages": new_messages,
        "change_request_files": records,
        # Discovery pipeline runs in delta mode for change-request sessions
        # regardless of the operator's --spec-discovery flag — the gatekeeper is
        # the whole point of the folder convention. Clear skip_discovery
        # so route_after_spec_review honors interview follow-ups instead
        # of short-circuiting to the gatekeeper.
        "skip_discovery": False,
    }


def _build_change_request_preamble(state: AgentState, phase: str) -> str:
    """Return the delta-mode prompt preamble for ``phase`` (one of
    ``"requirements"``, ``"architecture"``, ``"deployment"``,
    ``"patching"``, ``"tests"``). Empty string when not in
    change-request mode.

    The preamble lists active CR-N IDs and tells the LLM which artifacts
    to tag with `<!-- BEGIN/END CR-N -->` (specs) or `# CR-N:` /
    `// CR-N:` comments (code, tests, infra) so a downstream
    `grep -rn "CR-7" .` finds every artifact connected to the request.
    The full change-request text already sits in messages[1] — we only
    need to remind the LLM of the active set and the marker contract.
    """
    if not state.get("change_request_mode", False):
        return ""
    records = state.get("change_request_files", []) or []
    if not records:
        return ""
    cr_lines = [f"  - CR-{r['cr_id']}: {r.get('original_name', '?')}" for r in records]
    cr_ids_csv = ", ".join(f"CR-{r['cr_id']}" for r in records)

    phase_rules = {
        "requirements": (
            "This is an EXISTING project with prior SPEC_REQUIREMENTS.md. "
            "Ask DELTA-shaped questions only — \"what's changing because "
            "of these requests?\", \"what must NOT change?\", \"what's "
            "the acceptance test?\". Do NOT re-elicit baseline "
            "requirements that are already documented. "
            "Tag every passage you propose to add or modify in the spec "
            "with `<!-- BEGIN CR-N -->` / `<!-- END CR-N -->` markers "
            "naming the originating request, stacking IDs when one "
            "passage serves multiple CRs (`<!-- BEGIN CR-7,CR-8 -->`)."
        ),
        "architecture": (
            "Evaluate whether these CR-N changes are architecture-"
            "significant. If NONE are (pure logic / config tweaks), "
            "return modules=[] and complete=true — the architecture "
            "review cycle will short-circuit. If ANY are, ask only the "
            "delta-shaped architecture questions and tag affected "
            "passages with `<!-- BEGIN/END CR-N -->` markers."
        ),
        "deployment": (
            "Evaluate whether these CR-N changes are deployment- or "
            "infrastructure-significant (new service, new env var, new "
            "reverse-proxy route, container limit change, dependency "
            "added). If NONE are, return modules=[] and complete=true — "
            "the deployment review cycle short-circuits and existing "
            "infra files (compose, Dockerfile, Caddyfile, blueprint) "
            "stay untouched. If ANY are, ask the delta questions and "
            "tag affected passages with `<!-- BEGIN/END CR-N -->`."
        ),
        "patching": (
            "When you patch source files to satisfy these change "
            "requests, add ONE terse comment per modified function/"
            "class/region — language-appropriate (`# CR-N: …` in "
            "Python, `// CR-N: …` in JS/TS/Go/Rust/Java) — naming the "
            "originating request. New files get the same one-line "
            "comment under the module docstring / imports. Do NOT mark "
            "every line touched; one marker per region is the rule."
        ),
        "tests": (
            "When you generate tests for these change requests, name "
            "the new test functions with the `test_cr_N_<descriptive>` "
            "pattern (or the per-language idiom), and reference the CR "
            "in each test docstring (e.g. `\"\"\"Verifies CR-7: …\"\"\"`). "
            "When extending an existing test, add a single inline "
            "`# CR-N:` comment at the new assertion(s)."
        ),
    }
    rules = phase_rules.get(phase, "")

    return (
        "## Change-Request Mode Active\n\n"
        f"{len(records)} pending change request(s) drive this session "
        f"({cr_ids_csv}):\n"
        + "\n".join(cr_lines)
        + "\n\n"
        + "The full text of each request is in the conversation history "
        "(first user message). Each request's intent must be traceable "
        "via the `CR-N` ID through every artifact this session produces.\n\n"
        f"{rules}\n\n"
        "---\n\n"
    )


def route_after_start(state: AgentState) -> Literal[
    "requirements_discovery_node", "patching_node", "ingest_change_requests_node",
]:
    """START edge router. Module-level so tests can call it directly.

    Precedence:
      1. ``change_request_mode`` → ingest_change_requests_node (a populated
         change_requests/ folder always overrides skip_discovery so a
         misconfigured run still goes through the gatekeeper pipeline once
         PR-2 lands the delta routing).
      2. ``skip_discovery`` → patching_node (the bare existing-project path
         from before change-request mode existed).
      3. Default → requirements_discovery_node (greenfield discovery).
    """
    if state.get("change_request_mode", False):
        logger.info(
            "[router] change_request_mode active. "
            "Routing START → ingest_change_requests_node."
        )
        return "ingest_change_requests_node"
    if state.get("skip_discovery", False):
        logger.info("[router] spec discovery skipped. Routing START → patching_node.")
        return "patching_node"
    return "requirements_discovery_node"


# ---------------------------------------------------------------------------
# 6c. Generator Nodes for Three-Phase HITL Gates
# ---------------------------------------------------------------------------

async def requirements_discovery_node(state: AgentState) -> dict[str, Any]:
    """
    Exhaustive requirements discovery: calls the planning LLM with a structured
    cross-examination prompt across 8 sectors. Returns grouped questions as JSON.
    If the user has already answered previous questions, evaluates answers for
    remaining critical unknowns and generates follow-ups.
    """
    gateway = get_gateway()
    if gateway is None:
        logger.error("[reqs_disc] No gateway configured.")
        return {"node_state": {"discovery_complete": True, "error": "No gateway"}}

    messages = list(state.get("messages", []))
    question_count = state.get("node_state", {}).get("discovery_question_count", 0)

    # Determine if this is the first pass or a follow-up
    is_followup = question_count > 0

    if is_followup:
        prompt = f"""You are a Lead Systems Auditor. This is a FOLLOW-UP round (#{question_count + 1}).
Review the conversation above where the user answered your previous questions.
Your task:
1. Cross-reference the user's answers against all 8 sectors below.
2. Identify any REMAINING unknowns, gaps, or contradictions.
3. For each critical gap, generate a targeted follow-up question.
4. If ALL sectors are fully resolved across all 8 areas, output: {{"complete": true}}

Output JSON:
{{
  "modules": [
    {{"name": "INPUT VALIDATION", "questions": [
      {{"id": "Q1.1", "text": "...", "critical": true/false, "suggested_answer": "..."}}
    ]}},
    ...
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}}

Every question MUST include a "suggested_answer" — your best, most-probable
answer given the conversation context, project files, and prior responses.
Keep it short (1 line, concrete, actionable). The interview presents it to
the operator as a default they can press Enter to accept; a vague placeholder
defeats the purpose. If you genuinely have no signal, use the conservative
industry default for that sector and say so.
Return ONLY valid JSON. No markdown or explanation."""

    else:
        prompt = """You are a Lead Systems Auditor. Perform EXHAUSTIVE requirements discovery across ALL 8 sectors below.
For each sector, ask every question needed to eliminate unknowns. Be extremely thorough.

## Required Sectors

### 1. INPUT DATA VALIDATION
- Data types, value ranges, required vs optional fields, format constraints, nesting limits, encoding rules.

### 2. PAYLOAD FORMATTING
- JSON schema/XML/binary format, field naming conventions, pagination structure, error envelope format, versioning.

### 3. ERROR HANDLING BEHAVIORS
- HTTP status codes per scenario, retry logic (exponential backoff, max retries), circuit breaker thresholds, graceful degradation, dead-letter queues.

### 4. MULTI-USER EDGE CASES
- Concurrency model (optimistic/pessimistic locking), race conditions, idempotency requirements, distributed transaction boundaries, eventual consistency windows.

### 5. SECURITY CONTROLS
- Authentication method (JWT, OAuth2, API keys), token storage/rotation, RBAC/permission model, CORS policy, rate limiting, input sanitization, CSRF protection.

### 6. STRICT BUSINESS LOGIC RULES
- State machines and transitions, invariants that must hold, validation constraints, business rule precedence, conflict resolution.

### 7. DATA RETENTION BOUNDARIES
- TTLs for cached data, archival policies, GDPR/compliance requirements, audit log retention, backup schedules.

### 8. HIDDEN ASSUMPTIONS
- Platform requirements (OS, architecture), network topology assumptions, timezone/locale handling, third-party service availability, expected load profiles.

Output the EXACT JSON shape below — the key must be literally "modules"
(not "sectors", not "questions", not the section titles above). The harness
parses this shape strictly; any other top-level key yields zero questions
and the operator sees an empty interview screen.

{
  "modules": [
    {"name": "INPUT VALIDATION", "questions": [
      {"id": "Q1.1", "text": "...", "critical": true, "suggested_answer": "..."},
      {"id": "Q1.2", "text": "...", "critical": false, "suggested_answer": "..."}
    ]},
    {"name": "PAYLOAD FORMATTING", "questions": [...]},
    {"name": "ERROR HANDLING BEHAVIORS", "questions": [...]},
    {"name": "MULTI-USER EDGE CASES", "questions": [...]},
    {"name": "SECURITY CONTROLS", "questions": [...]},
    {"name": "STRICT BUSINESS LOGIC RULES", "questions": [...]},
    {"name": "DATA RETENTION BOUNDARIES", "questions": [...]},
    {"name": "HIDDEN ASSUMPTIONS", "questions": [...]}
  ],
  "complete": false,
  "summary": "Brief status of what's covered vs still unknown"
}

Mark critical items with "critical": true. Every question MUST include
"suggested_answer" — your best, most-probable answer given the conversation
context, project files, and sector intent. Keep it short (1 line, concrete,
actionable). The interview presents it as a default the operator can press
Enter to accept; a vague placeholder defeats the purpose. If you have no
signal, use the conservative industry default and say so. Return ONLY valid
JSON. No markdown, no explanation, no code blocks."""

    # Delta-mode preamble: when change_request_mode is active, prepend the
    # CR-N attribution rules and the "ask delta-shaped questions only"
    # instruction so the LLM doesn't re-elicit baseline requirements on
    # an existing-project run. No-op (empty string) when not in CR mode.
    prompt = _build_change_request_preamble(state, "requirements") + prompt
    messages.append({"role": "user", "content": prompt})

    from harness.gateway import NodeRole

    current_budget = state.get("budget_remaining_usd", 0.0)
    if current_budget <= 0:
        logger.warning("[reqs_disc] Budget exhausted ($%.4f); skipping discovery.", current_budget)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": "budget exhausted"},
            "budget_remaining_usd": current_budget,
        }

    try:
        response, budget = await gateway.dispatch(
            messages=list(messages), role=NodeRole.PLANNING,
            budget_remaining_usd=current_budget,
        )

        from harness.trust import validate_discovery_json
        discovery_data, trust_errors = validate_discovery_json(response.content)
        if trust_errors:
            logger.error("[reqs_disc] Discovery response failed trust validation: %s", trust_errors)
            return {
                "messages": messages,
                "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                "budget_remaining_usd": budget,
            }

        complete = discovery_data.get("complete", False)
        modules = discovery_data.get("modules", [])
        total_questions = sum(len(m.get("questions", [])) for m in modules)
        critical_count = sum(1 for m in modules for q in m.get("questions", []) if q.get("critical"))

        # Catch schema drift: a discovery response that parsed but had no
        # modules + complete=False means the LLM used a different top-level
        # key (e.g. "sectors", "components"). Surface it loudly so the
        # operator doesn't see an empty interview screen and silently
        # answer "DONE" to no questions — which is the exact symptom that
        # produced the empty Round 1 in session c371c744-….
        if not modules and not complete:
            top_keys = sorted(discovery_data.keys()) if isinstance(discovery_data, dict) else []
            logger.warning(
                "[reqs_disc] LLM response parsed cleanly but has ZERO modules "
                "and complete=False — likely the model used a non-canonical "
                "top-level key (expected 'modules', saw keys=%s). The "
                "interview will display no questions. Re-issuing as a "
                "follow-up round will retry with the canonical schema "
                "embedded in the prompt.",
                top_keys,
            )

        messages.append({"role": "assistant", "content": json.dumps(discovery_data, sort_keys=True)})

        logger.info("[reqs_disc] Round %d: %d questions across %d modules (%d critical). Complete=%s budget=$%.4f",
                     question_count + 1, total_questions, len(modules), critical_count, complete, budget)

        return {
            "messages": messages,
            "discovery_questions": discovery_data,
            "current_gate": "REQUIREMENTS",
            "budget_remaining_usd": budget,
            "node_state": {
                "current_node": "requirements_discovery",
                "discovery_complete": complete,
                "discovery_question_count": question_count + 1,
                "discovery_critical_remaining": critical_count if not complete else 0,
            },
        }
    except Exception as exc:
        logger.exception("[reqs_disc] Discovery failed: %s", exc)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": str(exc)},
            "budget_remaining_usd": current_budget,
        }


async def architecture_discovery_node(state: AgentState) -> dict[str, Any]:
    """
    Exhaustive architecture discovery: queries for deep technical variables
    across 8 sectors. Returns grouped questions as JSON, same format as requirements.
    """
    gateway = get_gateway()
    if gateway is None:
        logger.error("[arch_disc] No gateway configured.")
        return {"node_state": {"discovery_complete": True, "error": "No gateway"}}

    messages = list(state.get("messages", []))
    question_count = state.get("node_state", {}).get("discovery_question_count", 0)
    is_followup = question_count > 0

    if is_followup:
        prompt = f"""You are a Principal Infrastructure Architect. FOLLOW-UP round #{question_count + 1}.
Review the conversation above. Cross-reference answers. Find remaining gaps.
If all 8 architectural sectors are fully resolved, output {{"complete": true}} and nothing else.

Otherwise, output the EXACT JSON shape below — top-level key MUST be
literally "modules". The harness parses this shape strictly; any other
top-level key yields zero questions and the operator sees an empty
interview screen.

{{
  "modules": [
    {{"name": "STORAGE TOPOLOGY", "questions": [
      {{"id": "A1.1", "text": "...", "critical": true, "suggested_answer": "..."}}
    ]}}
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}}

Every question MUST include "suggested_answer" — your best, most-probable
answer given the conversation context, project files, and prior responses.
Keep it short (1 line, concrete, actionable). The interview presents it as
a default the operator can press Enter to accept; a vague placeholder
defeats the purpose. If you have no signal, use the conservative industry
default and say so. Return ONLY valid JSON. No markdown, no explanation, no
code fences."""

    else:
        prompt = """You are a Principal Infrastructure Architect. Perform EXHAUSTIVE architecture discovery across ALL 8 sectors.

## Required Sectors

### 1. DATABASE SCHEMA DEFINITIONS
- Tables, columns, primary keys, foreign keys, unique constraints, indexes (B-tree/hash/GIN), partitioning strategy, replication (master-slave/multi-master), connection pooling.

### 2. MICROSERVICE COMMUNICATION
- Protocol per service boundary (REST, WebSocket, gRPC, GraphQL, message queue), serialization format, service discovery, load balancing, circuit breaking, retry/backoff.

### 3. EXTERNAL API RATE-LIMITING
- Provider rate limits per API, token bucket/sliding window algorithm, throttling tiers, quota management, retry-after handling.

### 4. CONTAINER VOLUME STORAGE
- Named volumes vs bind mounts, persistent storage paths per service, backup strategy, tmpfs requirements for ephemeral data, NFS/EFS for shared state.

### 5. ENVIRONMENTAL SECRECY
- Secrets manager (HashiCorp Vault, AWS Secrets Manager, Doppler), env var injection, .env file handling, CI/CD secret masking, rotation policy.

### 6. SCALING PARAMETERS
- Horizontal vs vertical scaling, auto-scaling triggers (CPU > 70%, memory, request queue depth), min/max replicas, cold start mitigation.

### 7. OBSERVABILITY
- Structured logging format (JSON), log aggregation, metrics (Prometheus/Datadog), distributed tracing (OpenTelemetry), alerting thresholds, health check endpoints.

### 8. CI/CD PIPELINE HOOKS
- Build triggers (push, PR, tag), deployment gates (approval, test pass), rollback strategy, canary/blue-green deployment, environment promotion path.

Output the EXACT JSON shape below — the top-level key MUST be literally
"modules" (not "sectors", not "components"). Any other key yields zero
questions and the operator sees an empty interview screen.

{
  "modules": [
    {"name": "STORAGE TOPOLOGY", "questions": [
      {"id": "A1.1", "text": "...", "critical": true, "suggested_answer": "..."}
    ]},
    ... one entry per architectural sector above ...
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}

Mark critical items with "critical": true. Every question MUST include
"suggested_answer" — your best, most-probable answer given the conversation
context, project files, and sector intent. Keep it short (1 line, concrete,
actionable). The interview presents it as a default the operator can press
Enter to accept; a vague placeholder defeats the purpose. If you have no
signal, use the conservative industry default and say so. Return ONLY valid
JSON. No markdown, no explanation, no code blocks."""
    # Delta-mode preamble — see ``_build_change_request_preamble``. In
    # delta mode the LLM is told to short-circuit (modules=[], complete=
    # true) when no CR is architecture-significant, so light fixes don't
    # spin up the full architecture review cycle.
    prompt = _build_change_request_preamble(state, "architecture") + prompt
    messages.append({"role": "user", "content": prompt})

    from harness.gateway import NodeRole

    current_budget = state.get("budget_remaining_usd", 0.0)
    if current_budget <= 0:
        logger.warning("[arch_disc] Budget exhausted ($%.4f); skipping discovery.", current_budget)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": "budget exhausted"},
            "budget_remaining_usd": current_budget,
        }

    try:
        response, budget = await gateway.dispatch(
            messages=list(messages), role=NodeRole.PLANNING,
            budget_remaining_usd=current_budget,
        )

        from harness.trust import validate_discovery_json
        discovery_data, trust_errors = validate_discovery_json(response.content)
        if trust_errors:
            logger.error("[arch_disc] Discovery response failed trust validation: %s", trust_errors)
            return {
                "messages": messages,
                "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                "budget_remaining_usd": budget,
            }

        complete = discovery_data.get("complete", False)
        modules = discovery_data.get("modules", [])
        total_q = sum(len(m.get("questions", [])) for m in modules)
        critical_count = sum(1 for m in modules for q in m.get("questions", []) if q.get("critical"))

        messages.append({"role": "assistant", "content": json.dumps(discovery_data, sort_keys=True)})

        logger.info("[arch_disc] Round %d: %d questions (%d critical). Complete=%s budget=$%.4f",
                     question_count + 1, total_q, critical_count, complete, budget)

        return {
            "messages": messages,
            "discovery_questions": discovery_data,
            "current_gate": "ARCHITECTURE",
            "budget_remaining_usd": budget,
            "node_state": {
                "current_node": "architecture_discovery",
                "discovery_complete": complete,
                "discovery_question_count": question_count + 1,
                "discovery_critical_remaining": critical_count if not complete else 0,
            },
        }
    except Exception as exc:
        logger.exception("[arch_disc] Discovery failed: %s", exc)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": str(exc)},
            "budget_remaining_usd": current_budget,
        }


async def deployment_discovery_node(state: AgentState) -> dict[str, Any]:
    """
    Exhaustive deployment infrastructure discovery: calls the planning LLM
    to cross-examine the user across 4 deployment-specific sectors.
    Returns grouped questions as JSON. Follow-ups on subsequent passes.
    """
    gateway = get_gateway()
    if gateway is None:
        logger.error("[deploy_disc] No gateway configured.")
        return {"node_state": {"discovery_complete": True, "error": "No gateway"}}

    messages = list(state.get("messages", []))
    question_count = state.get("node_state", {}).get("discovery_question_count", 0)
    is_followup = question_count > 0

    # Optional org-wide policy from the ``deployment_defaults`` section of
    # config.json. When non-empty, the planning LLM is told to treat every
    # populated field as RESOLVED and skip emitting a question for it. When
    # empty, the prompt block below is omitted and the questionnaire runs
    # in its current full mode.
    deployment_defaults = state.get("deployment_defaults", {}) or {}
    resolved_block = ""
    if deployment_defaults:
        resolved_block = (
            "\n\n## Pre-resolved deployment policies (operator-supplied)\n\n"
            "The operator has already declared the following deployment "
            "policies via the ``deployment_defaults`` section of config.json. "
            "Treat every populated field below as RESOLVED — do NOT emit a "
            "question whose answer is already given here. Generate "
            "questions ONLY for sectors / fields not covered and for "
            "stack-specific details (e.g. per-service ports, volume paths) "
            "that this policy does not specify. If every sector is fully "
            'covered, return {"complete": true} on round 1 and skip the '
            "interview entirely. The values below carry into "
            "DEPLOYMENT_BLUEPRINT.md verbatim.\n\n"
            "```json\n"
            + json.dumps(deployment_defaults, indent=2, sort_keys=True)
            + "\n```"
        )

    if is_followup:
        prompt = f"""You are a Principal DevSecOps Engineer. FOLLOW-UP round #{question_count + 1}.
Review the conversation above. Cross-reference deployment answers. Find remaining gaps.
If all 4 deployment sectors are fully resolved, output {{"complete": true}} and nothing else.

Otherwise, output the EXACT JSON shape below — top-level key MUST be
literally "modules". Any other key yields zero questions on the
interview screen.

{{
  "modules": [
    {{"name": "NETWORK TOPOLOGY", "questions": [
      {{"id": "D1.1", "text": "...", "critical": true, "suggested_answer": "..."}}
    ]}}
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}}

Every question MUST include "suggested_answer" — your best, most-probable
answer given the conversation context, project files, and prior responses.
Keep it short (1 line, concrete, actionable). The interview presents it as
a default the operator can press Enter to accept; a vague placeholder
defeats the purpose. If you have no signal, use the conservative industry
default and say so. Return ONLY valid JSON. No markdown, no explanation, no
code fences.{resolved_block}"""

    else:
        prompt = """You are a Principal DevSecOps Systems Engineer and Lead SRE. Perform EXHAUSTIVE deployment infrastructure discovery across ALL 4 sectors below.

## Required Sectors

### 1. NETWORK TOPOLOGY
- External routing ports (HTTP 80, HTTPS 443, custom ports), reverse proxy paths (Caddy/Nginx rules), internal Docker network bridge mappings, DNS configuration (host resolution, custom domains), port collision parameters on host.

### 2. DATA & STORAGE PERSISTENCE
- Absolute path volume mounting points, shared database cluster parameters, write permissions (UID/GID mapping), backup paths, tmpfs vs bind mount decisions, NFS/EFS for multi-host setups.

### 3. SECRETS & IDENTITY MANAGEMENT
- Vault integrations (HashiCorp Vault, Doppler, AWS Secrets Manager), decryption methods, Keycloak realm secrets, database passwords, runtime environment variable configurations, .env file handling, CI/CD secret masking.

### 4. PARTIAL INFRASTRUCTURE SYNC
- Verify any pre-existing containers running on host, existing databases or shared network clusters, active ports already bound, avoid destruction or duplication of production resources, sidecar dependencies, legacy service compatibility.

Output the EXACT JSON shape below — top-level key MUST be literally
"modules". Any other key yields zero questions on the interview screen.

{
  "modules": [
    {"name": "RUNTIME PLATFORM", "questions": [
      {"id": "D1.1", "text": "...", "critical": true, "suggested_answer": "..."}
    ]},
    ... one entry per deployment sector above ...
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}

Mark critical items with "critical": true. Every question MUST include
"suggested_answer" — your best, most-probable answer given the conversation
context, project files, and sector intent. Keep it short (1 line, concrete,
actionable). The interview presents it as a default the operator can press
Enter to accept; a vague placeholder defeats the purpose. If you have no
signal, use the conservative industry default and say so. Return ONLY valid
JSON. No markdown, no explanation, no code blocks.""" + resolved_block

    # Delta-mode preamble — same shape as the requirements/architecture
    # nodes. In CR mode the LLM returns modules=[] and complete=true when
    # none of the pending changes touch infrastructure, so light app-only
    # tweaks bypass the deployment review cycle entirely.
    prompt = _build_change_request_preamble(state, "deployment") + prompt
    messages.append({"role": "user", "content": prompt})

    from harness.gateway import NodeRole

    current_budget = state.get("budget_remaining_usd", 0.0)
    if current_budget <= 0:
        logger.warning("[deploy_disc] Budget exhausted ($%.4f); skipping discovery.", current_budget)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": "budget exhausted"},
            "budget_remaining_usd": current_budget,
        }

    try:
        response, budget = await gateway.dispatch(
            messages=list(messages), role=NodeRole.PLANNING,
            budget_remaining_usd=current_budget,
        )

        from harness.trust import validate_discovery_json
        discovery_data, trust_errors = validate_discovery_json(response.content)
        if trust_errors:
            logger.error("[deploy_disc] Discovery response failed trust validation: %s", trust_errors)
            return {
                "messages": messages,
                "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                "budget_remaining_usd": budget,
            }

        complete = discovery_data.get("complete", False)
        modules = discovery_data.get("modules", [])
        total_q = sum(len(m.get("questions", [])) for m in modules)
        critical_count = sum(1 for m in modules for q in m.get("questions", []) if q.get("critical"))

        messages.append({"role": "assistant", "content": json.dumps(discovery_data, sort_keys=True)})

        logger.info("[deploy_disc] Round %d: %d questions (%d critical). Complete=%s budget=$%.4f",
                     question_count + 1, total_q, critical_count, complete, budget)

        return {
            "messages": messages,
            "discovery_questions": discovery_data,
            "current_gate": "DEPLOYMENT",
            "budget_remaining_usd": budget,
            "node_state": {
                "current_node": "deployment_discovery",
                "discovery_complete": complete,
                "discovery_question_count": question_count + 1,
                "discovery_critical_remaining": critical_count if not complete else 0,
            },
        }
    except Exception as exc:
        logger.exception("[deploy_disc] Discovery failed: %s", exc)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": str(exc)},
            "budget_remaining_usd": current_budget,
        }


async def write_spec_node(state: AgentState) -> dict[str, Any]:
    """
    Serializes the full discovery transcript into SPEC_REQUIREMENTS.md,
    SPEC_ARCHITECTURE.md, or DEPLOYMENT_BLUEPRINT.md based on current_gate.
    Compiles all Q&A from the conversation history into a comprehensive Markdown document.
    """
    gate = state.get("current_gate", "REQUIREMENTS")
    workspace = state.get("workspace_path", os.getcwd())
    output_dir = os.path.join(workspace, "docs")

    if gate == "REQUIREMENTS":
        path_key = "spec_requirements_path"
    elif gate == "ARCHITECTURE":
        path_key = "spec_architecture_path"
    else:
        path_key = "deployment_blueprint_path"

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        # docs/ exists but is not a directory, or perms blocked creation.
        logger.error("[write_spec] Cannot create %s: %s", output_dir, exc)
        return {
            path_key: "",
            "current_gate": gate,
            "node_state": {
                "current_node": "write_spec",
                "spec_written": False,
                "spec_write_error": f"makedirs failed: {exc}",
            },
        }

    messages = state.get("messages", [])

    # Build the specification from the full conversation
    sections: list[str] = []
    sections.append(f"# {gate.capitalize()} Specification\n")
    sections.append("*Auto-generated from exhaustive discovery process.*\n")

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user" and ("discovery" in content[:50].lower() or "You are a" in content[:100]):
            continue  # Skip the discovery prompts themselves

        if role == "assistant" and content.strip().startswith("{"):
            try:
                data = json.loads(content)
                if "modules" in data:
                    sections.append("\n## Discovery Questions\n")
                    for mod in data.get("modules", []):
                        sections.append(f"### {mod.get('name', 'Module')}\n")
                        for q in mod.get("questions", []):
                            marker = " **[CRITICAL]**" if q.get("critical") else ""
                            sections.append(f"- {q.get('id', '?')}:{marker} {q.get('text', '')}")
                    sections.append("")
            except json.JSONDecodeError:
                pass
        elif role in ("user", "assistant"):
            sections.append(f"### {role.upper()}\n{content[:3000]}\n")

    spec_content = "\n".join(sections)

    if gate == "REQUIREMENTS":
        spec_path = os.path.join(output_dir, "SPEC_REQUIREMENTS.md")
    elif gate == "ARCHITECTURE":
        spec_path = os.path.join(output_dir, "SPEC_ARCHITECTURE.md")
    else:
        spec_path = os.path.join(output_dir, "DEPLOYMENT_BLUEPRINT.md")

    # Change-request mode: prepend a revision header listing the active
    # CR IDs and preserve the prior content below. This stops the
    # destructive overwrite that would otherwise drop a previously
    # approved spec on every existing-project session.
    final_content = spec_content
    if state.get("change_request_mode", False) and os.path.isfile(spec_path):
        prior_content = ""
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                prior_content = f.read()
        except OSError as exc:
            logger.warning(
                "[write_spec] Could not read prior %s for revision merge: "
                "%s. Falling back to overwrite.", spec_path, exc,
            )
        if prior_content:
            cr_records = state.get("change_request_files", []) or []
            cr_ids = ", ".join(f"CR-{r['cr_id']}" for r in cr_records) or "CR-?"
            session_id = state.get("session_id", "<unknown-session>")
            revision_header = (
                f"## Revision: {cr_ids} — session {session_id}\n"
                f"\n"
                f"_(Existing spec preserved verbatim below; this section "
                f"captures the delta proposed by the listed change "
                f"requests. The discovery interview's inline "
                f"`<!-- BEGIN CR-N -->` / `<!-- END CR-N -->` markers in "
                f"the body of the spec link each modified passage to "
                f"its originating request.)_\n\n"
            )
            final_content = revision_header + spec_content + "\n\n---\n\n" + prior_content
            logger.info(
                "[write_spec] Change-request mode: prepending revision "
                "header for %s onto existing %s (%d chars preserved).",
                cr_ids, spec_path, len(prior_content),
            )

    try:
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(final_content)
        logger.info("[write_spec] %s written (%d chars).", spec_path, len(final_content))
    except OSError as exc:
        # Don't silently claim success — the gatekeeper that follows will try
        # to read this path. Propagate the failure so routing can react.
        logger.error("[write_spec] Failed to write %s: %s", spec_path, exc)
        return {
            path_key: "",
            "current_gate": gate,
            "node_state": {
                "current_node": "write_spec",
                "spec_written": False,
                "spec_write_error": str(exc),
            },
        }

    return {
        path_key: spec_path,
        "current_gate": gate,
        "node_state": {"current_node": "write_spec", "spec_written": True},
    }


# ---------------------------------------------------------------------------
# Reviewer LLM nodes (DOC_REVIEWER + CODE_REVIEWER)
# ---------------------------------------------------------------------------

_SPEC_REVIEW_SYSTEM_PROMPT = """You are an independent reviewer of a software requirements/architecture specification. \
A different LLM drafted the spec; your job is to critique it adversarially.

Return STRICT JSON with this exact top-level shape and NOTHING else (no prose, no markdown fences):
{
  "completeness": ["string description of what is missing"],
  "contradictions": ["..."],
  "ambiguity": ["..."],
  "missing_edge_cases": ["..."],
  "security_gaps": ["..."],
  "testability": ["..."],
  "followup_questions": [
    {"id": "R1", "text": "question for the human author", "critical": true}
  ]
}

Each array may be empty if no issues are found in that category. \
Follow-up questions should be precise and answerable by the human in one or two sentences. \
Mark a question critical=true only if its answer would change the architecture or invalidate the spec."""


_SPEC_REVISE_INSTRUCTION_TEMPLATE = """The original specification draft is below, followed by an independent reviewer's critique JSON. \
Produce a fully revised specification document that addresses every actionable item in the critique. \
Output ONLY the revised Markdown — no preamble, no postscript, no code fences.

## Original Spec ({gate})
{original_spec}

## Reviewer Critique JSON
{critique_json}
"""


def _review_followups_to_discovery_shape(critique: dict[str, Any], gate: str) -> list[dict[str, Any]]:
    """Convert reviewer follow-up questions into the modules-and-questions shape
    that the existing discovery_interview_loop renders natively. All follow-ups
    are marked critical=False so the user can finalize with DONE — the original
    discovery phase already enforced critical-unknowns; this second pass is for
    refinement, not blocking."""
    raw = critique.get("followup_questions", []) or []
    if not isinstance(raw, list):
        return []

    questions: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        qid = str(item.get("id", "")).strip() or f"R{idx + 1}"
        questions.append({"id": qid, "text": text, "critical": False})

    if not questions:
        return []

    module_name = f"Reviewer Follow-ups ({gate.capitalize()})"
    return [{"name": module_name, "questions": questions}]


async def review_and_revise_spec(
    spec_path: str,
    gate: str,
    *,
    gateway: Any,
    budget_remaining_usd: float,
    user_goal: str,
    llm_dispatch_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the independent doc-reviewer critique + revise pass on a spec
    file. Writes ``SPEC_{REQUIREMENTS,ARCHITECTURE}_REVIEW.md`` alongside
    the spec and overwrites ``spec_path`` with the revised version.

    Used by:
      - ``spec_review_node`` (graph path, after discovery).
      - ``harness.cli.cmd_run`` (pre-flight path, after
        ``synthesize_requirements``) — the reviewer fires whenever
        ``doc_reviewer_primary`` is configured, independent of
        ``--spec-discovery``.

    Returns a dict with: ``review_path``, ``critique`` (parsed dict),
    ``new_budget_usd``, ``token_usage_list`` (list of provider Usage
    objects for the caller to aggregate), ``ok`` (bool). On any
    non-fatal failure ``ok=False`` and the caller falls through.
    """
    from harness.gateway import NodeRole

    result: dict[str, Any] = {
        "ok": False,
        "review_path": None,
        "critique": None,
        "new_budget_usd": budget_remaining_usd,
        "token_usage_list": [],
    }

    if gate == "REQUIREMENTS":
        review_filename = "SPEC_REQUIREMENTS_REVIEW.md"
    elif gate == "ARCHITECTURE":
        review_filename = "SPEC_ARCHITECTURE_REVIEW.md"
    else:
        return result

    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            original_spec = f.read()
    except OSError as exc:
        logger.warning("[spec_review] Cannot read %s: %s", spec_path, exc)
        return result

    critique_user_prompt = (
        f"## User Goal\n{user_goal}\n\n"
        f"## Specification Under Review ({gate})\n{original_spec}\n\n"
        "Produce the JSON critique now."
    )
    critique_messages: list[MessageDict] = [
        MessageDict(role="system", content=_SPEC_REVIEW_SYSTEM_PROMPT),
        MessageDict(role="user", content=critique_user_prompt),
    ]

    # Resolve the doc_reviewer continue_on_length flag. JSON critique
    # continuation is RISKY — concatenating cycles often yields
    # malformed JSON; the parse fallback below treats it as empty and
    # skips revision. Default is False; see _llm_dispatch_comment in
    # config/config.json.
    _doc_continue_enabled = False
    if isinstance(llm_dispatch_config, dict):
        _doc_continue_map = (
            llm_dispatch_config.get("continue_on_length") or {}
        )
        _doc_continue_enabled = bool(
            _doc_continue_map.get(
                "doc_reviewer",
                _CONTINUE_ON_LENGTH_DEFAULTS.get("doc_reviewer", False),
            )
        )

    try:
        critique_response, new_budget = await gateway.dispatch(
            messages=critique_messages,
            role=NodeRole.DOC_REVIEWER,
            budget_remaining_usd=budget_remaining_usd,
        )
    except Exception as exc:
        logger.warning("[spec_review] Reviewer dispatch failed: %s — passing through.", exc)
        return result

    async def _critique_dispatch(msgs, budget_remaining):
        return await gateway.dispatch(
            messages=list(msgs),
            role=NodeRole.DOC_REVIEWER,
            budget_remaining_usd=budget_remaining,
        )

    critique_response, new_budget, _critique_chunks = await _continue_on_length(
        initial_response=critique_response,
        initial_budget=new_budget,
        messages=critique_messages,
        dispatch=_critique_dispatch,
        continue_prompt=(
            "You hit the output token cap mid-critique. Continue the "
            "JSON critique — emit additional issues / gaps / clarity "
            "items not yet included. Stay inside the same JSON object."
        ),
        enabled=_doc_continue_enabled,
        role_label="spec_review:critique",
    )
    if len(_critique_chunks) > 1:
        critique_response.content = "\n".join(
            c for c in _critique_chunks if c
        )
    result["new_budget_usd"] = new_budget
    result["token_usage_list"].append(critique_response.usage)

    try:
        from harness.trust import _strip_code_fences
        critique_text = _strip_code_fences(critique_response.content)
    except Exception:
        critique_text = critique_response.content.strip()

    try:
        critique = json.loads(critique_text)
        if not isinstance(critique, dict):
            raise ValueError("critique JSON must be an object")
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[spec_review] Critique was not valid JSON (%s) — passing through.", exc)
        return result
    result["critique"] = critique

    review_path = os.path.join(os.path.dirname(spec_path), review_filename)
    try:
        with open(review_path, "w", encoding="utf-8") as f:
            f.write(f"# {gate.capitalize()} Spec Review\n\n")
            f.write("*Generated by the independent doc-reviewer LLM.*\n\n")
            f.write("```json\n")
            f.write(json.dumps(critique, indent=2))
            f.write("\n```\n")
        result["review_path"] = review_path
    except OSError as exc:
        logger.warning("[spec_review] Failed to write %s: %s", review_path, exc)

    revise_prompt = _SPEC_REVISE_INSTRUCTION_TEMPLATE.format(
        gate=gate,
        original_spec=original_spec,
        critique_json=json.dumps(critique, indent=2, sort_keys=True),
    )
    revise_messages: list[MessageDict] = [
        MessageDict(role="system", content="You are a senior specification author. Output clean Markdown only."),
        MessageDict(role="user", content=revise_prompt),
    ]
    try:
        revised_response, new_budget = await gateway.dispatch(
            messages=revise_messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=new_budget,
        )

        async def _revise_dispatch(msgs, budget_remaining):
            return await gateway.dispatch(
                messages=list(msgs),
                role=NodeRole.PLANNING,
                budget_remaining_usd=budget_remaining,
            )

        revised_response, new_budget, _revise_chunks = await _continue_on_length(
            initial_response=revised_response,
            initial_budget=new_budget,
            messages=revise_messages,
            dispatch=_revise_dispatch,
            continue_prompt=(
                "You hit the output token cap mid-revise. Continue "
                "the revised spec markdown from where you stopped. "
                "Do not restart from the top."
            ),
            enabled=_doc_continue_enabled,
            role_label="spec_review:revise",
        )
        if len(_revise_chunks) > 1:
            revised_response.content = "\n".join(
                c for c in _revise_chunks if c
            )
        result["new_budget_usd"] = new_budget
        result["token_usage_list"].append(revised_response.usage)
    except Exception as exc:
        logger.warning("[spec_review] Revise dispatch failed: %s — leaving original spec.", exc)
        revised_response = None

    if revised_response is not None:
        revised_text = revised_response.content.strip()
        if revised_text:
            try:
                with open(spec_path, "w", encoding="utf-8") as f:
                    f.write(revised_text)
                logger.info("[spec_review] Revised spec written to %s.", spec_path)
            except OSError as exc:
                logger.warning("[spec_review] Failed to overwrite %s: %s", spec_path, exc)

    result["ok"] = True
    return result


async def spec_review_node(state: AgentState) -> dict[str, Any]:
    """
    Independent LLM critiques the freshly written spec, then the primary
    planning model revises it. Only fires for REQUIREMENTS and ARCHITECTURE
    gates; DEPLOYMENT short-circuits.

    Activation is purely by configuration: if `doc_reviewer_primary` is empty
    in .harness_config.json, this node is a no-op and the graph follows the
    existing path (write_spec_node → human_gatekeeper_node).
    """
    gate = state.get("current_gate", "REQUIREMENTS")
    loop_counter = dict(state.get("loop_counter", {}))
    counter = loop_counter.get("review_spec", 0)

    gateway = get_gateway()
    gateway_config = get_gateway_config()

    doc_reviewer_primary = ""
    max_cycles = 1
    if gateway_config is not None:
        doc_reviewer_primary = getattr(gateway_config, "doc_reviewer_primary", "") or ""
        max_cycles = int(getattr(gateway_config, "max_doc_review_cycles", 1) or 0)

    # Determine which spec path to operate on.
    if gate == "REQUIREMENTS":
        path_key = "spec_requirements_path"
    elif gate == "ARCHITECTURE":
        path_key = "spec_architecture_path"
    else:
        logger.info("[spec_review] gate=%s — out of reviewer scope, passing through.", gate)
        return {"node_state": {"current_node": "spec_review", "skipped": True}}

    spec_path = str(state.get(path_key, "") or "")
    budget = state.get("budget_remaining_usd", 0.0)

    if not doc_reviewer_primary:
        logger.info("[spec_review] doc_reviewer_primary not configured — skipping.")
        return {"node_state": {"current_node": "spec_review", "skipped": True}}
    if gateway is None:
        logger.info("[spec_review] No gateway available — skipping.")
        return {"node_state": {"current_node": "spec_review", "skipped": True}}
    if counter >= max_cycles:
        logger.info("[spec_review] cycle cap reached (%d/%d) — passing through.", counter, max_cycles)
        return {"node_state": {"current_node": "spec_review", "skipped": True}}
    if budget < 0.10:
        logger.info("[spec_review] budget too low ($%.4f) — skipping.", budget)
        return {"node_state": {"current_node": "spec_review", "skipped": True}}
    if not spec_path or not os.path.isfile(spec_path):
        logger.info("[spec_review] spec file missing (%s) — skipping.", spec_path)
        return {"node_state": {"current_node": "spec_review", "skipped": True}}

    # Short user-goal summary — keeps the reviewer prompt cheap and prevents
    # it from re-litigating the discovery process.
    messages = state.get("messages", [])
    user_goal = ""
    if len(messages) >= 2:
        user_goal = str(messages[1].get("content", ""))[:1000]

    # Run the shared critique + revise pass.
    review_result = await review_and_revise_spec(
        spec_path,
        gate,
        gateway=gateway,
        budget_remaining_usd=budget,
        user_goal=user_goal,
        llm_dispatch_config=state.get("llm_dispatch_config", {}),
    )
    if not review_result["ok"]:
        # Helper logged the reason; pass through.
        return {"node_state": {"current_node": "spec_review", "skipped": True}}

    new_budget = review_result["new_budget_usd"]
    review_path = review_result["review_path"]
    critique = review_result["critique"] or {}

    # Aggregate token tracker across whichever dispatches actually fired.
    token_tracker = state.get("token_tracker", {})
    for usage in review_result["token_usage_list"]:
        token_tracker = gateway.aggregate_tokens(token_tracker, usage)

    loop_counter["review_spec"] = counter + 1

    # Shape follow-up questions for the discovery interview loop. The loop
    # reads state["discovery_questions"], so populate that. We also reset the
    # discovery counters so the loop renders these as a fresh round.
    followups = _review_followups_to_discovery_shape(critique, gate)
    discovery_payload: dict[str, Any] = {
        "modules": followups,
        "complete": False,
        "summary": (
            f"Reviewer raised {sum(len(m['questions']) for m in followups)} "
            "follow-up question(s)."
        ) if followups else "Reviewer found no critical issues.",
    }
    delta: dict[str, Any] = {
        "messages": list(messages),
        "token_tracker": token_tracker,
        "budget_remaining_usd": new_budget,
        "loop_counter": loop_counter,
        "reviewer_followups": followups,
        "current_gate": gate,
        "node_state": {
            "current_node": "spec_review",
            "skipped": False,
            "review_path": review_path,
            "followup_count": sum(len(m["questions"]) for m in followups),
            # critical=False for every reviewer question (see helper) so the
            # user can DONE through after the second pass.
            "discovery_critical_remaining": 0,
            "discovery_complete": False,
            "discovery_question_count": 0,
        },
    }
    if gate == "REQUIREMENTS":
        delta["reviewer_comments_requirements"] = json.dumps(critique, sort_keys=True)
    if followups:
        delta["discovery_questions"] = discovery_payload
    return delta


_CODE_REVIEW_SYSTEM_PROMPT = """You are an independent reviewer of code generated by another LLM. \
Critique it adversarially for correctness, security, performance, idiomatic style, and missing tests.

Return STRICT JSON with this exact shape and NOTHING else (no prose, no markdown fences):
{
  "findings": [
    {
      "file": "path/to/file.ext",
      "line": 42,
      "severity": "high|medium|low",
      "category": "correctness|security|performance|idiomatic|missing_tests",
      "suggestion": "concrete description of the change to make"
    }
  ]
}

`findings` may be empty if the code is clean. Be specific: vague suggestions like "improve error handling" are useless. \
Each finding must name a file and (where applicable) a line number."""


async def code_review_node(state: AgentState) -> dict[str, Any]:
    """
    Independent LLM critiques freshly compiled code, then the patcher model
    incorporates the feedback. Activation is purely by configuration: empty
    code_reviewer_primary == no-op, falling through to security_scan_node.
    """
    loop_counter = dict(state.get("loop_counter", {}))
    counter = loop_counter.get("review_code", 0)
    modified_files = list(state.get("modified_files", []))
    workspace = state.get("workspace_path", os.getcwd())
    budget = state.get("budget_remaining_usd", 0.0)

    gateway = get_gateway()
    gateway_config = get_gateway_config()

    code_reviewer_primary = ""
    max_cycles = 1
    if gateway_config is not None:
        code_reviewer_primary = getattr(gateway_config, "code_reviewer_primary", "") or ""
        max_cycles = int(getattr(gateway_config, "max_code_review_cycles", 1) or 0)

    if not code_reviewer_primary:
        logger.info("[code_review] code_reviewer_primary not configured — skipping.")
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}
    if gateway is None:
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}
    if counter >= max_cycles:
        logger.info("[code_review] cycle cap reached (%d/%d) — passing through.", counter, max_cycles)
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}
    if budget < 0.10:
        logger.info("[code_review] budget too low ($%.4f) — skipping.", budget)
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}
    if not modified_files:
        logger.info("[code_review] no modified_files — skipping.")
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}

    # Snapshot up to 20 files, 2000 lines each, to bound token cost.
    snapshot_chunks: list[str] = []
    file_cap = 20
    line_cap = 2000
    for path in modified_files[:file_cap]:
        abs_path = path if os.path.isabs(path) else os.path.join(workspace, path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[:line_cap]
        except OSError as exc:
            snapshot_chunks.append(f"### {path}\n(could not read: {exc})\n")
            continue
        snapshot_chunks.append(
            f"### {path}\n```\n{''.join(lines)}\n```\n"
        )

    from harness.gateway import NodeRole

    critique_user_prompt = (
        "## Modified Files\n" + "\n".join(snapshot_chunks) +
        "\n\nProduce the JSON critique now."
    )
    critique_messages: list[MessageDict] = [
        MessageDict(role="system", content=_CODE_REVIEW_SYSTEM_PROMPT),
        MessageDict(role="user", content=critique_user_prompt),
    ]

    try:
        critique_response, new_budget = await gateway.dispatch(
            messages=critique_messages,
            role=NodeRole.CODE_REVIEWER,
            budget_remaining_usd=budget,
        )
    except Exception as exc:
        logger.warning("[code_review] Reviewer dispatch failed: %s — passing through.", exc)
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}

    # Continuation on finish_reason=="length" — opt-in via
    # llm_dispatch.continue_on_length.code_reviewer. JSON critique
    # continuation is RISKY (see _llm_dispatch_comment in
    # config/config.json); default is False.
    _code_continue_enabled = _resolve_continue_on_length(state, "code_reviewer")

    async def _code_critique_dispatch(msgs, budget_remaining):
        return await gateway.dispatch(
            messages=list(msgs),
            role=NodeRole.CODE_REVIEWER,
            budget_remaining_usd=budget_remaining,
        )

    critique_response, new_budget, _code_critique_chunks = await _continue_on_length(
        initial_response=critique_response,
        initial_budget=new_budget,
        messages=critique_messages,
        dispatch=_code_critique_dispatch,
        continue_prompt=(
            "You hit the output token cap mid-critique. Continue the "
            "JSON critique — emit additional findings only. Stay "
            "inside the same JSON object."
        ),
        enabled=_code_continue_enabled,
        role_label="code_review:critique",
    )
    if len(_code_critique_chunks) > 1:
        critique_response.content = "\n".join(
            c for c in _code_critique_chunks if c
        )

    token_tracker = gateway.aggregate_tokens(state.get("token_tracker", {}), critique_response.usage)

    try:
        from harness.trust import _strip_code_fences
        critique_text = _strip_code_fences(critique_response.content)
    except Exception:
        critique_text = critique_response.content.strip()

    try:
        critique = json.loads(critique_text)
        findings = critique.get("findings", []) if isinstance(critique, dict) else []
        if not isinstance(findings, list):
            findings = []
        # Schema-drift sentinel: response parsed cleanly, but the canonical
        # "findings" key is missing AND the LLM produced something else at
        # the top level (e.g. "issues" / "comments"). The previous behaviour
        # silently treated this as "code is clean" and skipped the security
        # gate — same silent-drift shape as the discovery 0-modules bug.
        if isinstance(critique, dict) and "findings" not in critique:
            other_keys = sorted(k for k in critique.keys() if not k.startswith("_"))
            if other_keys:
                logger.warning(
                    "[code_review] Response parsed cleanly but the canonical "
                    "'findings' key is missing — the LLM used non-canonical "
                    "top-level keys=%s. Treating as no-findings (code passes "
                    "review) BUT the security gate would skip; verify the "
                    "review JSON in docs/CODE_REVIEW.md before trusting the "
                    "result.", other_keys,
                )
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[code_review] Critique was not valid JSON (%s) — passing through.", exc)
        findings = []
        critique = {"findings": []}

    # Persist the critique regardless of whether findings exist.
    docs_dir = os.path.join(workspace, "docs")
    try:
        os.makedirs(docs_dir, exist_ok=True)
        review_path = os.path.join(docs_dir, "CODE_REVIEW.md")
        with open(review_path, "w", encoding="utf-8") as f:
            f.write("# Code Review\n\n")
            f.write("*Generated by the independent code-reviewer LLM.*\n\n")
            if not findings:
                f.write("No findings.\n")
            else:
                f.write("```json\n")
                f.write(json.dumps(critique, indent=2))
                f.write("\n```\n")
    except OSError as exc:
        logger.warning("[code_review] Failed to write CODE_REVIEW.md: %s", exc)

    loop_counter["review_code"] = counter + 1

    if not findings:
        logger.info("[code_review] No findings — passing through to security scan.")
        return {
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "reviewer_comments_code": json.dumps(critique, sort_keys=True),
            "node_state": {
                "current_node": "code_review",
                "skipped": False,
                "repatched": False,
                "findings_count": 0,
            },
        }

    # Re-patch call. Use the patcher format-reminder so the patcher model
    # returns the same SEARCH/REPLACE block syntax patching_node uses.
    from harness.patcher import process_llm_patch_output

    _CODE_REVIEW_FORMAT_REMINDER = """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the patch block syntax below. No explanations, no markdown fences, no text outside the blocks.

Valid blocks:
<<<CREATE_FILE>>>
file: path/to/file.ext
content:
<complete file contents>
<<<END_CREATE_FILE>>>

<<<REPLACE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to find>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>

<<<DELETE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to delete>
<<<END_DELETE_BLOCK>>>

<<<INSERT_AT_BLOCK>>>
file: path/to/file.ext
anchor: <function or class name>
placement: before|after
content:
<lines to insert>
<<<END_INSERT_AT_BLOCK>>>

Generate the patches that address every finding below. Output only the blocks — no other text."""

    repatch_user_prompt = (
        f"## Reviewer Findings (JSON)\n```json\n{json.dumps(critique, indent=2, sort_keys=True)}\n```\n\n"
        f"## Modified Files Snapshot\n" + "\n".join(snapshot_chunks) +
        f"\n\n{_CODE_REVIEW_FORMAT_REMINDER}"
    )
    repatch_messages: list[MessageDict] = [
        MessageDict(role="system", content="You are a senior software engineer. Apply the reviewer's feedback as patch blocks only."),
        MessageDict(role="user", content=repatch_user_prompt),
    ]

    try:
        repatch_response, new_budget = await gateway.dispatch(
            messages=repatch_messages,
            role=NodeRole.PATCHING,
            budget_remaining_usd=new_budget,
        )

        async def _code_repatch_dispatch(msgs, budget_remaining):
            return await gateway.dispatch(
                messages=list(msgs),
                role=NodeRole.PATCHING,
                budget_remaining_usd=budget_remaining,
            )

        repatch_response, new_budget, _code_repatch_chunks = await _continue_on_length(
            initial_response=repatch_response,
            initial_budget=new_budget,
            messages=repatch_messages,
            dispatch=_code_repatch_dispatch,
            continue_prompt=(
                "You hit the output token cap mid-repatch. Continue "
                "with the remaining patch blocks. Same DSL — no "
                "prose outside blocks."
            ),
            enabled=_code_continue_enabled,
            role_label="code_review:repatch",
        )
        if len(_code_repatch_chunks) > 1:
            repatch_response.content = "\n".join(
                c for c in _code_repatch_chunks if c
            )
    except Exception as exc:
        logger.warning("[code_review] Re-patch dispatch failed: %s — proceeding without re-patch.", exc)
        return {
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "reviewer_comments_code": json.dumps(critique, sort_keys=True),
            "node_state": {
                "current_node": "code_review",
                "skipped": False,
                "repatched": False,
                "findings_count": len(findings),
            },
        }

    token_tracker = gateway.aggregate_tokens(token_tracker, repatch_response.usage)

    allowed_paths = _build_patcher_allowlist(workspace)
    patch_results, new_modified_files = await process_llm_patch_output(
        repatch_response.content,
        workspace,
        modified_files,
        allowed_paths=allowed_paths,
    )
    success_count = sum(1 for r in patch_results if r.success)
    repatched = success_count > 0

    logger.info(
        "[code_review] Findings=%d, re-patched=%d/%d files (success=%s).",
        len(findings), success_count, len(patch_results), repatched,
    )

    # Audit #18 — record any files this node mutated since the last green
    # compile. The router consults this set before terminal exit. Existing
    # ``route_after_code_review`` already re-routes to compiler_node when
    # repatched=True, which will clear the set; tracking the file names
    # here is defence-in-depth for any future node that mutates without
    # setting its own re-verify flag.
    prior_pending = list(state.get("pending_mutations") or [])
    delta = [f for f in (new_modified_files or []) if f not in modified_files]
    if delta:
        prior_pending = prior_pending + delta

    return {
        "modified_files": new_modified_files,
        "token_tracker": token_tracker,
        "budget_remaining_usd": new_budget,
        "loop_counter": loop_counter,
        "reviewer_comments_code": json.dumps(critique, sort_keys=True),
        "pending_mutations": prior_pending,
        "node_state": {
            "current_node": "code_review",
            "skipped": False,
            "repatched": repatched,
            "findings_count": len(findings),
            "repatch_success": success_count,
            "repatch_total": len(patch_results),
        },
    }


async def generate_deployment_spec_node(state: AgentState) -> dict[str, Any]:
    """
    Runs workspace telemetry + architecture spec through the deployment
    synthesizer to produce DEPLOYMENT_BLUEPRINT.md. Sets current_gate = "DEPLOYMENT".
    """
    workspace = state.get("workspace_path", os.getcwd())
    output_dir = os.path.join(workspace, "docs")
    os.makedirs(output_dir, exist_ok=True)

    blueprint_path = os.path.join(output_dir, "DEPLOYMENT_BLUEPRINT.md")

    # Use the deploy module's telemetry + synthesis
    try:
        from harness.deploy import scan_workspace_telemetry

        telemetry = scan_workspace_telemetry(workspace)

        gateway = get_gateway()
        if gateway:
            arch_path = state.get("spec_architecture_path", os.path.join(output_dir, "SPEC_ARCHITECTURE.md"))
            arch_content = ""
            if os.path.isfile(arch_path):
                with open(arch_path, "r", encoding="utf-8") as f:
                    arch_content = f.read()[:5000]

            from harness.gateway import NodeRole
            prompt = f"""You are a DevOps Architect. Produce a DEPLOYMENT_BLUEPRINT.md based on the telemetry and architecture below.

## Workspace Telemetry
```json
{json.dumps(telemetry, indent=2, default=str, sort_keys=True)}
```

## Architecture Spec
{arch_content if arch_content else "(no SPEC_ARCHITECTURE.md)"}

## Output Sections
1. Container Inventory (services, base images, build contexts)
2. Network Topology (bridge networks, exposed ports)
3. Volume Configuration (named volumes, bind mounts)
4. Environment Variables (per service)
5. Health Check Configuration
6. Scaling & Resource Limits
7. Deployment Sequence (order of container startup)

Output as clean Markdown. Do NOT wrap the document in an outer
```markdown … ``` fence — emit the body directly, starting with the
first heading. Fences are reserved for code blocks INSIDE the document."""

            messages = [
                {"role": "system", "content": "You are a DevOps architect. Output clean Markdown — no outer code fences around the document."},
                {"role": "user", "content": prompt},
            ]
            current_budget = state.get("budget_remaining_usd", 0.0)
            if current_budget <= 0:
                logger.warning("[deployment_spec] Budget exhausted; falling back to deterministic blueprint.")
                content = f"# DEPLOYMENT_BLUEPRINT.md\n\nWorkspace: {workspace}\n\nTelemetry: {json.dumps(telemetry, indent=2, default=str)}\n\n(LLM blueprint skipped — budget exhausted.)"
                budget = current_budget
            else:
                response, budget = await gateway.dispatch(
                    messages=messages, role=NodeRole.PLANNING,
                    budget_remaining_usd=current_budget,
                )
                content = response.content.strip()
        else:
            content = f"# DEPLOYMENT_BLUEPRINT.md\n\nWorkspace: {workspace}\n\nTelemetry: {json.dumps(telemetry, indent=2, default=str)}"
            budget = state.get("budget_remaining_usd", 0.0)

        with open(blueprint_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("[deployment_spec] DEPLOYMENT_BLUEPRINT.md written (%d chars).", len(content))
    except Exception as exc:
        logger.warning("[deployment_spec] Failed: %s", exc)
        budget = state.get("budget_remaining_usd", 0.0)
        with open(blueprint_path, "w", encoding="utf-8") as f:
            f.write(f"# DEPLOYMENT_BLUEPRINT.md\n\nError: {exc}")

    return {
        "deployment_blueprint_path": blueprint_path,
        "current_gate": "DEPLOYMENT",
        "budget_remaining_usd": budget,
        "node_state": {"current_node": "generate_deployment_spec"},
    }


# ---------------------------------------------------------------------------
# 6cc. Installation Doc Synthesis (end-of-run, greenfield builds)
# ---------------------------------------------------------------------------

async def installation_doc_node(state: AgentState) -> dict[str, Any]:
    """End-of-run node that synthesises ``docs/INSTALLATION.md``.

    Best-effort: any failure is logged and the run still terminates
    cleanly via the END edge — a missing install doc must not roll back
    a build the operator already approved. Gated on
    ``state["install_doc"]`` so change-request runs skip the LLM call.

    Inputs read from state:
        * ``workspace_path`` — the generated project root.
        * ``spec_architecture_path`` — for the Build & Run section slice.
        * ``node_state["deployment"]["blueprint"]`` — when --deploy-dev
          produced one; ``None`` otherwise (the synth helper renders §5
          conditionally on that signal).
    """
    if not state.get("install_doc", False):
        logger.debug("[installation_doc] --install-doc=false; skipping.")
        return {}

    workspace_path = state.get("workspace_path", "")
    if not workspace_path or not os.path.isdir(workspace_path):
        logger.warning(
            "[installation_doc] Workspace path missing or not a directory (%r); "
            "skipping INSTALLATION.md generation.", workspace_path,
        )
        return {}

    architecture_path = state.get("spec_architecture_path") or os.path.join(
        workspace_path, "docs", "SPEC_ARCHITECTURE.md",
    )
    blueprint: Optional[dict[str, Any]] = None
    deployment_state = (state.get("node_state") or {}).get("deployment") or {}
    if isinstance(deployment_state, dict):
        candidate = deployment_state.get("blueprint")
        if isinstance(candidate, dict):
            blueprint = candidate

    output_dir = os.path.join(workspace_path, "docs")
    gateway = get_gateway()
    if gateway is None:
        logger.warning(
            "[installation_doc] LLM gateway unavailable; skipping INSTALLATION.md."
        )
        return {}

    from harness.cli import synthesize_installation

    try:
        install_path = await synthesize_installation(
            workspace_path=workspace_path,
            architecture_path=architecture_path,
            output_dir=output_dir,
            gateway=gateway,
            blueprint=blueprint,
        )
    except Exception as exc:  # noqa: BLE001 — never fail the run on a doc miss
        logger.warning(
            "[installation_doc] Synthesis failed; INSTALLATION.md not written: %s",
            exc,
        )
        return {}

    return {
        "installation_doc_path": install_path,
        "node_state": {"current_node": "installation_doc"},
    }


# ---------------------------------------------------------------------------
# 6d. Route After Discovery Interview
# ---------------------------------------------------------------------------

def route_after_discovery(state: AgentState) -> str:
    """
    Routes after discovery_interview_loop completes.

    Decision matrix:
        discovery_complete == true  → write_spec_node (serialize to .md)
        discovery_complete == false AND critical > 0 → loop back to discovery node
        discovery_complete == false AND no critical → write_spec_node (non-critical only)
        user typed 'DONE' with critical remaining → alert + loop back
        user typed 'SUSPEND' → save & quit (route to END)

    Loop ceiling: the per-gate discovery_question_count is compared against
    GatewayConfig.max_discovery_iterations (default 10). When the cap is
    reached we forward to write_spec_node with the current draft instead of
    looping again — prevents a runaway loop from a confused user or a
    misbehaving LLM that keeps emitting follow-ups.
    """
    node_state = state.get("node_state", {})
    complete = node_state.get("discovery_complete", False)
    critical = node_state.get("discovery_critical_remaining", 0)
    gate = state.get("current_gate", "REQUIREMENTS")
    rounds = int(node_state.get("discovery_question_count", 0) or 0)

    if node_state.get("hitl_suspend"):
        logger.info("[router] Discovery: developer chose to suspend. Routing to END.")
        return "__end__"

    # Loop-cap short-circuit. Honoured before the user_done_with_critical
    # branch so a user who keeps typing DONE doesn't trap us either.
    cap = 10
    try:
        gw_config = get_gateway_config()
        if gw_config is not None:
            cap = int(getattr(gw_config, "max_discovery_iterations", 10) or 10)
    except Exception:  # noqa: BLE001
        pass
    if rounds >= cap:
        logger.warning(
            "[router] Discovery hit max_discovery_iterations (%d). "
            "Forwarding to write_spec_node with the current draft.",
            cap,
        )
        return "write_spec_node"

    if node_state.get("user_done_with_critical"):
        # User tried to exit with critical unknowns — route back to the correct discovery node
        if gate == "DEPLOYMENT":
            return "deployment_discovery_node"
        elif gate == "ARCHITECTURE":
            return "architecture_discovery_node"
        else:
            return "discovery_interview_loop"

    if complete or (not complete and critical == 0):
        return "write_spec_node"

    # Still incomplete with critical unknowns → loop back to the right discovery node
    if gate == "REQUIREMENTS":
        return "requirements_discovery_node"
    elif gate == "ARCHITECTURE":
        return "architecture_discovery_node"
    else:
        return "deployment_discovery_node"


# ---------------------------------------------------------------------------
# 6e. Route After Human Gatekeeper
# ---------------------------------------------------------------------------

def route_after_gatekeeper(state: AgentState) -> str:
    """
    Routes based on current_gate and the gatekeeper's decision in node_state.
    
    The human_gatekeeper_node sets node_state.gatekeeper_action to:
        - "approve" → proceed to next phase
        - "refine"  → loop back to the current phase's generator
        - "manual"  → proceed (manual edits done)
        - "suspend" → save & quit (route to END)
    """
    gate = state.get("current_gate", "")
    action = state.get("node_state", {}).get("gatekeeper_action", "approve")

    if action == "suspend":
        logger.info("[router] Gatekeeper: developer chose to suspend. Routing to END.")
        return "__end__"

    if action == "refine":
        # Route back to the corresponding discovery node for re-generation
        if gate == "REQUIREMENTS":
            return "requirements_discovery_node"
        elif gate == "ARCHITECTURE":
            return "architecture_discovery_node"
        elif gate == "DEPLOYMENT":
            return "generate_deployment_spec_node"

    # approve or manual: proceed forward
    if gate == "REQUIREMENTS":
        return "architecture_discovery_node"
    elif gate == "ARCHITECTURE":
        return "patching_node"
    elif gate == "DEPLOYMENT":
        return "deployment_node"

    return "__end__"


# ---------------------------------------------------------------------------
# 7. Route After HITL: Always Back to Compiler
# ---------------------------------------------------------------------------

def route_after_security_scan(state: AgentState) -> Literal["repair_node", "human_intervention_node", "deployment_discovery_node", "deployment_node", "installation_doc_node", "compiler_node", "__end__"]:
    """
    Conditional edge router executed after security_scan_node completes.

    Decision matrix:
        No security findings AND Flutter project       → END (mobile builds don't
                                                          fit the docker-compose
                                                          deploy pipeline)
        No security findings AND dev_deployment=False  → END (operator opt-in gate)
        No security findings AND dev_deployment=True
          AND cd_discovery=True                        → deployment_discovery_node
                                                          (synthesise blueprint
                                                          via LLM interview)
        No security findings AND dev_deployment=True
          AND cd_discovery=False                       → deployment_node
                                                          (skip discovery; the
                                                          deploy step synthesises
                                                          the blueprint from
                                                          workspace telemetry
                                                          alone)
        Security findings AND sec_attempts < 2         → repair_node
        Security findings AND sec_attempts >= 2        → human_intervention_node
        budget_remaining <= 0                          → human_intervention_node

    Routes to repair_node so security findings travel through the same
    formatter (``_format_diagnostics_for_repair``) and escalation logic
    (cheap → reasoning model on round 3) that compile errors use. The
    repair_node prompt detects the scanner-prefixed error_codes
    populated by security_scan_node and switches its framing sentence
    to make the security context explicit to the LLM. After repair, the
    compiler verifies the fix and security_scan_node re-verifies clean.
    """
    budget_remaining: float = state.get("budget_remaining_usd", 0.0)
    loop_counter: dict[str, Any] = state.get("loop_counter", {})
    sec_attempts: int = loop_counter.get("security", 0)
    _sec_cfg_raw = state.get("security_scan_config", {}) or {}
    sec_cfg: dict[str, Any] = _sec_cfg_raw if isinstance(_sec_cfg_raw, dict) else {}
    max_sec_attempts: int = int(sec_cfg.get("max_security_fix_attempts", 2))
    # Hard ceiling for the HITL ping-pong loop. In auto-resume mode
    # (--hitl-repair=false / CI / no TTY) the HITL menu re-routes back
    # to compiler_node without resetting the security counter, so the
    # same finding can survive arbitrarily many trips through the gate
    # — turning into a thrash loop that only ends when something else
    # (semgrep timeout, budget, code-review cap) shadows it. Past this
    # ceiling the run terminates rather than pretend the loop is
    # productive. The default ratio is 3x; operators can tune via the
    # `hard_security_loop_ceiling` config key.
    from harness.security import _HARD_SECURITY_CEILING_MULTIPLIER
    hard_ceiling = int(sec_cfg.get(
        "hard_security_loop_ceiling",
        max_sec_attempts * _HARD_SECURITY_CEILING_MULTIPLIER,
    ))
    compiler_errors = state.get("compiler_errors", [])

    # Check budget first
    if budget_remaining <= 0.0:
        logger.warning("[router] Budget exhausted ($%.4f). Routing to HITL.", budget_remaining)
        return "human_intervention_node"

    # If no compiler_errors populated, security scan passed
    if not compiler_errors:
        # Audit #18 — defensive pre-exit verification. When the operator
        # opts in (compiler.pre_exit_verify=true) AND a post-green node has
        # mutated source since the last green compile AND we haven't
        # already burned a final-verify pass, re-run the compiler before
        # any terminal route. Capped at one re-verify per session via
        # ``loop_counter["final_verify"]`` so a flaky test can't trap us.
        pending = state.get("pending_mutations") or []
        already_verified = bool(loop_counter.get("final_verify", 0))
        if (
            state.get("pre_exit_verify", False)
            and pending
            and not already_verified
        ):
            logger.info(
                "[router] pre_exit_verify: %d pending mutation(s) since "
                "last green compile (%s). Re-running compiler before exit.",
                len(pending), pending[:5],
            )
            try:
                from harness.observability import emit_event
                emit_event(
                    "pre_exit_verify_triggered",
                    pending_count=len(pending),
                    pending_files=list(pending)[:20],
                )
            except Exception:  # noqa: BLE001 — telemetry must never block
                pass
            return "compiler_node"

        # Mobile short-circuit (M-1): Flutter projects don't fit the
        # docker-compose deployment model. Skip deployment_* and end after
        # the security scan passes. iOS builds need macOS anyway and would
        # fail in the Linux sandbox; Android artifacts live in
        # build/app/outputs/ for the user to pick up.
        from harness.impact import _is_flutter_project
        workspace_path = state.get("workspace_path", "")
        if workspace_path and _is_flutter_project(workspace_path):
            logger.info(
                "[router] Flutter project detected. Skipping deploy pipeline "
                "(M-1). Routing to installation_doc_node before END."
            )
            return "installation_doc_node"
        # Operator opt-in gate (--deploy-dev). When the flag is false the
        # harness stops after a clean security scan; the operator can
        # inspect the generated code, then re-run with --deploy-dev true
        # to enter the deployment phase. Stop via installation_doc_node so
        # greenfield runs still get their docs/INSTALLATION.md.
        if not state.get("dev_deployment", False):
            logger.info(
                "[router] Security scan clean. --deploy-dev not set; "
                "skipping deployment phase. Routing to installation_doc_node "
                "before END."
            )
            return "installation_doc_node"
        # --cd-discovery picks between the LLM-driven blueprint pipeline
        # (deployment_discovery_node → interview → gatekeeper →
        # deployment_node) and the telemetry-only fast path (straight to
        # deployment_node, which synthesises the blueprint from workspace
        # telemetry — and the deployment_defaults section of config.json
        # where present — without an LLM interview). Both end in
        # deployment_node.
        if not state.get("cd_discovery", False):
            logger.info(
                "[router] Security scan clean. --cd-discovery=false; "
                "skipping deployment discovery interview and routing "
                "straight to deployment_node (telemetry-only blueprint)."
            )
            return "deployment_node"
        logger.info("[router] Security scan clean. Routing to deployment discovery.")
        return "deployment_discovery_node"

    # Security findings exist — first check the HARD ceiling. Past this
    # point the HITL ping-pong is thrashing (most often: the LLM keeps
    # proposing a patch the spec-driven allowlist rejects, e.g. edits
    # into ``docs/SPEC_ARCHITECTURE.md``). Terminate the run rather than
    # keep looping. The build is incomplete — exit will be non-zero.
    if sec_attempts >= hard_ceiling:
        logger.error(
            "[router] Security HITL ping-pong hard ceiling reached "
            "(attempts=%d, ceiling=%d). %d finding(s) survived %d HITL "
            "resume(s) without being fixed. Terminating without deployment — "
            "operator must inspect findings manually.",
            sec_attempts, hard_ceiling, len(compiler_errors),
            sec_attempts - max_sec_attempts,
        )
        return "__end__"

    # Security findings exist — check soft attempt limit (route to HITL)
    if sec_attempts >= max_sec_attempts:
        logger.warning(
            "[router] Security fix limit reached (%d/%d). %d finding(s) remain. "
            "Routing to HITL (hard ceiling at %d).",
            sec_attempts, max_sec_attempts, len(compiler_errors), hard_ceiling,
        )
        return "human_intervention_node"

    logger.info(
        "[router] %d security finding(s) detected. Routing to repair_node for fix (attempt %d/%d).",
        len(compiler_errors), sec_attempts, max_sec_attempts,
    )
    return "repair_node"


# ---------------------------------------------------------------------------
# 6e. Route After Deployment: success → installation_doc, else compiler logic
# ---------------------------------------------------------------------------

def route_after_deployment(state: AgentState) -> Literal[
    "installation_doc_node",
    "security_scan_node",
    "repair_node",
    "human_intervention_node",
    "test_generation_node",
]:
    """Route after deployment_node terminates.

    Successful health check (``node_state.deployment.success == True``)
    short-circuits to ``installation_doc_node`` so greenfield runs end at
    a documented artifact instead of re-entering the compile/scan loop.
    Anything else delegates to the standard post-build router so failed
    deploys still route through repair / HITL / test generation just like
    a normal build failure.
    """
    ns = state.get("node_state") or {}
    deployment = ns.get("deployment") if isinstance(ns, dict) else None
    if isinstance(deployment, dict) and deployment.get("success") is True:
        logger.info(
            "[router] Deployment succeeded. Routing to installation_doc_node "
            "before END."
        )
        return "installation_doc_node"
    return route_after_compiler(state)


# ---------------------------------------------------------------------------
# 7. Route After HITL: Always Back to Compiler
# ---------------------------------------------------------------------------

def route_after_hitl(state: AgentState) -> Literal["compiler_node", "__end__"]:
    """
    After human intervention, always route back to compiler_node for re-validation.
    Exceptions:
        - Developer chose to abandon ([q]) → END with git rollback
        - Developer chose to suspend ([s]) → END without rollback

    Note: Memory cleanse for HITL resolution is handled inside
    compiler_node when exit_code == 0 after the re-validation build passes.
    """
    node_state: dict[str, Any] = state.get("node_state", {})
    if node_state.get("hitl_suspend", False):
        logger.info("[router] HITL: Developer chose to suspend. Routing to END.")
        return "__end__"
    if node_state.get("hitl_abandon", False):
        logger.info("[router] HITL: Developer chose to abandon. Routing to END.")
        return "__end__"
    logger.info("[router] HITL resolved. Routing to compiler_node for re-validation.")
    return "compiler_node"


# ---------------------------------------------------------------------------
# 8. Route After Planning: Always to Patching
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 9. Graph Builder — Assembles the Full StateGraph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Construct and return the complete LangGraph StateGraph with all nodes,
    edges, and conditional routing logic.

    Graph topology:

        START
          │
          ▼
    planning_node
          │
          ▼
    patching_node
          │
          ▼
    compiler_node ────── exit 0 ──────────► END
          │
          │ exit ≠ 0
          ▼
    ┌── route_after_compiler ──┐
    │                          │
    │ repairs < 3    repairs >= 3
    ▼                          ▼
    repair_node         human_intervention_node
    │                          │
    └──────► compiler_node ◄───┘
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        raise ImportError(
            "langgraph is required. Install with: pip install langgraph>=0.4.0"
        )

    # Create the graph bound to our AgentState type
    graph = StateGraph(AgentState)

    # Register core graph nodes
    graph.add_node("planning_node", planning_node)
    graph.add_node("patching_node", patching_node)
    graph.add_node("compiler_node", compiler_node)
    graph.add_node("repair_node", repair_node)
    graph.add_node("human_intervention_node", human_intervention_node)

    # Register security scan node for SAST + secret auditing (post-compile gatekeeper)
    from harness.security import security_scan_node as _security_scan_node
    graph.add_node("security_scan_node", _security_scan_node)  # type: ignore[type-var]

    # Register lintgate node for deterministic format verification
    from harness.lintgate import lintgate_node as _lintgate_node
    graph.add_node("lintgate_node", _lintgate_node)  # type: ignore[type-var]

    # Register speculative node for multi-variant branching
    from harness.speculative import speculate_node as _speculate_node
    graph.add_node("speculative_node", _speculate_node)  # type: ignore[type-var]

    # Register test-generation node — runs after speculative branching, before
    # lintgate, so the deterministic lint pass formats generated tests too.
    from harness.test_generation import (
        test_generation_node as _test_generation_node,
        route_after_test_generation as _route_after_test_generation,
    )
    graph.add_node("test_generation_node", _test_generation_node)  # type: ignore[type-var]

    # Register change-request ingest entry point and the one-shot
    # reverse-engineer architecture synthesis node that runs once on
    # first contact with a repo that lacks SPEC_ARCHITECTURE.md.
    graph.add_node("ingest_change_requests_node", ingest_change_requests_node)
    graph.add_node(
        "reverse_engineer_architecture_node",
        reverse_engineer_architecture_node,
    )

    # Register exhaustive discovery nodes
    graph.add_node("requirements_discovery_node", requirements_discovery_node)
    graph.add_node("architecture_discovery_node", architecture_discovery_node)
    graph.add_node("write_spec_node", write_spec_node)
    from harness.cli import discovery_interview_loop as _discovery_interview_loop
    graph.add_node("discovery_interview_loop", _discovery_interview_loop)  # type: ignore[type-var]

    # Register human gatekeeper node for final review
    from harness.cli import human_gatekeeper_node as _human_gatekeeper_node
    graph.add_node("human_gatekeeper_node", _human_gatekeeper_node)  # type: ignore[type-var]

    # Register reviewer LLM nodes (DOC_REVIEWER + CODE_REVIEWER). Each is a
    # no-op when its corresponding *_reviewer_primary slot is unset in config.
    graph.add_node("spec_review_node", spec_review_node)
    graph.add_node("code_review_node", code_review_node)

    # Register deployment spec node
    graph.add_node("generate_deployment_spec_node", generate_deployment_spec_node)

    # =====================================================================
    # Discovery pipeline (exhaustive zero-unknowns protocol):
    # START → requirements_discovery_node → discovery_interview_loop
    #   loop: discovery_interview ← route_after_discovery → requirements_discovery (if incomplete)
    #   done: discovery_interview → route_after_discovery → write_spec_node
    #   write_spec_node → human_gatekeeper_node (REQUIREMENTS final review)
    #   gatekeeper: approve → architecture_discovery_node
    #   architecture_discovery_node → discovery_interview_loop
    #   loop: discovery_interview ← route_after_discovery → architecture_discovery (if incomplete)
    #   done: discovery_interview → route_after_discovery → write_spec_node
    #   write_spec_node → human_gatekeeper_node (ARCHITECTURE final review)
    #   gatekeeper: approve → patching_node
    # =====================================================================

    # =====================================================================
    # START routing: change_request_mode wins (ingest → patching for PR-1;
    # ingest → discovery delta-mode → gatekeeper → patching in PR-2+).
    # Otherwise: skip_discovery → patching_node, else → discovery pipeline.
    # `route_after_start` is module-level (see below) so tests can call it
    # without building the full graph.
    # =====================================================================
    graph.add_conditional_edges(
        START,
        route_after_start,
        {
            "requirements_discovery_node": "requirements_discovery_node",
            "patching_node": "patching_node",
            "ingest_change_requests_node": "ingest_change_requests_node",
        },
    )

    # ingest_change_requests_node hands off to the one-shot reverse-
    # engineer pass (no-op when SPEC_ARCHITECTURE.md already exists),
    # which then hands off to the discovery pipeline so change requests
    # flow through requirements → interview → spec → gatekeeper →
    # patching. The ingest node sets ``skip_discovery=False`` in its
    # returned state so the downstream nodes don't short-circuit to
    # gatekeeper-only.
    graph.add_edge(
        "ingest_change_requests_node", "reverse_engineer_architecture_node",
    )
    graph.add_edge(
        "reverse_engineer_architecture_node", "requirements_discovery_node",
    )

    # Requirements discovery loop
    graph.add_edge("requirements_discovery_node", "discovery_interview_loop")
    graph.add_conditional_edges(
        "discovery_interview_loop",
        route_after_discovery,
        {
            "requirements_discovery_node": "requirements_discovery_node",
            "architecture_discovery_node": "architecture_discovery_node",
            "deployment_discovery_node": "deployment_discovery_node",
            "write_spec_node": "write_spec_node",
            # Self-loop: user typed DONE with critical unknowns → re-display menu
            "discovery_interview_loop": "discovery_interview_loop",
            "__end__": END,
        },
    )

    # After write_spec, route through the doc-reviewer LLM (no-op if reviewer
    # is not configured). The reviewer either revises the spec and asks the
    # user follow-up questions, or short-circuits to the human gatekeeper.
    graph.add_edge("write_spec_node", "spec_review_node")

    def route_after_spec_review(state: AgentState) -> Literal[
        "discovery_interview_loop", "human_gatekeeper_node"
    ]:
        """The human gatekeeper is the FINAL review gate. The LLM reviewer
        either bails (no model configured / cycle cap / no follow-ups) and we
        route straight to the gatekeeper, or it produces follow-up questions
        and we route through one more pass of the discovery interview loop
        before the gatekeeper sees the upgraded spec."""
        if state.get("skip_discovery", False):
            return "human_gatekeeper_node"
        node_state = state.get("node_state", {}) or {}
        if node_state.get("skipped", False):
            return "human_gatekeeper_node"
        followups = state.get("reviewer_followups", []) or []
        if followups:
            logger.info(
                "[router] spec_review produced %d follow-up question(s) — routing to discovery interview.",
                sum(len(m.get("questions", [])) for m in followups),
            )
            return "discovery_interview_loop"
        return "human_gatekeeper_node"

    graph.add_conditional_edges(
        "spec_review_node",
        route_after_spec_review,
        {
            "discovery_interview_loop": "discovery_interview_loop",
            "human_gatekeeper_node": "human_gatekeeper_node",
        },
    )

    # Gatekeeper routes: approve → next phase, refine → loop back
    graph.add_conditional_edges(
        "human_gatekeeper_node",
        route_after_gatekeeper,
        {
            "requirements_discovery_node": "requirements_discovery_node",
            "architecture_discovery_node": "architecture_discovery_node",
            "generate_deployment_spec_node": "generate_deployment_spec_node",
            "patching_node": "patching_node",
            "deployment_node": "deployment_node",
            "__end__": END,
        },
    )

    # Architecture discovery loop (entered from gatekeeper approve)
    graph.add_edge("architecture_discovery_node", "discovery_interview_loop")

    # =====================================================================
    # Code generation pipeline (after ARCHITECTURE gate approved):
    # patching → speculative → lintgate → compiler
    # =====================================================================
    graph.add_edge("patching_node", "speculative_node")
    # speculative_node → test_generation_node → conditional edge:
    #   - tests passed (or skipped) → lintgate_node
    #   - tests failed → repair_node (TEST_FAILURE diagnostics surfaced)
    #   - env_misconfig (no LLM gateway / max iterations) → human_intervention_node
    graph.add_edge("speculative_node", "test_generation_node")
    graph.add_conditional_edges(
        "test_generation_node",
        _route_after_test_generation,
        {
            "lintgate_node": "lintgate_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
        },
    )
    graph.add_edge("lintgate_node", "compiler_node")

    # =====================================================================
    # Compiler → repair loop + security gate
    #
    # On a clean compile, divert through code_review_node first. The reviewer
    # may apply a re-patch that needs re-validation, so it routes back to
    # compiler_node if it re-patched; otherwise it proceeds to security_scan.
    # The reviewer is a no-op when code_reviewer_primary is unset in config,
    # so this path is behavior-compatible with the pre-reviewer flow.
    # =====================================================================
    graph.add_conditional_edges(
        "compiler_node",
        route_after_compiler,
        {
            # Clean compile → code reviewer (no-op pass-through if unconfigured).
            "security_scan_node": "code_review_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
            # pytest exit=5 with source present → generate tests instead of
            # burning a repair iteration on a non-error.
            "test_generation_node": "test_generation_node",
        },
    )

    def route_after_code_review(state: AgentState) -> Literal[
        "compiler_node", "security_scan_node"
    ]:
        """If the reviewer re-patched, re-validate by going back through the
        compiler. If not (no findings, cycle cap, or unconfigured), proceed
        straight to security_scan_node — exactly today's behavior."""
        node_state = state.get("node_state", {}) or {}
        if node_state.get("repatched", False):
            logger.info("[router] code_review re-patched — re-running compiler.")
            return "compiler_node"
        return "security_scan_node"

    graph.add_conditional_edges(
        "code_review_node",
        route_after_code_review,
        {
            "compiler_node": "compiler_node",
            "security_scan_node": "security_scan_node",
        },
    )

    # After repair, go directly to compiler (skip lintgate to avoid reformatting
    # the file between repair attempts, which would break SEARCH/REPLACE matching).
    graph.add_edge("repair_node", "compiler_node")

    # =====================================================================
    # Deployment discovery pipeline (after security scan clean):
    # security_scan → deployment_discovery_node → discovery_interview_loop
    #   loop: discovery_interview ← route_after_discovery → deployment_discovery (if incomplete)
    #   done: discovery_interview → route_after_discovery → write_spec_node
    #   write_spec_node → human_gatekeeper_node (DEPLOYMENT final review)
    #   gatekeeper: approve → deployment_node → END
    #   gatekeeper: refine → deployment_discovery_node
    # =====================================================================

    # Register deployment discovery node
    graph.add_node("deployment_discovery_node", deployment_discovery_node)

    # Register installation_doc_node — best-effort end-of-run docs that
    # writes <workspace>/docs/INSTALLATION.md from telemetry + manifests
    # + the Build & Run section of SPEC_ARCHITECTURE.md (+ the deployment
    # blueprint when present). Gated on state["install_doc"] so the no-op
    # path is a single state-pass-through for change-request runs.
    graph.add_node("installation_doc_node", installation_doc_node)
    graph.add_edge("installation_doc_node", END)

    # Route security_scan clean → deployment discovery (or installation_doc
    # for Flutter / --deploy-dev=false success exits); findings → repair_node
    # so security fixes go through the same _format_diagnostics_for_repair +
    # escalation path as compile errors.
    graph.add_conditional_edges(
        "security_scan_node",
        route_after_security_scan,
        {
            "deployment_discovery_node": "deployment_discovery_node",
            # Fast-path for --deploy-dev=true + --cd-discovery=false:
            # the router routes straight to deployment_node, which
            # synthesises the blueprint from workspace telemetry alone
            # (with state.deployment_defaults pulled from the
            # ``deployment_defaults`` section of config.json where set).
            "deployment_node": "deployment_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
            "installation_doc_node": "installation_doc_node",
            "__end__": END,
        },
    )

    # Deployment discovery → interview loop
    graph.add_edge("deployment_discovery_node", "discovery_interview_loop")

    # Register deployment node
    from harness.deploy import deployment_node as _deployment_node
    graph.add_node("deployment_node", _deployment_node)  # type: ignore[type-var]

    # Deployment conditional edges. On success the router below short-
    # circuits to installation_doc_node (which then routes to END);
    # otherwise it falls through to the standard route_after_compiler
    # outcomes (repair / HITL / re-scan).
    graph.add_conditional_edges(
        "deployment_node",
        route_after_deployment,
        {
            "installation_doc_node": "installation_doc_node",
            "security_scan_node": "security_scan_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
            "test_generation_node": "test_generation_node",
        },
    )

    # After HITL resolution, go back to compiler (or END if abandoned)
    graph.add_conditional_edges(
        "human_intervention_node",
        route_after_hitl,
        {
            "compiler_node": "compiler_node",
            "__end__": END,
        },
    )

    logger.info("[graph] Full StateGraph topology assembled successfully.")
    return graph


# ---------------------------------------------------------------------------
# 12. Compile Helper — Builds the Graph with Checkpointer
# ---------------------------------------------------------------------------

def compile_graph(checkpointer: Any = None) -> Any:
    """
    Compile the StateGraph with an optional checkpointer for persistence.

    Args:
        checkpointer: A LangGraph-compatible checkpointer (MemorySaver, AsyncSqliteSaver, etc.).
                       If None, the graph runs without persistence.

    Returns:
        A compiled LangGraph runnable ready for .ainvoke() or .astream().
    """
    graph = build_graph()
    if checkpointer is not None:
        compiled = graph.compile(checkpointer=checkpointer)
    else:
        compiled = graph.compile()
    return compiled


# ---------------------------------------------------------------------------
# 13. Graph Execution Entry Point
# ---------------------------------------------------------------------------

async def _rewind_suspended_checkpoint(compiled_graph: Any, config: dict[str, Any]) -> None:
    """If the resumed checkpoint ended at END via hitl_suspend, rewind it.

    LangGraph treats Save & Quit ([s]) as a normal terminal transition: the
    suspend flag flips on, ``route_after_hitl`` routes to ``__end__``, and
    the next ``ainvoke(None)`` short-circuits because there's no pending
    work. The fix is to stamp a "resumed" state onto the checkpoint as if
    ``human_intervention_node`` had just returned with the user pressing
    [r] Resume — outgoing edges re-fire and the graph routes back to
    ``compiler_node``.

    A no-op when the checkpoint isn't in the suspended-terminal state:
        - state.next is non-empty: graph is paused mid-flight; normal resume.
        - hitl_suspend is False: graph ended naturally (exit 0, abandon, etc.).
    """
    try:
        state = await compiled_graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001 — defensive; never block resume
        logger.debug("[run_graph] Could not read checkpoint state for rewind check: %s", exc)
        return

    if state is None:
        return
    # state.next is a tuple of pending node names; empty when at END.
    if getattr(state, "next", None):
        return
    values = getattr(state, "values", None) or {}
    node_state = values.get("node_state", {}) or {}
    if not node_state.get("hitl_suspend"):
        return

    # Distinguish suspend source. Discovery-interview suspends must NOT
    # rewind through human_intervention_node — that route ends up at
    # compiler_node and re-runs the entire build/security pipeline,
    # throwing away work the user already completed (code, tests, security
    # scan) before the discovery phase even started.
    suspended_from = node_state.get("suspended_from")
    # Back-compat: pre-tag checkpoints have no suspended_from. Infer from
    # current_gate + current_node — discovery nodes set current_node to
    # "<phase>_discovery" before stamping hitl_suspend via the interview loop.
    if not suspended_from:
        cur_node = node_state.get("current_node", "")
        gate = values.get("current_gate", "")
        if (
            gate in ("REQUIREMENTS", "ARCHITECTURE", "DEPLOYMENT")
            and isinstance(cur_node, str)
            and cur_node.endswith("_discovery")
        ):
            suspended_from = "discovery_interview"
        else:
            suspended_from = "hitl_menu"

    if suspended_from == "discovery_interview":
        # Re-enter the gate-appropriate discovery node so its unconditional
        # outgoing edge fires straight into discovery_interview_loop, which
        # then re-renders the cached discovery_questions checkpointed by the
        # prior session. aupdate_state(as_node=...) records "this node just
        # returned this state" without re-executing the node — so the LLM
        # is NOT re-called and no budget is burned on rewind.
        gate = values.get("current_gate", "")
        gate_to_node = {
            "REQUIREMENTS": "requirements_discovery_node",
            "ARCHITECTURE": "architecture_discovery_node",
            "DEPLOYMENT": "deployment_discovery_node",
        }
        rewind_node = gate_to_node.get(gate, "deployment_discovery_node")
        cleared = dict(node_state)
        cleared["hitl_suspend"] = False
        cleared.pop("suspended_from", None)
        cleared.pop("user_done_with_critical", None)
        logger.info(
            "[run_graph] Resume rewind: discovery-interview suspend in %s phase. "
            "Re-firing %s → discovery_interview_loop with cached questions.",
            gate or "UNKNOWN", rewind_node,
        )
        try:
            await compiled_graph.aupdate_state(
                config,
                {"node_state": cleared},
                as_node=rewind_node,
            )
        except Exception as exc:  # noqa: BLE001 — log but don't crash resume
            logger.warning(
                "[run_graph] Failed to rewind discovery-suspend checkpoint: %s. "
                "Resume may no-op; user can re-run discovery via a fresh session.",
                exc,
            )
        return

    # Default path: HITL menu Save & Quit. Mirror the [r] Resume branch of
    # hitl_menu_loop — clear suspend flags, mark HITL resolved, reset loop
    # counter to allow one more repair cycle.
    cleared = dict(node_state)
    cleared["hitl_suspend"] = False
    cleared["hitl_active"] = False
    cleared["hitl_awaiting_input"] = False
    cleared["hitl_resolved"] = True
    cleared.pop("suspended_from", None)

    gw = get_gateway()
    max_repair = (
        int(getattr(gw.config, "max_patch_repair_iterations", 5))
        if gw is not None else 3
    )
    # total_repairs = max-1 → one more repair attempt before HITL re-triggers.
    next_total = max(0, max_repair - 1)

    logger.info(
        "[run_graph] Resume rewind: prior session ended via Save & Quit. "
        "Clearing hitl_suspend and resetting loop counter (total_repairs=%d) "
        "so the graph re-enters compiler_node.",
        next_total,
    )

    try:
        await compiled_graph.aupdate_state(
            config,
            {
                "node_state": cleared,
                "loop_counter": {
                    "patching": 0,
                    "repair": 0,
                    "compiler": 0,
                    "total_repairs": next_total,
                },
            },
            as_node="human_intervention_node",
        )
    except Exception as exc:  # noqa: BLE001 — log but don't crash resume
        logger.warning(
            "[run_graph] Failed to rewind suspended checkpoint: %s. "
            "Resume will likely no-op; user may need to start a fresh session.",
            exc,
        )


async def run_graph(
    *,
    workspace_path: str,
    prompt: str,
    build_command: str,
    allow_network: bool = False,
    budget_usd: float = 2.00,
    session_id: str = "",
    checkpointer: Any = None,
    thread_id: Optional[str] = None,
    spec_override: Optional[str] = None,
    skip_discovery: bool = False,
    is_resume: bool = False,
    lintgate_config: Optional[dict[str, Any]] = None,
    deployment_config: Optional[dict[str, Any]] = None,
    deployment_defaults: Optional[dict[str, Any]] = None,
    sandbox_config: Optional[dict[str, Any]] = None,
    test_generation_config: Optional[dict[str, Any]] = None,
    speculative_config: Optional[dict[str, Any]] = None,
    compiler_config: Optional[dict[str, Any]] = None,
    change_request_mode: bool = False,
    change_requests_dir_abs: str = "",
    archive_target_dir: str = "",
    change_requests_config: Optional[dict[str, Any]] = None,
    dev_deployment: bool = False,
    cd_discovery: bool = False,
    install_doc: bool = False,
    repo_memory_config: Optional[dict[str, Any]] = None,
    repo_index_config: Optional[dict[str, Any]] = None,
    llm_dispatch_config: Optional[dict[str, Any]] = None,
) -> AgentState:
    """
    Execute the full agent graph from start to finish.

    This is the primary async entry point called by the CLI layer.

    Args:
        workspace_path: Absolute path to the target repository.
        prompt: The engineering task description.
        build_command: The shell command to build/verify the project.
        allow_network: Whether the sandbox permits outbound network.
        budget_usd: Hard dollar cap for LLM calls.
        session_id: Human-readable session identifier (falls back to UUIDv4).
        checkpointer: LangGraph checkpointer for persistence. If None, runs ephemerally.
        thread_id: LangGraph thread ID for checkpoint lookups. Auto-generated if None.

    Returns:
        The final AgentState after graph completion.
    """
    import uuid

    # Generate session identifiers
    if not session_id:
        session_id = str(uuid.uuid4())
    if thread_id is None:
        thread_id = session_id  # Use session_id as thread_id for simplicity

    # Build initial state
    initial_state = create_initial_state(
        workspace_path=workspace_path,
        initial_prompt=prompt,
        build_command=build_command,
        allow_network=allow_network,
        budget_usd=budget_usd,
        session_id=session_id,
        spec_override=spec_override,
        skip_discovery=skip_discovery,
        change_request_mode=change_request_mode,
        change_requests_dir_abs=change_requests_dir_abs,
        archive_target_dir=archive_target_dir,
        change_requests_config=change_requests_config,
        dev_deployment=dev_deployment,
        cd_discovery=cd_discovery,
        install_doc=install_doc,
    )

    # Per-node config sections — read by lintgate_node and deployment_node
    # respectively. These are free-form dicts on the state; nodes consult
    # them via state.get("lintgate_config", {}) etc.
    if lintgate_config is not None:
        initial_state["lintgate_config"] = lintgate_config
    if deployment_config is not None:
        initial_state["deployment_config"] = deployment_config
    if deployment_defaults is not None:
        initial_state["deployment_defaults"] = deployment_defaults
    if test_generation_config is not None:
        initial_state["test_generation_config"] = test_generation_config
    if speculative_config is not None:
        initial_state["speculative_config"] = speculative_config
    if change_requests_config is not None:
        initial_state["change_requests_config"] = change_requests_config
    if repo_memory_config is not None:
        initial_state["repo_memory_config"] = repo_memory_config
    if repo_index_config is not None:
        initial_state["repo_index_config"] = repo_index_config
    if llm_dispatch_config is not None:
        initial_state["llm_dispatch_config"] = llm_dispatch_config
    # Plumb the smoke-check flag into state so compiler_node can read it
    # without reaching out to config (which the graph module doesn't
    # touch directly today).
    if compiler_config is not None:
        initial_state["run_prod_import_smoke_check"] = bool(
            compiler_config.get("run_prod_import_smoke_check", True)
        )
        # Audit #18 — opt-in defensive re-compile before terminal exit when
        # a post-green node mutated source. Default False; the per-node
        # re-routes (route_after_code_review → compiler when repatched)
        # already cover the known cases.
        initial_state["pre_exit_verify"] = bool(
            compiler_config.get("pre_exit_verify", False)
        )

    # Pre-flight toolchain adaptation: flip ``allow_network`` and
    # ``read_only_root`` to match the build command's install needs NOW so
    # the very first compile doesn't waste a cycle on an offline / RO root
    # for a command that needs ``pip install`` / ``npm install``. The
    # ``docker_image`` half of toolchain adaptation is now a no-op — the
    # harness's kitchen-sink BUILDER_IMAGE has every supported toolchain,
    # so there's no per-command image to pick. compiler_node's own call to
    # this helper is idempotent — it becomes a no-op once we've pre-adapted
    # here.
    (
        adapted_cfg,
        adapted_allow_network,
        image_was_adapted,
        network_was_adapted,
        ro_root_was_adapted,
    ) = _apply_toolchain_adaptation(build_command, sandbox_config, allow_network)
    if image_was_adapted:
        logger.info(
            "[run_graph] Pre-flight sandbox docker_image set to %r "
            "to match build toolchain implied by: %s",
            adapted_cfg["docker_image"], build_command,
        )
    if network_was_adapted:
        logger.info(
            "[run_graph] Pre-flight allow_network=True because build command "
            "requires registry access: %s", build_command,
        )
        initial_state["allow_network"] = adapted_allow_network
    if ro_root_was_adapted:
        logger.info(
            "[run_graph] Pre-flight sandbox.read_only_root=False because build "
            "command installs packages into system locations: %s", build_command,
        )
    if sandbox_config is not None or image_was_adapted or ro_root_was_adapted:
        initial_state["sandbox_config"] = adapted_cfg

    # Compile graph with checkpointer
    compiled_graph = compile_graph(checkpointer=checkpointer)

    # Runtime configuration for LangGraph
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "session_id": session_id,
        }
    }

    logger.info(
        "[run_graph] Starting graph execution. thread_id=%s session_id=%s is_resume=%s",
        thread_id, session_id, is_resume,
    )

    # Resume mode: do NOT pass a full initial_state — every key in the
    # input dict overwrites the corresponding channel in the checkpointed
    # state, so a fresh AgentState would reset messages, loop_counter,
    # current_gate, node_state, etc. back to their zero values and the
    # graph would re-enter at START → requirements_discovery_node even
    # though the prior session was deep in architecture_discovery.
    # Passing None tells LangGraph "no input update, just continue from
    # the checkpoint" — the graph picks up at exactly the node that was
    # in-flight when the session was saved.
    if is_resume:
        invoke_input: Optional[AgentState] = None
        logger.info(
            "[run_graph] Resume mode: invoking with no input update; "
            "the graph will continue from the last checkpointed node."
        )
        # Save & Quit ([s] in hitl_menu_loop) routes through route_after_hitl
        # to __end__, leaving the checkpoint at the terminal pseudo-node with
        # node_state.hitl_suspend=True. A naive ainvoke(None) on that
        # checkpoint returns immediately — LangGraph sees no pending work —
        # so the user's `teane resume` would no-op (Final exit_code=N,
        # 0 nodes executed). Rewind the checkpoint as if
        # human_intervention_node just produced a "resume" outcome
        # (hitl_suspend cleared, loop counter rolled back), so the outgoing
        # edge from HITL re-fires and routes to compiler_node — identical
        # to the user having pressed [r] Resume instead of [s] Save & Quit.
        await _rewind_suspended_checkpoint(compiled_graph, config)
    else:
        invoke_input = initial_state

    # Bind the active session_id to the asyncio context so every downstream
    # Gateway.dispatch (across all roles, all nodes, all concurrent
    # speculative variants) can read it for per-call debug-dump filenames.
    # See harness/observability.py: active_session_scope.
    from harness.observability import active_session_scope

    with active_session_scope(session_id):
        # Execute the graph — ainvoke streams all state updates and returns final state
        final_state: AgentState = await compiled_graph.ainvoke(invoke_input, config)

    logger.info("[run_graph] Graph execution complete. Final exit_code=%d", final_state.get("exit_code", -1))
    return final_state
