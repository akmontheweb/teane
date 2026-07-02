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
from harness.spec_files import list_spec_files, read_spec_file

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


# ---------------------------------------------------------------------------
# Flow — which top-level command spawned this graph run. Set by the CLI
# layer (`cmd_build` / `cmd_patch` / `cmd_deploy`); stored in AgentState so
# resume can route back to the right entry edge after a crash. The legacy
# `teane run` command is gone; tests and direct callers default to BUILD.
# ---------------------------------------------------------------------------
FLOW_BUILD = "build"
FLOW_PATCH = "patch"
FLOW_DEPLOY = "deploy"
FLOW_TEST = "test"
_VALID_FLOWS = frozenset({FLOW_BUILD, FLOW_PATCH, FLOW_DEPLOY, FLOW_TEST})


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
    # Sorted ``"<code>::<message>"`` fingerprints of the diagnostics from the
    # CURRENT compiler run, written by compiler_node every time it executes
    # (empty list on exit_code == 0). The rotation is: at the top of
    # compiler_node, whatever is in ``last_diag_fingerprints`` is copied into
    # ``prior_diag_fingerprints`` (i.e. yesterday's "current" is today's
    # "prior") before the new value overwrites it.
    last_diag_fingerprints: list[str]
    # Sorted fingerprints from the PREVIOUS compiler run, populated by
    # compiler_node's rotate-then-write step. Read by repair_node's
    # diagnostic formatter: any group whose fingerprint is in this set is
    # unconditionally promoted to top-N in the next prompt, regardless of
    # cascade rank. This is the empirical override that catches cases
    # where the cascade prior mis-classified a non-upstream error as "may
    # resolve on its own" (e.g. a TS2769 overload mismatch deferred for
    # 3 rounds, then HITL).
    prior_diag_fingerprints: list[str]
    # Raw count of error-severity diagnostics from the CURRENT and PREVIOUS
    # compiler runs. Rotated alongside the fingerprint sets and used by
    # repair_node's no_progress gate as a second signal: when many tests
    # share the same root cause they collapse to ONE fingerprint, so the
    # fingerprint set can stay the same size while the raw count drops
    # (10 → 9 → 8 …). Crediting raw-count shrinkage as progress prevents
    # the no_progress counter from firing on real wins that the
    # fingerprint set can't see.
    last_diag_count: int
    prior_diag_count: int
    # Error codes the LLM explicitly requested to promote into top-N on
    # the NEXT repair iteration (Phase 1.2 escape hatch). Populated by
    # repair_node when it parses ``<<<PROMOTE_DEFERRED>>>`` blocks from
    # the LLM response; consumed by ``_format_diagnostics_for_repair``
    # which bumps any matching group past the cascade prior. Cleared
    # after consumption so a promotion only lasts one round.
    promoted_codes_next_round: list[str]
    token_tracker: TokenTrackerDict
    # Mostly int counters per node ("planning", "patching", "repair", ...),
    # but a few sentinel-string entries piggy-back on the dict too
    # (e.g. "missing_dep_last_symbol", "replace_block_misses_per_file").
    loop_counter: dict[str, Any]
    allow_network: bool
    build_command: str
    budget_remaining_usd: float
    # Snapshot of the session budget cap at boot. Used by HITL display
    # surfaces (cli.hitl_menu_loop) to show ``$remaining / $cap``
    # without hard-coding $2.00. Read-only after create_initial_state.
    budget_initial_usd: float
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
    # installation_doc_node fires at the terminal success edges
    # (--deploy-dev=false clean security scan, and after
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
    # Story-mode fields (Agile decomposition + per-story TDD). When
    # ``decomposition_enabled`` is True, route_after_gatekeeper(ARCHITECTURE)
    # branches to decomposition_node instead of patching_node; the per-story
    # loop sets current_story_id / current_batch_id / story_scope_files as it
    # walks the planned stories table in ``<workspace>/.teane/state.db``. All
    # default to safe values (False / "" / 0 / []) so today's monolithic flow
    # is preserved when ``--no-stories`` is passed or no decomposition has
    # run. MUST be declared here — LangGraph drops unregistered fields on
    # the channel layer (see the discovery_questions warning above).
    decomposition_enabled: bool
    current_story_id: str
    current_batch_id: int
    story_defects_open: int
    stories_db_path: str
    story_scope_files: list[str]
    # Structured architecture summary parsed from
    # ``docs/SPEC_ARCHITECTURE.md`` §11 (the fenced ``jsonc`` block the
    # ``arch_doc_generator`` skill emits). Populated lazily — first by
    # ``decomposition_node`` when it runs (agile mode), and otherwise by
    # ``patching_node`` on its first turn after the arch doc exists.
    # Consumers (patching preamble, batch planner, traceability) read
    # the resolved endpoints / components / contract paths from here so
    # they do not re-derive them on every node call. Empty dict = either
    # the arch doc has no §11 block, the schema_version is unrecognised,
    # or the file is absent — every consumer must tolerate this and
    # fall back to the prose handoff in ``messages[0]``.
    # MUST be declared here — LangGraph drops unregistered channel keys
    # silently (see the discovery_questions warning above).
    arch_summary: dict[str, Any]
    # Snapshot of ``modified_files`` taken when story_loop_node picked the
    # current story. story_complete_node diffs against this to attribute
    # only newly-touched files to the active STORY-N row in file_links.
    # Without the snapshot we'd over-attribute (story 2 inherits story 1's
    # files) — link_file de-dups via INSERT OR IGNORE so it's harmless to
    # the schema, but the TRACEABILITY.md view would show false coupling.
    story_modified_baseline: list[str]
    # Per-batch scope of touched files. Populated by ``patching_node``
    # while ``current_batch_id`` is non-zero; cleared by the batch-commit
    # node when the batch lands. Consumer nodes (code_review,
    # test_generation, security_scan) read this via
    # ``_scope_files_for_batch`` so per-batch verification gates only see
    # files this batch wrote, not the cumulative session set.
    # ``modified_files`` remains the session-cumulative source of truth
    # for git staging and traceability.
    batch_modified_files: list[str]
    # Phase K — per-batch verification-gate progress markers used for
    # mid-batch crash resume. Outer key is ``str(current_batch_id)``;
    # inner dict carries boolean flags like ``compile_passed`` and
    # ``review_passed``. ``compiler_node`` and ``code_review_node`` set
    # their respective flags on clean pass; ``route_after_story_loop``
    # skips already-passed gates when ``batch_complete=True`` so a
    # resumed session doesn't re-run compile+review just to land in the
    # gate where the crash actually happened. ``batch_commit_node``
    # pops the batch's entry as part of its state reset.
    batch_gate_progress: dict[str, dict[str, bool]]
    # Story keys whose patching turn has already run in the CURRENT
    # batch (Phase E.3 cursor-advance fix). Without this, ``story_loop_node``
    # re-picks the same story every cycle because (a) ``_next_story_in_batch``
    # orders in_progress rows before planned rows so a once-patched story
    # always sorts first, and (b) nothing in the per-batch flow marks the
    # story ``done`` between patching turns (story_complete_node is
    # per-story-mode only; batch_commit_node fires at end-of-batch). The
    # router would then bounce patching ⇄ story_loop on STORY-1 until the
    # global ``consecutive_zero_patch_rounds`` tripwire escalated to HITL —
    # a session-burning loop. ``batch_planner_node`` initialises this to
    # ``[]`` for each new batch; ``story_loop_node`` appends
    # ``current_story_id`` on every entry; ``_next_story_in_batch`` skips
    # any story whose key is in the list. Reset to ``[]`` by
    # ``batch_commit_node`` so the next batch starts clean.
    batch_patched_story_keys: list[str]
    # Opt-in: when True (sourced from ``agile_defaults.commit_on_story`` in
    # config.json — there is no longer a CLI flag for this knob),
    # story_complete_node runs `git add . && git commit -m "<STORY-N>:
    # <title>"` after a green story and records the SHA in the commits
    # table. No-op when the workspace isn't a git repo. Default False —
    # git commits are a side effect users should opt into.
    commit_on_story: bool
    # Story planner / TDD knobs from the CLI. ``story_batch_size`` caps how
    # many independent stories the planner pulls into one batch (default
    # DEFAULT_BATCH_SIZE in harness.story_loop).
    #
    # ``story_repair_cap`` (default 3): max total_repairs before
    # ``story_complete_node`` parks the story as ``blocked`` and records
    # a defect. Phase E.3 changed the live routing so the per-batch
    # verification chain runs once after every story has patched, then
    # routes to ``batch_commit_node`` directly — bypassing
    # ``story_complete_node`` in batch-mode runs. The cap is therefore
    # only enforced today when a session runs in legacy non-batch
    # story-mode (no ``batch_planner_node`` in play, e.g. unit tests
    # that bypass the planner). The session-level repair budget that
    # actually governs batch-mode runs is
    # ``gateway.config.max_patch_repair_iterations`` (consulted inside
    # ``route_after_compiler``).
    story_batch_size: int
    story_repair_cap: int
    # Top-level command flow that spawned this run. One of "build" / "patch"
    # / "deploy" (see FLOW_* constants). `route_after_start` consults this
    # field FIRST — it determines the entry edge so the deploy command can
    # short-circuit straight to the deployment chain without touching the
    # discovery / patching nodes, and the patch command can route through
    # `reverse_spec_node` when `--generate-specs` was resolved active. Read
    # by resume so a crashed flow continues at the right edge.
    flow: str
    # Whether the patch flow should reverse-engineer SPEC_REQUIREMENTS.md /
    # SPEC_ARCHITECTURE.md from the existing codebase before reconciling.
    # Set by `cmd_patch` after resolving the `--generate-specs` tri-state.
    # Inert (False) on every other flow.
    generate_specs: bool
    # Merged ``.harness_config.json`` contents (operator config + bundled
    # defaults). Populated by ``create_initial_state`` from the ``config``
    # kwarg the CLI hands in. Nodes that need operator-tunable knobs read
    # them via ``state["harness_config"]`` rather than re-loading the file.
    # The v5 end-of-session audit reads ``harness_config["traceability"]
    # ["enforce"]`` (default True) — see ``installation_doc_node``.
    harness_config: dict[str, Any]

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
    decomposition_enabled: bool = False,
    stories_db_path: str = "",
    commit_on_story: bool = False,
    story_batch_size: int = 5,
    story_repair_cap: int = 3,
    flow: str = FLOW_BUILD,
    generate_specs: bool = False,
    config: Optional[dict[str, Any]] = None,
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
            + _build_and_emit_system_prompt(workspace_path, build_command, config=config)
        )
        # When a user-approved spec already exists (from the pre-flight
        # product_spec_dir refinement), skip the graph's discovery
        # pipeline completely. Otherwise write_spec_node would overwrite
        # the approved SPEC_REQUIREMENTS.md with a minimal
        # conversation-history compilation.
        skip_discovery = True
    else:
        system_prompt = _build_and_emit_system_prompt(workspace_path, build_command, config=config)
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
        budget_initial_usd=budget_usd,
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
        decomposition_enabled=bool(decomposition_enabled),
        current_story_id="",
        current_batch_id=0,
        story_defects_open=0,
        stories_db_path=stories_db_path,
        story_scope_files=[],
        arch_summary={},
        story_modified_baseline=[],
        batch_modified_files=[],
        batch_gate_progress={},
        batch_patched_story_keys=[],
        commit_on_story=bool(commit_on_story),
        story_batch_size=int(story_batch_size),
        story_repair_cap=int(story_repair_cap),
        flow=flow if flow in _VALID_FLOWS else FLOW_BUILD,
        generate_specs=bool(generate_specs),
        # Make the merged config dict visible to downstream nodes.
        # Operators tune ``traceability.enforce`` (Phase 6 audit gate),
        # ``security.*``, ``test_generation.*`` etc. via this dict.
        # Always present (empty dict when no config was passed) so
        # readers can ``.get("section", {}).get("key", default)`` without
        # None-checks.
        harness_config=dict(config or {}),
    )


# ---------------------------------------------------------------------------
# 2a. Batch-scope helpers (per-batch verification pipeline, Phase D)
# ---------------------------------------------------------------------------

def _extend_batch_scope(
    state: "AgentState", new_modified_files: list[str]
) -> list[str]:
    """Append files newly touched by this node to ``batch_modified_files``.

    Used inside patching_node and repair_node returns. Computes the set
    of files added by THIS invocation (``new_modified_files`` minus the
    pre-call ``state["modified_files"]``) and appends them to the
    existing batch scope, preserving insertion order and de-duplicating.

    When ``current_batch_id`` is 0 (monolithic / non-batch mode) the
    function still returns a coherent value — the unchanged existing
    list — so callers can unconditionally pass the result back into the
    state delta without branching."""
    existing_batch = list(state.get("batch_modified_files") or [])
    if not int(state.get("current_batch_id") or 0):
        return existing_batch
    pre_call = set(state.get("modified_files") or [])
    new_in_this_call = [f for f in new_modified_files if f not in pre_call]
    if not new_in_this_call:
        return existing_batch
    seen: set[str] = set(existing_batch)
    out = list(existing_batch)
    for f in new_in_this_call:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _batch_gate_passed(
    state: "AgentState", gate: str,
) -> bool:
    """Return True when ``current_batch_id``'s ``gate`` flag is set
    in ``batch_gate_progress`` (Phase K).

    No batch active → False. Used by routers to skip already-passed
    gates on resume."""
    bid = int(state.get("current_batch_id") or 0)
    if bid <= 0:
        return False
    bgp = state.get("batch_gate_progress") or {}
    if not isinstance(bgp, dict):
        return False
    entry = bgp.get(str(bid)) or {}
    return bool(entry.get(gate))


def _mark_batch_gate(
    state: "AgentState", gate: str,
) -> dict[str, dict[str, bool]]:
    """Return a NEW ``batch_gate_progress`` dict with the current
    batch's ``gate`` flag set to True (Phase K).

    The result is the new value to fold back into the state-delta dict
    returned by a node. No-op (returns the existing dict) when there's
    no active batch."""
    bid = int(state.get("current_batch_id") or 0)
    existing = state.get("batch_gate_progress") or {}
    if not isinstance(existing, dict):
        existing = {}
    if bid <= 0:
        return dict(existing)
    out: dict[str, dict[str, bool]] = {
        k: dict(v) for k, v in existing.items()
    }
    entry = dict(out.get(str(bid)) or {})
    entry[gate] = True
    out[str(bid)] = entry
    return out


def _scope_files_for_consumer(state: "AgentState") -> list[str]:
    """Return the file set that per-batch verification gates should read.

    In batch-mode (``current_batch_id`` non-zero) callers consume
    ``batch_modified_files`` so review / test-gen only see what this
    batch touched. Outside batch-mode they fall back to the cumulative
    session ``modified_files`` — preserving pre-batch behavior.

    Empty ``batch_modified_files`` in batch-mode falls back to
    ``modified_files`` too — the patching phase may not have populated
    the batch list yet on the very first invocation."""
    if int(state.get("current_batch_id") or 0):
        batch = list(state.get("batch_modified_files") or [])
        if batch:
            return batch
    return list(state.get("modified_files") or [])


# Files that conventionally live at workspace root and are exempt from the
# "all source under <root>/" enforcement when a source root is detected.
# patching_node / repair_node / test_generation_node compose the patcher
# Built-in default root-level files the patcher is allowed to modify.
# This matches patcher.root_files in config/config.json. The operator can override
# the list in config.json; for full dynamic configuration support, thread config
# through _build_patcher_allowlist (future enhancement).
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
    # supports JS stacks (vendor/Dockerfile.builder), and the React skill
    # expects these at the workspace root. Without them in the static
    # set, every LLM patch to package.json or tsconfig.json was rejected
    # before the build could repair.
    "package.json", "package-lock.json",
    "npm-shrinkwrap.json",
    "tsconfig.json", "tsconfig.base.json",
    ".npmrc", ".nvmrc", ".node-version",
    # Container deployment — Dockerfile and docker-compose files must be
    # in the allowlist for deployment discovery and synthesis to work.
    # The deployment phase may generate or modify these, and repair nodes
    # may need to adjust them for build fixes.
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Caddyfile", ".dockerignore",
    # Common dev-experience dotfiles. The runtime root scan only picks
    # these up when they already exist on disk, so a fresh greenfield
    # rejected the LLM's first attempt to CREATE them ("[patcher] Skill
    # allowlist rejected patch to .eslintrc.json / .prettierrc /
    # .env.example"). Seeding them statically lets the LLM author
    # idiomatic configs on round 1.
    ".env.example", ".env.sample", ".env.template",
    ".eslintrc", ".eslintrc.json", ".eslintrc.js", ".eslintrc.cjs",
    ".eslintrc.yaml", ".eslintrc.yml", ".eslintignore",
    ".prettierrc", ".prettierrc.json", ".prettierrc.js",
    ".prettierrc.yaml", ".prettierrc.yml", ".prettierignore",
    ".babelrc", ".babelrc.json", ".babelrc.js",
    ".editorconfig", ".gitattributes",
    ".browserslistrc",
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
    ".eslintrc", ".prettierrc", ".babelrc", "tsconfig",
)


def _is_node_config_file(name: str) -> bool:
    """True when ``name`` is a Node/JS tool config worth allowing at root.

    Catches the open-ended families that don't have one canonical filename:
      - ``*.config.{js,cjs,mjs,ts,json}`` — jest, vite, next, tailwind,
        postcss, playwright, rollup, webpack, etc.
      - ``.eslintrc*`` / ``.prettierrc*`` / ``.babelrc*`` — each ships in
        bare, ``.json``, ``.js``, ``.cjs``, and ``.yaml`` forms.
      - ``tsconfig*`` — Vite/Next/Angular/Nx scaffolds emit several
        variants alongside the base file (``tsconfig.app.json``,
        ``tsconfig.node.json``, ``tsconfig.build.json``, etc.).
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


def _read_config_root_files() -> list[str]:
    """Return the operator-configured ``patcher.root_files`` list from
    config.json (deduped, str-only). Defaults to ``[]`` on any
    error.

    The static :data:`_ROOT_ALLOWLIST_FILES` set is the harness's
    built-in baseline. ``config.json`` ``patcher.root_files`` is
    documented as "merged with dynamically scanned entries (...) to
    form the final allowlist", but historically only ``CommandValidator``
    /``SecurityScanPolicy`` consumed it — the spec-driven and
    filesystem-fallback allowlist builders only saw the built-in
    static set. This helper closes that gap: operators who widen
    ``patcher.root_files`` (commit 0b75d0e added Java / modern Python
    /Node tooling) now see those entries in both allowlist tiers.

    Read is best-effort; any config error falls back to ``[]`` so the
    patcher still works on a malformed config (the static set keeps
    real builds running).
    """
    try:
        from harness.cli import _strip_comments, load_raw_config
        cfg = _strip_comments(load_raw_config())
    except Exception:  # noqa: BLE001 — config absence must not break the patcher
        return []
    patcher_cfg = cfg.get("patcher") if isinstance(cfg, dict) else None
    if not isinstance(patcher_cfg, dict):
        return []
    raw = patcher_cfg.get("root_files")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        stripped = entry.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            out.append(stripped)
    return out


def _read_extra_allowlist_globs() -> list[str]:
    """Return operator-supplied extra allowlist entries from config.json
    (Phase 3(d)). Defaults to ``[]`` when absent.

    The patcher's built-in allowlist refuses paths it has never seen the
    spec or filesystem declare. Some files that operators legitimately
    want to edit don't fit the spec contract — ``LICENSE``,
    ``CHANGELOG.md``, ``.editorconfig``, ``.dockerignore``, top-level
    ``*.lock`` files. Rather than widen the built-in list (and quietly
    expand the LLM's blast radius for every project), expose a config
    knob so operators can opt-in per-repo:

        "patcher": {
          "extra_allowlist_globs": ["LICENSE", "CHANGELOG.md", ".editorconfig"]
        }

    Reading is best-effort: any config-loading error falls back to ``[]``
    so the patcher still works on a malformed config. The list is appended
    to BOTH the spec-driven and filesystem-fallback allowlists so the
    knob behaves the same regardless of which tier built the rest of the
    list.
    """
    try:
        from harness.cli import _strip_comments, load_raw_config
        cfg = _strip_comments(load_raw_config())
    except Exception:  # noqa: BLE001 — config absence must not break the patcher
        return []
    patcher_cfg = cfg.get("patcher") if isinstance(cfg, dict) else None
    if not isinstance(patcher_cfg, dict):
        return []
    raw = patcher_cfg.get("extra_allowlist_globs")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            out.append(entry.strip())
    return out


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
    Phase 3(d): operator-supplied ``patcher.extra_allowlist_globs`` is
    appended to whichever tier wins.
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
    else:
        allowlist = _filesystem_allowlist(workspace_path)
    if allowlist is not None:
        extra = _read_extra_allowlist_globs()
        for entry in extra:
            if entry not in allowlist:
                allowlist.append(entry)
        if extra:
            logger.info(
                "[allowlist] extra_allowlist_globs appended: %s",
                extra,
            )
    return allowlist


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
    # Operator-widened config.json patcher.root_files. Without this
    # merge, additions to config.json (commit 0b75d0e widened root_files
    # for Java / Python / Node toolchains) never reached the spec-driven
    # allowlist — the LLM's first patch to a config-listed file like
    # ``compose.yml`` or ``.env.local`` was rejected even though the
    # operator had explicitly opted it in.
    for entry in _read_config_root_files():
        if entry not in allowlist:
            allowlist.append(entry)
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

    # Operator-widened config.json patcher.root_files (same merge as the
    # spec-driven tier). Greenfield projects skip this branch entirely
    # via the ``return None`` above; we land here for both source-rooted
    # workspaces (the ``if roots:`` branch) and stale-fallback workspaces.
    for entry in _read_config_root_files():
        if entry not in allowlist:
            allowlist.append(entry)
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


def _render_core_languages_directive(config: Optional[dict[str, Any]] = None) -> str:
    """Render the locked core-technology stack directive injected at the
    top of every system prompt.

    The harness is intentionally bounded to a small, opinionated set:
    Python or Java for backend, React + TypeScript + TailwindCSS (built
    with Vite) for web. The directive tells the LLM exactly which stacks
    are in-bounds, which are explicitly off-limits, and how to refuse a
    user request that strays outside the box. ``config`` lets the caller
    pin a specific backend choice; when omitted, both options are listed.
    """
    backend_choice: Optional[str] = None
    web_extras: list[str] = []
    if isinstance(config, dict):
        try:
            from harness.cli import (
                resolve_core_languages,
                _REQUIRED_WEB_LANGUAGES,
            )
            resolved = resolve_core_languages(config)
            backend_choice = resolved.get("backend_language")
            web_list = resolved.get("web_language") or []
            web_extras = [
                item for item in web_list
                if isinstance(item, str) and item not in _REQUIRED_WEB_LANGUAGES
            ]
        except Exception:  # noqa: BLE001 — fall back to the dual presentation
            backend_choice = None
            web_extras = []

    if backend_choice == "Python":
        backend_block = (
            "- **Selected backend:** Python (FastAPI or Flask). Spring Boot "
            "is the alternative when an operator picks Java in "
            "`core_languages.backend_language`, but THIS run is Python."
        )
    elif backend_choice == "Java":
        backend_block = (
            "- **Selected backend:** Java with Spring Boot. Python "
            "(FastAPI / Flask) is the alternative when an operator picks "
            "Python in `core_languages.backend_language`, but THIS run is "
            "Java."
        )
    else:
        backend_block = (
            "- **Option A:** Python (FastAPI or Flask).\n"
            "- **Option B:** Java with Spring Boot.\n"
            "Pick exactly ONE option per project based on the spec."
        )

    extras_block = ""
    if web_extras:
        extras_block = (
            "- **Operator-enabled extras:** " + ", ".join(web_extras) +
            ". Use these libraries when they are the natural fit — for "
            "example, prefer Radix UI primitives over hand-rolled "
            "dropdowns / dialogs / popovers and style them with "
            "Tailwind utility classes.\n"
        )

    return (
        "## Locked Core Technology Stack (MANDATORY)\n"
        "This harness only supports the technology stack below. Do NOT "
        "introduce any other framework, language, or build tool — the "
        "patcher, sandbox, skills, and style guides have been shaped "
        "exclusively around this matrix.\n"
        "\n"
        "### 1. Frontend Stack (mandatory)\n"
        "- **Library:** React (functional components + hooks; no class "
        "components).\n"
        "- **Language:** TypeScript with `strict` mode enabled in "
        "`tsconfig.json`.\n"
        "- **Styling:** Tailwind CSS (no styled-components, no Emotion, "
        "no global CSS overrides beyond the Tailwind entry file).\n"
        "- **Build Tool:** Vite.\n"
        f"{extras_block}"
        "- **Forbidden:** Next.js, Vue, Svelte, Angular, Nuxt, SvelteKit, "
        "Remix, plain JavaScript, jQuery, Bootstrap, Bulma, Material UI "
        "as a styling layer.\n"
        "\n"
        "### 2. Backend Stack (select ONE based on requirements)\n"
        f"{backend_block}\n"
        "- **Forbidden:** Node.js / Express / Fastify / NestJS, Ruby / "
        "Rails, Go, PHP / Laravel, .NET / C#, Rust, Elixir, Scala, "
        "Kotlin (server), Deno, Bun.\n"
        "\n"
        "### Agentic Behaviour & Rules\n"
        "1. **Bounding Box.** If the user (or the spec) asks for a "
        "technology outside this stack (e.g. \"Build a Ruby on Rails "
        "app\", \"Use Vue.js\", \"Write the API in Go\"), POLITELY "
        "REFUSE and explicitly state that this harness only supports "
        "Python or Java backends and React + TypeScript + Tailwind "
        "frontends. Propose the closest in-stack equivalent (e.g. "
        "\"I can build the same app with FastAPI + React instead\") "
        "and stop until the operator confirms.\n"
        "2. **API-First Communication.** Frontend and backend MUST be "
        "decoupled. The frontend communicates with the backend "
        "exclusively over RESTful HTTP/JSON APIs. No server-rendered "
        "HTML, no GraphQL by default, no shared in-process calls.\n"
        "3. **Type Safety.** Every JSON shape returned by the backend "
        "MUST have a matching TypeScript interface on the frontend; "
        "when a backend response schema changes, the frontend interface "
        "MUST change in the same patch.\n"
        "4. **No Placeholders.** Generate complete, functional code. "
        "Do NOT emit generic placeholders like `// Add logic here`, "
        "`pass  # implement me`, or `throw new Error('not implemented')` "
        "unless the operator explicitly asked for a partial scaffold.\n"
    )


def _build_system_prompt(
    workspace_path: str,
    build_command: str,
    config: Optional[dict[str, Any]] = None,
) -> str:
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
    # actually use. A pure FastAPI project doesn't need the React skill
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
            f"(React + TypeScript + Tailwind, Vite-built browser bundles) "
            f"belongs in the client-side root; HTTP handlers, models, and "
            f"background workers belong in the server-side root. When the choice is "
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

    core_languages_directive = _render_core_languages_directive(config)

    return f"""You are an expert software engineer with deep knowledge of the codebase below.

{core_languages_directive}
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
  - `uv pip install --system -r requirements.txt && pytest`  → declare
    every tool the build invokes (pytest-asyncio, ruff, mypy, etc.) in
    `requirements.txt`. `pytest` itself is pre-installed in the sandbox
    but adding it to `requirements.txt` is fine — the project will need
    it when installed outside the sandbox.
  - `uv pip install --system -e '.[dev]' && pytest`  → declare those
    tools under `[project.optional-dependencies].dev` in `pyproject.toml`.
  - `npm install && npm run build && npm test`  → the test runner referenced
    by the `test` script (Vitest) must be in `package.json` `devDependencies`.
  - `mvn -B test`  → Maven Surefire runs JUnit automatically; declare JUnit
    Jupiter and any extra test deps under `<scope>test</scope>` in `pom.xml`.
  - `./gradlew test` / `gradle test`  → declare JUnit Jupiter under
    `testImplementation` in `build.gradle`.

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

### REWRITE_FILE — escape hatch, NOT the default
```
<<<REWRITE_FILE>>>
file: path/to/existing/file.ext
content:
<complete new file contents>
<<<END_REWRITE_FILE>>>
```

REWRITE_FILE overwrites an existing file wholesale. Only use it when the
JUDGE'S VERDICT banner explicitly says "You MAY now emit REWRITE_FILE
on X" — that flag fires only after the SAME file:line has been the
blocker for 3+ rounds and surgical REPLACE_BLOCKs have demonstrably
failed to converge. For everyday edits, use REPLACE_BLOCK.

Because REWRITE_FILE clobbers the file, you MUST supply the COMPLETE
corrected contents (imports, every class, every def, every existing
test). A partial rewrite deletes whatever you didn't include —
recovery is expensive. Post-patch parse validation still applies: if
your REWRITE_FILE produces unparseable Python or JSON the patcher rolls
the file back to its pre-patch state and reports the block as failed.

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
- Minimum acceptable shape for a Python project (uv is pre-installed in
  the sandbox and 10-30× faster than pip; ALWAYS prefer `uv pip install`
  over plain `pip install`, see harness/skills/makefile_python.md):
  ```
  .PHONY: build test
  build:
  	uv pip install --system -r requirements.txt
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


def _build_and_emit_system_prompt(
    workspace_path: str,
    build_command: str,
    config: Optional[dict[str, Any]] = None,
) -> str:
    """Wrap :func:`_build_system_prompt` so callers get the same prompt but
    a ``system_prompt_built`` observability event also lands in the log.

    Kept separate from ``_build_system_prompt`` so unit tests that only care
    about prompt content (the majority) stay synchronous and don't have to
    stub out observability.
    """
    prompt = _build_system_prompt(workspace_path, build_command, config=config)
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
    ".ruff_cache", ".idea", ".vscode", ".gradle",
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


async def _maybe_discovery_saturation_check(
    *,
    gate: str,
    question_count: int,
    complete: bool,
    discovery_data: dict[str, Any],
    messages: list[dict[str, Any]],
    budget: float,
) -> tuple[bool, float]:
    """LLM-judgment discovery saturation check (#3).

    Asks a cheap LLM whether the just-completed discovery round has
    saturated the section — i.e. the prior answers + workspace evidence
    already cover the spec well enough that another round of follow-ups
    would burn tokens without changing the outcome. Returns
    ``(override_complete, new_budget)``. Caller treats the override as
    "set complete=True and zero out critical_remaining" when True.

    Fail-open: returns ``(complete, budget)`` unchanged when the call
    can't or shouldn't run (disabled, no gateway, before second round,
    already complete, budget too low, or any dispatch failure). The
    deterministic interview cap (``max_discovery_iterations``) keeps the
    loop bounded regardless.
    """
    if complete:
        return True, budget
    # Only fire from the SECOND round onward — round 1 always emits the
    # initial questionnaire and there's nothing yet to be saturated.
    if question_count < 1:
        return complete, budget
    gw = get_gateway()
    if gw is None or not str(getattr(gw.config, "repair_primary", "") or "").strip():
        return complete, budget
    if not bool(getattr(gw.config, "llm_judgment_discovery_saturation", True)):
        return complete, budget
    if budget < 0.01:
        return complete, budget

    modules = discovery_data.get("modules") if isinstance(discovery_data, dict) else None
    pending_lines: list[str] = []
    for m in (modules or []):
        for q in (m.get("questions") or []):
            pending_lines.append(
                f"  - [{m.get('name', '?')}] {str(q.get('text', ''))[:140]} "
                f"(critical={bool(q.get('critical'))}, "
                f"suggested={str(q.get('suggested_answer', ''))[:100]})"
            )
    pending_block = "\n".join(pending_lines[:30]) or "(none)"

    prompt = (
        f"You are deciding whether the {gate} discovery phase is SATURATED "
        "— that is, whether the operator's prior answers and the "
        "deterministic workspace evidence already give us enough to write "
        "the spec without another round of follow-up questions. "
        "Saturated = YES only when:\n"
        "  - No CRITICAL pending question would change the spec materially.\n"
        "  - The remaining non-critical questions are either redundant with "
        "answers already given or fully resolved by workspace evidence "
        "shown in the conversation above.\n"
        "Default to NO when uncertain — wasting a round is cheap; "
        "skipping a real follow-up is not.\n\n"
        "Respond with STRICT JSON ONLY (no prose, no markdown, no code "
        'fences): {"saturated": true|false, "reason": "<one short sentence>"}\n\n'
        f"Round just completed: {question_count + 1}\n"
        f"Pending questions this round:\n{pending_block}\n"
    )

    try:
        from harness.gateway import NodeRole
        check_messages = list(messages) + [{"role": "user", "content": prompt}]
        response, new_budget = await gw.dispatch(
            messages=check_messages,
            role=NodeRole.REPAIR,
            budget_remaining_usd=budget,
        )
        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        parsed: Any = json.loads(raw) if raw else {}
        if not isinstance(parsed, dict):
            return complete, new_budget
        verdict = bool(parsed.get("saturated", False))
        reason = str(parsed.get("reason", ""))[:200]
        if verdict:
            logger.info(
                "[judgment:discovery_saturation/%s] Round %d declared "
                "saturated; skipping further rounds. Reason: %s",
                gate.lower(), question_count + 1, reason,
            )
        return verdict, new_budget
    except Exception as exc:  # noqa: BLE001 — judgment must never break the loop
        logger.warning(
            "[judgment:discovery_saturation/%s] Check failed (%s); "
            "falling back to existing complete=%s.",
            gate.lower(), exc, complete,
        )
        return complete, budget


# Sector taxonomies that mirror the headings inside
# ``harness/skills/docgen/{requirements,architecture}_discovery.md``. Kept
# in code (rather than re-parsed from the markdown each round) because the
# focus helper validates the LLM's response against this exact set: if the
# .md file's sector names ever drift, these tuples are the single place to
# update. Anyone editing the .md MUST update the tuple, and vice versa —
# the test pack guards both directions.
_REQUIREMENTS_SECTORS: tuple[str, ...] = (
    "USER ROLES & PERSONAS",
    "FEATURES & USER STORIES",
    "INPUT VALIDATION & PAYLOAD FORMAT",
    "EDGE CASES & BOUNDARY CONDITIONS",
    "ERROR HANDLING & RETRY BEHAVIOR",
    "SECURITY CONTROLS & THREAT MODEL",
    "ABUSE & MISUSE CASES",
    "CONCURRENCY & MULTI-USER SEMANTICS",
    "BUSINESS LOGIC & STATE MACHINES",
    "COMPLIANCE & DATA CLASSIFICATION",
    "OBSERVABILITY & SUCCESS METRICS",
    "DATA RETENTION & LIFECYCLE",
    "HIDDEN ASSUMPTIONS & ENVIRONMENT",
)

_ARCHITECTURE_SECTORS: tuple[str, ...] = (
    "DATA MODEL & OWNERSHIP",
    "COMPONENT INTERFACES & CONTRACTS",
    "TRUST BOUNDARIES & SECURITY ZONES",
    "EXTERNAL DEPENDENCIES & RATE LIMITS",
    "STORAGE TOPOLOGY",
    "SECRETS & CONFIGURATION MANAGEMENT",
    "DEPLOYMENT TOPOLOGY",
    "SCALING & PERFORMANCE BUDGETS",
    "FAILURE DOMAINS & RESILIENCE PATTERNS",
    "OBSERVABILITY & ALERTING",
    "CI/CD & RELEASE STRATEGY",
    "DATA LIFECYCLE",
)


_FOCUS_MIN = 3
_FOCUS_MAX = 5


def _render_focus_block(focus: list[str]) -> str:
    """Render the focused-sector instruction block spliced into the
    follow-up prompt's ``{FOCUS_SECTORS_BLOCK}`` placeholder. Empty list
    → empty string so the prompt flows as if focus never ran.
    """
    if not focus:
        return ""
    bullets = "\n".join(f"- {name}" for name in focus)
    return (
        "## Focus this round\n\n"
        "Based on prior-round answers, concentrate this round's audit on "
        "the following sectors first:\n"
        f"{bullets}\n\n"
        "Re-audit the remaining sectors only if you spot a critical gap "
        "while answering the focused ones. Asking 1-2 sharp questions per "
        "focused sector beats asking a generic question in every sector.\n"
    )


async def _maybe_discovery_followup_focus(
    *,
    gate: str,
    question_count: int,
    sectors: tuple[str, ...],
    messages: list[dict[str, Any]],
    budget: float,
) -> tuple[Optional[list[str]], float]:
    """LLM-judgment follow-up focus picker (#6).

    Asks a cheap LLM which 3-5 sectors most warrant re-auditing in this
    follow-up round, given the conversation so far. Returns ``(focus_list,
    new_budget)``. Caller renders the focus list into the follow-up prompt
    via ``_render_focus_block`` and substitutes ``{FOCUS_SECTORS_BLOCK}``.

    Fail-open: returns ``(None, budget)`` whenever the call can't or
    shouldn't run (disabled by config, no gateway, no repair model routed,
    first round, budget too low, or dispatch / parse failure). Caller
    treats ``None`` as "render the empty block and proceed unfocused".

    Args:
        gate: ``"REQUIREMENTS"`` or ``"ARCHITECTURE"`` — for logging only.
        question_count: Rounds completed BEFORE this one. Focus runs only
            from the second round onward (question_count >= 1).
        sectors: The canonical sector list for this gate. Returned focus
            entries are validated against this set; unknowns are dropped.
        messages: Conversation history (includes prior-round answers).
        budget: Remaining session budget in USD.
    """
    if question_count < 1:
        return None, budget
    gw = get_gateway()
    if gw is None or not str(getattr(gw.config, "repair_primary", "") or "").strip():
        return None, budget
    if not bool(getattr(gw.config, "llm_judgment_discovery_followup_focus", True)):
        return None, budget
    if budget < 0.01:
        return None, budget

    sector_list = "\n".join(f"- {name}" for name in sectors)
    prompt = (
        f"You are picking which {gate} sectors most need follow-up "
        f"in the next discovery round.\n\n"
        f"Sectors:\n{sector_list}\n\n"
        f"Pick the {_FOCUS_MIN}-{_FOCUS_MAX} sectors with the BIGGEST "
        "remaining gap based on the conversation above. Prefer sectors "
        "where:\n"
        "  - The operator's prior answers were vague, contradictory, or "
        "marked as guesses.\n"
        "  - A critical question went unanswered or was deferred.\n"
        "  - Downstream gates (security, data, scaling) materially "
        "depend on this sector and it is still soft.\n\n"
        "Respond with STRICT JSON ONLY (no prose, no markdown, no code "
        f'fences): {{"focus": ["<SECTOR NAME>", ...]}}\n\n'
        f"Use the EXACT names from the sector list above. Max "
        f"{_FOCUS_MAX} entries. If every sector is equally soft, pick "
        f"the {_FOCUS_MIN} that block downstream gates the hardest.\n"
    )

    # Track the post-dispatch budget out here so the exception path can
    # return the spend that actually happened rather than the pre-call
    # budget. The judgment call is cheap (~$0.001) but ignoring it leaks
    # token accounting; the existing saturation_check helper does swallow
    # it — I'm not matching that quirk here.
    new_budget = budget
    try:
        from harness.gateway import NodeRole
        check_messages = list(messages) + [{"role": "user", "content": prompt}]
        response, new_budget = await gw.dispatch(
            messages=check_messages,
            role=NodeRole.REPAIR,
            budget_remaining_usd=budget,
        )
        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        parsed: Any = json.loads(raw) if raw else {}
        if not isinstance(parsed, dict):
            return None, new_budget
        raw_focus = parsed.get("focus")
        if not isinstance(raw_focus, list):
            return None, new_budget

        known = set(sectors)
        validated: list[str] = []
        for item in raw_focus:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if normalized in known and normalized not in validated:
                validated.append(normalized)
            if len(validated) >= _FOCUS_MAX:
                break

        if len(validated) < _FOCUS_MIN:
            # Under the floor → don't risk a too-narrow follow-up.
            # Falling back to the full unfocused prompt is safer than
            # asking 1-2 questions in an LLM-chosen tiny subset.
            logger.info(
                "[judgment:discovery_followup_focus/%s] LLM picked only "
                "%d valid sector(s) (below floor=%d); falling back to "
                "unfocused follow-up.",
                gate.lower(), len(validated), _FOCUS_MIN,
            )
            return None, new_budget

        logger.info(
            "[judgment:discovery_followup_focus/%s] Round %d focus: %s",
            gate.lower(), question_count + 1, validated,
        )
        return validated, new_budget
    except Exception as exc:  # noqa: BLE001 — judgment must never break the loop
        logger.warning(
            "[judgment:discovery_followup_focus/%s] Picker failed (%s); "
            "falling back to unfocused follow-up.",
            gate.lower(), exc,
        )
        return None, new_budget


async def _maybe_judgment_llm(
    *,
    prompt: str,
    budget_remaining_usd: float,
    purpose: str,
    enabled: bool,
) -> tuple[Optional[str], float]:
    """Cheap one-shot LLM-judgment call shared by the four kill-switched
    judgment additions (HITL escalation summary, patcher rejection
    diagnosis, pre-flight autofix classification, discovery saturation).

    Dispatches under ``NodeRole.JUDGMENT`` — same cheap model as REPAIR
    but a distinct cache-drift bucket. These calls ship tiny user-only
    prompts (no shared system message), so binding them to REPAIR would
    flip the gateway's prefix-hash record back and forth every iteration
    and force auto-cache misses on the real repair-loop dispatch.

    Fail-open by design: returns ``(None, budget_unchanged)`` whenever
    the call can't or shouldn't run (disabled by config, no gateway, no
    repair model routed, budget too low, or the dispatch raises) so
    callers fall back to their existing deterministic behaviour without
    a guard around every call site.

    A $0.01 floor is enforced on top of the gateway's hard guardrail —
    judgment calls are cheap (~$0.001) but a 0-budget call would still
    raise inside ``Gateway.dispatch``.
    """
    if not enabled:
        return None, budget_remaining_usd
    gw = get_gateway()
    if gw is None or not str(getattr(gw.config, "repair_primary", "") or "").strip():
        return None, budget_remaining_usd
    if budget_remaining_usd < 0.01:
        return None, budget_remaining_usd
    try:
        from harness.gateway import NodeRole
        messages = [{"role": "user", "content": prompt}]
        response, new_budget = await gw.dispatch(
            messages=messages,
            role=NodeRole.JUDGMENT,
            budget_remaining_usd=budget_remaining_usd,
        )
        content = (response.content or "").strip()
        if not content:
            return None, new_budget
        return content, new_budget
    except Exception as exc:  # noqa: BLE001 — judgment must never break the loop
        logger.warning(
            "[judgment:%s] LLM call failed (%s); falling back to deterministic path.",
            purpose, exc,
        )
        return None, budget_remaining_usd


async def _repair_malformed_json(
    *,
    raw_text: str,
    schema_hint: str,
    dispatch: Any,
    budget_remaining_usd: float,
    purpose: str,
) -> tuple[Optional[Any], float]:
    """One-shot JSON-repair pass for strict-schema nodes.

    When a reviewer / discovery / reflection dispatch returns text that
    doesn't parse as JSON, callers historically discarded the whole
    response (empty findings, terminated interview, dropped verdict).
    That's the most severe form of LLM starvation — the model DID
    produce signal, and the code threw it away over a formatting slip.

    This helper re-prompts the model with the offending text plus the
    schema and asks it to emit JSON only. On success, returns
    ``(parsed_object, new_budget)``. On persistent failure, returns
    ``(None, new_budget)`` and the caller falls back to its existing
    empty-critique / skip-injection behaviour.

    ``dispatch`` is an async callable
    ``(messages, budget) -> (response, new_budget)`` — captured by the
    caller so this helper stays role-agnostic. ``purpose`` is a short
    tag used only for log lines.
    """
    if not raw_text or not raw_text.strip():
        return None, budget_remaining_usd
    if budget_remaining_usd < 0.01:
        return None, budget_remaining_usd
    truncated = raw_text.strip()
    if len(truncated) > 8000:
        truncated = truncated[:8000] + "\n...(truncated for repair prompt)"
    repair_prompt = (
        "Your previous response could not be parsed as JSON. Re-emit the "
        "same content as VALID JSON that matches the schema below. "
        "Output ONLY the JSON object — no prose, no markdown fences, no "
        "commentary before or after.\n\n"
        f"## Expected schema\n{schema_hint}\n\n"
        f"## Your previous (unparseable) response\n{truncated}\n\n"
        "Emit the corrected JSON now."
    )
    try:
        response, new_budget = await dispatch(
            [{"role": "user", "content": repair_prompt}],
            budget_remaining_usd,
        )
    except Exception as exc:  # noqa: BLE001 — repair must never break the loop
        logger.warning(
            "[json_repair:%s] Re-dispatch failed (%s); falling back.",
            purpose, exc,
        )
        return None, budget_remaining_usd
    repaired_text = (getattr(response, "content", "") or "").strip()
    if not repaired_text:
        return None, new_budget
    try:
        from harness.trust import _strip_code_fences
        repaired_text = _strip_code_fences(repaired_text)
    except Exception:  # noqa: BLE001
        pass
    try:
        parsed = json.loads(repaired_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "[json_repair:%s] Repair pass still not valid JSON (%s); "
            "falling back.", purpose, exc,
        )
        return None, new_budget
    logger.info(
        "[json_repair:%s] Recovered valid JSON after one re-prompt.", purpose,
    )
    return parsed, new_budget


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

    dropped_n = len(messages) - len(cleansed)
    logger.info(
        "[memory_cleanse] Trimmed mid-loop messages: %d → %d (kept system + "
        "planning + last assistant; dropped %d intermediate turns).",
        len(messages), len(cleansed), dropped_n,
    )
    # Phase 2.1 — decision-point logging. Mid-loop cleanse is a known
    # signal-loss site (it drops intermediate assistant chatter and
    # related user feedback). Surface dropped role-counts so a HITL
    # post-mortem can tell whether the cleanse ate something important.
    try:
        from harness.observability import emit_event as _emit_drop
        kept_roles = [str(m.get("role", "?")) for m in cleansed]
        dropped_roles: dict[str, int] = {}
        for m in messages:
            if m in cleansed:
                continue
            r = str(m.get("role", "?"))
            dropped_roles[r] = dropped_roles.get(r, 0) + 1
        _emit_drop(
            "dropped_from_prompt",
            site="repair_iteration_cleanse",
            dropped_count=dropped_n,
            kept_count=len(cleansed),
            reason="trim_mid_loop_chatter_before_next_repair_dispatch",
            kept_roles=kept_roles,
            dropped_roles=dropped_roles,
        )
    except Exception:  # noqa: BLE001
        pass
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
# Operators can override via
# ``llm_dispatch.patching_read_file_cap`` in config.json (clamped to
# [1, 30] at read time — see :func:`_resolve_patching_read_file_cap`).
_PATCHING_READ_FILE_CAP = 10

# Cap on continuation cycles for nodes that opt into
# llm_dispatch.continue_on_length.<role>. See the comment on that
# section in config/config.json for the per-role risk profile.
# Operators can override via
# ``llm_dispatch.max_continuation_cycles`` in config.json (clamped to
# [1, 10] at read time — see :func:`_resolve_max_continuation_cycles`).
_MAX_CONTINUATION_CYCLES = 5

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


def _resolve_max_continuation_cycles(
    state_or_cfg: Any,
) -> int:
    """Resolve the cap on continuation cycles.

    Accepts either an ``AgentState`` (reads
    ``state.llm_dispatch_config.max_continuation_cycles``) or the
    ``llm_dispatch_config`` dict directly. Clamps to ``[1, 10]``; falls
    back to :data:`_MAX_CONTINUATION_CYCLES` when the config omits it or
    supplies a non-int / out-of-range value.
    """
    if isinstance(state_or_cfg, dict) and "llm_dispatch_config" in state_or_cfg:
        cfg = state_or_cfg.get("llm_dispatch_config", {}) or {}
    else:
        cfg = state_or_cfg or {}
    raw = cfg.get("max_continuation_cycles", _MAX_CONTINUATION_CYCLES)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _MAX_CONTINUATION_CYCLES
    return max(1, min(10, val))


def _resolve_patching_read_file_cap(state: "AgentState") -> int:
    """Resolve the cap on ``read_file`` rounds inside one patching turn.

    Reads ``state.llm_dispatch_config.patching_read_file_cap`` and
    clamps to ``[1, 30]``; falls back to
    :data:`_PATCHING_READ_FILE_CAP` when the config omits it or supplies
    a non-int / out-of-range value.
    """
    cfg = state.get("llm_dispatch_config", {}) or {}
    raw = cfg.get("patching_read_file_cap", _PATCHING_READ_FILE_CAP)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _PATCHING_READ_FILE_CAP
    return max(1, min(30, val))


async def _continue_on_length(
    *,
    initial_response: Any,
    initial_budget: float,
    messages: list["MessageDict"],
    dispatch: Any,
    continue_prompt: str,
    enabled: bool,
    role_label: str,
    max_cycles: int = _MAX_CONTINUATION_CYCLES,
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
        and continuation_cycles < max_cycles
    ):
        continuation_cycles += 1
        logger.info(
            "[%s] hit output token cap (cycle %d/%d) — requesting "
            "continuation.",
            role_label, continuation_cycles, max_cycles,
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
        continuation_cycles >= max_cycles
        and getattr(response, "finish_reason", "stop") == "length"
    ):
        logger.warning(
            "[%s] LLM still truncated after %d continuation cycle(s); "
            "accepting what landed and moving on.",
            role_label, max_cycles,
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

    read_file_cap = _resolve_patching_read_file_cap(state)
    rounds = 0
    while rounds < read_file_cap:
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

    if rounds >= read_file_cap:
        logger.info(
            "[patching_tool_loop] hit cap of %d read_file rounds; "
            "patching proceeds with what the LLM has so far.",
            read_file_cap,
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
                    f"{read_file_cap} times — the cap is reached. "
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
            max_cycles=_resolve_max_continuation_cycles(state),
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
        # Story mode: when current_story_id is set, scope this turn to
        # the named STORY-N's acceptance criteria + file hints, with the
        # STORY-N marker contract. No-op (empty string) when no story
        # is active — monolithic flow (--no-stories) is byte-identical.
        story_preamble = _build_story_preamble(state, "patching")
        # Architecture-summary handoff: render endpoint / component /
        # contract tables from docs/SPEC_ARCHITECTURE.md §11 so the
        # patcher does not re-derive paths and schema names on every
        # turn. Empty string when the arch doc has no §11 block —
        # patching falls back to the prose document the system prompt
        # already carries.
        arch_preamble, resolved_arch_summary = _build_arch_summary_preamble(state)

        # Python import convention: the prod-import smoke check imports
        # every ``*.py`` file with its dotted path relative to the
        # workspace root (e.g. ``server.main``, ``server.routers.search``
        # — see ``_walk_prod_python_modules``). Top-level imports inside
        # those files (``from routers import search``,
        # ``from database import engine``) blow up with
        # ``ModuleNotFoundError`` even when the target file exists,
        # because the dotted-name discovery requires the package
        # prefix. Without this rule the repair loop thrashes on the
        # symptom (``No module named 'fastapi'``-shaped errors for
        # first-party packages) and never fixes the cause. The rule is
        # emitted unconditionally; non-Python projects ignore it
        # silently.
        _IMPORT_CONVENTION_RULE = (
            "[PYTHON IMPORT CONVENTION — STRICTLY ENFORCED]\n"
            "If you emit any ``*.py`` files, every intra-project import "
            "MUST be absolute from the workspace root. The harness "
            "discovers production modules by walking the workspace and "
            "joining path segments with dots; ``server/main.py`` is "
            "imported as ``server.main`` during the prod-import smoke "
            "check, NOT as ``main``. Inside that file you MUST therefore "
            "write ``from server.routers import search`` and "
            "``from server.database import engine`` — NOT "
            "``from routers import search`` or ``from database import "
            "engine``. The same rule applies to every package subdirectory "
            "you create (``client/`` if Python, ``app/``, ``api/``, etc.). "
            "Top-level imports of first-party packages will fail the "
            "smoke check with ``ModuleNotFoundError`` even though the "
            "file is on disk, and the repair loop cannot recover from "
            "this without your help.\n\n"
        )
        # Inject a format reminder to ensure the LLM outputs patch blocks
        _FORMAT_REMINDER = allowlist_preamble + cr_preamble + story_preamble + arch_preamble + _IMPORT_CONVENTION_RULE + """[CRITICAL FORMAT INSTRUCTION]
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
            + story_preamble
            + _IMPORT_CONVENTION_RULE
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
            max_cycles=_resolve_max_continuation_cycles(state),
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

        # Layer 3 — global no-progress failsafe. Snapshot the budget at
        # the last moment patching produced progress; route_after_patching
        # / route_after_compiler escalate to HITL when spend since that
        # marker exceeds the threshold.
        from harness.no_progress import update_and_check as _np_update_and_check
        _np_update_and_check(
            loop_counter,
            budget_remaining_usd=new_budget,
            progress_made=(success_count > 0),
            threshold_usd=float(
                state.get("no_progress_budget_usd") or 1.50
            ),
        )

        # Zero-patch tripwires. Both counters bump in lock-step now so
        # Layer 1 (global HITL at ≥2 consecutive_zero) and Layer 2
        # (story-level auto-advance at STORY_ZERO_PATCH_CAP) are both
        # live regardless of mode:
        #
        # - Story mode (``current_story_id`` set): bump the per-story
        #   counter in ``story_zero_patch_rounds[story_key]`` AND the
        #   global ``consecutive_zero_patch_rounds``. Layer 2 (story
        #   auto-advance at ≥STORY_ZERO_PATCH_CAP=3) usually catches
        #   the stall first; Layer 1 is the system-wide safety net for
        #   the case where the same story keeps getting re-picked (or
        #   adjacent stories all stall identically) — without it, story
        #   mode could spin indefinitely between the patcher and
        #   story_loop_node when every story is vacuous.
        # - Monolithic mode: only the global counter exists.
        #
        # Both counters reset on a successful patch round to avoid
        # false escalations triggered by a long-tail single-story stall.
        current_story_id = state.get("current_story_id") or ""
        if success_count == 0:
            loop_counter["consecutive_zero_patch_rounds"] = (
                loop_counter.get("consecutive_zero_patch_rounds", 0) + 1
            )
        else:
            loop_counter["consecutive_zero_patch_rounds"] = 0
        # Distinguish "LLM emitted patches, all rejected by allowlist" from
        # the generic "0 patches applied" case. The former is high-signal:
        # the model is deliberately targeting paths outside the allowlist
        # and no amount of retry will get it back on track without either
        # a rewire or operator intervention. Escalates to HITL faster than
        # the generic zero-patch counter — see route_after_compiler.
        all_rejected_by_allowlist = (
            success_count == 0
            and len(patch_results) > 0
            and len(allowlist_rejections) == fail_count
        )
        if all_rejected_by_allowlist:
            loop_counter["consecutive_all_allowlist_rejected_rounds"] = (
                loop_counter.get("consecutive_all_allowlist_rejected_rounds", 0) + 1
            )
        else:
            loop_counter["consecutive_all_allowlist_rejected_rounds"] = 0
        if current_story_id:
            sz = dict(loop_counter.get("story_zero_patch_rounds", {}) or {})
            if success_count == 0:
                sz[current_story_id] = int(sz.get(current_story_id, 0) or 0) + 1
            else:
                sz[current_story_id] = 0
            loop_counter["story_zero_patch_rounds"] = sz
            zero_rounds_for_log = sz[current_story_id]
        else:
            zero_rounds_for_log = loop_counter["consecutive_zero_patch_rounds"]

        logger.info(
            "[patching_node] Patches applied. tokens_in=%d tokens_out=%d cost=$%.6f budget_left=$%.4f "
            "patches=%d succeed=%d fail=%d zero_rounds=%d%s",
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost_usd,
            new_budget,
            len(patch_results), success_count, fail_count,
            zero_rounds_for_log,
            f" story={current_story_id}" if current_story_id else "",
        )

        return {
            "messages": messages,
            "modified_files": modified_files,
            "batch_modified_files": _extend_batch_scope(state, modified_files),
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            # Cache the resolved §11 summary so subsequent patching
            # turns (and code_review / test_generation downstream) skip
            # the disk read. Empty dict here means "no §11 block on
            # disk" — re-loading would just hit the same miss.
            "arch_summary": resolved_arch_summary,
            "node_state": {
                "current_node": "patching",
                "patch_complete": True,
                "patch_success": success_count,
                "patch_fail": fail_count,
                "allowlist_rejections": allowlist_rejections,
                "patch_failures": patch_failures,
                "allowed_paths": allowed_paths,
                "zero_rounds": zero_rounds_for_log,
                "current_story_id": current_story_id,
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
    Java JDK + Maven + Gradle, Node + npm, SQLite, Playwright +
    Chromium, and the make/git glue all into one container. So there
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
    to reach a registry (pip/npm) — OR invokes ``make``.

    The ``make`` clause exists because the LLM-generated ``Makefile`` (per
    ``harness/skills/makefile_python.md``) conventionally puts
    ``pip install -r requirements.txt`` (or ``npm install``) INSIDE
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
        "pip install", "pip3 install", "npm install",
        "poetry install",
        # uv is the harness-preferred Python installer (see
        # harness/skills/makefile_python.md). All three subcommands hit
        # PyPI: `uv pip install -r requirements.txt`, `uv sync` (resolve
        # + install from pyproject.toml / uv.lock), `uv add <pkg>` (add
        # + install).
        "uv pip install", "uv sync", "uv add",
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
#     anything else (npm, docker) the container needs a different base
#     image, which only the operator can change.
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
    # Shell-style "<cmd>: command not found" (covers npm, etc.)
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
    re.compile(r"(?m)^(?P<sym>node|npm): not found\s*$"),
)

# Node/npm-side missing dependency. Distinct from _PYTHON_MODULE_MISS_*
# because the autofix path adds the dep to package.json instead of
# requirements.txt, and the LLM repair prompt needs to know it's a Node
# package miss (not a shell-command miss).
#
# Real-world coverage:
#   Plain Node     → "Cannot find module 'X'" (with single OR double quotes)
#   Vite + PostCSS → "[vite:css] [postcss] Cannot find module 'X'"
#   Webpack        → "Module not found: Error: Can't resolve 'X'"
#   Next.js prod   → "Cannot find module 'next/X'" (same pattern works)
#
# We deliberately exclude paths that start with "./" or "/" or "."
# inside the capture — those are user-code relative imports (link-check
# territory, not missing-package territory). Only bare package names
# (with optional scopes like @scope/pkg) get tagged as MISSING_DEP.
_NODE_MODULE_MISS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Bare Node "Cannot find module 'X'" — the canonical shape used by
    # require()/node_modules resolution. Allows scoped packages (@scope/pkg)
    # and sub-paths (pkg/sub) but rejects anything that looks like a
    # relative import (starts with ./ or / or ../).
    re.compile(
        r"Cannot find module ['\"]"
        r"(?P<sym>(?:@[a-zA-Z0-9][\w.\-]*/)?[a-zA-Z][\w.\-]*(?:/[\w.\-]+)*)"
        r"['\"]"
    ),
    # Webpack: "Module not found: Error: Can't resolve 'X'"
    re.compile(
        r"Module not found:\s*Error:\s*Can't resolve ['\"]"
        r"(?P<sym>(?:@[a-zA-Z0-9][\w.\-]*/)?[a-zA-Z][\w.\-]*(?:/[\w.\-]+)*)"
        r"['\"]"
    ),
)

# Composite — preserved for callers that don't care about the source kind.
_ENV_MISCONFIG_PATTERNS: tuple[re.Pattern[str], ...] = (
    *_PYTHON_MODULE_MISS_PATTERNS,
    *_SHELL_COMMAND_MISS_PATTERNS,
    *_NODE_MODULE_MISS_PATTERNS,
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
        (npm, docker) needs a different base image and the
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
    # Node/npm package misses — kind="node" so compiler_node routes to
    # the package.json autofix path instead of requirements.txt.
    for pattern in _NODE_MODULE_MISS_PATTERNS:
        match = pattern.search(tail)
        if match:
            sym = match.group("sym").strip("'\"")
            # Skip relative paths defensively — the regex already excludes
            # them but a future loosening shouldn't accidentally tag user
            # code as a missing npm package.
            if sym.startswith(".") or sym.startswith("/"):
                continue
            if workspace_path and _node_module_exists_in_workspace(sym, workspace_path):
                logger.info(
                    "[env_misconfig] '%s' looks like a missing npm package "
                    "but a local file by that name exists in %s — treating "
                    "as a user-code import bug, not env misconfig.",
                    sym, workspace_path,
                )
                return None
            return (sym, "node")
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


def _node_module_exists_in_workspace(module: str, workspace_path: str) -> bool:
    """True when ``module`` matches a local JS/TS source path in the
    workspace tree (not a node_modules package).

    Distinguishes a genuine missing-npm-package failure from a typo'd
    local import. The Node "Cannot find module 'X'" error fires for
    both; the autofix path must only modify package.json when X really
    is a third-party package.

    Treats only single-segment names (the first path component) as
    candidates — `foo/bar` only matches when a `foo` directory or
    `foo.{js,jsx,ts,tsx}` file exists somewhere under the workspace.
    Scoped packages (`@scope/pkg`) never match a local file.
    """
    if not module or not workspace_path or not os.path.isdir(workspace_path):
        return False
    if module.startswith("@"):
        return False
    from harness.impact import _NEVER_SOURCE_DIRS
    head = module.split("/", 1)[0].strip()
    if not head:
        return False
    try:
        for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
                and d != "node_modules"
            ]
            if head in sub_dirs:
                return True
            for ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
                if f"{head}{ext}" in sub_files:
                    return True
    except OSError:
        return False
    return False


# Test-runner "no tests collected" markers across stacks. Each runner
# emits a distinct message + a non-zero exit code when the collector
# found zero tests to execute:
#   * pytest       → exit 5
#   * Jest         → exit 1 (unless --passWithNoTests)
#   * Vitest       → exit 1
#   * Mocha        → exit 1 with --fail-zero (v10+) or when files missing
#   * Maven        → exit 1 with failIfNoTests=true (Surefire/Failsafe)
#   * Gradle test  → exit 1 with --fail-on-no-matching-tests
# The batch's verification unit simply has no tests yet — the repair
# LLM cannot fix this from inside the loop. We confirm via the literal
# marker line (so a config error that happens to produce the same exit
# code isn't misclassified) AND require the build command to actually
# exercise one of these runners.
_NO_TESTS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # pytest — "==== no tests ran in 0.01s ===="
    re.compile(r"(?m)^=+\s*no tests ran in [\d.]+s\s*=*$"),
    re.compile(r"(?m)^no tests ran in [\d.]+s\s*$"),
    # Jest — "No tests found, exiting with code 1"
    re.compile(r"(?im)^\s*no tests found, exiting with code \d+\s*$"),
    re.compile(r"(?im)^\s*no tests found related to files changed\b"),
    # Vitest — "No test files found, exiting with code 1"
    re.compile(r"(?im)^\s*no test files? found, exiting with code \d+\s*$"),
    re.compile(r"(?im)^\s*no test files? found\b"),
    # Mocha — "Error: No test files found"
    re.compile(r"(?im)^\s*error: no test files? found\b"),
    # Maven Surefire / Failsafe — "There are no tests to run." /
    # "No tests were executed" (variants across plugin versions).
    re.compile(r"(?im)^.*there are no tests? to run\b.*$"),
    re.compile(r"(?im)^.*no tests were executed\b.*$"),
    # Gradle test task with --fail-on-no-matching-tests
    re.compile(r"(?im)^.*no tests found for given includes\b.*$"),
    re.compile(r"(?im)^.*no tests were found for the following includes\b.*$"),
)


# Substrings that identify a test-runner invocation in the build command.
# Used to gate `_is_no_tests_collected` — without a runner in the command,
# a marker match in the output is almost certainly incidental (e.g. an
# error message that quotes one of these strings).
_TEST_RUNNER_TOKENS: tuple[str, ...] = (
    "pytest",           # Python
    "vitest",           # Web
    "jest",             # Web
    "mocha",            # Web
    "karma",            # Web
    "playwright",       # Web e2e
    "cypress",          # Web e2e
    "mvn",              # Java (Maven)
    "maven",            # Java (Maven)
    "gradle",           # Java (Gradle)
    "gradlew",          # Java (Gradle wrapper)
    "junit",            # Java (direct)
    "surefire",         # Java (Maven plugin)
    "failsafe",         # Java (Maven plugin)
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


_COMMAND_BLOCKED_RULE_PATTERN = re.compile(
    r"\bMatched Rule:\s*(?P<rule>whitelist_missing:[\w\-\.]+|blocked_pattern:[^\s]+)",
    re.IGNORECASE,
)


def _is_command_blocked_by_security(raw_output: str) -> Optional[str]:
    """When the sandbox's CommandValidator refused the build command,
    return the matched validator rule (e.g. ``whitelist_missing:cd`` or
    ``blocked_pattern:\\bsudo\\b``). Returns ``None`` when the build
    output isn't a security-validator block.

    The CommandValidator's exception (raised in
    :meth:`harness.sandbox.SandboxRunner.run` before any subprocess is
    spawned) is stringified into ``BuildResult.raw_output`` as::

        [SECURITY BLOCKED]: Command 'cd' is not in the allowed commands
          Command: cd server && uv pip install ...
          Matched Rule: whitelist_missing:cd
          Tip: Configure 'security.allowed_commands' ... in .harness_config.json

    This detector keys off the ``Matched Rule:`` token so we don't false-
    positive on user app code that happens to print the string
    ``[SECURITY BLOCKED]`` in its own logs.

    A blocked command is NOT something the repair LLM can fix from
    inside the workspace — the CommandValidator's config lives in the
    GLOBAL ``~/.harness/...`` config and the patcher allowlist refuses
    to write there. Without short-circuiting, repair burns its full
    iteration budget rejecting patches at the allowlist and the
    operator gets a misleading HITL summary built from later (different)
    failure modes once the build command happens to slip through (e.g.
    the prod-import smoke check, which composes a separate command).
    """
    if not raw_output:
        return None
    if "[SECURITY BLOCKED]" not in raw_output:
        return None
    match = _COMMAND_BLOCKED_RULE_PATTERN.search(raw_output)
    if match is None:
        return None
    return match.group("rule")


def _is_no_tests_collected(exit_code: int, raw_output: str, build_command: str) -> bool:
    """True when a non-zero build exit came from a test runner reporting
    "no tests collected" rather than from an actual compile/test failure.

    Detected across stacks (see :data:`_NO_TESTS_PATTERNS` and
    :data:`_TEST_RUNNER_TOKENS`):
      * pytest exit 5 → "no tests ran in Xs"
      * Jest / Vitest / Mocha exit 1 → "no tests found" / "no test files found"
      * Maven Surefire/Failsafe exit 1 → "there are no tests to run"
      * Gradle test exit 1 → "no tests found for given includes"

    Treating any of these as a generic build failure burns repair
    iterations on a problem the repair LLM cannot fix (there's nothing
    to repair — the runner found nothing). ``compiler_node`` folds the
    exit to 0 when source files exist, or hands off to HITL when the
    workspace is empty.
    """
    if exit_code == 0 or not raw_output:
        return False
    cmd_lower = build_command.lower()
    if not any(tok in cmd_lower for tok in _TEST_RUNNER_TOKENS):
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

    Long build logs (Java dependency stacks, npm install walls) tend to
    put the root-cause error near the START and
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
    # Rescue middle-band "Caused by:" chains before returning. uv / cargo
    # / gradle / maven emit 3-6 levels of ``Caused by:`` and the salient
    # ROOT line is usually the deepest — which for a >3.7KB log lands in
    # the dropped middle. Sweep item #4: session db6bfcbe lost the uv
    # wheel-install root at literally the word "Caused" because of this
    # slice. Lift ``Caused by:`` lines into a small preamble so the LLM
    # sees them even when head+tail miss the region.
    causal_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("Caused by:") or stripped.startswith("caused by:"):
            causal_lines.append(line.rstrip())
        # Cap to avoid a runaway chain crowding out the head/tail views.
        if len(causal_lines) >= 12:
            break
    causal_block = ""
    if causal_lines:
        # Deduplicate consecutive identical lines while preserving order —
        # some tools repeat the same ``Caused by:`` line across contexts.
        seen: set[str] = set()
        unique_causal: list[str] = []
        for c in causal_lines:
            key = c.strip()
            if key not in seen:
                seen.add(key)
                unique_causal.append(c)
        causal_block = (
            "--- (Caused-by chain lifted from middle of log) ---\n"
            + "\n".join(unique_causal)
            + "\n--- (end lifted chain) ---\n"
        )
    return (
        f"{head}\n"
        f"... [truncated {dropped} chars from the middle of {total}-char build log] ...\n"
        f"{causal_block}"
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
    elif "npm install" in build_cmd:
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


# First-level subdirs the install-step composer should NOT descend into
# when sniffing for Python manifests. Mirrors the build-detection skip set
# in cli.py so monorepo backends in `server/` get installed but build
# artefacts, virtualenvs, doc folders, etc. don't get probed.
_INSTALL_STEP_SUBDIR_SKIP = frozenset({
    "node_modules", "__pycache__", ".git", "build", "dist", "venv", ".venv",
    "env", "target", "out", "docs", "product_spec", "change_requests",
    "tests", "test", "__tests__", "client", "frontend", "web", "ui",
})


# Shared install-time venv path inside the sandbox container. Lives under
# /tmp so it's writable by non-root sandbox users — `uv pip install --system`
# targets /usr/local/lib/python3.11/dist-packages, which a non-root user
# can't write to, so wheel installs fail with `Permission denied (os error
# 13)` before the smoke check / pytest ever runs. `--system-site-packages`
# keeps the builder image's pre-baked deps visible for imports while still
# routing all new writes into the writable venv.
_PROD_SMOKE_VENV_PATH = "/tmp/teane-venv"


# Substrings that mark a line as the real error rather than install-
# progress chatter. Used by the prod-smoke fallback diagnostic to lift
# error lines to the head of the message so the HITL summariser's per-
# error truncation doesn't drop them.
_SMOKE_SALIENT_MARKERS: tuple[str, ...] = (
    "error:", "ERROR", "Error:", "Permission denied", "Caused by:",
    "FAIL:", "Traceback", "fatal:", "ModuleNotFoundError",
    "ImportError",
)


def _surface_salient_errors(raw_output: str) -> str:
    """Return a diagnostic message that puts error-marker lines first,
    then a trailing slice of the raw output for context.

    Pure-text helper, no I/O. When the raw output starts with progress
    chatter (uv's "Downloading … (N MiB)" / "Downloaded …" / "Prepared
    N packages …") and the real failure appears further down, naively
    slicing ``raw_output[-4000:]`` keeps the failure inside the window
    but pushes it past the per-diagnostic head truncation the HITL
    summariser applies. Lifting marker lines to the head guarantees
    the LLM sees the real cause regardless of where the truncation
    boundary lands.
    """
    if not raw_output:
        return ""
    salient: list[str] = []
    for line in raw_output.splitlines():
        if any(m in line for m in _SMOKE_SALIENT_MARKERS):
            salient.append(line)
    if not salient:
        return raw_output[-4000:]
    # Cap salient at 20 lines so a runaway error stream can't crowd out
    # the trailing context entirely.
    head = "\n".join(salient[:20])
    tail = raw_output[-2000:]
    return f"{head}\n--- (build output tail) ---\n{tail}"


# uv prints ``you require <name> ∅`` when the resolver has non-empty
# candidate packages but the version constraint intersects to the empty
# set. The name uv shows is PEP503-normalised (``.``/``_``/``-`` → ``-``
# lower-case), so the repair LLM — reading the file source, which may
# spell it ``pdfminer.six`` — keeps prescribing a rename that is already
# done. Session a7e0bef1 burned 4 iterations on this exact confusion
# before HITL. Detecting this pattern and lifting the offending manifest
# line into the diagnostic short-circuits the misdiagnosis.
_UV_EMPTY_VERSION_SET_RE = re.compile(r"you require ([A-Za-z0-9._-]+) ∅")


def _pep503_normalize(name: str) -> str:
    """PEP 503 name normalisation: collapse ``-``/``_``/``.`` runs to
    a single ``-`` and lower-case. Used to match uv's error-display
    name against manifest lines that may spell the same package
    differently."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _uv_empty_version_set_hint(
    raw_output: str, workspace_path: Optional[str],
) -> Optional[str]:
    """When uv's output includes ``you require <NAME> ∅``, return a
    hint that (a) names PEP503 normalisation so ``pdfminer-six`` in the
    error is not mistaken for a different package than ``pdfminer.six``
    in the file, and (b) quotes the offending constraint from
    ``requirements.txt`` / ``pyproject.toml`` so the LLM sees the real
    culprit — a bounded range that doesn't overlap any released
    version. Returns None when the pattern isn't present.

    Common cause: LLM chose ``<2024.0`` on a calendar-versioned package
    whose releases look like ``20231228`` — 2024.0 < 20231228 → empty
    range. Without this hint the LLM reads ``pdfminer-six`` (uv's
    normalised display) as a different package from ``pdfminer.six``
    (its own file) and keeps prescribing a rename that changes nothing.
    """
    if not raw_output:
        return None
    m = _UV_EMPTY_VERSION_SET_RE.search(raw_output)
    if not m:
        return None
    reported = m.group(1)
    normalized = _pep503_normalize(reported)
    matched_line: Optional[str] = None
    if workspace_path and os.path.isdir(workspace_path):
        for manifest in ("requirements.txt", "requirements-dev.txt",
                         "pyproject.toml"):
            path = os.path.join(workspace_path, manifest)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        stripped = line.strip()
                        if not stripped or stripped.startswith("#"):
                            continue
                        # Extract the package-name token: up to the first
                        # constraint / extra / comment / quote character.
                        head = re.split(
                            r"[<>=!~;\[\s,'\"]", stripped, maxsplit=1,
                        )[0]
                        if _pep503_normalize(head) == normalized:
                            matched_line = f"{manifest}:{lineno}: {stripped}"
                            break
                if matched_line:
                    break
            except OSError:
                continue
    lines = [
        "[uv version-constraint failure detected]",
        (
            f"uv reports `you require {reported} ∅` — the trailing ∅ is "
            f"the EMPTY VERSION SET (no released version satisfies the "
            f"constraint), NOT 'package not found'. The package exists; "
            f"the version range you wrote doesn't overlap any release."
        ),
    ]
    if matched_line:
        lines.append(f"Offending constraint: `{matched_line}`.")
        lines.append(
            "Fix: widen or correct the version bound so at least one "
            "released version satisfies it. Common trap: bounds like "
            "`<2024.0` on a calendar-versioned package whose releases "
            "look like `20231228` — 2024.0 is numerically less than "
            "20231228, so the range is empty. Use e.g. `<20250101` for "
            "calendar-versioned packages, or drop the upper bound."
        )
    else:
        lines.append(
            f"Search for `{reported}` (any of `{reported.replace('-', '.')}` "
            f"/ `{reported.replace('-', '_')}` / `{reported}`) in "
            f"`requirements.txt` / `pyproject.toml` and widen its bound. "
            "Common trap: bounds like `<2024.0` on a calendar-versioned "
            "package whose releases look like `20231228` — empty range."
        )
    lines.append(
        f"Note: uv normalises names per PEP 503 (`.`/`_`/`-` runs → `-`, "
        f"lower-case). `{reported}` in the error is the same package "
        f"however it is spelled in the manifest — do NOT rename it."
    )
    return "\n".join(lines)


def _uv_venv_prefix() -> str:
    """Shell prefix that creates (idempotent) and activates the sandbox
    venv so subsequent ``uv pip install`` and ``python3`` calls in the
    same chained command target a user-writable location.

    Emitted as a single chain element so callers can ``&&``-prepend it to
    an existing install/pytest command. The venv is rebuilt on every fresh
    sandbox container (``/tmp`` is ephemeral); within a container the
    ``test -d`` guard makes re-activation a no-op.

    Note: no subshell parens here. In bash ``&&`` and ``||`` are equal-
    precedence and left-associative, so ``test -d X || uv venv X && …``
    parses identically to ``(test -d X || uv venv X) && …``. The earlier
    parenthesised form was rejected by the sandbox security validator
    (it read ``(test`` as the command name and matched
    ``whitelist_missing:(test``); the unparenthesised form has the same
    semantics and parses cleanly.
    """
    return (
        f"test -d {_PROD_SMOKE_VENV_PATH}/bin "
        f"|| uv venv --system-site-packages {_PROD_SMOKE_VENV_PATH} "
        f"&& . {_PROD_SMOKE_VENV_PATH}/bin/activate"
    )


def _compose_prod_smoke_install_step(workspace_path: str) -> Optional[str]:
    """Build a comprehensive install step for the prod-import smoke check
    by sniffing **all** Python manifests in the workspace (root + first-
    level subdirs) and chaining the install commands.

    Independent of ``build_command`` — the build command may be ``make
    build`` (which the smoke check can't safely use as install) or a
    bare ``pip install pytest`` seed (which doesn't install the project's
    deps). This sniff is the authoritative source for what gets pip-
    installed before the import check runs.

    Returns the chained install command, or ``None`` if no Python manifest
    exists anywhere in the workspace (in which case the caller should
    skip the smoke check — there's nothing to import).

    Supported tech stack only (Python). Java/Node smoke checks aren't
    Python-side imports so the install step here doesn't need to cover
    them; the actual ``mvn`` / ``npm`` build runs separately downstream.
    """
    if not os.path.isdir(workspace_path):
        return None

    install_cmds: list[str] = []

    def has_file(*parts: str) -> bool:
        return os.path.isfile(os.path.join(workspace_path, *parts))

    # Root-level Python manifest takes precedence — single-project repo.
    if has_file("pyproject.toml"):
        install_cmds.append("uv pip install -e .")
    elif has_file("requirements.txt"):
        install_cmds.append("uv pip install -r requirements.txt")

    # Plus any Python manifest one level deep (monorepo: server/ + client/).
    # Skipped subdirs match the cli.py subdir-detection skip set so we don't
    # accidentally pip-install the frontend or recurse into build output.
    try:
        entries = sorted(os.listdir(workspace_path))
    except OSError:
        entries = []
    for entry in entries:
        if entry.startswith("."):
            continue
        if entry in _INSTALL_STEP_SUBDIR_SKIP:
            continue
        full = os.path.join(workspace_path, entry)
        if not os.path.isdir(full):
            continue
        sub_pyproject = os.path.join(full, "pyproject.toml")
        sub_req = os.path.join(full, "requirements.txt")
        if os.path.isfile(sub_pyproject):
            install_cmds.append(f"uv pip install -e {entry}")
        elif os.path.isfile(sub_req):
            install_cmds.append(
                f"uv pip install -r {entry}/requirements.txt"
            )
        # Optional dev requirements alongside.
        sub_req_dev = os.path.join(full, "requirements-dev.txt")
        if os.path.isfile(sub_req_dev):
            install_cmds.append(
                f"uv pip install -r {entry}/requirements-dev.txt"
            )

    if not install_cmds:
        return None

    # pytest itself — pre-baked in the builder image but harmless to
    # re-state, and required if the harness runs outside the builder image.
    install_cmds.append("uv pip install pytest")
    # Prepend the venv prefix so every `uv pip install` writes to the
    # user-writable /tmp venv instead of /usr/local/lib/python3.11/dist-
    # packages (non-root sandboxes can't write there → Permission denied).
    return " && ".join([_uv_venv_prefix(), *install_cmds])


# A top-level module is considered a project module (not a missing third-
# party dep) if it appears at the workspace root or in a first-level
# Python subdir as a package / module file. Used by the smoke check's
# DEPS_NOT_INSTALLED classifier — a ModuleNotFoundError on a name that
# IS a project module means the file is broken (repair-able); the same
# error on a name that isn't means the env is missing a dependency
# (install-step or manifest fix).
def _project_top_level_names(workspace_path: str) -> set[str]:
    if not os.path.isdir(workspace_path):
        return set()
    names: set[str] = set()
    try:
        for entry in os.listdir(workspace_path):
            if entry.startswith(".") or entry in _SMOKE_CHECK_SKIP_DIRS:
                continue
            full = os.path.join(workspace_path, entry)
            if os.path.isdir(full):
                # Any Python directory at root is a candidate top-level
                # package. (Subdirs of monorepos count too — server/, app/,
                # core/, etc. — because the smoke check imports them as
                # `server.foo`, where `server` is the resolved top-level.)
                names.add(entry)
            elif entry.endswith(".py"):
                names.add(entry[:-3])
    except OSError:
        pass
    return names


_MODULE_NAME_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")


def _detect_python_manifest(workspace_path: str) -> str:
    """Return the canonical Python dep-manifest path for ``workspace_path``,
    relative to the workspace root. Picks the most prominent existing
    file so the DEPS_NOT_INSTALLED diagnostic points the repair LLM at
    a path that actually exists.

    Probe order: root pyproject.toml → root requirements.txt → first
    subdir pyproject.toml → first subdir requirements.txt → fall back to
    a fresh ``requirements.txt`` at workspace root (LLM will need to
    create it). Mirrors the install-step composer's probe order so the
    diagnostic points at the SAME file the install step would read.
    """
    if os.path.isfile(os.path.join(workspace_path, "pyproject.toml")):
        return "pyproject.toml"
    if os.path.isfile(os.path.join(workspace_path, "requirements.txt")):
        return "requirements.txt"
    try:
        entries = sorted(os.listdir(workspace_path))
    except OSError:
        entries = []
    for entry in entries:
        if entry.startswith(".") or entry in _INSTALL_STEP_SUBDIR_SKIP:
            continue
        full = os.path.join(workspace_path, entry)
        if not os.path.isdir(full):
            continue
        if os.path.isfile(os.path.join(full, "pyproject.toml")):
            return f"{entry}/pyproject.toml"
        if os.path.isfile(os.path.join(full, "requirements.txt")):
            return f"{entry}/requirements.txt"
    return "requirements.txt"


def _classify_smoke_failure(
    module: str,
    exc_type: str,
    message: str,
    project_top_names: set[str],
) -> tuple[str, str]:
    """Classify a single prod-smoke failure as either a third-party
    DEPS_NOT_INSTALLED error or a project-side bug.

    Returns ``(error_code, repair_hint)`` where ``error_code`` is one of:
      - ``DEPS_NOT_INSTALLED:<package>`` — missing pip-installable dep
      - ``PROD_IMPORT_SMOKE:<exc_type>`` — original code-side failure
    and ``repair_hint`` is the guidance line to attach to the diagnostic
    message (empty string for the unchanged case).
    """
    if exc_type != "ModuleNotFoundError":
        return f"PROD_IMPORT_SMOKE:{exc_type}", ""
    m = _MODULE_NAME_RE.search(message)
    if not m:
        return f"PROD_IMPORT_SMOKE:{exc_type}", ""
    missing = m.group(1)
    top = missing.split(".", 1)[0]
    if top in project_top_names:
        # Project module missing — code bug, leave the original tag.
        return f"PROD_IMPORT_SMOKE:{exc_type}", ""
    # Third-party dep. The fix is to add it to requirements.txt /
    # pyproject.toml — NOT to invent a project module with that name.
    hint = (
        f" — `{top}` is a third-party package, not project code. Add it "
        f"to requirements.txt (or pyproject.toml `[project.dependencies]`). "
        f"Do NOT create a `{top}/` directory or `{top}.py` file in the "
        f"workspace."
    )
    return f"DEPS_NOT_INSTALLED:{top}", hint


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
    # Sandbox CommandValidator refused the smoke command (e.g. ``(test``
    # was rejected before the prefix-stripping fix landed). Repair LLM
    # cannot reach the validator config, so surface a tagged diagnostic
    # the caller can promote into ``build_command_blocked`` on node_state.
    # Without this short-circuit the smoke check falls through to the
    # coarse PROD_IMPORT_SMOKE catch-all, and the router dispatches to
    # repair_node — burning iterations on a problem the LLM cannot fix.
    blocked_rule = _is_command_blocked_by_security(result.raw_output or "")
    if blocked_rule:
        logger.warning(
            "[prod-smoke] Smoke install command blocked by sandbox security "
            "validator (rule=%s). Surfacing as BUILD_COMMAND_BLOCKED.",
            blocked_rule,
        )
        return [{
            "error_code": "BUILD_COMMAND_BLOCKED",
            "message": (
                f"The composed prod-smoke install command was rejected by "
                f"the sandbox security validator (matched rule: "
                f"{blocked_rule}). This is a harness-internal config "
                f"issue, not user code — the repair LLM cannot fix it "
                f"because the validator config lives outside the "
                f"workspace allowlist."
            ),
            "file": "<harness:security-validator>",
            "line": 0,
            "column": 0,
            "severity": "error",
            "semantic_context": "",
            "matched_rule": blocked_rule,
        }]
    # Parse FAIL: lines from the output. The smoke script prints them
    # one per line right after the FAILURES header.
    diagnostics: list[dict[str, Any]] = []
    seen = False
    project_top = _project_top_level_names(workspace_path)
    missing_third_party: dict[str, list[str]] = {}
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
        error_code, hint = _classify_smoke_failure(
            module, exc_type, message, project_top,
        )
        if error_code.startswith("DEPS_NOT_INSTALLED:"):
            pkg = error_code.split(":", 1)[1]
            missing_third_party.setdefault(pkg, []).append(module)
            continue
        diagnostics.append({
            "error_code": error_code,
            "message": (
                f"Production module `{module}` failed to import "
                f"({exc_type}): {message}{hint}"
            ),
            "file": module.replace(".", "/") + ".py",
            "line": 0,
            "column": 0,
            "severity": "error",
            "semantic_context": "",
        })
    # Collapse all third-party DEPS_NOT_INSTALLED failures into a single
    # diagnostic. The repair LLM needs one actionable message ("add these
    # to requirements.txt") — surfacing 27 separate cascade failures sends
    # it on a wild goose chase trying to fix project code that's fine.
    if missing_third_party:
        packages = sorted(missing_third_party.keys())
        affected_count = sum(len(v) for v in missing_third_party.values())
        manifest_target = _detect_python_manifest(workspace_path)
        diagnostics.insert(0, {
            "error_code": "DEPS_NOT_INSTALLED",
            "message": (
                f"{len(packages)} third-party Python package(s) failed to "
                f"import — they are not installed in the build sandbox and "
                f"cascade across {affected_count} production module(s). "
                f"Missing: {', '.join(packages)}. "
                f"Fix: ensure each is declared in `{manifest_target}` so "
                f"the build's install step picks it up. Do NOT create "
                f"project modules with these names."
            ),
            "file": manifest_target,
            "line": 0,
            "column": 0,
            "severity": "error",
            "semantic_context": "",
        })
        logger.warning(
            "[prod-smoke] %d third-party package(s) missing from build env: "
            "%s. Surfacing as single DEPS_NOT_INSTALLED diagnostic instead "
            "of %d per-module cascade failures.",
            len(packages), ", ".join(packages), affected_count,
        )
    if not diagnostics:
        # Fall back to a single coarse diagnostic carrying the tail of
        # the output so the LLM has something to work with. The tail
        # size must be big enough to keep the toolchain's chained
        # "Caused by:" stack — uv truncates its own wheel-install
        # failures across 3–5 nested causes, and capping at 1500 chars
        # cut them off at literally the word "Caused" in session
        # db6bfcbe. 4000 leaves room for the full cause chain plus a
        # handful of "Downloaded" progress lines for context.
        #
        # Salient lines first: the HITL summariser truncates each
        # diagnostic's message and only the leading portion reaches the
        # LLM. If the raw tail starts with install-progress chatter
        # ("Downloading pydantic-core (2.0MiB)") the real error line
        # ("error: Failed to install …  Permission denied") gets buried
        # past the truncation boundary — surface ERROR markers up front
        # so the first 1500 chars hit the LLM's prompt.
        raw = result.raw_output or ""
        uv_hint = _uv_empty_version_set_hint(raw, workspace_path)
        if uv_hint:
            # Distinct error_code so the fingerprint doesn't blur with
            # generic PROD_IMPORT_SMOKE failures — persistence across
            # rounds then keys off "version-constraint" specifically,
            # and the reflection LLM sees the same tag if it recurs.
            error_code = "UV_VERSION_CONSTRAINT_EMPTY"
            message = f"{uv_hint}\n\n{_surface_salient_errors(raw)}"
        else:
            error_code = "PROD_IMPORT_SMOKE"
            message = _surface_salient_errors(raw)
        diagnostics.append({
            "error_code": error_code,
            "message": message,
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
            if len(content) > 20000:
                content = content[:20000] + "\n# ...(truncated)\n"
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


_DEFAULT_REPAIR_DIAGNOSTIC_CAP = 24
_DEFAULT_REPAIR_INVENTORY_CAP = 100
_EOS_REPAIR_DIAGNOSTIC_CAP = 30
_EOS_REPAIR_INVENTORY_CAP = 150


# Critical config files that should ALWAYS appear in the repair-prompt
# workspace inventory (prepended ahead of the cap), regardless of how
# many other files were modified this session. These are the paths the
# LLM most often CREATE_FILE's into a "File already exists with
# different content" rejection because they're scaffolded once early and
# fall out of the inventory window in later iterations. Limited to the
# supported tech stack (Python / Java / React+TS+Tailwind+Vite + the
# usual Docker/Make plumbing).
_CRITICAL_CONFIG_BASENAMES = frozenset({
    # Python
    "requirements.txt", "requirements-dev.txt", "pyproject.toml",
    "setup.py", "setup.cfg", "Pipfile", "Pipfile.lock", "poetry.lock",
    "uv.lock",
    # Java
    "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
    "settings.gradle.kts", "gradle.properties", "gradlew", "gradlew.bat",
    # Node / React / TS / Vite / Tailwind
    "package.json", "package-lock.json", "tsconfig.json", "tsconfig.node.json",
    "vite.config.ts", "vite.config.js", "tailwind.config.js",
    "tailwind.config.ts", "postcss.config.js", "postcss.config.cjs",
    ".eslintrc.json", ".eslintrc.cjs", ".prettierrc",
    # Build / container plumbing
    "Makefile", "makefile", "GNUmakefile", "Dockerfile",
    "docker-compose.yml", "docker-compose.yaml",
    ".env", ".env.example", ".gitignore",
})


def _is_critical_config_path(path: str) -> bool:
    """True when ``path``'s basename is in the supported-tech critical
    config set. Path is the workspace-relative form recorded in
    ``modified_files`` (e.g. ``server/requirements.txt``).
    """
    if not path:
        return False
    basename = path.rsplit("/", 1)[-1]
    return basename in _CRITICAL_CONFIG_BASENAMES


_CR_EXTRA_FILE_CAP = 6


def _cr_impact_augment(
    state: "AgentState", diag_files: list[str],
) -> list[str]:
    """Return additional file paths to surface to the repair LLM when
    in change-request mode (Phase L).

    Augments the existing diagnostic file list with up to
    :data:`_CR_EXTRA_FILE_CAP` extra files:

    - **Shared utilities the session has touched.** Intersect
      ``modified_files`` with ``DependencyGraph.high_fanout_files`` so
      we only surface utilities that THIS CR actually amended.
    - **Immediate callers** of every diag file via
      ``DependencyGraph.immediate_callers_of``. When a CR amends a
      utility's signature, its callers are the most likely places a
      regression surfaces.

    Empty list outside CR mode, on any analyzer error, or when the
    workspace has nothing to add. Best-effort: this helper must never
    block a repair attempt.
    """
    if not state.get("change_request_mode"):
        return []
    workspace = state.get("workspace_path") or ""
    if not workspace:
        return []
    seen: set[str] = set(diag_files)
    extras: list[str] = []
    try:
        from harness.impact import DependencyGraph

        graph_obj = DependencyGraph(workspace)
        graph_obj.build()

        # High-fanout files the CR touched.
        touched = {
            p for p in (state.get("modified_files") or []) if p
        }
        if touched:
            top = graph_obj.high_fanout_files(top_k=10)
            for fp, _fanout in top:
                if len(extras) >= _CR_EXTRA_FILE_CAP:
                    break
                # Compare on relpath so absolute/relative variants match.
                rel = os.path.relpath(fp, workspace)
                if rel in touched and rel not in seen:
                    extras.append(rel)
                    seen.add(rel)

        # Immediate callers of every diagnostic file.
        if len(extras) < _CR_EXTRA_FILE_CAP:
            callers = graph_obj.immediate_callers_of(
                list(diag_files), top_k=_CR_EXTRA_FILE_CAP,
            )
            for fp in callers:
                if len(extras) >= _CR_EXTRA_FILE_CAP:
                    break
                rel = os.path.relpath(fp, workspace)
                if rel in seen:
                    continue
                extras.append(rel)
                seen.add(rel)
    except Exception as exc:  # noqa: BLE001 — never block repair
        logger.debug(
            "[repair_node] CR impact augment skipped: %s", exc,
        )
        return []
    return extras


def _repair_file_caps(state: "AgentState") -> tuple[int, int]:
    """Return ``(diagnostic_cap, inventory_cap)`` for the current repair.

    Phase J — at the end-of-session repair (when
    ``node_state.end_of_session_phase`` is set by
    ``end_of_session_regression_node``), security-scan repairs may
    have touched shared utilities. The failing tests can cascade
    across many files the LLM didn't directly patch, so the default
    12+50 caps starve the model of the cross-file context it needs.
    Bumping to 30+150 gives the senior reasoning model enough surface
    to catch a shared-utility regression without exploding prompts on
    a per-batch repair where the smaller caps work well.

    The caps are configurable via the gateway's config:
    ``end_of_session_repair_diagnostic_cap`` (default
    :data:`_EOS_REPAIR_DIAGNOSTIC_CAP`) and
    ``end_of_session_repair_inventory_cap`` (default
    :data:`_EOS_REPAIR_INVENTORY_CAP`).
    """
    ns = state.get("node_state") or {}
    in_eos = bool(
        isinstance(ns, dict) and ns.get("end_of_session_phase")
    )
    if not in_eos:
        return _DEFAULT_REPAIR_DIAGNOSTIC_CAP, _DEFAULT_REPAIR_INVENTORY_CAP
    gw = get_gateway()
    diag_cap = _EOS_REPAIR_DIAGNOSTIC_CAP
    inv_cap = _EOS_REPAIR_INVENTORY_CAP
    if gw is not None:
        diag_cap = int(getattr(
            gw.config, "end_of_session_repair_diagnostic_cap",
            _EOS_REPAIR_DIAGNOSTIC_CAP,
        ))
        inv_cap = int(getattr(
            gw.config, "end_of_session_repair_inventory_cap",
            _EOS_REPAIR_INVENTORY_CAP,
        ))
    return diag_cap, inv_cap


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


_PY_IMPORT_FROM_RE = re.compile(
    r"^\s*from\s+([a-zA-Z_][\w.]*)\s+import\b",
    re.MULTILINE,
)
_PY_IMPORT_RE = re.compile(
    r"^\s*import\s+([a-zA-Z_][\w.]*(?:\s*,\s*[a-zA-Z_][\w.]*)*)",
    re.MULTILINE,
)


def _first_party_imports_for(
    workspace_path: str, test_rel_path: str, *, max_depth: int = 2,
) -> list[str]:
    """Parse the test file's ``import`` / ``from X import Y`` statements
    and return the WORKSPACE-RELATIVE paths of first-party modules the
    test depends on.

    "First-party" = the import target resolves to a file under
    ``workspace_path``. Third-party packages (``pytest``,
    ``unittest.mock``, ``httpx``, …) are skipped because they live in
    site-packages, not the workspace.

    ``max_depth`` controls how deep into ``server.services.search`` we
    walk while looking for a matching source file:
      - depth=1 → ``server/services/search.py`` only
      - depth=2 → also try ``server/services/search/__init__.py`` etc.

    Used by the prefetch pass (Fix #2): the repair LLM otherwise burns
    rounds emitting READ_FILE for the modules it knows it needs to see.
    Pre-attaching them removes that round-trip entirely — same shape
    Claude Code achieves implicitly via Read tool calls.

    Returns workspace-relative paths in declaration order, deduped.
    Empty list when the test file can't be read or contains no
    first-party imports.
    """
    if not workspace_path or not test_rel_path:
        return []
    abs_test = os.path.join(workspace_path, test_rel_path)
    try:
        with open(abs_test, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
    except OSError:
        return []
    modules: list[str] = []
    for m in _PY_IMPORT_FROM_RE.finditer(src):
        modules.append(m.group(1))
    for m in _PY_IMPORT_RE.finditer(src):
        for name in m.group(1).split(","):
            name = name.strip().split(" as ")[0].strip()
            if name:
                modules.append(name)
    out: list[str] = []
    seen: set[str] = set()
    for mod in modules:
        if mod.split(".")[0] in _STDLIB_TOPLEVEL:
            continue  # cheap stdlib filter — see frozenset below
        rel_candidates: list[str] = []
        parts = mod.split(".")
        # Walk shortest-first so ``server.services.search`` resolves to
        # ``server/services/search.py`` before ``server/services/__init__.py``
        # (the more-specific target is what the test actually depends on).
        for depth in range(len(parts), 0, -1):
            base = "/".join(parts[:depth])
            rel_candidates.append(base + ".py")
            rel_candidates.append(base + "/__init__.py")
        if max_depth < len(parts):
            rel_candidates = rel_candidates[: 2 * max_depth]
        for rel in rel_candidates:
            if rel in seen:
                continue
            abs_path = os.path.join(workspace_path, rel)
            if os.path.isfile(abs_path):
                seen.add(rel)
                out.append(rel)
                break  # one hit per imported module — the most specific
    return out


# Conservative stdlib + common-third-party top-level packages. The list is
# deliberately not exhaustive — anything we DON'T list and that resolves
# to a real file under workspace_path is treated as first-party. False
# negatives just mean the LLM has to READ_FILE the module itself; false
# positives (a project module named "json" or "os") are vastly less
# common and the prefetch only adds context, never overrides anything.
_STDLIB_TOPLEVEL: frozenset[str] = frozenset({
    "abc", "argparse", "ast", "asyncio", "base64", "bisect", "builtins",
    "calendar", "collections", "concurrent", "contextlib", "copy", "csv",
    "ctypes", "datetime", "decimal", "difflib", "dis", "email", "enum",
    "errno", "fnmatch", "fractions", "functools", "gc", "glob", "gzip",
    "hashlib", "heapq", "hmac", "html", "http", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "linecache", "locale", "logging",
    "math", "mimetypes", "multiprocessing", "numbers", "operator", "os",
    "pathlib", "pickle", "pkgutil", "platform", "plistlib", "posix",
    "pprint", "queue", "random", "re", "secrets", "select", "shelve",
    "shutil", "signal", "smtplib", "socket", "sqlite3", "ssl", "stat",
    "string", "struct", "subprocess", "sys", "sysconfig", "tempfile",
    "textwrap", "threading", "time", "timeit", "token", "tokenize",
    "traceback", "types", "typing", "unicodedata", "unittest", "urllib",
    "uuid", "warnings", "weakref", "xml", "zipfile", "zoneinfo",
    # Common third-party we never want to prefetch as "first-party":
    "pytest", "pytest_asyncio", "httpx", "requests", "fastapi", "starlette",
    "pydantic", "sqlalchemy", "redis", "fakeredis", "aiohttp", "anyio",
    "trio", "uvicorn", "click", "typer", "rich", "yaml", "toml",
    "numpy", "pandas", "torch", "tensorflow", "sklearn", "scipy",
    "matplotlib", "plotly", "boto3", "botocore", "google", "openai",
    "anthropic", "langchain", "langgraph", "langsmith", "tiktoken",
    "psycopg2", "asyncpg", "alembic", "marshmallow", "attrs",
    "celery", "kombu", "amqp", "billiard",
})


def _conftest_chain_for_test(
    workspace_path: str, test_rel_path: str,
) -> list[str]:
    """Return the chain of ``conftest.py`` paths that pytest would apply
    to the test at ``test_rel_path``, in pytest's precedence order
    (workspace-root first, leaf last). Honours the rule that pytest only
    picks up a ``conftest.py`` from a directory on the path from the
    rootdir down to the test file.

    Output paths are workspace-relative. Skips any directory above
    ``workspace_path`` (we don't claim to know where pytest's rootdir is
    set when it's above the workspace).

    Used by :func:`_collect_conftests_for_failing_tests` to surface the
    correct conftest tree to the repair LLM — workspaces with
    overlapping trees (e.g. ``tests/conftest.py`` AND
    ``server/tests/conftest.py``) otherwise produce repair attempts on
    the wrong file because the LLM has no way to know which one pytest
    is actually loading.
    """
    if not test_rel_path or not workspace_path:
        return []
    rel = test_rel_path.replace("\\", "/").lstrip("./")
    if rel.startswith("/"):
        rel = rel.lstrip("/")
    parts = [p for p in rel.split("/") if p]
    if not parts:
        return []
    # Strip the filename — pytest looks at DIRECTORIES, not files.
    parts = parts[:-1]
    chain: list[str] = []
    # Workspace-root conftest, then each ancestor down to the test's dir.
    candidates: list[str] = ["conftest.py"]
    for i in range(1, len(parts) + 1):
        candidates.append("/".join(parts[:i]) + "/conftest.py")
    for rel_candidate in candidates:
        abs_path = os.path.join(workspace_path, rel_candidate)
        if os.path.isfile(abs_path):
            chain.append(rel_candidate)
    return chain


def _collect_conftests_for_failing_tests(
    workspace_path: str,
    compiler_errors: list[DiagnosticObjectDict],
    *,
    max_unique: int = 6,
) -> list[tuple[str, list[str]]]:
    """For every distinct failing-test path in ``compiler_errors``, return
    ``(test_rel_path, conftest_chain)`` so the prompt can show which
    conftest tree pytest is loading for each failing test.

    Multiple failing tests may share the same chain — only the FIRST
    occurrence of each unique chain is returned (the caller wants a
    representative example per chain, not a row per failing test).
    Capped at ``max_unique`` to bound token cost; large failure sets
    rarely span more than 2-3 distinct chains anyway.

    Returns ``[]`` when there are no test failures with file info.
    Synthetic markers (``<harness:...>``) are skipped.
    """
    if not compiler_errors:
        return []
    out: list[tuple[str, list[str]]] = []
    seen_chains: set[tuple[str, ...]] = set()
    for err in compiler_errors:
        f = str(err.get("file", "") or "").strip()
        if not f or f.startswith("<"):
            continue
        rel = f
        # Normalise absolute paths back to workspace-relative when possible.
        if os.path.isabs(rel):
            try:
                rel = os.path.relpath(rel, workspace_path)
            except ValueError:
                continue
        if rel.startswith(".."):
            continue
        chain = _conftest_chain_for_test(workspace_path, rel)
        if not chain:
            continue
        key = tuple(chain)
        if key in seen_chains:
            continue
        seen_chains.add(key)
        out.append((rel, chain))
        if len(out) >= max_unique:
            break
    return out


def _format_conftest_chains_for_repair(
    workspace_path: str,
    chains: list[tuple[str, list[str]]],
    *,
    max_lines_per_file: int = 120,
    record_hashes_into: Optional[dict[str, str]] = None,
) -> str:
    """Render the conftest chains as a Markdown block for the repair
    prompt. Empty string when ``chains`` is empty so callers can
    concatenate unconditionally.

    Each chain is shown in pytest precedence order (root first → leaf
    last). When multiple distinct chains exist (two test trees with
    independent conftests, as in session cf3fcd27's
    ``tests/conftest.py`` + ``server/tests/conftest.py`` split), the
    block explicitly calls out the split so the LLM stops patching the
    wrong one. The content for each conftest is line-numbered and
    truncated at ``max_lines_per_file`` so large fixtures don't blow the
    token budget — the LLM can READ_FILE for the rest if it needs to.
    """
    if not chains:
        return ""
    parts: list[str] = ["\n## Active conftest.py files for failing tests"]
    if len(chains) >= 2:
        parts.append(
            "**Multiple distinct conftest chains** — the failing tests live "
            "under separate test trees with independent fixture sets. "
            "Patching the wrong tree is a no-op for the failing test. Match "
            "your patch's file path against the chain shown for the test "
            "you're fixing."
        )
    parts.append(
        "Pytest loads every ``conftest.py`` on the path from the rootdir "
        "down to the failing test, root first. Anything in a LATER "
        "conftest can override an EARLIER one. The chain below is the "
        "fixture surface area for each failing test."
    )
    for test_rel, chain in chains:
        parts.append(f"\n### Chain for `{test_rel}`")
        parts.append(
            "Precedence order (load order — first is rootmost):\n"
            + "\n".join(f"  {i+1}. `{c}`" for i, c in enumerate(chain))
        )
        for conftest_rel in chain:
            abs_path = os.path.join(workspace_path, conftest_rel)
            rendered, file_hash = _render_file_with_line_numbers_and_hash(
                abs_path,
            )
            if rendered is None:
                continue
            if record_hashes_into is not None and file_hash is not None:
                record_hashes_into[conftest_rel] = file_hash
            # Cap lines to keep large fixture modules from dominating.
            rendered_lines = rendered.splitlines()
            truncated = ""
            if len(rendered_lines) > max_lines_per_file:
                rendered = "\n".join(rendered_lines[:max_lines_per_file])
                truncated = (
                    f"\n  ... ({len(rendered_lines) - max_lines_per_file} "
                    "more lines elided — emit READ_FILE if you need them)"
                )
            parts.append(
                f"\n#### `{conftest_rel}`\n```\n{rendered}{truncated}\n```"
            )
    return "\n".join(parts) + "\n"


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


def _build_patcher_rejection_diagnosis_prompt(
    *,
    rejections: list[Any],
    patch_failures: list[Any],
    allowed_paths: list[str],
    modified_files: list[str],
) -> str:
    """Compose the prompt for the patcher-rejection diagnosis call (#4).

    Pass the raw rejection records, the patch-failure records, the
    currently allowed roots, and the modified-files inventory. The
    helper trims each section so the prompt stays inside the cheap
    model's working window. Returns the prompt string; the caller
    dispatches via :func:`_maybe_judgment_llm`.
    """
    rej_paths = sorted(
        {str(r.get("file", "")) for r in (rejections or []) if r.get("file")}
    )
    rej_block = "\n".join(f"  - {p}" for p in rej_paths[:15]) or "  (none)"

    pf_lines: list[str] = []
    for pf in (patch_failures or [])[:8]:
        if not isinstance(pf, dict):
            continue
        op = str(pf.get("operation", "?"))
        fp = str(pf.get("file", "?"))
        reason = str(pf.get("reason", "") or pf.get("error", ""))[:240]
        pf_lines.append(f"  - {op} on {fp} :: {reason}")
    pf_block = "\n".join(pf_lines) or "  (none)"

    allowed_block = (
        "\n".join(f"  - {p}" for p in allowed_paths[:20])
        if allowed_paths else "  (unconstrained)"
    )
    inv_shown = modified_files[:25]
    inv_block = (
        "\n".join(f"  - {p}" for p in inv_shown)
        + (f"\n  - (+{len(modified_files) - len(inv_shown)} more)"
           if len(modified_files) > len(inv_shown) else "")
        if modified_files else "  (none)"
    )

    return (
        "You are diagnosing why a code-patcher rejected the previous "
        "repair attempt's hunks. For each failure listed below, classify "
        "it into exactly one of these categories:\n"
        "  ALLOWLIST_MISS  — patch targeted a path outside the allowed roots\n"
        "  STALE_CONTEXT   — REPLACE_BLOCK search lines don't match disk "
        "(file changed or model guessed)\n"
        "  WRONG_FILE      — patch targets a file that does not exist or "
        "the right code lives elsewhere\n"
        "  FORMAT_ERROR    — malformed block (missing markers, bad header)\n\n"
        "Then emit ONE corrective instruction per distinct category — at "
        "most 4 short imperative sentences total. Reference the actual "
        "file names. Concrete advice ('move app/server.py under src/' or "
        "'emit READ_FILE for routes.py first; the file has 38 lines now') "
        "beats generic advice ('be more careful'). Do NOT restate the "
        "category labels in your reply; just give the directives.\n\n"
        "## Allowlist rejections (paths the patcher refused)\n"
        f"{rej_block}\n\n"
        "## Other patch failures (operation :: reason)\n"
        f"{pf_block}\n\n"
        "## Currently allowed roots\n"
        f"{allowed_block}\n\n"
        "## Files already in the workspace (do NOT CREATE_FILE these)\n"
        f"{inv_block}\n"
    )


def _build_preflight_autofix_prompt(
    *, symbols: list[str], build_command: str, sandbox_image: str,
) -> str:
    """Compose the prompt for the pre-flight autofix classifier (#2).

    The classifier gets the list of unique missing symbols plus the build
    command and the sandbox image so it can reason about whether each
    symbol can plausibly be installed via the build's package manager.
    The expected response shape is a strict JSON object so the parser
    can read it deterministically without LLM-shape gymnastics.
    """
    return (
        "You are classifying missing-dependency error symbols so the "
        "harness can skip futile autofix attempts. For EACH symbol below, "
        "decide whether it is:\n"
        "  MANIFEST_FIXABLE — a library/package that the build's package "
        "manager (pip / npm / cargo / go mod / maven / gradle) could "
        "install if appended to the workspace manifest. Examples: "
        "'requests', 'pytest-asyncio', 'lodash', '@types/node'.\n"
        "  TOOLCHAIN_MISMATCH — a system binary, language runtime, or "
        "package manager itself that the sandbox image lacks. No edit to "
        "requirements.txt / package.json can fix it; the sandbox image "
        "or build_command must change. Examples: 'pip', 'npm', 'node', "
        "'cargo', 'go', 'docker', 'make', 'gcc', 'java'.\n\n"
        "Respond with STRICT JSON ONLY — no prose, no markdown, no code "
        "fences. Shape:\n"
        '{"classifications": {"<symbol>": "MANIFEST_FIXABLE" | "TOOLCHAIN_MISMATCH", ...}}\n\n'
        f"Build command: {build_command or '(unknown)'}\n"
        f"Sandbox image: {sandbox_image or '(default)'}\n"
        "Symbols to classify:\n"
        + "\n".join(f"  - {s}" for s in symbols)
        + "\n"
    )


def _parse_preflight_verdict(
    raw: str, symbols: list[str],
) -> list[str]:
    """Parse the strict-JSON response from the pre-flight classifier (#2).

    Returns the list of symbols the LLM tagged TOOLCHAIN_MISMATCH, in the
    same order they appeared in ``symbols``. Returns the empty list on
    any parse failure or when no symbol is non-fixable — the caller's
    deterministic autofix path then runs as usual.
    """
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        logger.debug(
            "[judgment:preflight_autofix] Non-JSON verdict; skipping (raw=%r).",
            text[:200],
        )
        return []
    classifications = parsed.get("classifications") if isinstance(parsed, dict) else None
    if not isinstance(classifications, dict):
        return []
    non_fixable: list[str] = []
    sym_set = {s.lower() for s in symbols}
    for sym, verdict in classifications.items():
        if not isinstance(sym, str) or not isinstance(verdict, str):
            continue
        if sym.lower() not in sym_set:
            continue
        if verdict.strip().upper() == "TOOLCHAIN_MISMATCH":
            non_fixable.append(sym)
    return non_fixable


# Phase 2.2 fix B — repair-prompt prelude that makes the LLM trace
# before patching. Prepended onto the test-failure and default repair
# prompts (NOT the security/escalation paths — those have their own
# framing). The directive forces a one-sentence root-cause statement
# before any patch synthesis, which catches the class of bug where
# the failing assertion is several steps downstream of the actual fault
# (e.g. test asserts on amendment-selection output but the bug is in
# the regex that builds the period key). Empirically the repair LLM
# pattern-matches on the failing test name and patches the WRONG layer
# unless it's explicitly told to walk the data flow first.
_TRACE_FIRST_DIRECTIVE = (
    "## Trace before you patch\n\n"
    "Before writing ANY patch block, do this for each failing test "
    "below:\n"
    "1. Identify the EXACT data the assertion compared (the values "
    "pytest / vitest / etc. printed alongside `assert X == Y`). \n"
    "2. Trace BACK through the call chain: which function produced the "
    "wrong value? Which function fed it the inputs that led there? "
    "Walk until you hit the FIRST function in the chain whose output "
    "diverges from what the test setup implies it should be.\n"
    "3. Write ONE sentence — the root-cause sentence — naming that "
    "function, the file it lives in, and why its output is wrong "
    "given its inputs.\n"
    "4. Only then write the patch. Patch the function named in step 3, "
    "not the function the test names directly (unless they're the "
    "same).\n\n"
    "Do NOT skip steps 1–3. The failing test's name is often misleading "
    "— it describes the OUTER behaviour, not the inner function that's "
    "broken. Patching the layer the test names without tracing first "
    "is how repair loops burn iterations on the wrong file.\n\n"
    "---\n\n"
)


# Diagnostic-code prefixes / message fragments that strongly indicate a
# missing-dependency / environment-not-installed failure rather than a
# bug in the source code under repair. When the persistent failing-set
# is dominated by these, the right repair is to fix the install/build
# step (e.g. emit `cd <subdir> && npm install` before tsc runs, or add
# a missing package to package.json), NOT to keep editing the source.
# See session 7c30bce2 where 102 TS2307 errors persisted across 4 rounds
# because the harness's build command never ran `npm install` in the
# client subdir — the judge kept pointing the repair LLM at App.tsx
# while the real fix was upstream of the compile.
_INSTALL_ERROR_CODE_PREFIXES = (
    "TS2307",        # TypeScript "Cannot find module"
    "MISSING_DEP",   # harness-emitted missing dep
    "MODULENOTFOUND", # Python ModuleNotFoundError (uppercased)
    "IMPORTERROR",   # Python ImportError (uppercased)
)
_INSTALL_ERROR_MSG_FRAGMENTS = (
    "cannot find module",
    "cannot find type definition",
    "modulenotfounderror",
    "module not found",
    "no module named",
    # pytest-asyncio missing or not configured — the collector raises
    # "async def functions are not natively supported" for every async
    # test in the suite. Extractor tags these as AssertionError so none
    # of the *_CODE_PREFIXES match; the message fragment is the only
    # signal that lets the reflection judge point at requirements.txt /
    # pyproject.toml instead of the test source. Observed in session
    # 116667f5 where 4/4 diagnostics carried this shape and the judge
    # burned repair rounds on server/api/routes/search.py.
    "async def functions are not natively supported",
)


def _diagnostics_look_like_install_failure(
    diagnostics: list[dict[str, Any]],
) -> bool:
    """True when the majority of persistent diagnostics look like missing
    deps / unresolved imports — i.e. the kind of error that disappears
    once ``npm install`` / ``pip install`` actually runs. Used by the
    repair-reflection prompt to bias the judge toward classifying the
    real blocker as build/install rather than source code."""
    if not diagnostics:
        return False
    matched = 0
    for d in diagnostics:
        code = str(d.get("error_code", "") or "").upper()
        msg = str(d.get("message", "") or "").lower()
        if any(code.startswith(pfx) for pfx in _INSTALL_ERROR_CODE_PREFIXES):
            matched += 1
            continue
        if any(frag in msg for frag in _INSTALL_ERROR_MSG_FRAGMENTS):
            matched += 1
    # Majority threshold — a handful of "Cannot find module" alongside
    # real type errors is normal during repair; we only want to flip the
    # judge when the failing-set is dominated by them.
    return matched * 2 >= len(diagnostics) and matched >= 1


# Regex patterns for module names embedded in ModuleNotFoundError /
# ImportError / TS2307 messages. Module-level so the workspace-source
# helper doesn't recompile on every diagnostic scan.
_MISSING_MODULE_PATTERNS = (
    re.compile(r"[Nn]o module named ['\"]?([\w\.]+)"),
    re.compile(r"[Cc]annot find module ['\"]([\w\.@/\-]+)"),
)


def _missing_module_matches_workspace_source(
    diagnostics: list[dict[str, Any]], workspace_path: str,
) -> Optional[str]:
    """When a ModuleNotFoundError-style diagnostic names a module whose
    top-level package IS a workspace source directory (i.e. the code is
    right there on disk), return that name. Returns None when no such
    diagnostic exists.

    Signals a PATH/CWD/build-wiring failure (test runner in the wrong
    CWD, missing PYTHONPATH, ``cd <subdir> && pytest`` putting source
    root out of sys.path) rather than a missing pip dependency. Without
    this signal :func:`_diagnostics_look_like_install_failure` fires on
    the same shape and the reflection judge steers repair toward the
    manifest — which already declares everything needed — while the
    real fix (build command / test invocation) never gets addressed.
    See session 3193a24f: three repair rounds burned adding a root
    ``conftest.py`` and ``server/__init__.py`` before HITL was reached,
    because the missing name (``server``) was reported as a dependency
    even though ``server/`` was a scaffolded source directory.
    """
    if not diagnostics or not workspace_path or not os.path.isdir(workspace_path):
        return None
    for d in diagnostics:
        msg = str(d.get("message", "") or "")
        for pat in _MISSING_MODULE_PATTERNS:
            m = pat.search(msg)
            if not m:
                continue
            # Take the top-level package name only — sys.path resolution
            # keys on the leading segment (``server`` from
            # ``server.app.config``).
            top = m.group(1).split(".", 1)[0].strip()
            if not top:
                break
            candidate = os.path.join(workspace_path, top)
            if os.path.isdir(candidate):
                return top
            break
    return None


def _build_repair_reflection_prompt(
    *,
    prior_diagnostics_count: int,
    current_diagnostics_count: int,
    resolved_fingerprints: list[str],
    persisted_fingerprints: list[str],
    new_fingerprints: list[str],
    top_persisted_diagnostics: list[dict[str, Any]],
    install_failure_likely: bool = False,
    path_wiring_module: Optional[str] = None,
    build_output_tail: str = "",
) -> str:
    """Compose the prompt for the per-round repair-reflection judgment
    (Phase 2.2).

    The judgment LLM gets a structured before/after view of the failing
    diagnostics (now including each error's file + line) and is asked
    whether the previous round addressed the highest-priority error.
    It must answer with strict JSON so the parser can act on the verdict
    without LLM-shape gymnastics.

    The verdict is consumed by repair_node, which — when DISTRACTION /
    REGRESSION — injects ``real_blocker`` as a system message into the
    upcoming dispatch, giving the repair LLM a second opinion before
    it writes patches.

    Anti-hallucination contract (fix A):
    ------------------------------------
    Earlier versions of this prompt passed only error codes + truncated
    messages and demanded "citing a specific file/symbol/error" — so
    the LLM had no choice but to invent file/symbol context. The
    fabricated guidance got injected as authoritative direction and
    repair would chase phantoms for dozens of rounds. The fix passes
    real ``file:line`` for every persisted diagnostic and rephrases the
    instructions to ground the answer in that data, with an explicit
    escape hatch ("write \"insufficient data — investigate <file>'s
    data flow\"") for the case where messages are too thin to localize.
    """
    res_block = (
        "\n".join(f"  - {fp}" for fp in resolved_fingerprints[:5])
        if resolved_fingerprints else "  (none)"
    )
    pers_block = (
        "\n".join(f"  - {fp}" for fp in persisted_fingerprints[:5])
        if persisted_fingerprints else "  (none)"
    )
    new_block = (
        "\n".join(f"  - {fp}" for fp in new_fingerprints[:5])
        if new_fingerprints else "  (none)"
    )

    def _fmt_loc(diag: dict[str, Any]) -> str:
        f = str(diag.get("file", "") or "").strip()
        ln_raw = diag.get("line", 0)
        try:
            ln = int(ln_raw) if ln_raw is not None else 0
        except (TypeError, ValueError):
            ln = 0
        if f and ln > 0:
            return f"{f}:{ln}"
        if f:
            return f
        return "<no location>"

    if top_persisted_diagnostics:
        lines: list[str] = []
        for d in top_persisted_diagnostics[:3]:
            code = str(d.get("error_code", "?"))
            msg = str(d.get("message", ""))
            loc = _fmt_loc(d)
            lines.append(f"  - [{code}] {loc} :: {msg[:180]}")
        top_block = "\n".join(lines)
    else:
        top_block = "  (no persistent errors)"

    # Fix A — enrich the judge with the raw build-output tail when the top
    # persistent error's message is bare (equals the error code, or too short
    # to localize). Without this, an AssertionError from pytest whose only
    # useful info lives in the `--tb=long` traceback (e.g. "assert True is
    # False where True = <Session>.is_active") reaches the judge as just
    # "[AssertionError] tests/foo.py:42 :: AssertionError". The judge then
    # hits the "insufficient data" escape hatch, and the repair loop chases
    # ghosts for the rest of max_iterations (session 116667f5). Passing the
    # tail unconditionally would double the judge's token cost; gating on a
    # bare top message keeps the extra cost to sessions that need it.
    def _top_message_is_bare(diags: list[dict[str, Any]]) -> bool:
        if not diags:
            return False
        top = diags[0]
        code = str(top.get("error_code", "") or "").strip()
        msg = str(top.get("message", "") or "").strip()
        if not msg:
            return True
        if code and msg.lower() == code.lower():
            return True
        return len(msg) < 40
    tail_block = ""
    if build_output_tail and _top_message_is_bare(top_persisted_diagnostics):
        _tail_snippet = build_output_tail[-2500:]
        tail_block = (
            "\nBUILD OUTPUT TAIL (last ~2.5KB — the top persistent error's "
            "message is a bare exception type, so the full traceback below is "
            "your primary source of grounding for ``real_blocker``):\n"
            "---\n"
            f"{_tail_snippet}\n"
            "---\n\n"
        )

    # Whether the LLM has ANY concrete file:line anchor to substitute into
    # the "insufficient data — investigate <file>" escape hatch. When this
    # is False (all prior errors were resolved but new ones appeared with
    # no locations — e.g. pytest AssertionError from the collector), the
    # placeholder text ``<file>`` has nothing to bind to; the prompt below
    # switches to a distinct fallback string so the LLM never ships a
    # literal ``<file>`` downstream. Observed in session 116667f5.
    _has_anchor_location = any(
        str(d.get("file", "") or "").strip()
        and not str(d.get("file", "")).strip().startswith("<")
        for d in top_persisted_diagnostics
    )
    if _has_anchor_location:
        _generic_fallback_rule = (
            "  - If the top-error messages are too generic to localize "
            "(e.g. just a bare exception type), set ``real_blocker`` to "
            "the literal string \"insufficient data — investigate "
            "<file>'s data flow into the assertion\" with <file> "
            "REPLACED by an actual path from the Top persistent errors "
            "above. Do NOT leave the literal string ``<file>`` in your "
            "response — the harness treats an unsubstituted placeholder "
            "as a broken verdict and drops it. The harness will "
            "recognise the substituted form and skip the misleading "
            "injection instead of treating a guess as authoritative.\n\n"
        )
    else:
        # No persisted diagnostics carry a file:line, so ``<file>`` has
        # nothing to substitute. Tell the LLM to emit a distinct plain
        # sentence — the parser rejects any leftover ``<file>`` anyway.
        _generic_fallback_rule = (
            "  - No diagnostic in the failing set carries a file:line "
            "anchor (all prior errors resolved and the new ones lack "
            "locations, or all messages are too generic to localize). "
            "Set ``real_blocker`` to the literal string \"insufficient "
            "data — no diagnostic locations available\". Do NOT invent "
            "a file path and do NOT emit the placeholder ``<file>``; "
            "the harness will recognise this fallback and skip the "
            "misleading injection instead of treating a guess as "
            "authoritative.\n\n"
        )

    # The path-wiring hint overrides the install-failure hint: when the
    # missing module exists as a workspace source directory, the failure
    # is upstream of both the source and the manifest — it's in the test
    # invocation (wrong CWD, missing PYTHONPATH, ``cd <subdir> && pytest``
    # putting source root out of sys.path). Suppressing the install hint
    # in this case stops the judge from telling the repair LLM to add a
    # source directory to requirements.txt (a nonsensical fix that
    # burned three rounds in session 3193a24f).
    if path_wiring_module:
        path_wiring_hint_block = (
            "\nPATH/WIRING FAILURE HINT — read before answering:\n"
            f"  The missing module ``{path_wiring_module}`` EXISTS as a "
            "top-level workspace source directory (the harness checked). "
            "The manifest is fine — the failure is in the test-runner "
            "invocation:\n"
            "    - pytest may be running from a subdir "
            f"(e.g. ``cd {path_wiring_module} && pytest``) so tests "
            f"importing ``from {path_wiring_module}.X`` cannot resolve "
            f"the top-level ``{path_wiring_module}`` package on "
            "sys.path;\n"
            "    - PYTHONPATH may be unset so the workspace root isn't "
            "on the import path;\n"
            "    - a build-command wiring bug may be running the tests "
            "in the wrong directory.\n"
            "  Set ``real_blocker`` to a one-sentence description of "
            "the WIRING bug (which command, wrong CWD, or missing env "
            "var) and DO NOT recommend editing requirements.txt / "
            "pyproject.toml / package.json — those already contain "
            f"what's needed, and ``{path_wiring_module}`` is a source "
            "directory, not a pip package. Recommend fixing the build "
            "command / test-runner invocation instead.\n\n"
        )
        install_hint_block = ""
    else:
        path_wiring_hint_block = ""
        # Optional install-failure hint. Inserted only when the harness's
        # heuristic flags the persistent set as dominated by missing-module /
        # unresolved-import errors. Acts as a second-opinion nudge — the
        # judge can still rule it a code issue if the data warrants, but the
        # hint stops the LLM from chasing source-level fixes when the real
        # blocker is upstream of the compile.
        install_hint_block = (
            "\nENVIRONMENT-FAILURE HINT — read before answering:\n"
            "  The harness's pre-check on this round's failing diagnostics "
            "found that the majority look like missing-dependency / "
            "unresolved-import errors (TS2307 / MISSING_DEP / "
            "ModuleNotFoundError / ImportError / \"Cannot find module\"). "
            "When the imported names are declared in the workspace's "
            "package.json / requirements.txt / pyproject.toml, the right "
            "fix is the BUILD or INSTALL step, not the source code:\n"
            "    - the install step may not be running in the subdir where "
            "the deps actually live (root-delegating-to-subdir layouts);\n"
            "    - the build command may invoke the compiler before deps "
            "are installed;\n"
            "    - a package may be missing from the manifest entirely.\n"
            "  If your inspection confirms this pattern, set "
            "``real_blocker`` to a one-sentence description that points "
            "the repair LLM at the BUILD configuration (package.json "
            "scripts, Makefile, requirements.txt, install command in CI) "
            "rather than the source file. Patching source code will NOT "
            "make these errors go away.\n\n"
        ) if install_failure_likely else ""

    return (
        "You are auditing the previous repair iteration's outcome to "
        "tell the harness whether to keep going on the same track or "
        "redirect. The repair LLM applied patches to fix compile/test "
        "errors; the build was re-run; here is the delta in failing "
        "diagnostics.\n\n"
        f"Diagnostics before this round: {prior_diagnostics_count}\n"
        f"Diagnostics after this round:  {current_diagnostics_count}\n\n"
        f"Resolved (previous round → gone):\n{res_block}\n\n"
        f"Persisted (still failing):\n{pers_block}\n\n"
        f"New (introduced by this round's patches):\n{new_block}\n\n"
        f"Top persistent errors (with file:line):\n{top_block}\n\n"
        f"{tail_block}"
        f"{path_wiring_hint_block}"
        f"{install_hint_block}"
        "Answer ONE structured question: did the previous round make "
        "PROGRESS on the highest-priority error, or did it spend the "
        "iteration on a distraction? Use these definitions:\n"
        "  PROGRESS — the most-critical persistent error from the prior "
        "round was either resolved OR meaningfully changed (e.g. its "
        "message moved closer to a fix). Even partial progress counts.\n"
        "  DISTRACTION — the most-critical error is unchanged AND the "
        "round's patches addressed lower-priority items (test mocks, "
        "lint fixes, cosmetics) instead of the blocker.\n"
        "  REGRESSION — the round introduced new failures that didn't "
        "exist before, and the original blocker is also unchanged.\n\n"
        "GROUNDING RULES — read carefully, this is where prior versions "
        "of this prompt failed:\n"
        "  - Your answer must be grounded ONLY in the diagnostic data "
        "above. The file:line locations shown are the ONLY files you "
        "may name. Do NOT invent file paths, symbol names, or call-site "
        "guesses that are not present above.\n"
        "  - EXCEPTION for install / unresolved-import / test-plugin "
        "errors (TS2307, MISSING_DEP, MODULENOTFOUND, IMPORTERROR, "
        "UV_VERSION_CONSTRAINT and any diagnostic whose message includes "
        "\"cannot find module\", \"no module named\", or \"async def "
        "functions are not natively supported\" — the last of which "
        "means pytest-asyncio is missing from the manifest or "
        "`asyncio_mode` is not configured): the file:line shown is the "
        "IMPORT / test-collection site, NOT where the fix lives. For "
        "these codes the fix belongs in the manifest — `package.json`, "
        "`requirements.txt`, `pyproject.toml`, `Cargo.toml`, `pytest.ini` "
        "— or in the build/install command. You MAY name that manifest "
        "even when it does not appear in the diagnostics above; the "
        "grounding rule above does not apply to install-class fixes. "
        "Do NOT recommend editing the import site.\n"
        f"{_generic_fallback_rule}"
        "Respond with STRICT JSON ONLY — no prose, no markdown, no code "
        "fences. Shape:\n"
        '{"verdict": "PROGRESS" | "DISTRACTION" | "REGRESSION", '
        '"real_blocker": "<one-sentence description grounded in the '
        'file:line locations above, OR the literal insufficient-data '
        'sentence shown in the grounding rules>", '
        '"recommendation": "<one short imperative sentence for the next '
        'repair LLM, e.g. \\"Edit src/services/AuthService.ts lines '
        '20/25/73 to call jwt.sign(payload, secret, {expiresIn: ...}) '
        'instead of passing options as the second arg.\\""}\n'
    )


def _reflection_verdict_is_low_signal(verdict: dict[str, str]) -> bool:
    """Return True when the reflection verdict's ``real_blocker`` is the
    prompt's ``insufficient data`` escape-hatch sentinel — i.e. the judge
    could not localize the failure and emitted a placeholder rather than a
    grounded diagnosis.

    Callers use this to (a) skip promoting the verdict into the "JUDGE'S
    VERDICT" banner (there is nothing actionable to inject), and (b) skip
    ticking ``consecutive_distraction_rounds`` so a stream of low-signal
    verdicts does not race the circuit-breaker to HITL when the underlying
    problem may be fixable if given more context (see Fix A which enriches
    the judge's view of bare exception-type errors). Observed in session
    116667f5: six rounds of oscillating DISTRACTION/REGRESSION verdicts,
    each promoted verbatim as the banner directive despite the sentinel
    containing zero actionable information.
    """
    blocker = (verdict.get("real_blocker") or "").strip().lower()
    return blocker.startswith("insufficient data")


def _reflection_grounds_in_diagnostics(
    verdict: dict[str, str],
    compiler_errors: list[dict[str, Any]],
) -> bool:
    """Fix C — return True iff the reflection's real_blocker or
    recommendation references a file that is actually present in the
    current ``compiler_errors``. Used to gate injection of the
    reflection verdict into the repair prompt: when the LLM names a
    location that doesn't intersect the real failing set, the
    diagnosis is almost certainly fabricated and forwarding it to
    repair just anchors the next round on a phantom.

    The check is deliberately lenient — it matches on either the full
    relative path or the basename, and on either side of the verdict
    payload. The intent is "did the reflection model engage with the
    actual diagnostic locations at all?", not "did it pinpoint the
    exact line." False positives (text accidentally contains a file
    name) are acceptable; false negatives are not, because they'd
    mute a real injection.

    Special case: the explicit "insufficient data" escape-hatch
    sentence introduced by fix A is treated as deliberately ungrounded
    — we return False so it does NOT get injected. The grounding-rule
    text in the prompt tells the LLM the harness will skip it.
    """
    blocker = (verdict.get("real_blocker") or "").strip().lower()
    recommendation = (verdict.get("recommendation") or "").strip().lower()
    if not blocker:
        return False
    # Honour the fix-A escape hatch — but ONLY when the recommendation is
    # also vague. Observed in session cf3fcd27: the judge wrote
    # "insufficient data — investigate <file>" in real_blocker AND
    # "Mock the service function that triggers the RuntimeError at
    # services/edgar.py:48" in recommendation — i.e. it hedged on the
    # blocker line but knew the fix. Treating that as ungrounded buries
    # the actionable half of the verdict. We now ignore the blocker text
    # in that case and ground purely against the recommendation.
    if "insufficient data" in blocker:
        haystack = recommendation
        if not recommendation:
            return False
    else:
        haystack = blocker + " " + recommendation
    files_seen: set[str] = set()
    for err in compiler_errors:
        f = str(err.get("file", "") or "").strip()
        if not f or f.startswith("<"):
            continue  # skip synthetic markers like "<harness:...>"
        files_seen.add(f.lower())
        # Also accept basename — LLMs often shorten paths.
        basename = os.path.basename(f).lower()
        if basename:
            files_seen.add(basename)
    if not files_seen:
        # No file info to ground against — fall open (don't penalise
        # diagnostics that lack file info; injection MAY still help).
        return True
    if any(name in haystack for name in files_seen):
        return True
    # Install-class escape hatch — mirror of the reflection prompt's
    # grounding-rules exception. For TS2307 / MISSING_DEP / MODULE_NOT_
    # FOUND / IMPORTERROR / UV_VERSION_CONSTRAINT, the diagnostic's
    # file:line is the IMPORT site (App.tsx:2 for "cannot find module
    # 'react-router-dom'"), NOT where the fix goes. The reflection is
    # told to name package.json / requirements.txt / pyproject.toml /
    # Cargo.toml instead — files that legitimately won't appear in
    # ``compiler_errors``. Without this branch the strict grounding
    # check mutes those verdicts as "ungrounded" even though they are
    # the correct answer. See sweep item #3.
    codes = {str(e.get("error_code", "") or "").upper() for e in compiler_errors}
    is_install_class = (
        any(c.startswith(pfx) for c in codes
            for pfx in _INSTALL_ERROR_CODE_PREFIXES)
        or any(frag in str(e.get("message", "") or "").lower()
               for e in compiler_errors
               for frag in _INSTALL_ERROR_MSG_FRAGMENTS)
    )
    if is_install_class:
        for manifest in (
            "package.json", "requirements.txt", "pyproject.toml",
            "cargo.toml", "package-lock.json",
            # pytest-asyncio "not natively supported" is manifest-class
            # too — the fix is adding the dep or setting asyncio_mode in
            # pytest.ini / pyproject.toml. Accept the config file as a
            # grounded target so the verdict isn't muted.
            "pytest.ini", "setup.cfg", "tox.ini",
        ):
            if manifest in haystack:
                return True
    return False


def _verdict_named_files(
    verdict: dict[str, str],
    compiler_errors: list[dict[str, Any]],
) -> list[str]:
    """Return the relative paths of files that BOTH appear in the judge's
    real_blocker/recommendation text AND are present in the current
    ``compiler_errors`` failing set.

    Used by the judge-ignored gate (Fix #3): after the repair LLM applies
    patches, we check whether any of these files were modified. If the judge
    named ``services/edgar.py`` and the LLM patched ``test_api.py`` instead,
    the round is a structural distraction regardless of what the next
    reflection verdict ends up saying.

    Returns full relative paths (not basenames) so the touched-files check
    can match on path-suffix. Empty list when nothing matches — caller must
    treat that as "no enforcement this round."
    """
    blocker = (verdict.get("real_blocker") or "").lower()
    recommendation = (verdict.get("recommendation") or "").lower()
    if not blocker:
        return []
    # Same recommendation-fallback as _reflection_grounds_in_diagnostics:
    # an "insufficient data" blocker with a concrete recommendation is
    # still actionable — pull file names from the recommendation alone.
    if "insufficient data" in blocker:
        if not recommendation:
            return []
        haystack = recommendation
    else:
        haystack = blocker + " " + recommendation
    matched: list[str] = []
    seen: set[str] = set()
    for err in compiler_errors:
        f = str(err.get("file", "") or "").strip()
        if not f or f.startswith("<"):
            continue
        if f in seen:
            continue
        basename = os.path.basename(f).lower()
        if f.lower() in haystack or (basename and basename in haystack):
            matched.append(f)
            seen.add(f)
    return matched


def _verdict_named_file_lines(
    verdict: dict[str, str],
    compiler_errors: list[dict[str, Any]],
) -> list[tuple[str, int]]:
    """Extract ``(file, line)`` tuples from the judge's ``real_blocker`` /
    ``recommendation`` text that also appear in ``compiler_errors``.

    Used by the persistent-blocker directive (Fixes #2/#3 from the audit).
    When the judge names the SAME ``file:line`` two rounds running, the
    repair LLM's last patch either missed the target or landed a cosmetic
    change nearby. Round N+1's banner promotes that fact to a hard
    directive ("your patch MUST alter line 126 of test_edgar.py"), so
    the LLM stops nibbling around the target.

    Grounding rule: the tuple must be present in the CURRENT failing set.
    A stale ``real_blocker`` string that mentions a file:line pair the
    compiler is no longer complaining about is dropped — we never
    hard-target a file the build has already moved past.

    Returns ``[]`` when nothing matches; caller treats that as "no
    persistence directive this round."
    """
    blocker = (verdict.get("real_blocker") or "")
    recommendation = (verdict.get("recommendation") or "")
    if not blocker and not recommendation:
        return []
    haystack = blocker + " " + recommendation
    # Match "<any path>.<py|json|ts|tsx|js|jsx|md>:<line>" and
    # "<path> line <line>" / "line <line> of <path>" / "line <line> in <path>".
    # File chars stay conservative — no whitespace, no punctuation that would
    # cross word boundaries. Line 0 is filtered out (summary-row artefact from
    # ``_parse_pytest_summary``, not a real location).
    _colon = re.compile(
        r"(?P<file>[\w./\-]+\.(?:py|json|ts|tsx|js|jsx|md))"
        r":(?P<line>\d+)\b"
    )
    _line_of = re.compile(
        r"line\s+(?P<line>\d+)\s+(?:of|in)\s+"
        r"(?P<file>[\w./\-]+\.(?:py|json|ts|tsx|js|jsx|md))",
        re.IGNORECASE,
    )
    _file_line = re.compile(
        r"(?P<file>[\w./\-]+\.(?:py|json|ts|tsx|js|jsx|md))\s+"
        r"(?:on\s+|at\s+)?line\s+(?P<line>\d+)",
        re.IGNORECASE,
    )
    candidates: set[tuple[str, int]] = set()
    for pat in (_colon, _line_of, _file_line):
        for m in pat.finditer(haystack):
            try:
                lineno = int(m.group("line"))
            except (TypeError, ValueError):
                continue
            if lineno <= 0:
                continue
            candidates.add((m.group("file"), lineno))
    if not candidates:
        return []
    # Ground the tuples against compiler_errors — file suffix match on
    # either full path or basename, line number must equal. Preserves the
    # form the compiler reported (full relative path), so subsequent
    # comparisons across rounds use a stable key.
    matched: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for err in compiler_errors:
        err_file = str(err.get("file", "") or "").strip()
        try:
            err_line = int(err.get("line", 0) or 0)
        except (TypeError, ValueError):
            err_line = 0
        if not err_file or err_line <= 0 or err_file.startswith("<"):
            continue
        err_basename = os.path.basename(err_file).lower()
        for cand_file, cand_line in candidates:
            if cand_line != err_line:
                continue
            cand_low = cand_file.lower()
            if (
                cand_low == err_file.lower()
                or err_file.lower().endswith(cand_low)
                or cand_low.endswith(err_file.lower())
                or (err_basename and cand_low.endswith(err_basename))
            ):
                key = (err_file, err_line)
                if key not in seen:
                    matched.append(key)
                    seen.add(key)
                break
    return matched


def _related_test_files(
    source_file: str,
    workspace_path: str,
    *,
    max_matches: int = 5,
) -> list[str]:
    """Return relative paths of test files whose basename matches
    ``test_<stem>.py`` for the given ``source_file`` and live under a
    ``tests/`` / ``test/`` directory in the workspace.

    Used by the "fix may belong in the test file" hint (session
    b92043caq): when the reflection judge persistently names a source
    file (``backend/services/edgar.py`` → 4 rounds of "coroutine not
    awaited"), the LLM tends to keep patching the source even when the
    real fix is in the test's mock setup. Surfacing the related test
    path gives the LLM a concrete second target to consider.

    Path resolution is basename-only — we don't try to match full
    module paths because tests are often placed in a mirror layout
    (``tests/backend/services/test_edgar.py``) OR flat
    (``backend/tests/test_edgar.py``) OR top-level (``tests/test_edgar.py``).
    Basename covers all three without hard-coding a project layout.

    Returns ``[]`` when no candidate matches or ``workspace_path``
    can't be walked. Never raises.
    """
    if not source_file or not workspace_path:
        return []
    stem = os.path.splitext(os.path.basename(source_file))[0]
    if not stem or stem.startswith("__"):
        return []
    target = f"test_{stem}.py"
    matches: list[str] = []
    try:
        for root, dirs, files in os.walk(workspace_path):
            # Skip obviously irrelevant dirs. VCS + package caches +
            # deps trees dominate os.walk runtime without ever
            # containing a project test file. Keep the exclusion set
            # narrow so we don't accidentally skip a legitimate
            # ``tests/`` layout the operator happened to place under
            # e.g. ``vendor/``.
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d not in {
                    "node_modules", "__pycache__", ".pytest_cache",
                    ".mypy_cache", ".venv", "venv", "dist", "build",
                }
            ]
            rel_root = os.path.relpath(root, workspace_path)
            # Only accept matches under a directory named ``tests`` or
            # ``test`` somewhere in the relative path — a bare
            # ``test_foo.py`` sitting at workspace root next to source
            # is much more likely to be an accidental collision than
            # a real test target.
            path_parts = {p.lower() for p in rel_root.split(os.sep)}
            if not ({"tests", "test"} & path_parts):
                continue
            if target in files:
                rel_path = os.path.normpath(os.path.join(rel_root, target))
                # Never suggest editing the file that IS the persistent
                # blocker source — happens when someone named their
                # source file test_x.py by mistake.
                if os.path.normpath(rel_path) != os.path.normpath(source_file):
                    matches.append(rel_path)
                    if len(matches) >= max_matches:
                        break
    except OSError:
        return []
    return matches


def _shared_root_cause_fanout(
    compiler_errors: list[dict[str, Any]],
    *,
    threshold: int = 3,
) -> list[tuple[str, list[str]]]:
    """Group current failing diagnostics by ``error_code`` and return any
    (code, files) pairs where the same code appears across ≥ ``threshold``
    distinct files.

    Used by the JUDGE'S VERDICT banner (Fix #3): when one root cause spans
    many test files (e.g. 10 tests all hitting an EDGAR mock guard with
    the same RuntimeError), the LLM otherwise patches one per round and
    the no_progress gate fires on partial wins. The banner adds a
    "patch ALL of them in one response" directive sourced from this list.

    Returns ``[]`` when no code clears the threshold — the directive is
    silent and the banner reads identically to the single-site case.
    """
    files_by_code: dict[str, set[str]] = {}
    for err in compiler_errors:
        code = str(err.get("error_code", "") or "").strip()
        f = str(err.get("file", "") or "").strip()
        if not code or not f or f.startswith("<"):
            continue
        files_by_code.setdefault(code, set()).add(f)
    return [
        (code, sorted(files))
        for code, files in files_by_code.items()
        if len(files) >= threshold
    ]


def _patches_touched_judge_files(
    patch_results: list[Any], judge_named: list[str],
    *,
    include_attempts: bool = False,
) -> bool:
    """True iff at least one patch touched a file the judge named (suffix /
    basename match — LLMs and the harness use slightly different path
    roots, e.g. ``services/edgar.py`` vs ``server/services/edgar.py``).

    By default only *successful*, non-no-op patches count. When
    ``include_attempts=True``, failed patches with a non-empty file also
    count — for the judge-ignored gate we want to distinguish "the LLM
    chose the wrong file" (real distraction) from "the LLM chose the right
    file but the patch's REPLACE_BLOCK search missed" (mechanical failure,
    not distraction). The patcher's per-file rejection diagnosis already
    surfaces the search-miss reason to the next round, so escalating with
    a judge-ignored banner on top causes the LLM to thrash between
    "address the rejection" and "stop ignoring the judge".
    """
    if not judge_named:
        return True  # nothing to enforce
    judge_norm = [(p, os.path.basename(p)) for p in judge_named]
    for r in patch_results:
        success = getattr(r, "success", False)
        no_op = getattr(r, "no_op", False)
        if not include_attempts:
            if not success or no_op:
                continue
        else:
            # Attempt-mode: count both successes (non-no-op) and failures.
            # Skip only no-ops since those neither succeeded nor were tried.
            if no_op:
                continue
        f = str(getattr(r, "file", "") or "")
        if not f:
            continue
        f_base = os.path.basename(f)
        for j, j_base in judge_norm:
            if f == j or f.endswith("/" + j) or j.endswith("/" + f):
                return True
            if f_base and j_base and f_base == j_base:
                return True
    return False


def _parse_repair_reflection_verdict(
    raw: str,
) -> Optional[dict[str, str]]:
    """Parse the strict-JSON response from the repair-reflection judgment.

    Returns a dict with keys ``verdict``, ``real_blocker``,
    ``recommendation`` (all strings) when the response is well-formed,
    or ``None`` on any parse failure / missing keys. The caller treats
    None as "skip reflection injection, proceed as normal."
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        logger.debug(
            "[judgment:repair_reflection] Non-JSON verdict; skipping (raw=%r).",
            text[:200],
        )
        return None
    if not isinstance(parsed, dict):
        return None
    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in {"PROGRESS", "DISTRACTION", "REGRESSION"}:
        return None
    real_blocker = str(parsed.get("real_blocker", "")).strip()
    recommendation = str(parsed.get("recommendation", "")).strip()
    if not real_blocker and verdict != "PROGRESS":
        # PROGRESS verdicts can omit the blocker; the other two need it.
        return None
    # Escape-hatch placeholder guard — mirror of the prompt-side change.
    # If the LLM shipped the literal ``<file>`` (or the ``<no location>``
    # marker the harness uses when a diagnostic has no file field) still
    # unsubstituted in ``real_blocker``, treat the sentence as broken and
    # rewrite it to the plain fallback string. Downstream grounding logic
    # already special-cases "insufficient data" text, so the rewritten
    # form still routes correctly — it just no longer carries a template
    # placeholder into logs, events, or system-message injections.
    # See session 116667f5 where deepseek-v4-flash returned the literal
    # ``<file>`` template because ``top_persisted_diagnostics`` was empty.
    _placeholder_markers = ("<file>", "<no location>", "<file:line>")
    if any(marker in real_blocker for marker in _placeholder_markers):
        logger.warning(
            "[judgment:repair_reflection] Escape-hatch template returned "
            "with unsubstituted placeholder (%r); rewriting to plain "
            "'insufficient data — no diagnostic locations available'.",
            real_blocker[:200],
        )
        real_blocker = "insufficient data — no diagnostic locations available"
    return {
        "verdict": verdict,
        "real_blocker": real_blocker,
        "recommendation": recommendation,
    }


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
# Contrast with non-installable symbols (npm, node, docker) where the
# base image itself is wrong and only a config change can unblock the
# run; those still short-circuit to HITL.
_PIP_INSTALLABLE_SYMBOLS: frozenset[str] = frozenset({
    "pytest", "pytest-asyncio", "pytest-cov", "pytest-mock", "pytest-xdist",
    "ruff", "mypy", "black", "isort", "flake8", "pylint",
    "coverage", "tox", "nox", "poetry",
})


def _npm_root_package(symbol: str) -> str:
    """Return the installable npm package name for ``symbol``, stripping
    any sub-path. Mirrors the strip logic in
    ``autofix._try_missing_npm_dep`` so the LLM's diagnostic points at
    the same string the autofix would write into ``package.json``.

    Examples:
        ``react-router-dom/dist/x`` → ``react-router-dom``
        ``@scope/pkg/sub``         → ``@scope/pkg``
        ``next``                    → ``next``
    """
    if not symbol:
        return symbol
    if symbol.startswith("@"):
        parts = symbol.split("/", 2)
        return "/".join(parts[:2]) if len(parts) >= 2 else symbol
    return symbol.split("/", 1)[0]


def _repairable_dep_hint(
    symbol: str, build_command: str, miss_kind: str = "python",
) -> str:
    """Repair-friendly diagnostic for a missing test / lint tool. Reaches
    the repair LLM via compiler_errors and points it at the smallest
    possible patch. Distinct from :func:`_env_misconfig_hint`, which is
    sent to HITL because no in-container patch can fix it.

    ``miss_kind`` toggles Python vs Node phrasing: for ``node``, the fix
    target is ``package.json`` (not ``requirements.txt``), and we strip
    any sub-path so the LLM writes the installable root name instead of
    ``pkg/dist/x`` (session-4d1f9e1c-adjacent misread).
    """
    if miss_kind == "node":
        root = _npm_root_package(symbol)
        sub_note = ""
        if root != symbol:
            sub_note = (
                f" The reported miss was `{symbol}` (a sub-path); the "
                f"npm-installable package is its root `{root}` — do "
                f"NOT put `{symbol}` in `package.json`.\n\n"
            )
        return (
            f"Build failed: node cannot find module `{symbol}`. This "
            f"means the npm package `{root}` is not installed in the "
            f"workspace's `node_modules`. The sandbox runs "
            f"`{build_command.strip()}`, which invokes node after the "
            f"install step — but `{root}` isn't in `package.json` (or "
            f"the install step didn't run in the subdir where "
            f"`package.json` lives).\n\n"
            f"{sub_note}"
            f"Fix in ONE place: add `{root}` to `dependencies` (or "
            f"`devDependencies` if it's a build-time-only tool) in "
            f"`package.json`. If `package.json` does not exist, CREATE "
            f"it with a minimal shape that includes `{root}`.\n"
            f"Do not edit the import site (`import '{symbol}'`) — the "
            f"import statement is correct; the missing piece is the "
            f"dependency declaration."
        )
    return (
        f"Build failed: '{symbol}' is required by the build command but is "
        f"not declared as a dependency. The sandbox runs "
        f"`{build_command.strip()}`, which invokes `{symbol}` after the "
        f"install step — but '{symbol}' isn't in the workspace's "
        f"dependency manifest, so pip never installs it.\n\n"
        f"Fix in ONE place:\n"
        f"  - If the install step is `(uv )?pip install -r requirements.txt`: "
        f"add `{symbol}` to `requirements.txt`. If the file does not "
        f"exist yet, CREATE it with one dependency per line "
        f"(including `{symbol}`).\n"
        f"  - If the install step is `(uv )?pip install -e '.[dev]'`: add "
        f"`{symbol}` to `[project.optional-dependencies].dev` in "
        f"`pyproject.toml`. If the section does not exist, CREATE it.\n"
        f"Do not change the build_command or docker_image — the package "
        f"is pip-installable and the current image is correct."
    )


def _env_misconfig_hint(symbol: str, build_command: str) -> str:
    """Build the actionable HITL message for an env-misconfig hit.

    Distinct from :func:`_repairable_dep_hint` — this message flows to
    HITL because no in-container edit will unblock the run (the missing
    runtime is baked into the docker image, not installable from
    inside). Name the exact config file the operator must touch so the
    operator (and any LLM reading the log) knows where to look and, more
    importantly, does NOT try to patch it from inside the repair loop —
    the patcher allowlist blocks harness-internal configs regardless.
    """
    # Pick the most likely installer for the missing symbol.
    py_symbols = {"pytest", "ruff", "mypy", "black", "poetry", "tox", "nox"}
    if symbol.lower() in py_symbols or "python" in symbol.lower():
        installer = f"pip install {symbol}"
    elif symbol in {"npm", "node"}:
        installer = (
            f"use a node-bearing docker_image (e.g. node:20-slim) — "
            f"'{symbol}' is not installable from inside the container"
        )
    else:
        installer = f"install {symbol} before running the build"
    return (
        f"Build container is missing '{symbol}'. The repair LLM cannot "
        f"fix this — it's a sandbox/dependency setup issue that lives "
        f"OUTSIDE the workspace. Do NOT emit patches against "
        f".harness_config.json or any harness-internal file; the "
        f"patcher allowlist will reject them and the docker_image "
        f"setting is stored in the operator's global config anyway.\n\n"
        f"Operator fix (one of):\n"
        f"  - Update the workspace's `build_command` (typically in "
        f"`Makefile`, `package.json` scripts, or `pyproject.toml` "
        f"`[tool.harness]` if present) to prepend the install step: "
        f"`{installer} && {build_command.strip()}`.\n"
        f"  - Change the harness's `sandbox.docker_image` in the "
        f"operator config (`~/.harness/config.json` or `~/.harness/"
        f"config.yaml`, NOT any file in the workspace) to an image "
        f"that ships `{symbol}` pre-installed (e.g. `node:20-slim` for "
        f"node/npm, `python:3.11-slim` for python)."
    )


# Matches ` && cd <token> && ...` inside a shell command. The token can
# be a bare dir name or a simple relative path; we intentionally ignore
# quoted / substituted / absolute forms so an operator-customised
# `cd "/opt/build" && ...` never trips the stale-cd guard. The leading
# `&&` requirement skips `cd &&` typos that would already fail earlier.
_CD_TARGET_RE = re.compile(r"&&\s*cd\s+([A-Za-z0-9_./-]+)\s*&&")


def _first_missing_cd_target(
    build_command: str, workspace_path: str,
) -> Optional[str]:
    """Return the first `cd <dir>` target in build_command that does
    not exist under workspace_path, or None if every target resolves.

    Used by compiler_node to detect a stale build_command (typically a
    `cd backend && ...` frozen from a pre-reset workspace snapshot) and
    trigger a re-resolve instead of running a build that will exit-2
    with `sh: cd: can't cd to <dir>` and no structured diagnostics.

    Absolute paths and anything containing shell substitutions are
    ignored — those are operator-authored and the harness should not
    second-guess them.
    """
    if not build_command or "cd " not in build_command:
        return None
    for match in _CD_TARGET_RE.finditer(build_command):
        target = match.group(1).strip()
        if not target or target.startswith("/") or "$" in target:
            continue
        full = os.path.join(workspace_path, target)
        if not os.path.isdir(full):
            return target
    return None


def _first_out_of_spec_cd_target(
    build_command: str, workspace_path: str,
) -> Optional[tuple[str, list[str]]]:
    """Return (offending_target, spec_roots) when build_command's `cd X`
    references a subdir that is NOT in the SPEC_ARCHITECTURE.md-declared
    roots. Returns None when there is no spec, no `cd` target, or all
    targets are spec-approved.

    Complements :func:`_first_missing_cd_target`. That one catches the
    case where the cd target simply doesn't exist on disk. This one
    catches the harder case where the LLM already created a rogue
    directory (e.g. `backend/` because the stale build_command told it
    to) but the spec allowlist forbids writing there — the LLM sees a
    passing `cd` but every patch it emits under that dir gets 100%
    rejected, and the repair loop burns budget.

    Returned spec_roots are the raw root paths from the spec so the
    caller can rewrite the build_command against them (e.g. swap
    `cd backend` for `cd server` when Python backend lives under
    `server/`).
    """
    if not build_command or "cd " not in build_command:
        return None
    layout = _read_spec_layout(workspace_path)
    if layout is None or not getattr(layout, "has_layout", False):
        return None
    spec_roots = [
        r.path.strip("/").split("/", 1)[0]
        for r in getattr(layout, "roots", []) or []
        if getattr(r, "path", None)
    ]
    spec_roots = [r for r in spec_roots if r]
    if not spec_roots:
        return None
    spec_root_set = set(spec_roots)
    for match in _CD_TARGET_RE.finditer(build_command):
        target = match.group(1).strip()
        if not target or target.startswith("/") or "$" in target:
            continue
        # Top-level dir of the cd target — the spec roots are top-level
        # by contract.
        top = target.split("/", 1)[0]
        if top not in spec_root_set:
            return target, spec_roots
    return None


def _rewrite_cd_target(
    build_command: str, old_target: str, new_target: str,
) -> str:
    """Rewrite the first `&& cd <old_target> &&` occurrence in
    build_command to point at ``new_target``. Preserves surrounding
    tokens exactly. Returns build_command unchanged when no match.
    """
    pattern = re.compile(
        r"(&&\s*cd\s+)" + re.escape(old_target) + r"(\s*&&)"
    )
    return pattern.sub(rf"\g<1>{new_target}\g<2>", build_command, count=1)


# Matches `sh: N: cd: can't cd to <dir>` and `bash: cd: <dir>: No such
# file or directory` — the two shells that show up as the shell in
# harness sandbox images (dash for the default docker image, bash for
# alpine variants). Both forms carry the failing dir in group(1).
_CD_FAILURE_PATTERNS = (
    re.compile(r"sh:\s*\d+:\s*cd:\s*can'?t\s+cd\s+to\s+(\S+)"),
    re.compile(r"bash:\s*cd:\s*(\S+):\s*No such file or directory"),
    re.compile(r"cd:\s*(\S+):\s*No such file or directory"),
)


def _detect_cd_failure_from_output(raw_output: str) -> Optional[str]:
    """Return the failing directory name when raw_output shows the
    shell couldn't cd into it. Returns None when no cd failure is
    present.

    Compiler_node uses this to short-circuit to HITL: the LLM cannot
    fix a build_command wiring problem (the allowlist would reject any
    patch to the missing directory anyway).
    """
    if not raw_output or "cd" not in raw_output:
        return None
    for pattern in _CD_FAILURE_PATTERNS:
        m = pattern.search(raw_output)
        if m:
            token = m.group(1).strip().rstrip(":,")
            if token:
                return token
    return None


def _pick_spec_backend_root(workspace_path: str, spec_roots: list[str]) -> Optional[str]:
    """Return the first spec root that carries a Python manifest, or the
    first Node-manifest root as a fallback. Used to rewire a stale
    `cd backend` to the actual spec-approved backend root.
    """
    py_manifests = ("pyproject.toml", "requirements.txt")
    node_manifests = ("package.json",)
    node_fallback: Optional[str] = None
    for root in spec_roots:
        root_dir = os.path.join(workspace_path, root)
        if not os.path.isdir(root_dir):
            continue
        for m in py_manifests:
            if os.path.isfile(os.path.join(root_dir, m)):
                return root
        if node_fallback is None:
            for m in node_manifests:
                if os.path.isfile(os.path.join(root_dir, m)):
                    node_fallback = root
                    break
    return node_fallback


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

    # Install commands (pip install -e ., npm install -g)
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
    #
    # is_greenfield is recovered from the session flow. In greenfield
    # the detector ignores LLM-scaffolded Makefiles, so an LLM emitting
    # a partial Makefile mid-session can't hijack the build command
    # away from the per-stack baseline. See cli.py
    # _detect_default_build_command for the contract.
    is_greenfield_compile = bool(state.get("flow") == "build")
    adapted_build_cmd: Optional[str] = None
    if build_cmd.strip() == "make build" and not any(
        os.path.exists(os.path.join(workspace, name))
        for name in ("Makefile", "makefile", "GNUmakefile")
    ):
        from harness.cli import _detect_default_build_command
        late = _detect_default_build_command(
            workspace, is_greenfield=is_greenfield_compile,
        )
        if late and late != "make build":
            logger.info(
                "[compiler_node] Workspace has no Makefile; adapting build command "
                "from default 'make build' to detected: %s", late,
            )
            adapted_build_cmd = late
            build_cmd = late

    # Mid-session upgrade: when the build_cmd is the bare "pip install <tool>
    # && pytest -q" fallback (chosen earlier because the workspace had no
    # manifest yet) AND the detector now returns something more specific —
    # typically because the first patching pass wrote requirements.txt /
    # pyproject.toml (greenfield case) or because autofix R4 wrote it to
    # install a missing pip-installable dep — re-detect so the next
    # compile actually installs the project's deps. The path check is
    # deliberately delegated to ``_detect_default_build_command`` itself
    # so monorepo layouts (``server/requirements.txt`` + ``client/...``)
    # are caught too; an explicit ``os.path.isfile`` check at workspace
    # root misses them and the bare-pytest cmd loops forever on
    # ModuleNotFoundError. The `!= "make build"` guard remains as defence
    # in depth — in greenfield the detector won't return it anyway, in
    # brownfield it prevents replacing an explicit pip+pytest seed with
    # a freshly-emitted LLM Makefile (unlikely but possible during patch).
    if (
        adapted_build_cmd is None
        and "pip install" in build_cmd
        and "-r" not in build_cmd
    ):
        from harness.cli import _detect_default_build_command
        re_detected = _detect_default_build_command(
            workspace, is_greenfield=is_greenfield_compile,
        )
        if re_detected and re_detected != build_cmd and re_detected != "make build":
            logger.info(
                "[compiler_node] Workspace gained a dependency manifest mid-session; "
                "upgrading build command from %r to detected %r so installed deps "
                "are honored.",
                build_cmd, re_detected,
            )
            adapted_build_cmd = re_detected
            build_cmd = re_detected

    # Second mid-session upgrade: greenfield runs where the detector ALREADY
    # ran (cached cmd has `-r`) but the workspace gained an additional
    # manifest the previous detection didn't see — typically
    # `requirements-dev.txt` written by the test_generation pass after the
    # first compile cycle locked in the build command. Re-detect and adopt
    # only when the new command is a strict superset that adds an install
    # step (same prefix up to the trailing `&& python3 -m pytest -q` tail),
    # so an operator-customised brownfield command can never get replaced.
    if (
        adapted_build_cmd is None
        and is_greenfield_compile
        and "uv pip install -r" in build_cmd
    ):
        from harness.cli import _detect_default_build_command
        re_detected = _detect_default_build_command(
            workspace, is_greenfield=True,
        )
        # Source of truth lives in harness/cli.py — single import avoids
        # the two sides drifting if the canonical pytest invocation changes.
        from harness.cli import _PYTEST_RUN as _CLI_PYTEST_RUN
        tail = f" && {_CLI_PYTEST_RUN}"
        if (
            re_detected
            and re_detected != build_cmd
            and re_detected.endswith(tail)
            and build_cmd.endswith(tail)
            and re_detected[: -len(tail)].startswith(build_cmd[: -len(tail)])
        ):
            logger.info(
                "[compiler_node] Workspace gained an additional manifest mid-session; "
                "upgrading build command from %r to detected %r so the extra install "
                "step is honored.",
                build_cmd, re_detected,
            )
            adapted_build_cmd = re_detected
            build_cmd = re_detected

    # Fourth mid-session upgrade: build_cmd references `cd <subdir>` but
    # the subdir does not exist in the workspace. Without this, the build
    # exits with `sh: cd: can't cd to <subdir>` (exit 2, zero structured
    # diagnostics), the router sends the LLM back to repair, and the LLM
    # either creates files under a directory the allowlist forbids (100%
    # rejected → repair loop) or gives up. Root cause is usually a stale
    # build_command frozen from a pre-reset workspace snapshot; the fix
    # is to re-resolve now that the workspace has stabilised. See cli.py
    # `resolve_build_command`.
    if adapted_build_cmd is None:
        missing_cd = _first_missing_cd_target(build_cmd, workspace)
        if missing_cd is not None:
            try:
                from harness.cli import (
                    _strip_comments,
                    load_raw_config,
                    resolve_build_command as _resolve_bc,
                )
                cfg = _strip_comments(load_raw_config())
            except Exception:  # noqa: BLE001 — config read is advisory here
                cfg = {}
                _resolve_bc = None  # type: ignore[assignment]
            re_resolved = None
            if _resolve_bc is not None:
                try:
                    re_resolved = _resolve_bc(
                        cfg if isinstance(cfg, dict) else {},
                        workspace,
                        is_greenfield=is_greenfield_compile,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[compiler_node] Stale-cd re-resolve raised %s; "
                        "leaving build_cmd unchanged so the router can "
                        "surface the cd failure to HITL.", exc,
                    )
            if re_resolved and re_resolved != build_cmd:
                logger.warning(
                    "[compiler_node] Build command references `cd %s` but that "
                    "directory does not exist in the workspace; re-resolving. "
                    "Old: %r  New: %r",
                    missing_cd, build_cmd, re_resolved,
                )
                adapted_build_cmd = re_resolved
                build_cmd = re_resolved

    # Fifth mid-session upgrade: build_cmd's `cd X` target exists on disk
    # but is NOT one of the spec-approved roots. Root cause: the LLM
    # (mis)followed a stale `cd backend` in the system prompt and
    # created `backend/` even though the spec puts code under `server/`.
    # Every patch it now emits under backend/ gets rejected by the
    # spec-driven allowlist (roots=['client', 'server']). Rewriting the
    # build_command to point at a spec-approved root makes the two
    # signals in the LLM's system prompt agree and lets the patcher
    # accept the next batch. See _first_out_of_spec_cd_target for the
    # detection contract.
    if adapted_build_cmd is None:
        out_of_spec = _first_out_of_spec_cd_target(build_cmd, workspace)
        if out_of_spec is not None:
            bad_target, spec_roots = out_of_spec
            new_target = _pick_spec_backend_root(workspace, spec_roots)
            if new_target and new_target != bad_target:
                rewritten = _rewrite_cd_target(build_cmd, bad_target, new_target)
                if rewritten != build_cmd:
                    logger.warning(
                        "[compiler_node] Build command's `cd %s` conflicts with "
                        "spec-driven allowlist roots %s; rewriting to `cd %s` so "
                        "the patcher and the build agree. Old: %r  New: %r",
                        bad_target, spec_roots, new_target, build_cmd, rewritten,
                    )
                    adapted_build_cmd = rewritten
                    build_cmd = rewritten
            else:
                logger.warning(
                    "[compiler_node] Build command's `cd %s` conflicts with "
                    "spec-driven allowlist roots %s but no spec root has a "
                    "usable manifest yet; leaving build_cmd unchanged so the "
                    "router can surface this to HITL.",
                    bad_target, spec_roots,
                )

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
            "because build command installs packages (pip/npm) into "
            "system locations the read-only root FS would block: %s",
            build_cmd,
        )

    # Pre-build link check: every relative import in JS/TS/Python source
    # must resolve to an existing file. Catches the failure mode where
    # codegen drops a component import (CIOD's App.jsx importing
    # './components/Dashboard' when only DashboardPage exists). Short-
    # circuits the build the same way prod_import_smoke_check does, but
    # cheaper (pure-Python AST/regex, no sandbox).
    if state.get("run_link_check", True):
        try:
            from harness.link_check import (
                broken_links_to_diagnostics,
                scan_workspace_for_broken_imports,
            )
            broken = scan_workspace_for_broken_imports(workspace)
        except Exception as exc:  # noqa: BLE001 — link check is advisory
            logger.warning("[compiler_node] Link check failed (%s); skipping.", exc)
            broken = []
        if broken:
            link_diags = broken_links_to_diagnostics(broken)
            logger.info(
                "[compiler_node] Pre-build link check found %d unresolved "
                "relative import(s); short-circuiting build.",
                len(link_diags),
            )
            short_circuit_state = dict(state.get("node_state", {}) or {})
            short_circuit_state["current_node"] = "compiler"
            short_circuit_state["link_check_failed"] = True
            short_circuit_state["last_build_output"] = (
                f"Pre-build link check failed: {len(link_diags)} relative "
                "import(s) do not resolve to any file on disk. The actual "
                "build was skipped. Fix the imports (or create the missing "
                "files) before the build is attempted."
            )
            # Rotate survival-tracking fingerprints on the short-circuit
            # path too — see _rotate_diag_fingerprints_delta for why.
            return {
                "exit_code": 1,
                "compiler_errors": link_diags,
                "node_state": short_circuit_state,
                "loop_counter": loop_counter,
                **_rotate_diag_fingerprints_delta(state, link_diags),
            }

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
    # The install step is composed fresh from the workspace, NOT extracted
    # from build_cmd. Build commands take many shapes (`make build`, a bare
    # `pip install pytest && pytest -q` seed, a `cd subdir && ...` form);
    # naively splitting on `&&` misses the project's actual deps in every
    # case except the simplest single-manifest layout. The composer
    # inspects the workspace directly and chains installs for every
    # supported Python manifest at root + first-level subdirs. If it
    # returns None, there's no Python code to smoke-import yet — skip.
    composed_install_step = (
        _compose_prod_smoke_install_step(workspace) if smoke_enabled else None
    )
    if smoke_enabled and composed_install_step:
        install_step = composed_install_step
        logger.info(
            "[prod-smoke] Composed install step from workspace manifests: %s",
            install_step,
        )
        # The composed install step always runs `uv pip install`, which
        # needs network access. The outer `allow_network` may still be
        # False when build_cmd is e.g. `make build` and the toolchain
        # adapter didn't flip it on. Force network just for the smoke
        # check — without it pip can't reach PyPI and every import fails
        # with `ModuleNotFoundError`, indistinguishable from a missing
        # manifest entry.
        smoke_errors = await _run_prod_import_smoke_check(
            workspace_path=workspace,
            sandbox_config=sandbox_cfg,
            allow_network=True,
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
            # Promote a smoke-emitted BUILD_COMMAND_BLOCKED diag onto
            # node_state so route_after_compiler short-circuits to HITL
            # instead of dispatching repair on a harness-internal failure.
            for diag in smoke_errors:
                if diag.get("error_code") == "BUILD_COMMAND_BLOCKED":
                    short_circuit_state["build_command_blocked"] = True
                    short_circuit_state["build_command_blocked_rule"] = (
                        diag.get("matched_rule", "")
                    )
                    short_circuit_state["last_build_output"] = (
                        "Sandbox security validator rejected the composed "
                        "install command. This is harness-internal config, "
                        "not user code — routing to HITL."
                    )
                    break
            # Rotate survival-tracking fingerprints — see
            # _rotate_diag_fingerprints_delta. Regression fix for the
            # runaway repair loop in session 7e4cba32.
            return {
                "exit_code": 1,
                "compiler_errors": smoke_errors,
                "node_state": short_circuit_state,
                "loop_counter": loop_counter,
                **_rotate_diag_fingerprints_delta(state, smoke_errors),
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
    #   - Everything else (npm/node/docker, single-segment local
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
            # land the requirements.txt edit.
            # Node "Cannot find module 'X'" is ALWAYS repairable too — any
            # bare-name miss is an npm-installable package and autofix R7
            # (`_try_missing_npm_dep`) writes it to package.json. Scoped
            # packages (@scope/pkg) and sub-paths (pkg/sub) included.
            # Shell `command not found` is only repairable when the symbol
            # is a pip-installable Python tool listed in
            # ``_PIP_INSTALLABLE_SYMBOLS``; everything else (docker, etc.)
            # needs an operator-side image swap.
            env_misconfig_is_repairable = (
                miss_kind in ("python", "node")
                or env_misconfig_symbol.lower() in _PIP_INSTALLABLE_SYMBOLS
            )
            if env_misconfig_is_repairable:
                logger.info(
                    "[compiler_node] Missing '%s' (kind=%s) is pip-installable. "
                    "Routing through repair loop so autofix / LLM can amend the "
                    "dep manifest.",
                    env_misconfig_symbol, miss_kind,
                )
                msg = _repairable_dep_hint(
                    env_misconfig_symbol, build_cmd, miss_kind=miss_kind,
                )
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

    # Sandbox CommandValidator block — the build command itself was
    # refused before any subprocess ran. The validator config (allowed_
    # commands / blocked_patterns) lives in the GLOBAL config; the patcher
    # allowlist cannot write to it, so repair_node would burn its full
    # iteration budget producing rejected patches. Detected from the raw
    # output (the exception text is what BuildResult carries) and surfaced
    # via node_state so route_after_compiler can short-circuit to HITL.
    cmd_blocked_rule: Optional[str] = None
    if exit_code != 0:
        cmd_blocked_rule = _is_command_blocked_by_security(raw_log)
        if cmd_blocked_rule:
            logger.warning(
                "[compiler_node] Build command blocked by security validator "
                "(rule=%s). The validator config is global, not workspace-"
                "scoped — repair loop cannot reach it. Short-circuiting to HITL.",
                cmd_blocked_rule,
            )

    # Detect `sh: N: cd: can't cd to <dir>` (exit 2, zero structured
    # diagnostics). The upstream Fix 2/Fix 3 self-heal covers the common
    # case, but survives only when the workspace already carries a spec-
    # approved manifest to point at. When both self-heals leave the
    # build_command pointing at a missing dir, another repair round has
    # nothing to fix — the failure is entirely in build_command wiring,
    # not in the LLM's code. Route to HITL with a build_command_cd_missing
    # tag so the operator can rewire and resume.
    cd_missing_dir: Optional[str] = None
    if exit_code != 0 and not compiler_errors and not cmd_blocked_rule:
        cd_missing_dir = _detect_cd_failure_from_output(raw_log)
        if cd_missing_dir:
            logger.warning(
                "[compiler_node] Shell failed to cd into '%s' — the build "
                "command references a directory that does not exist in the "
                "workspace and repair cannot create it (allowlist would "
                "reject or the LLM has no signal to write there). "
                "Short-circuiting to HITL.", cd_missing_dir,
            )

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
    if cmd_blocked_rule:
        node_state["build_command_blocked"] = True
        node_state["build_command_blocked_rule"] = cmd_blocked_rule
    if cd_missing_dir:
        node_state["build_command_cd_missing"] = True
        node_state["build_command_cd_missing_dir"] = cd_missing_dir

    # First-failure snapshot. Without this the HITL escalation summary
    # only sees the FINAL round's build output — which lies to the
    # operator when the failure mode changes mid-session. Session
    # db6bfcbe ran rounds 1–3 with the build command blocked by the
    # security validator; round 4's repair happened to take a different
    # code path that surfaced a downstream uv install failure, and the
    # summariser (seeing only the uv tail) hallucinated a pinned dep
    # that didn't exist. Frozen on the first non-zero-exit round so
    # later iterations don't overwrite it; the most-recent output is
    # still available via ``last_build_output``.
    if exit_code != 0 and not node_state.get("first_failure_build_output"):
        node_state["first_failure_build_output"] = raw_log
        node_state["first_failure_build_command"] = build_cmd
        node_state["first_failure_round"] = int(
            loop_counter.get("total_repairs", 0) or 0
        )
        node_state["first_failure_compiler_errors"] = [
            dict(e) for e in (compiler_errors or [])[:8]
        ]

    # Test-runner "no tests collected" is NOT a build failure — the runner
    # just had nothing to run. Two dispositions:
    #   * Workspace has source → fold to exit 0 and let the graph advance.
    #     Per the batch-verification design (memory:
    #     [[project_per_batch_pipeline]]) an empty test set in a batch is a
    #     valid intermediate state — test_generation_node may have run and
    #     produced 0 tests, or the stack has no test files by design. The
    #     repair LLM cannot fix a non-error; routing anywhere but forward
    #     just burns iterations.
    #   * Workspace has no source → route to HITL. The prior patching pass
    #     produced nothing usable (e.g. allowlist rejected every patch);
    #     the operator needs to fix layout/config, not run repair.
    #
    # Deliberately do NOT populate `compiler_errors` here — a non-empty
    # errors list would flip downstream routers into repair even after the
    # exit fold.
    #
    # Covers pytest exit 5, Jest/Vitest/Mocha exit 1, Maven Surefire /
    # Gradle test exit 1 (see `_NO_TESTS_PATTERNS`).
    if exit_code != 0 and not compiler_errors and _is_no_tests_collected(
        exit_code, raw_log, build_cmd,
    ):
        has_source = _workspace_has_source_files(workspace)
        if has_source:
            logger.warning(
                "[compiler_node] Test runner reported no tests collected "
                "(exit=%d) but workspace has source files — treating as "
                "success and advancing the graph.",
                exit_code,
            )
            exit_code = 0
        else:
            # Preserved for the router's HITL branch. `no_tests_has_source`
            # stays False so the router picks the "empty workspace" path.
            node_state["no_tests_collected"] = True
            node_state["no_tests_has_source"] = False
            logger.warning(
                "[compiler_node] Test runner reported no tests collected "
                "(exit=%d) AND workspace has no source files. Routing to HITL.",
                exit_code,
            )

    # Rotate the survival-tracking fingerprints: yesterday's "current"
    # becomes today's "prior", and the new diagnostics' shape becomes the
    # new "current". repair_node reads ``prior_diag_fingerprints`` to
    # detect which groups survived between two compile rounds. Cleared
    # to [] on success. Warnings are excluded — they don't drive the
    # repair loop, so survival of a warning is irrelevant.
    return_dict: dict[str, Any] = {
        "exit_code": exit_code,
        "compiler_errors": compiler_errors,
        "loop_counter": loop_counter,
        "node_state": node_state,
        "prior_diag_fingerprints": list(state.get("last_diag_fingerprints") or []),
        "last_diag_fingerprints": (
            _fingerprint_diagnostics(compiler_errors) if exit_code != 0 else []
        ),
        "prior_diag_count": int(state.get("last_diag_count") or 0),
        "last_diag_count": (
            sum(
                1 for e in compiler_errors
                if str(e.get("severity", "error")).lower() != "warning"
            )
            if exit_code != 0 else 0
        ),
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
        # Phase K — mark batch-mode's compile gate as passed so
        # ``route_after_story_loop`` skips re-running this chain on a
        # resumed session. No-op outside batch-mode.
        if int(state.get("current_batch_id") or 0):
            return_dict["batch_gate_progress"] = _mark_batch_gate(
                state, "compile_passed",
            )

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

    # Phase 1.1 — progress-based budget. The previous repair iteration's
    # patches were applied, then compiler_node ran again; if the resulting
    # fingerprint set is missing AT LEAST ONE fingerprint from the prior
    # round, the previous round made real progress (some earlier error is
    # gone). When this fails, we tick a separate ``no_progress_repairs``
    # counter that route_after_compiler uses for the HITL gate, while
    # ``total_repairs`` keeps incrementing for telemetry. Net effect: rounds
    # that demonstrably move the failing-set forward don't count against the
    # budget, so a session that improves 251 → 9 → 3 → 1 → 0 fits in budget
    # even if it took 4 rounds, while a session that goes 3 → 3 → 3 hits
    # HITL at the no_progress cap. The first repair iteration has no prior
    # to compare against — credit it neutrally (don't tick no_progress).
    prior_fps_set = set(state.get("prior_diag_fingerprints") or [])
    current_fps_set = set(state.get("last_diag_fingerprints") or [])
    prior_diag_count = int(state.get("prior_diag_count") or 0)
    current_diag_count = int(state.get("last_diag_count") or 0)
    is_first_iteration = loop_counter["total_repairs"] == 1
    # The prior reading is uninformative when it has no fingerprints AND no
    # diagnostic count — that means the previous compiler run failed before
    # producing any parseable errors (e.g. crashed in `npm install` so the
    # TS-compile diagnostics never materialised). Going from prior=0 to a
    # non-zero current is the first real signal, not a regression — without
    # this guard the no_progress counter ticks on the very round that
    # finally surfaced real errors. Treat as neutral, same as the first
    # iteration. See session 7c30bce2 for the original symptom.
    has_meaningful_prior = bool(prior_fps_set) or prior_diag_count > 0
    fps_shrank = bool(prior_fps_set - current_fps_set)
    # Raw-count shrinkage is the secondary signal — when many tests share
    # one fingerprint (e.g. 10 tests all hitting the same EDGAR-mock
    # guard collapse to ONE ``RuntimeError::Real EDGAR HTTP call …``
    # fingerprint), fixing tests one-at-a-time leaves the set the same
    # size but reduces the raw count by 1. Without this signal the
    # no_progress gate fires on a loop that's actually making real
    # progress. Only counts as progress when there WAS a prior count to
    # compare against (current_diag_count==0 means success — handled
    # separately by the loop terminator).
    count_shrank = (
        prior_diag_count > 0
        and current_diag_count > 0
        and current_diag_count < prior_diag_count
    )
    prior_round_made_progress = fps_shrank or count_shrank
    if is_first_iteration or not has_meaningful_prior:
        prior_round_made_progress = True  # neutral; nothing to evaluate yet
    if not prior_round_made_progress:
        loop_counter["no_progress_repairs"] = (
            loop_counter.get("no_progress_repairs", 0) + 1
        )
        logger.info(
            "[repair_node] Prior repair did not shrink failing fingerprints "
            "OR raw count (fps prior=%d/current=%d shared=%d; count "
            "prior=%d/current=%d). no_progress_repairs=%d.",
            len(prior_fps_set), len(current_fps_set),
            len(prior_fps_set & current_fps_set),
            prior_diag_count, current_diag_count,
            loop_counter["no_progress_repairs"],
        )
    else:
        # Reset on progress so non-consecutive stalls don't accumulate.
        # Matches the spirit of consecutive_zero_patch_rounds: a stretch
        # of progress earns back the budget.
        if loop_counter.get("no_progress_repairs", 0) > 0:
            _signal = (
                "fingerprint set" if fps_shrank else "raw diagnostic count"
            )
            logger.info(
                "[repair_node] Prior repair shrank %s (fps prior=%d → "
                "current=%d; count prior=%d → current=%d). Resetting "
                "no_progress_repairs from %d to 0.",
                _signal,
                len(prior_fps_set), len(current_fps_set),
                prior_diag_count, current_diag_count,
                loop_counter["no_progress_repairs"],
            )
        loop_counter["no_progress_repairs"] = 0

    # Phase 2.2 — per-round repair reflection. Cheap LLM judges whether
    # the PREVIOUS round actually addressed the highest-priority error.
    # When it didn't (verdict = DISTRACTION or REGRESSION), the verdict's
    # ``real_blocker`` is injected as a system message into the upcoming
    # dispatch so the repair LLM sees a second opinion before it writes
    # its next patches. Only runs from iteration 2 onward — iteration 1
    # has nothing to reflect on. Strictly fail-open: any error in the
    # reflection path leaves repair behavior unchanged.
    reflection_verdict: Optional[dict[str, str]] = None
    if (
        not is_first_iteration
        and bool(prior_fps_set)  # need a prior to compare against
    ):
        gw_for_reflection = get_gateway()
        reflection_enabled = bool(
            gw_for_reflection is not None
            and getattr(
                gw_for_reflection.config,
                "llm_judgment_repair_reflection",
                True,
            )
        )
        if reflection_enabled:
            # Diff the fingerprint sets and grab a few top persistent
            # error messages for the prompt body. The fingerprint sets
            # use Phase 3(a) normalisation; re-normalise when matching
            # against raw compiler_errors so quoted-span variants align.
            resolved = sorted(prior_fps_set - current_fps_set)
            persisted = sorted(current_fps_set & prior_fps_set)
            new_fps = sorted(current_fps_set - prior_fps_set)
            # Phase 2.2 fix A — pass full diagnostic dicts (with file/line)
            # so the reflection LLM can ground its answer in real locations
            # instead of hallucinating file/symbol names to satisfy the
            # earlier "cite a file/symbol" prompt demand.
            top_persisted_diagnostics: list[dict[str, Any]] = []
            _seen_fps: set[str] = set()
            _reflection_errors: list[DiagnosticObjectDict] = (
                state.get("compiler_errors", []) or []
            )
            for err in _reflection_errors:
                code = str(err.get("error_code", "?"))
                msg = str(err.get("message", ""))
                fp = f"{code}::{_normalize_diagnostic_message(msg)}"
                if fp in persisted and fp not in _seen_fps:
                    _seen_fps.add(fp)
                    top_persisted_diagnostics.append(dict(err))
                if len(top_persisted_diagnostics) >= 3:
                    break
            persistent_errors_for_scoring = [
                err for err in _reflection_errors
                if (str(err.get("error_code", "?")) + "::" +
                    _normalize_diagnostic_message(str(err.get("message", ""))))
                in persisted
            ]
            install_failure_likely = _diagnostics_look_like_install_failure(
                # Score against the FULL persistent failing-set, not just
                # the top-3 we show — the heuristic should see the whole
                # distribution before deciding the pattern.
                persistent_errors_for_scoring,
            )
            # Detect the sub-case where the "missing" module is actually a
            # workspace source dir — a wiring/CWD failure, not an install
            # failure. Suppress the install-failure hint here so the judge
            # doesn't send the repair LLM to add a source dir name to
            # requirements.txt (session 3193a24f burned 3 rounds on this).
            _workspace_for_wiring = state.get("workspace_path") or ""
            path_wiring_module = _missing_module_matches_workspace_source(
                persistent_errors_for_scoring, _workspace_for_wiring,
            )
            if path_wiring_module:
                install_failure_likely = False
                logger.info(
                    "[repair_node] Reflection input flagged as PATH/WIRING "
                    "failure (missing module %r is a workspace source dir). "
                    "Judge prompt will include the wiring hint and suppress "
                    "the install-failure hint.",
                    path_wiring_module,
                )
                try:
                    from harness.observability import emit_event as _emit
                    _emit(
                        "repair_reflection_path_wiring_detected",
                        module=path_wiring_module,
                        persisted_count=len(persistent_errors_for_scoring),
                    )
                except Exception:  # noqa: BLE001
                    pass
            elif install_failure_likely:
                logger.info(
                    "[repair_node] Reflection input flagged as likely "
                    "install/environment failure (majority of persistent "
                    "errors are missing-module / unresolved-import). "
                    "Judge prompt will include the install-failure hint."
                )
            reflection_prompt = _build_repair_reflection_prompt(
                prior_diagnostics_count=len(prior_fps_set),
                current_diagnostics_count=len(current_fps_set),
                resolved_fingerprints=resolved,
                persisted_fingerprints=persisted,
                new_fingerprints=new_fps,
                top_persisted_diagnostics=top_persisted_diagnostics,
                install_failure_likely=install_failure_likely,
                path_wiring_module=path_wiring_module,
                build_output_tail=str(
                    state.get("node_state", {}).get("last_build_output", "")
                    or ""
                ),
            )
            verdict_raw, new_budget = await _maybe_judgment_llm(
                prompt=reflection_prompt,
                budget_remaining_usd=state.get("budget_remaining_usd", 0.0),
                purpose="repair_reflection",
                enabled=True,
            )
            # Persist the (possibly reduced) budget back to state so the
            # main repair dispatch sees the up-to-date number.
            if isinstance(new_budget, (int, float)) and new_budget != state.get("budget_remaining_usd", 0.0):
                state = cast(AgentState, dict(state))
                state["budget_remaining_usd"] = float(new_budget)
            reflection_verdict = _parse_repair_reflection_verdict(verdict_raw or "")
            # One-shot JSON-repair when the verdict didn't parse — the
            # judge often produced signal but wrapped it wrong. Better
            # to spend one small judgment call than to silently drop
            # the reasoning.
            if reflection_verdict is None and (verdict_raw or "").strip():
                _gw = get_gateway()
                if _gw is not None:
                    from harness.gateway import NodeRole as _NodeRole
                    async def _reflection_repair_dispatch(msgs, bud):
                        return await _gw.dispatch(
                            messages=list(msgs),
                            role=_NodeRole.JUDGMENT,
                            budget_remaining_usd=bud,
                        )
                    _reflection_schema = (
                        "A JSON object with exactly three keys: "
                        "'verdict' (one of 'PROGRESS', 'DISTRACTION', "
                        "'REGRESSION'), 'real_blocker' (string — the "
                        "underlying reason the repair loop is stuck; "
                        "may be empty only when verdict='PROGRESS'), "
                        "and 'recommendation' (string). No other keys."
                    )
                    repaired_verdict, repaired_budget = await _repair_malformed_json(
                        raw_text=verdict_raw or "",
                        schema_hint=_reflection_schema,
                        dispatch=_reflection_repair_dispatch,
                        budget_remaining_usd=state.get("budget_remaining_usd", 0.0),
                        purpose="repair_reflection",
                    )
                    if isinstance(repaired_budget, (int, float)) and repaired_budget != state.get("budget_remaining_usd", 0.0):
                        state = cast(AgentState, dict(state))
                        state["budget_remaining_usd"] = float(repaired_budget)
                    if isinstance(repaired_verdict, dict):
                        reflection_verdict = _parse_repair_reflection_verdict(
                            json.dumps(repaired_verdict)
                        )
            if reflection_verdict:
                logger.info(
                    "[repair_node] Reflection verdict: %s. Blocker: %s",
                    reflection_verdict["verdict"],
                    reflection_verdict["real_blocker"][:200],
                )
                # Consecutive-DISTRACTION circuit breaker. The existing
                # ``no_progress_repairs`` gate resets on any fingerprint
                # shrinkage, so an LLM that oscillates the failing set
                # never trips it. This counter listens to the reflection
                # verdict directly: DISTRACTION/REGRESSION ticks it up,
                # PROGRESS resets it. ``route_after_compiler`` consults
                # the counter against ``max_consecutive_distraction_rounds``
                # and escalates to HITL when it saturates.
                _v = reflection_verdict["verdict"]
                _low_signal = _reflection_verdict_is_low_signal(reflection_verdict)
                if _v in {"DISTRACTION", "REGRESSION"} and not _low_signal:
                    loop_counter["consecutive_distraction_rounds"] = (
                        loop_counter.get("consecutive_distraction_rounds", 0) + 1
                    )
                    logger.info(
                        "[repair_node] consecutive_distraction_rounds=%d "
                        "(verdict=%s).",
                        loop_counter["consecutive_distraction_rounds"], _v,
                    )
                elif _v in {"DISTRACTION", "REGRESSION"} and _low_signal:
                    # Fix B — the judge fell back to "insufficient data" (its
                    # escape hatch for bare exception-type errors). The verdict
                    # word oscillates on the failing-count delta but carries no
                    # actionable content, so ticking the distraction circuit-
                    # breaker would race us to HITL on noise instead of a real
                    # distraction signal. Hold that counter steady.
                    #
                    # Fix D (session 116667f5 escalation) — but tick a SECOND
                    # counter that measures how many consecutive rounds the
                    # judge could not localize. When it hits the escalation
                    # threshold the repair prompt injects the raw build-output
                    # tail so the repair LLM sees the full pytest traceback
                    # (assertion-rewrite lines, attribute values, etc.) that
                    # the structured diagnostics lack. Without this the loop
                    # was silently spinning until ``total_repairs`` ran out.
                    loop_counter["consecutive_low_signal_rounds"] = (
                        loop_counter.get("consecutive_low_signal_rounds", 0) + 1
                    )
                    logger.info(
                        "[repair_node] Reflection verdict %s but real_blocker "
                        "is the 'insufficient data' sentinel — holding "
                        "consecutive_distraction_rounds at %d, "
                        "consecutive_low_signal_rounds=%d.",
                        _v,
                        loop_counter.get("consecutive_distraction_rounds", 0),
                        loop_counter["consecutive_low_signal_rounds"],
                    )
                else:  # PROGRESS — earn back the budget.
                    if loop_counter.get("consecutive_distraction_rounds", 0) > 0:
                        logger.info(
                            "[repair_node] Reflection verdict PROGRESS; "
                            "resetting consecutive_distraction_rounds from "
                            "%d to 0.",
                            loop_counter["consecutive_distraction_rounds"],
                        )
                    loop_counter["consecutive_distraction_rounds"] = 0
                    # Any PROGRESS verdict also clears the low-signal streak —
                    # the judge is engaging with the diagnostics again, so the
                    # raw-tail escalation is no longer needed on the next round.
                    if loop_counter.get("consecutive_low_signal_rounds", 0) > 0:
                        logger.info(
                            "[repair_node] Reflection verdict PROGRESS; "
                            "resetting consecutive_low_signal_rounds from "
                            "%d to 0.",
                            loop_counter["consecutive_low_signal_rounds"],
                        )
                    loop_counter["consecutive_low_signal_rounds"] = 0
                try:
                    from harness.observability import emit_event as _emit
                    _emit(
                        "repair_reflection_verdict",
                        iteration=loop_counter["total_repairs"],
                        verdict=reflection_verdict["verdict"],
                        real_blocker=reflection_verdict["real_blocker"][:500],
                        recommendation=reflection_verdict["recommendation"][:500],
                        prior_count=len(prior_fps_set),
                        current_count=len(current_fps_set),
                        consecutive_distraction_rounds=loop_counter.get(
                            "consecutive_distraction_rounds", 0
                        ),
                    )
                except Exception:  # noqa: BLE001
                    pass

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
        dropped_warn = len(raw_errors) - len(errors)
        logger.info(
            "[repair_node] Filtered %d warning-severity diagnostic(s); "
            "%d error(s) passed through to repair.",
            dropped_warn, len(errors),
        )
        # Phase 2.1 — decision-point logging.
        try:
            from harness.observability import emit_event as _emit_drop
            _emit_drop(
                "dropped_from_prompt",
                site="warning_severity_filter",
                dropped_count=dropped_warn,
                kept_count=len(errors),
                reason="warnings_do_not_block_build",
                examples=[
                    {
                        "code": str(e.get("error_code", "?")),
                        "message_excerpt": str(e.get("message", ""))[:80],
                    }
                    for e in raw_errors
                    if str(e.get("severity", "error")).lower() == "warning"
                ][:5],
            )
        except Exception:  # noqa: BLE001
            pass

    # LLM-judgment pre-flight autofix classifier (#2). On the FIRST repair
    # iteration, when the diagnostics carry MISSING_DEP or
    # DEP_RESOLUTION_CONFLICT, ask a cheap LLM whether each unique missing
    # symbol is "manifest-fixable" (autofix can add it) or
    # "toolchain-mismatch" (system binary / interpreter / package manager
    # the sandbox image lacks — only an image swap can fix it). When ANY
    # symbol is toolchain-mismatch we set env_misconfig and short-circuit
    # back to compiler_node; the existing route_after_compiler env_misconfig
    # branch then sends to HITL on the next pass. Saves the two compile
    # cycles the deterministic same-symbol tripwire (at N=3) currently
    # burns before reaching the same conclusion.
    gateway = get_gateway()
    if (
        gateway is not None
        and loop_counter.get("total_repairs", 0) == 1
        and bool(getattr(
            gateway.config, "llm_judgment_preflight_autofix", True,
        ))
    ):
        dep_symbols = sorted({
            str(e.get("missing_symbol", "") or "").strip()
            for e in errors
            if str(e.get("error_code", "")).upper() in {
                "MISSING_DEP", "DEP_RESOLUTION_CONFLICT",
            }
            and str(e.get("missing_symbol", "") or "").strip()
        })
        if dep_symbols:
            preflight_prompt = _build_preflight_autofix_prompt(
                symbols=dep_symbols,
                build_command=str(state.get("build_command", "") or ""),
                sandbox_image=str(
                    (state.get("sandbox_config") or {}).get("docker_image", "")
                    or ""
                ),
            )
            verdict, new_budget = await _maybe_judgment_llm(
                prompt=preflight_prompt,
                budget_remaining_usd=state.get("budget_remaining_usd", 0.0),
                purpose="preflight_autofix_judgment",
                enabled=True,
            )
            if verdict:
                state = cast(AgentState, dict(state))
                state["budget_remaining_usd"] = new_budget
                non_fixable = _parse_preflight_verdict(verdict, dep_symbols)
                if non_fixable:
                    sym = non_fixable[0]
                    logger.warning(
                        "[repair_node] Pre-flight classifier marked '%s' as "
                        "toolchain-mismatch (non-fixable from inside the "
                        "loop). Short-circuiting to HITL on the next router "
                        "pass instead of running autofix on a dead end.",
                        sym,
                    )
                    short_circuit_state = dict(state.get("node_state", {}) or {})
                    short_circuit_state["current_node"] = "repair"
                    short_circuit_state["env_misconfig"] = True
                    short_circuit_state["env_misconfig_symbol"] = sym
                    short_circuit_state["preflight_autofix_verdict"] = {
                        "non_fixable": non_fixable,
                        "all_classified": dep_symbols,
                    }
                    return {
                        "node_state": short_circuit_state,
                        "loop_counter": loop_counter,
                        "budget_remaining_usd": new_budget,
                    }

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
            "batch_modified_files": _extend_batch_scope(state, autofix_modified_files),
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
            diag_cap, _ = _repair_file_caps(state)
            # Phase L — in CR mode, augment the diagnostic file list
            # with up to +6 likely cascade sources: high-fanout
            # shared utilities the CR touched + immediate callers of
            # any failing file. The repair LLM otherwise sees only the
            # failing test/file and would not realise a shared util
            # the CR amended is the root cause.
            cr_extra = _cr_impact_augment(state, diag_files)
            # Fix #2 — first-party-import prefetch. For every failing
            # test file, parse its imports and pre-attach the workspace
            # modules it depends on (recursively bounded by depth=2).
            # The repair LLM otherwise emits a READ_FILE block on every
            # round to see the same module-under-test; with prefetch,
            # the content is already in the prompt and the round
            # converges in one dispatch. Caps the extra files at 6 so
            # token cost stays bounded; the LLM can READ_FILE for the
            # tail if it really needs it.
            import_prefetch: list[str] = []
            _seen_imp: set[str] = set(diag_files) | set(cr_extra)
            for _diag_file in diag_files:
                if len(import_prefetch) >= 6:
                    break
                if not _diag_file.endswith(".py"):
                    continue
                for _imp_rel in _first_party_imports_for(
                    workspace_path, _diag_file,
                ):
                    if _imp_rel in _seen_imp:
                        continue
                    _seen_imp.add(_imp_rel)
                    import_prefetch.append(_imp_rel)
                    if len(import_prefetch) >= 6:
                        break
            files_for_preflight = (
                list(diag_files) + cr_extra + import_prefetch
            )
            preflight_pairs = _collect_workspace_file_content(
                workspace_path, files_for_preflight,
                max_files=diag_cap,
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
    # Workspace-aware conftest chain (Fix #3). When tests fail, surface
    # the EXACT chain of ``conftest.py`` files pytest is loading for
    # each failing test, with content inlined. The repair LLM otherwise
    # has no way to tell which conftest is in scope when a workspace
    # has overlapping test trees (e.g. ``tests/conftest.py`` AND
    # ``server/tests/conftest.py``) and ends up patching the wrong one.
    # Runs every iteration so a conftest the LLM patched mid-session is
    # re-rendered with the new bytes — feeds the patcher's drift
    # detector via ``files_seen_by_llm`` so subsequent patches against
    # the updated conftest match.
    _conftest_chains = _collect_conftests_for_failing_tests(
        workspace_path, errors,
    )
    if _conftest_chains:
        error_summary += _format_conftest_chains_for_repair(
            workspace_path, _conftest_chains,
            record_hashes_into=files_seen_by_llm,
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
    # Pass the prior round's diagnostic fingerprints so the formatter can
    # promote any group that survived past the cascade prior. See
    # ``_format_diagnostics_for_repair`` for the layered defense (Layer 3).
    # Also pass any codes the LLM requested via PROMOTE_DEFERRED on the
    # PREVIOUS round (Phase 1.2 escape hatch).
    # Phase 4 — emit structured JSON payload alongside markdown unless
    # the operator has disabled it via config.
    _gw_for_payload = get_gateway()
    _emit_structured_payload = bool(
        _gw_for_payload is None
        or getattr(
            _gw_for_payload.config,
            "repair_structured_diagnostic_payload",
            True,
        )
    )
    error_summary += _format_diagnostics_for_repair(
        errors,
        prior_fingerprints=set(state.get("prior_diag_fingerprints") or []),
        promoted_codes=set(state.get("promoted_codes_next_round") or []),
        emit_structured_payload=_emit_structured_payload,
    )

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
    # Phase 3(e) — tolerate mixed-prefix repair contexts. The original
    # ``all(...)`` flipped the entire prompt frame to non-security
    # whenever even one stray compile error appeared alongside scanner
    # findings (e.g. when semgrep's ``extra.fix`` leaves a syntactically
    # broken patch behind and the next compile surfaces both). Use a
    # supermajority threshold so the dominant repair context wins.
    _SECURITY_PREFIX_DOMINANCE_THRESHOLD = 0.8
    _security_count = sum(
        1 for e in errors
        if str(e.get("error_code", "")).upper().startswith(_SECURITY_PREFIXES)
    )
    is_security_repair = bool(errors) and (
        _security_count / len(errors) >= _SECURITY_PREFIX_DOMINANCE_THRESHOLD
    )

    # Detect repair driven by harness-generated test failures. The
    # test_generation_node tags each diagnostic with an error_code starting
    # with "TEST_FAILURE" so we can swap in framing that tells the LLM these
    # are unit-test failures (not compile errors) and that fixing the
    # implementation is preferred over weakening the test assertion.
    # Phase 3(e) — same supermajority logic as is_security_repair so a
    # single stray compile diagnostic alongside test failures doesn't
    # flip the entire prompt out of the test-failure framing.
    _test_failure_count = sum(
        1 for e in errors
        if str(e.get("error_code", "")).upper().startswith("TEST_FAILURE")
    )
    is_test_failure_repair = bool(errors) and (
        _test_failure_count / len(errors) >= _SECURITY_PREFIX_DOMINANCE_THRESHOLD
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
    prior_patch_failures = state.get("node_state", {}).get("patch_failures") or []

    # LLM-judgment patcher-rejection diagnosis (#4). When the previous
    # attempt produced any rejections, run a cheap classifier+adviser call
    # so the repair LLM sees actionable guidance ABOVE the raw rejection
    # dumps. The classifier sorts each failure into a category (allowlist
    # miss / stale context / wrong file / format error) and emits one
    # corrective instruction per category. Without this the loop just
    # re-reads "your patches were rejected" round after round and tends
    # to re-emit the same broken hunk.
    if prior_rejections or prior_patch_failures:
        diag_prompt = _build_patcher_rejection_diagnosis_prompt(
            rejections=prior_rejections,
            patch_failures=prior_patch_failures,
            allowed_paths=state.get("node_state", {}).get("allowed_paths") or [],
            modified_files=sorted(
                {p for p in (state.get("modified_files") or []) if p}
            ),
        )
        diagnosis, new_budget = await _maybe_judgment_llm(
            prompt=diag_prompt,
            budget_remaining_usd=state.get("budget_remaining_usd", 0.0),
            purpose="patcher_rejection_diagnosis",
            enabled=bool(getattr(
                gateway.config,
                "llm_judgment_patcher_rejection_diagnosis",
                True,
            )),
        )
        if diagnosis:
            state = cast(AgentState, dict(state))
            state["budget_remaining_usd"] = new_budget
            error_summary += (
                "\n## Patcher-rejection diagnosis (LLM advisory)\n"
                "An auxiliary judgment LLM read the rejected patches from "
                "your previous attempt and classified each failure. Treat "
                "the following as a directive for THIS round — applying it "
                "correctly is what unblocks the loop:\n\n"
                f"{diagnosis}\n"
            )
            logger.info(
                "[repair_node] Patcher-rejection diagnosis attached (%d chars).",
                len(diagnosis),
            )

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
    error_summary += _format_prior_patch_failures(prior_patch_failures)

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
        # Phase J: in end-of-session repair the SNAPSHOT_CAP doubles
        # (12→30 diagnostic, 50→150 inventory) so a security-repair
        # that touched a shared utility doesn't leave the EoS repair
        # blind to the cascade.
        _, SNAPSHOT_CAP = _repair_file_caps(state)
        # Critical config files are PREPENDED so they always survive the
        # cap. These are the files the LLM most often CREATE_FILE's into
        # rejection ("File already exists with different content") because
        # they're scaffolded once early and never seen again unless we
        # actively keep them in the prompt.
        critical = [p for p in inventory_files if _is_critical_config_path(p)]
        rest = [p for p in inventory_files if not _is_critical_config_path(p)]
        ordered = critical + rest
        shown = ordered[:SNAPSHOT_CAP]
        extra = max(0, len(ordered) - SNAPSHOT_CAP)
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

    # Fix D — low-signal escalation. When the judge has returned the
    # ``insufficient data — investigate <file>`` sentinel on two or more
    # consecutive rounds, the structured diagnostics alone are not enough
    # for the repair LLM to converge (session 116667f5: 6 rounds spent
    # guessing at the value ``session.is_active`` returned because the
    # bare AssertionError message hid it). Inject the raw build-output
    # tail alongside the structured summary so the LLM sees the full
    # pytest traceback (assertion-rewrite lines, ``+  where`` value
    # explanations, attribute traversals) that the parser cannot reduce
    # to a single message. Gated on the counter so we don't pay the
    # extra tokens on healthy sessions where the diagnostics suffice.
    _low_signal_streak = int(
        loop_counter.get("consecutive_low_signal_rounds", 0) or 0
    )
    if errors and _low_signal_streak >= 2:
        raw_output = state.get("node_state", {}).get("last_build_output", "")
        if raw_output:
            error_summary += (
                "\n## Raw Build Output (low-signal escalation)\n"
                "The reflection judge has been unable to localize the "
                "failing diagnostic for "
                f"{_low_signal_streak} consecutive rounds — the "
                "structured diagnostics above collapse to bare exception "
                "types. The full pytest traceback below is your primary "
                "source of grounding this round: look for ``E   assert "
                "<expr>``, ``E    +  where <val> = <obj>.<attr>``, and "
                "similar assertion-rewrite lines that name the resolved "
                "values so you can see what the assertion actually "
                "compared.\n"
                f"```\n{_slice_build_output_for_repair(raw_output)}\n```\n"
            )
            logger.info(
                "[repair_node] Low-signal escalation: injected raw build "
                "output tail into repair prompt "
                "(consecutive_low_signal_rounds=%d).",
                _low_signal_streak,
            )
            try:
                from harness.observability import emit_event as _emit_ls
                _emit_ls(
                    "repair_low_signal_escalation",
                    iteration=loop_counter.get("total_repairs", 0),
                    consecutive_low_signal_rounds=_low_signal_streak,
                )
            except Exception:  # noqa: BLE001
                pass

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
        # Phase J: at end-of-session repair, jump straight to the
        # reasoning model. The failing tests at this gate ran after
        # security-scan repairs, so a one-shot diagnosis benefits
        # from the bigger model on attempt #1 rather than burning
        # cheap-model rounds first. Toggle via
        # ``end_of_session_force_reasoning_model`` (default True).
        _eos_ns = state.get("node_state") or {}
        _eos_active = bool(
            isinstance(_eos_ns, dict) and _eos_ns.get("end_of_session_phase")
        )
        _eos_force = bool(getattr(
            gateway.config, "end_of_session_force_reasoning_model", True,
        ))
        if _eos_active and _eos_force:
            use_escalation = True

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
        # Phase L: when the session is processing change-requests, the
        # repair LLM should consider cross-cutting cascade sources
        # (shared utils, immediate callers) the CR may have amended.
        # _cr_impact_augment has already widened the file content
        # surfaced into error_summary, but the framing line tells the
        # LLM why and where to look. Concatenates BEFORE the EoS
        # preamble so an end-of-session CR repair gets both.
        cr_preamble = ""
        if state.get("change_request_mode"):
            cr_preamble = (
                "## Change-request session\n\n"
                "This session is implementing change requests against "
                "an existing codebase. The failing tests below may "
                "involve features OUTSIDE the CR's stated scope — when "
                "that happens, the CR likely amended a shared utility "
                "(`utils.py`, `models.py`, a base class, a config "
                "loader, etc.) whose callers regressed. The diagnostic "
                "file list below has been augmented with the most-"
                "imported workspace files THIS CR touched plus the "
                "immediate callers of every failing file — use those "
                "as the primary suspect set before patching anything "
                "named only by the test.\n\n---\n\n"
            )

        # Phase J: prepend an end-of-session framing block when this
        # repair was triggered by the final regression gate. The
        # failing tests there ran after security-scan repairs may have
        # touched shared utilities, so the LLM needs to think about
        # cross-file cascades, not just the directly-named files.
        eos_preamble = ""
        if _eos_active:
            eos_preamble = (
                "## End-of-session regression\n\n"
                "This is the FINAL pre-deployment regression check. "
                "The failing tests below ran after the security-scan "
                "repair loop already landed patches in this session, "
                "so a likely cause is a SHARED UTILITY the security "
                "fix touched that's imported by code unrelated to the "
                "security finding itself. Before patching the named "
                "files, look at:\n"
                "- imports in the failing-test file and follow them "
                "into shared modules\n"
                "- recent modifications listed in the workspace "
                "inventory below — any of those that look like "
                "utilities, helpers, or `models.py` are prime suspects\n"
                "- whether a test expectation changed under your feet "
                "vs whether the implementation changed\n\n"
                "Then write the minimum diff that restores green tests "
                "WITHOUT regressing the security fix that was just "
                "applied.\n\n---\n\n"
            )

        if is_security_repair:
            repair_prompt = (
                cr_preamble + eos_preamble
                +"The deterministic security gate flagged the following vulnerabilities "
                "in code that has already passed the build. Generate precise SEARCH/REPLACE "
                "patches that REMOVE the root cause without regressing existing tests. "
                "Prefer the minimum diff: do not refactor unrelated code, do not weaken "
                "the security control elsewhere, and if a finding requires a dependency "
                "upgrade, write the new version into the manifest rather than vendoring "
                f"a patched copy.\n\n{error_summary}"
            )
        elif is_test_failure_repair:
            repair_prompt = (
                cr_preamble + eos_preamble
                + _TRACE_FIRST_DIRECTIVE
                +"The harness-generated unit tests just failed when executed in the "
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
                cr_preamble + eos_preamble
                +f"The build has failed {total_repairs} time(s) despite previous fix attempts. "
                f"The simpler model could not resolve these errors. You are a senior reasoning model. "
                f"Carefully analyze the errors and produce a definitive fix.\n\n{error_summary}"
            )
        else:
            repair_prompt = (
                cr_preamble + eos_preamble
                + _TRACE_FIRST_DIRECTIVE
                +"The build failed with the following errors. Generate precise SEARCH/REPLACE "
                f"patches to fix them.\n\n{error_summary}"
            )
        # Soft turn-budget warning (audit #19). Injected as a system
        # message so the LLM treats it with authority. Fires only on the
        # last two repair iterations; quiet otherwise. With Phase 1.1
        # the warning is keyed on ``no_progress_repairs`` (the actual HITL
        # gate), not raw ``total_repairs``. A model that has spent 5 total
        # rounds but made progress in every one shouldn't see a "last
        # iteration" warning — its budget is intact. Conversely, a model
        # that has stalled twice will see the warning at total=2.
        _no_progress_for_warning = int(loop_counter.get("no_progress_repairs", 0))
        budget_warning = _repair_budget_warning(_no_progress_for_warning, max_repair_attempts)
        if budget_warning is not None:
            messages.append(MessageDict(role="system", content=budget_warning))
            try:
                from harness.observability import emit_event
                emit_event(
                    "repair_budget_warning",
                    total_repairs=total_repairs,
                    no_progress_repairs=_no_progress_for_warning,
                    cap=max_repair_attempts,
                    remaining=max_repair_attempts - _no_progress_for_warning,
                )
            except Exception:  # noqa: BLE001 — telemetry must not block
                pass

        # Phase 2.2 — inject the reflection verdict as an authoritative
        # system message ahead of the repair prompt. The model that just
        # finished an iteration was, by definition, blind to whether its
        # patches addressed the right error; the cheap reflection judge
        # has perspective and tells it explicitly. For PROGRESS verdicts
        # we don't inject anything (don't waste tokens on "you did fine"),
        # only on DISTRACTION or REGRESSION where the next round needs to
        # change direction.
        #
        # Fix C — file-intersection gate. The reflection LLM is cheap
        # and small; when its real_blocker doesn't name any file that
        # is present in the current compiler_errors, the diagnosis is
        # almost certainly hallucinated (the previous loop spent dozens
        # of rounds chasing "the filing list retrieval logic" while the
        # actual bug was in a regex — see post-mortem on session
        # cf3fcd27). Skip the injection in that case rather than letting
        # noise become authoritative.
        #
        # SRS-trim (Fix #1+#2): when the verdict is DISTRACTION/REGRESSION
        # we ALSO rebuild ``messages`` from scratch — drop the giant SRS
        # at messages[0] (180k+ chars on real sessions) plus the iter-1..N-1
        # chatter, and lead the dispatch with the judge's banner. The
        # SRS-driven repair LLM was reading the spec and patching
        # cosmetics; with the SRS gone the verdict is the only authority
        # left in front of the model.
        if (
            reflection_verdict is not None
            and reflection_verdict["verdict"] in {"DISTRACTION", "REGRESSION"}
            and reflection_verdict["real_blocker"]
            and not _reflection_verdict_is_low_signal(reflection_verdict)
            and _reflection_grounds_in_diagnostics(
                reflection_verdict, state.get("compiler_errors", []) or []
            )
        ):
            judge_named_files = _verdict_named_files(
                reflection_verdict, state.get("compiler_errors", []) or []
            )
            ignored_last_round = bool(
                loop_counter.get("judge_ignored_last_round")
            )
            ignored_files_prev = list(
                loop_counter.get("judge_named_files_last_round") or []
            )
            ignored_touched_prev = list(
                loop_counter.get("judge_round_touched_files") or []
            )
            # Fix C — put a stable preamble at the top so the LLM
            # provider's prefix cache matches from round to round. The
            # verdict word (DISTRACTION vs REGRESSION) is the noisiest
            # field: it can flip round-to-round on the same underlying
            # blocker just because the failing-count delta changed sign
            # (session 116667f5 logged three cache_prefix_drift events
            # driven purely by the ``Verdict:`` line at offset 67).
            # Moving the volatile fields to the end keeps a couple hundred
            # chars of banner in-cache regardless of how the verdict
            # oscillates.
            reflection_banner_lines = [
                "=== JUDGE'S VERDICT — READ THIS FIRST AND OBEY ===",
                (
                    "The reflection judge audited the previous repair round "
                    "and issues the authoritative directive below. Treat it "
                    "as course-correction: apply the required action to your "
                    "next patch set exactly as written. If the directive "
                    "seems to disagree with the failing diagnostics you see, "
                    "assume your reading of the diagnostics is what's wrong "
                    "— the judge saw the delta between rounds; you did not."
                ),
                "----- directive -----",
                f"Real blocker: {reflection_verdict['real_blocker']}",
            ]
            if reflection_verdict["recommendation"]:
                reflection_banner_lines.append(
                    f"REQUIRED ACTION: {reflection_verdict['recommendation']}"
                )
            if judge_named_files:
                reflection_banner_lines.append(
                    "YOUR PATCHES THIS ROUND MUST MODIFY at least one of: "
                    + ", ".join(judge_named_files)
                )
            # Persistent-blocker directive (Fixes #2/#3). When the judge's
            # real_blocker names the SAME (file, line) two rounds running,
            # the previous repair round either missed the target entirely
            # or landed a cosmetic change nearby — session 674bfdbd cycled
            # 8+ rounds on `test_edgar.py:126` with 0-1 patches per round,
            # each leaving line 126 untouched. Promote that fact to a
            # non-negotiable directive: patch MUST alter the exact line.
            _current_named_lines = _verdict_named_file_lines(
                reflection_verdict,
                state.get("compiler_errors", []) or [],
            )
            _prev_named_lines_raw = list(
                loop_counter.get("judge_named_file_lines_last_round") or []
            )
            _prev_named_lines: set[tuple[str, int]] = set()
            for _pair in _prev_named_lines_raw:
                # Stored as list-of-lists / list-of-tuples depending on
                # roundtrip through JSON — normalise on read.
                if isinstance(_pair, (list, tuple)) and len(_pair) == 2:
                    try:
                        _prev_named_lines.add((str(_pair[0]), int(_pair[1])))
                    except (TypeError, ValueError):
                        continue
            _persistent_lines = [
                (f, ln) for (f, ln) in _current_named_lines
                if (f, ln) in _prev_named_lines
            ]
            if _persistent_lines:
                _display = ", ".join(
                    f"{f}:{ln}" for f, ln in _persistent_lines[:5]
                )
                # Streak counter: how many CONSECUTIVE rounds this exact
                # location has been the blocker. 2 rounds → normal
                # persistent directive. 3+ rounds → also unlock
                # REWRITE_FILE as the escape hatch. Tracked per-file so
                # a fresh blocker in a different file resets the streak
                # correctly on that file only.
                _streak_raw = loop_counter.get(
                    "persistent_blocker_streak_per_file"
                ) or {}
                _streak: dict[str, int] = (
                    dict(_streak_raw) if isinstance(_streak_raw, dict) else {}
                )
                _persistent_files_set = {f for f, _ in _persistent_lines}
                for _f in _persistent_files_set:
                    _streak[_f] = _streak.get(_f, 1) + 1
                # Reset streaks for files that dropped out of the
                # persistent set — keeps the counter honest across
                # blocker migrations.
                for _f in list(_streak.keys()):
                    if _f not in _persistent_files_set:
                        _streak.pop(_f, None)
                loop_counter["persistent_blocker_streak_per_file"] = _streak
                _max_streak = max(_streak.values(), default=0)
                reflection_banner_lines.append(
                    "PERSISTENT BLOCKER — the SAME location has failed "
                    f"two+ rounds running: {_display}. Your previous "
                    "patch either missed the target or landed a cosmetic "
                    "change nearby. MANDATORY: your search block(s) THIS "
                    "round MUST include the exact failing line — read "
                    "the current file (READ_FILE) if needed to lift the "
                    "surrounding context, then emit a REPLACE_BLOCK / "
                    "DELETE_BLOCK that textually contains the failing "
                    "line. A patch that does not touch these lines will "
                    "be treated as no progress."
                )
                if _max_streak >= 3:
                    _stuck_files = sorted({
                        f for f, s in _streak.items() if s >= 3
                    })[:5]
                    reflection_banner_lines.append(
                        "ESCAPE HATCH — this location has been stuck for "
                        f"3+ rounds. You MAY now emit <<<REWRITE_FILE>>> "
                        f"on: {', '.join(_stuck_files)}. Use it ONLY if "
                        "your surgical patches keep missing — REWRITE_FILE "
                        "replaces the entire file content, so you must "
                        "supply the FULL corrected file body (imports, "
                        "class defs, all tests). Format: same as "
                        "CREATE_FILE, but the block name is REWRITE_FILE / "
                        "END_REWRITE_FILE. Post-patch parse validation "
                        "still applies: broken syntax rolls the file "
                        "back."
                    )
                    # "Fix may belong in the test file" hint. When a
                    # source file has been the persistent blocker for
                    # 3+ rounds without converging, the fix is often
                    # actually in the test's mock setup or assertion,
                    # NOT the source file the traceback points at.
                    # Session b92043caq stalled on
                    # ``backend/services/edgar.py:96`` for 4 rounds while
                    # the real bug was an AsyncMock context-manager
                    # setup in ``backend/tests/test_edgar.py``. Surface
                    # the sibling test path so the LLM has a concrete
                    # second target to consider before REWRITE_FILE'ing
                    # the source.
                    _workspace = str(state.get("workspace_path", "") or "")
                    _companion_tests: list[str] = []
                    for _sf in _stuck_files:
                        # Only look for a companion when the stuck file
                        # is NOT itself a test — patching a test with a
                        # test is not the failure mode this catches.
                        _sf_norm = _sf.lower()
                        if "test_" in os.path.basename(_sf_norm):
                            continue
                        _companion_tests.extend(
                            _related_test_files(_sf, _workspace)
                        )
                    # Dedupe + cap to keep the banner focused.
                    _companion_tests = list(dict.fromkeys(_companion_tests))[:5]
                    if _companion_tests:
                        reflection_banner_lines.append(
                            "TEST-FILE HINT — before REWRITE_FILE'ing the "
                            "source, check whether the fix belongs in the "
                            f"related test file(s): {', '.join(_companion_tests)}. "
                            "This pattern (persistent blocker in a source "
                            "file after 3+ rounds) is often caused by a "
                            "mock setup bug in the test: `session.get()` "
                            "patched with `return_value=`, `AsyncMock` "
                            "used as an async-context-manager without "
                            "wiring `__aenter__`, `json_side_effect` "
                            "assigned to `mock.json` instead of "
                            "`mock.json.side_effect`, etc. Read the test "
                            "file (READ_FILE) and consider whether the "
                            "traceback is telling you the source is wrong "
                            "OR that the test's harness is lying to it. "
                            "Patch whichever side is actually broken."
                        )
            else:
                # No persistent lines this round — clear per-file streaks
                # so a NEW blocker on a stale-streak file doesn't inherit
                # a bogus count.
                loop_counter.pop("persistent_blocker_streak_per_file", None)
            # Verdict word last: this is the noisiest single field across
            # rounds and belongs after everything the cache can plausibly
            # match.
            reflection_banner_lines.append(
                f"Verdict category (meta): {reflection_verdict['verdict']}"
            )
            # Fan-out directive (Fix #3): see ``_shared_root_cause_fanout``.
            # Logic lives in the helper so it can be unit-tested without
            # standing up a full repair_node fixture.
            _shared_root_causes = _shared_root_cause_fanout(
                state.get("compiler_errors", []) or [],
            )
            if _shared_root_causes:
                _code, _files_list = _shared_root_causes[0]
                _display_files = _files_list[:8]
                _extra = max(0, len(_files_list) - len(_display_files))
                _files_str = ", ".join(_display_files)
                if _extra:
                    _files_str += f" (+{_extra} more)"
                reflection_banner_lines.append(
                    f"FAN-OUT: error code {_code} appears across "
                    f"{len(_files_list)} files: {_files_str}. They share "
                    "ONE root cause. Emit patches for ALL of them in this "
                    "single response — patching one per round is treated "
                    "as no progress and will terminate the loop."
                )
            reflection_msg = "\n".join(reflection_banner_lines)
            if ignored_last_round and ignored_files_prev:
                touched_str = (
                    ", ".join(ignored_touched_prev[:6])
                    if ignored_touched_prev else "(none)"
                )
                reflection_msg = (
                    "=== YOU IGNORED THE JUDGE LAST ROUND ===\n"
                    f"The judge named {', '.join(ignored_files_prev)} as the "
                    f"blocker. Your patches instead touched: {touched_str}. "
                    "Cosmetic mock-target renames and unrelated tweaks will "
                    "cause this loop to terminate. Patch the named file(s) "
                    "this round — no exceptions.\n\n"
                    + reflection_msg
                )
            # Build the focused dispatch list. Skip messages[0] (the SRS),
            # keep small autofix system notes, keep the last assistant turn
            # so the model sees its own previous patch attempt as delta
            # context. apply_repair_iteration_cleanse already ran for
            # iter ≥ 2, so ``messages`` is mostly minimal — we just need to
            # also drop the SRS itself, which it deliberately preserves.
            _focused: list[MessageDict] = [
                MessageDict(role="system", content=reflection_msg),
                MessageDict(role="system", content=(
                    "You are the repair LLM. Your only job this turn is to "
                    "fix the failing diagnostic the judge named above. The "
                    "spec/SRS has been intentionally trimmed for this turn "
                    "so it does not compete for your attention. Emit "
                    "patches in the block DSL only — no prose."
                )),
            ]
            _last_assistant_msg: Optional[MessageDict] = None
            for _m in reversed(messages):
                if _m.get("role") == "assistant":
                    _last_assistant_msg = _m
                    break
            if _last_assistant_msg is not None:
                _focused.append(_last_assistant_msg)
            # Preserve any non-SRS system messages already accumulated this
            # turn (autofix note, budget_warning). Skip messages[0] which
            # is the SRS and any system messages that look like full
            # planning prompts (>8000 chars heuristic — the SRS and the
            # original planning user message both exceed this; reflection
            # / autofix / status notes do not).
            for _idx, _m in enumerate(messages):
                if _idx == 0:
                    continue  # the SRS
                if _m.get("role") != "system":
                    continue
                content = str(_m.get("content", "") or "")
                if len(content) > 8000:
                    continue  # likely a planning blob
                _focused.append(_m)
            _before = len(messages)
            messages = _focused
            logger.info(
                "[repair_node] %s verdict — rebuilt repair messages: %d → %d "
                "(dropped SRS + chatter; led with judge banner; judge_files=%s, "
                "ignored_prior=%s).",
                reflection_verdict["verdict"], _before, len(messages),
                judge_named_files or "(none)", ignored_last_round,
            )
            try:
                from harness.observability import emit_event as _emit_lead
                _emit_lead(
                    "repair_reflection_promoted_to_lead",
                    verdict=reflection_verdict["verdict"],
                    judge_named_files=judge_named_files,
                    ignored_prior_round=ignored_last_round,
                    ignored_files_prev=ignored_files_prev,
                    messages_before=_before,
                    messages_after=len(messages),
                )
            except Exception:  # noqa: BLE001 — telemetry must not block
                pass
        elif (
            reflection_verdict is not None
            and reflection_verdict["verdict"] in {"DISTRACTION", "REGRESSION"}
            and reflection_verdict["real_blocker"]
            and _reflection_verdict_is_low_signal(reflection_verdict)
        ):
            # Fix B — the judge fell back to the "insufficient data" sentinel.
            # Promoting it to the banner gave the repair LLM no actionable
            # direction and cost a cache-prefix miss on the noise (session
            # 116667f5). Skip injection; let the ordinary error_summary do
            # the talking.
            logger.info(
                "[repair_node] Reflection verdict is low-signal "
                "('insufficient data' sentinel) — skipping banner "
                "injection (verdict=%s).",
                reflection_verdict["verdict"],
            )
            try:
                from harness.observability import emit_event as _emit_skip
                _emit_skip(
                    "repair_reflection_injection_skipped",
                    reason="low_signal_sentinel",
                    verdict=reflection_verdict["verdict"],
                    real_blocker=reflection_verdict["real_blocker"][:500],
                )
            except Exception:  # noqa: BLE001
                pass
        elif (
            reflection_verdict is not None
            and reflection_verdict["verdict"] in {"DISTRACTION", "REGRESSION"}
            and reflection_verdict["real_blocker"]
        ):
            logger.info(
                "[repair_node] Reflection real_blocker did not reference "
                "any file present in compiler_errors — skipping injection "
                "(verdict=%s, blocker=%s).",
                reflection_verdict["verdict"],
                reflection_verdict["real_blocker"][:200],
            )
            try:
                from harness.observability import emit_event as _emit_skip
                _emit_skip(
                    "repair_reflection_injection_skipped",
                    reason="no_file_intersection",
                    verdict=reflection_verdict["verdict"],
                    real_blocker=reflection_verdict["real_blocker"][:500],
                )
            except Exception:  # noqa: BLE001
                pass

        # Append the repair prompt first
        messages.append(MessageDict(role="user", content=repair_prompt))
        # Then append the strict format reminder (same as patching_node).
        # In change-request mode prepend the CR-N attribution rules so
        # repair patches also carry the marker comments.
        _REPAIR_FORMAT_REMINDER = _build_change_request_preamble(
            state, "patching"
        ) + """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the block syntax below. Do NOT include any explanations,
markdown code fences, or text outside the blocks. Your entire response must be parseable
as one or more blocks.

Patch blocks (emit these to fix the build):

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

File-read block (use SPARINGLY — strictly budgeted):

<<<READ_FILE>>>
file: path/to/file.ext
<<<END_READ_FILE>>>

READ_FILE budget: at most 2 READ_FILE rounds per repair iteration. The
harness resolves your reads and re-dispatches without consuming an
iteration — but only twice. A third READ_FILE WILL BE STRIPPED and you
will be forced to emit patches with whatever context you have. So read
only files you genuinely need to see and have not been shown; everything
else, patch directly.

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
            max_cycles=_resolve_max_continuation_cycles(state),
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
            parse_patch_blocks,
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

        # Forced-patch escape valve: if the post-cap response is STILL a
        # READ_FILE-only emission with no parseable patches, the model is
        # spiralling — it would otherwise get its READ_FILE stripped and
        # land 0 patches, tripping consecutive_zero. Spend one more
        # dispatch with an explicit "no more READ_FILE, patch now" prompt
        # so the model is forced to commit. Without this, a model that
        # correctly diagnosed the fix (recorded in reasoning_content) but
        # kept asking for files burns a full repair iteration on nothing.
        post_cap_reads = _parse_read_blocks(response.content)
        if post_cap_reads:
            stripped_preview = _strip_read_blocks(response.content)
            if not parse_patch_blocks(stripped_preview):
                logger.info(
                    "[repair_node] READ_FILE cap hit and post-cap response "
                    "carries %d more READ_FILE block(s) with no patches "
                    "(%s). Issuing forced-patch retry.",
                    len(post_cap_reads),
                    [r[0] for r in post_cap_reads],
                )
                messages.append(MessageDict(
                    role="assistant", content=response.content,
                ))
                messages.append(MessageDict(
                    role="user",
                    content=(
                        "[Forced-patch] Your READ_FILE budget for this "
                        "repair iteration is exhausted. The harness will "
                        "IGNORE any further READ_FILE blocks. Based on "
                        "the files you have already been shown above, "
                        "emit your best-guess patches NOW using "
                        "REPLACE_BLOCK / CREATE_FILE / DELETE_BLOCK / "
                        "INSERT_AT_BLOCK only. If you are uncertain, "
                        "favour the narrowest plausible fix — a "
                        "near-miss patch is more useful than zero "
                        "patches. No prose, no READ_FILE, no markdown."
                    ),
                ))
                response, new_budget = await _dispatch_repair(messages, new_budget)
                try:
                    from harness.observability import emit_event
                    emit_event(
                        "repair_forced_patch_retry",
                        unresolved_read_files=[r[0] for r in post_cap_reads],
                        total_repairs=loop_counter.get("total_repairs", 0),
                    )
                except Exception:  # noqa: BLE001 — telemetry must not block
                    pass
        # If the LLM is still emitting READ_FILE after the forced retry,
        # ignore them (strip below) and let the rest of the response apply.

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
        # Phase 1.2 — capture PROMOTE_DEFERRED escape-hatch requests.
        # Parse, strip from the patch payload, stash on state for the
        # NEXT round's formatter to consume. Promotion only lasts one
        # round (single-use credit), which keeps the budget bounded.
        try:
            from harness.patcher import (
                parse_promote_deferred_blocks as _parse_promote_blocks,
                strip_promote_deferred_blocks as _strip_promote_blocks,
            )
            promoted_for_next = _parse_promote_blocks(patch_payload)
            patch_payload = _strip_promote_blocks(patch_payload)
            if promoted_for_next:
                logger.info(
                    "[repair_node] PROMOTE_DEFERRED captured for next round: %s",
                    promoted_for_next,
                )
                try:
                    from harness.observability import emit_event
                    emit_event(
                        "promote_deferred_request",
                        codes=list(promoted_for_next),
                        total_repairs=loop_counter.get("total_repairs", 0),
                    )
                except Exception:  # noqa: BLE001 — telemetry must not block
                    pass
        except Exception:  # noqa: BLE001 — escape-hatch parsing must not block patches
            logger.exception(
                "[repair_node] PROMOTE_DEFERRED capture failed; continuing without."
            )
            promoted_for_next = []
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
        # Phase 2.1 — decision-point logging. Allowlist refusals are a
        # quiet failure mode: the LLM's patch never lands, the repair
        # round burns a slot, and unless you grep for "Rejected by
        # allowlist" the loss is invisible. One structured event per
        # rejection lets a post-mortem reconstruct exactly what files
        # the harness refused to let the LLM touch.
        if allowlist_rejections:
            try:
                from harness.observability import emit_event as _emit_drop
                _emit_drop(
                    "dropped_from_prompt",
                    site="patcher_allowlist_rejection",
                    dropped_count=len(allowlist_rejections),
                    kept_count=success_count,
                    reason="path_not_in_patcher_allowlist",
                    examples=[
                        {"file": r["file"], "operation": r["operation"]}
                        for r in allowlist_rejections[:5]
                    ],
                )
            except Exception:  # noqa: BLE001
                pass
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
        # See patching_node — mirror the all-allowlist-rejected counter so
        # the same short-circuit fires whether the rejection happened in
        # the initial patching pass or in a downstream repair round.
        # `fail_count` here excludes no-ops (real_success_count subtracted
        # already), so equality with allowlist_rejections is exact.
        _non_success = len(patch_results) - success_count
        all_rejected_by_allowlist_repair = (
            real_success_count == 0
            and len(patch_results) > 0
            and len(allowlist_rejections) == _non_success
        )
        if all_rejected_by_allowlist_repair:
            loop_counter["consecutive_all_allowlist_rejected_rounds"] = (
                loop_counter.get("consecutive_all_allowlist_rejected_rounds", 0) + 1
            )
        else:
            loop_counter["consecutive_all_allowlist_rejected_rounds"] = 0

        # Persistent-blocker persistence (Fixes #2/#3). Save the
        # ``(file, line)`` tuples the judge named this round so the NEXT
        # round's banner can detect two-round persistence and inject a
        # hard "your patch MUST alter line N" directive. Save only on
        # DISTRACTION/REGRESSION (a persistent stuck state); clear on
        # PROGRESS so stale locations from earlier failures don't leak.
        # Serialise as list-of-list so JSON checkpointing round-trips
        # cleanly.
        if (
            reflection_verdict is not None
            and reflection_verdict["verdict"] in {"DISTRACTION", "REGRESSION"}
            and not _reflection_verdict_is_low_signal(reflection_verdict)
        ):
            _named_file_lines_now = _verdict_named_file_lines(
                reflection_verdict, state.get("compiler_errors", []) or []
            )
            if _named_file_lines_now:
                loop_counter["judge_named_file_lines_last_round"] = [
                    [f, ln] for (f, ln) in _named_file_lines_now
                ]
            else:
                loop_counter.pop("judge_named_file_lines_last_round", None)
        else:
            loop_counter.pop("judge_named_file_lines_last_round", None)

        # Judge-ignored gate (Fix #3). When the reflection verdict named
        # specific files as the real blocker, check that the round's
        # patches actually touched at least one of them. If not, the
        # round is a structural distraction even if the next reflection
        # verdict somehow comes back PROGRESS — record it so the NEXT
        # round's reflection banner can lead with "YOU IGNORED THE JUDGE"
        # and force the repair LLM back onto the blocker. Match by path
        # suffix / basename so judge-named ``services/edgar.py`` and
        # touched ``server/services/edgar.py`` both count.
        if (
            reflection_verdict is not None
            and reflection_verdict["verdict"] in {"DISTRACTION", "REGRESSION"}
            and not _reflection_verdict_is_low_signal(reflection_verdict)
        ):
            _judge_named_now = _verdict_named_files(
                reflection_verdict, state.get("compiler_errors", []) or []
            )
            if _judge_named_now:
                _touched_judge_files = _patches_touched_judge_files(
                    patch_results, _judge_named_now,
                )
                # Soft-pass: if no SUCCESS touched a judge file but at least
                # one ATTEMPTED patch did, treat as mechanical failure (not
                # distraction). The patcher's per-file rejection diagnosis
                # already tells the LLM why its REPLACE_BLOCK missed; piling
                # on a judge-ignored banner makes the LLM flip between two
                # contradictory directives. See session 7c30bce2: round 4
                # patched client/package.json AND attempted client/src/App.tsx
                # (quote-style search miss) — the attempt on the right file
                # should have counted as compliance.
                _attempted_judge_files = (
                    _touched_judge_files
                    or _patches_touched_judge_files(
                        patch_results, _judge_named_now, include_attempts=True,
                    )
                )
                if not _attempted_judge_files:
                    loop_counter["judge_ignored_last_round"] = True
                    loop_counter["judge_named_files_last_round"] = _judge_named_now
                    loop_counter["judge_round_touched_files"] = sorted({
                        r.file for r in patch_results
                        if getattr(r, "success", False)
                        and not getattr(r, "no_op", False)
                        and r.file
                    })
                    logger.warning(
                        "[repair_node] Judge-ignored: round %d patched %s "
                        "but judge named %s. Next round's banner will "
                        "escalate.",
                        loop_counter.get("total_repairs", 0),
                        loop_counter["judge_round_touched_files"] or "(none)",
                        _judge_named_now,
                    )
                    try:
                        from harness.observability import emit_event as _emit_ji
                        _emit_ji(
                            "repair_judge_ignored",
                            iteration=loop_counter.get("total_repairs", 0),
                            judge_named_files=_judge_named_now,
                            touched_files=loop_counter[
                                "judge_round_touched_files"
                            ],
                            verdict=reflection_verdict["verdict"],
                        )
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    if not _touched_judge_files:
                        logger.info(
                            "[repair_node] Judge-attempted: round %d had no "
                            "SUCCESS on judge-named %s, but an attempt was "
                            "recorded — soft-passing as compliance (patcher "
                            "rejection diagnosis carries the mechanical "
                            "fix-up directive).",
                            loop_counter.get("total_repairs", 0),
                            _judge_named_now,
                        )
                    # Clear stale flags — this round complied with the judge.
                    loop_counter.pop("judge_ignored_last_round", None)
                    loop_counter.pop("judge_named_files_last_round", None)
                    loop_counter.pop("judge_round_touched_files", None)
        else:
            # PROGRESS verdict or no verdict — clear any stale flags so the
            # next round doesn't get a phantom "you ignored the judge"
            # banner from an earlier failure that's since been addressed.
            loop_counter.pop("judge_ignored_last_round", None)
            loop_counter.pop("judge_named_files_last_round", None)
            loop_counter.pop("judge_round_touched_files", None)

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

        # Layer 3 — global no-progress failsafe. Mirror the patching_node
        # call so route_after_compiler can escalate to HITL when repair
        # has been burning budget without landing real patches. Uses
        # ``real_success_count`` (excludes idempotency no-ops) to match
        # the existing ``consecutive_zero_patch_rounds`` semantics.
        from harness.no_progress import update_and_check as _np_update_and_check
        _np_update_and_check(
            loop_counter,
            budget_remaining_usd=new_budget,
            progress_made=(real_success_count > 0),
            threshold_usd=float(
                state.get("no_progress_budget_usd") or 1.50
            ),
        )

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
            "batch_modified_files": _extend_batch_scope(state, modified_files),
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            # Phase 1.2 — single-round promotion credit. The codes the
            # LLM requested in this round's response surface on the NEXT
            # round's diagnostic formatter, then get cleared. If empty
            # (no request was made), this still overwrites any prior
            # round's list — which is intentional: each request is
            # consumed-on-use, not stacked.
            "promoted_codes_next_round": list(promoted_for_next),
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


def _infer_hitl_trigger(state: AgentState, *, max_repair: int) -> str:
    """Return a human-readable label for *why* the run is at HITL.

    Routers that escalate to HITL via a conditional edge
    (``security_scan_node``, ``route_after_patching``,
    ``route_after_compiler``) cannot mutate state, so they cannot set
    ``hitl_trigger`` themselves. This helper inspects the same loop
    counters and node_state flags the routers consulted and returns
    the most specific label that fits.

    Ordering matters — most-specific first. ``persistent_build_failure``
    stays last as the catch-all when ``exit_code != 0`` and nothing
    more precise matched; otherwise it would shadow the security /
    zero-patch / no-progress signals that all coexist with exit_code != 0.

    Returns ``"unknown"`` only when the state genuinely carries no
    distinguishing signal (e.g. an operator-driven manual HITL from a
    debugger).
    """
    from harness.no_progress import tripped as _np_tripped_inf

    loop_counter = state.get("loop_counter", {}) or {}
    node_state = state.get("node_state", {}) or {}
    budget_remaining = float(state.get("budget_remaining_usd", 0.0) or 0.0)

    sec_cfg_raw = state.get("security_scan_config", {}) or {}
    sec_cfg = sec_cfg_raw if isinstance(sec_cfg_raw, dict) else {}
    max_sec_attempts = int(sec_cfg.get("max_security_fix_attempts", 2))
    sec_attempts = int(loop_counter.get("security", 0) or 0)
    consecutive_zero = int(
        loop_counter.get("consecutive_zero_patch_rounds", 0) or 0
    )

    # v5 Phase 7 BUG #6: the end-of-session traceability gate
    # (installation_doc_node) sets node_state.traceability_blocked +
    # exit_code=1 when the audit fails. Must take precedence over
    # the generic persistent_build_failure catch-all so HITL UX and
    # outside-actions can route to the coverage-gap branch instead
    # of the "open failing files in your IDE" branch.
    if node_state.get("traceability_blocked"):
        return "traceability_block"
    if node_state.get("decomposition_failed"):
        return "decomposition_validation_failed"
    if node_state.get("decomposition_missing"):
        return "decomposition_missing"
    if node_state.get("env_misconfig"):
        sym = node_state.get("env_misconfig_symbol", "")
        return f"env_misconfig:{sym}" if sym else "env_misconfig"
    if node_state.get("build_command_cd_missing"):
        d = node_state.get("build_command_cd_missing_dir", "")
        return f"build_command_cd_missing:{d}" if d else "build_command_cd_missing"
    if budget_remaining <= 0.0:
        return "budget_exhausted"
    if _np_tripped_inf(loop_counter):
        return "no_progress_failsafe"
    if sec_attempts >= max_sec_attempts and max_sec_attempts > 0:
        return f"security_fix_limit:{sec_attempts}/{max_sec_attempts}"
    consecutive_all_rejected = int(
        loop_counter.get("consecutive_all_allowlist_rejected_rounds", 0) or 0
    )
    if consecutive_all_rejected >= 1:
        return f"all_allowlist_rejected:{consecutive_all_rejected}"
    if consecutive_zero >= 2:
        return f"zero_patch_loop:{consecutive_zero}"
    # Consecutive-DISTRACTION circuit breaker. The route gate compares
    # loop_counter["consecutive_distraction_rounds"] against
    # ``gw.config.max_consecutive_distraction_rounds`` — we reuse the
    # same threshold here so the HITL label matches the actual gate.
    _gw_for_inf = get_gateway()
    _distraction_cap = (
        int(getattr(_gw_for_inf.config, "max_consecutive_distraction_rounds", 3))
        if _gw_for_inf is not None else 3
    )
    consecutive_distraction = int(
        loop_counter.get("consecutive_distraction_rounds", 0) or 0
    )
    if consecutive_distraction >= _distraction_cap:
        return f"reflection_distraction_loop:{consecutive_distraction}"
    if int(loop_counter.get("total_repairs", 0) or 0) >= max_repair:
        return "repair_loop_limit"
    exit_code_raw = state.get("exit_code", -1)
    exit_code = int(exit_code_raw) if exit_code_raw is not None else -1
    if exit_code != 0:
        return "persistent_build_failure"
    return "unknown"


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

    # Determine why we were invoked. Trigger inference reads
    # loop_counter / node_state itself via _infer_hitl_trigger;
    # this function only needs budget_remaining for the LLM
    # escalation summary's budget arithmetic below.
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
    trigger_reason = _infer_hitl_trigger(state, max_repair=max_repair)

    # Inject trigger reason into state so the menu can display it
    state_dict: dict[str, Any] = dict(state)
    state_dict["node_state"] = dict(state_dict.get("node_state") or {})
    state_dict["node_state"]["current_node"] = "human_intervention"
    state_dict["node_state"]["hitl_trigger"] = trigger_reason
    state_dict["node_state"]["hitl_active"] = True
    state_dict["node_state"]["hitl_awaiting_input"] = True

    # LLM-judgment kill-switched escalation summary (#1). For loop-stuck
    # triggers — repair limit, persistent failure, or no-progress
    # tripwires — generate a one-paragraph operator briefing that
    # explains why the loop couldn't fix it and what to try manually.
    # The bare trigger string ("repair_loop_limit") tells the operator
    # the cap was hit; this tells them what the loop was actually doing
    # and why it failed. Triggers with already-precise messages
    # (budget_exhausted, env_misconfig:<symbol>) skip the call.
    summary_eligible = trigger_reason in {
        "repair_loop_limit", "persistent_build_failure",
    }
    if summary_eligible and gw is not None:
        prompt = _build_hitl_escalation_summary_prompt(state, trigger_reason)
        summary, new_budget = await _maybe_judgment_llm(
            prompt=prompt,
            budget_remaining_usd=budget_remaining,
            purpose="hitl_escalation_summary",
            enabled=bool(getattr(
                gw.config, "llm_judgment_hitl_escalation_summary", True,
            )),
        )
        state_dict["budget_remaining_usd"] = new_budget
        if summary:
            state_dict["node_state"]["hitl_escalation_summary"] = summary
            logger.info(
                "[human_intervention_node] LLM escalation summary attached (%d chars).",
                len(summary),
            )

    # Delegate to the CLI layer's interactive menu loop.
    # This blocks on stdin until the developer makes a choice.
    from harness.cli import hitl_menu_loop

    updated_state = hitl_menu_loop(state_dict)

    # Layer 3 — reset the no-progress failsafe on every HITL exit. The
    # operator just inspected the situation and chose to resume / suspend;
    # post-intervention spend should start with a fresh budget marker
    # rather than inheriting the pre-pause tally that triggered the
    # escalation. Without this reset, a single "resume" choice would
    # immediately re-trip the failsafe on the very next patching turn.
    try:
        from harness.no_progress import reset as _np_reset
        lc = dict(updated_state.get("loop_counter", {}) or {})
        _np_reset(lc)
        updated_state["loop_counter"] = lc
    except Exception as _np_exc:  # noqa: BLE001 — failsafe reset is non-fatal
        logger.debug(
            "[human_intervention_node] no-progress reset skipped: %s",
            _np_exc,
        )

    # Extract the node_state back — hitl_menu_loop returns a full state dict
    return updated_state


def _build_hitl_escalation_summary_prompt(
    state: AgentState, trigger_reason: str,
) -> str:
    """Compose a compact prompt for the HITL escalation summary call (#1).

    Pulls the latest compiler errors, the last patcher rejections /
    patch failures, the modified-files inventory, and the loop counters
    so the LLM sees the same evidence the operator would have to dig out
    of the logs. Bounded so the prompt stays in the cheap-model budget.
    """
    node_state = state.get("node_state", {}) or {}
    loop_counter = state.get("loop_counter", {}) or {}
    errors = state.get("compiler_errors", []) or []
    rejections = node_state.get("allowlist_rejections") or []
    patch_failures = node_state.get("patch_failures") or []
    last_build_output = str(node_state.get("last_build_output", "") or "")
    modified_files = sorted({p for p in (state.get("modified_files") or []) if p})

    err_lines: list[str] = []
    for err in errors[:8]:
        # Per-error message cap. The previous 200-char cap silently
        # truncated chained "Caused by:" stacks (uv, cargo, gradle) at
        # the FIRST cause — in session db6bfcbe it cut the uv wheel-
        # install failure at literally the word "Caused", so the
        # summariser hallucinated a different root cause. 1500 is wide
        # enough to keep the full toolchain stack while still bounding
        # the prompt at 8 * 1500 = 12k for the eight-error worst case.
        err_lines.append(
            f"- {err.get('error_code', '?')} {err.get('file', '?')}:"
            f"{err.get('line', 0)} — "
            f"{str(err.get('message', ''))[:1500]}"
        )
    err_block = "\n".join(err_lines) if err_lines else "(no structured diagnostics)"

    rej_paths = sorted({str(r.get("file", "")) for r in rejections if r.get("file")})
    rej_block = (
        ", ".join(rej_paths[:10]) if rej_paths else "(none)"
    )

    pf_lines: list[str] = []
    for pf in patch_failures[:5]:
        if isinstance(pf, dict):
            pf_lines.append(
                f"- {pf.get('operation', '?')} on {pf.get('file', '?')}: "
                f"{str(pf.get('reason', '') or pf.get('error', ''))[:160]}"
            )
    pf_block = "\n".join(pf_lines) if pf_lines else "(none)"

    inv_block = (
        ", ".join(modified_files[:15])
        + (f" (+{len(modified_files) - 15} more)" if len(modified_files) > 15 else "")
        if modified_files else "(none)"
    )

    # First-failure context. When the build's failure mode changes
    # across repair rounds (e.g. round 1 = build command refused by the
    # security validator → round 4 = downstream uv install failure),
    # showing the summariser ONLY the most recent tail produces wrong
    # post-mortems. Pulled from compiler_node's snapshot frozen on the
    # first non-zero-exit round; skipped when it duplicates the most
    # recent output (no failure-mode shift). See session db6bfcbe.
    first_output = str(node_state.get("first_failure_build_output", "") or "")
    first_cmd = str(node_state.get("first_failure_build_command", "") or "")
    first_round = node_state.get("first_failure_round")
    first_errors = node_state.get("first_failure_compiler_errors") or []
    first_block: str = ""
    if first_output and first_output != last_build_output:
        first_err_lines: list[str] = []
        for err in first_errors[:5]:
            if isinstance(err, dict):
                first_err_lines.append(
                    f"- {err.get('error_code', '?')} {err.get('file', '?')}:"
                    f"{err.get('line', 0)} — "
                    f"{str(err.get('message', ''))[:200]}"
                )
        first_err_block = (
            "\n".join(first_err_lines)
            if first_err_lines else "(no structured diagnostics)"
        )
        first_block = (
            "\nFIRST-ROUND failure (round "
            f"{first_round if first_round is not None else '?'}; "
            "may differ from the recent tail above if the failure mode "
            "shifted mid-session):\n"
            f"  Build command: {first_cmd or '(unknown)'}\n"
            f"  Diagnostics:\n{first_err_block}\n"
            f"  Tail of first build output:\n"
            f"{first_output[-800:]}\n"
        )

    return (
        "You are summarising why the harness's build → repair loop has stopped "
        "making progress and is handing off to a human operator. Produce ONE "
        "short paragraph (4–6 sentences, no markdown headers, no bullet "
        "lists) that:\n"
        "  1. Names the root cause in concrete terms (which file/symbol/dep, "
        "not just 'tests fail').\n"
        "  2. States why the repair loop could not fix it (e.g. patcher kept "
        "rejecting the path, autofix ran out, same symptom recurred).\n"
        "  3. Recommends the single most likely manual fix.\n"
        "Be specific: cite filenames, missing symbols, or rejected paths from "
        "the evidence below. Do NOT paraphrase the trigger reason; the "
        "operator already sees it. When the FIRST-ROUND failure differs from "
        "the recent tail, the FIRST-ROUND error is almost always the real "
        "root cause — later rounds just expose downstream symptoms once the "
        "original problem is bypassed or papered over.\n\n"
        f"Trigger: {trigger_reason}\n"
        f"Repair iterations spent: {loop_counter.get('total_repairs', 0)}\n"
        f"Consecutive zero-patch rounds: {loop_counter.get('consecutive_zero_patch_rounds', 0)}\n"
        f"Build exit code: {state.get('exit_code', -1)}\n\n"
        f"Recent compiler errors:\n{err_block}\n\n"
        f"Recently rejected patch paths (allowlist): {rej_block}\n\n"
        f"Recent patcher rejections (other):\n{pf_block}\n\n"
        f"Workspace files touched this session: {inv_block}\n\n"
        f"Tail of last build output (truncated):\n"
        f"{last_build_output[-4000:] if last_build_output else '(empty)'}"
        f"{first_block}"
    )
    # Phase 2.1 — decision-point logging. The tail truncation hides the
    # FIRST stack frames of multi-stage builds (webpack's initial import
    # resolution error, javac's first cannot-find-symbol, etc.) when the
    # log is long. Emit a structured event so post-mortems can tell
    # whether the truncation chopped the signal vs. genuinely had
    # nothing important earlier in the log.
    if last_build_output and len(last_build_output) > 4000:
        try:
            from harness.observability import emit_event as _emit_drop
            _emit_drop(
                "dropped_from_prompt",
                site="hitl_summary_build_log_truncation",
                dropped_count=len(last_build_output) - 4000,
                kept_count=4000,
                reason="hitl_summary_keeps_last_4000_chars",
                full_log_size=len(last_build_output),
            )
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 5. Helper Utilities
# ---------------------------------------------------------------------------

# Phase 3(a) — message normalization for fingerprinting. Many compilers
# (TypeScript especially) embed concrete types and identifiers inside the
# error message itself, not just in the location. The same morally
# identical bug at rounds N and N+1 can produce different raw messages
# because a partial fix changed one of the types in the inferred chain.
# Example: ``Type 'string[]' is not assignable to type 'number[]'`` →
# ``Type 'string[]' is not assignable to type 'boolean[]'`` after a fix
# attempt that swapped one concrete to another. Without normalization,
# the fingerprint changes, Layer 3's survival promotion misses it, and
# the cascade prior (which dropped it last round) keeps dropping it.
# We strip:
#   * single-quoted spans  ('foo', 'string[]', 'Foo<Bar>')  →  '*'
#   * back-quoted spans   (`foo`, `Foo`)                    →  `*`
#   * double-quoted spans ("foo")                           →  "*"
# Numeric tokens with no semantic value (line/col numbers occasionally
# inlined in messages) are NOT stripped — they're rare in error messages
# (location is a separate field) and stripping them risks collapsing
# distinct errors that genuinely differ by line.
_FINGERPRINT_QUOTED_SPAN = re.compile(r"'[^']{1,200}'|`[^`]{1,200}`|\"[^\"]{1,200}\"")


def _normalize_diagnostic_message(msg: str) -> str:
    """Normalise a diagnostic message into a fingerprint-stable form.

    Replaces concrete identifiers and type spellings inside quotes with
    a generic placeholder so the message shape (which represents the
    error class) stays stable across rounds even when the LLM's partial
    fix swapped one type/symbol for another. See module-level docstring
    of _FINGERPRINT_QUOTED_SPAN for the failure mode this addresses.

    Whitespace is collapsed to single spaces so subtle reformatting
    (e.g. compiler v1 prints ``X is not assignable to Y`` vs v2
    ``X\nis not assignable to\nY``) doesn't create false divergence.
    """
    if not msg:
        return ""
    collapsed = _FINGERPRINT_QUOTED_SPAN.sub("'*'", msg)
    return " ".join(collapsed.split())


def _fingerprint_diagnostics(
    errors: list[DiagnosticObjectDict],
) -> list[str]:
    """Return a sorted, deduped list of ``"<code>::<normalised_message>"``
    fingerprints for error-severity diagnostics in ``errors``.

    Used by compiler_node to stash the current round's diagnostic shape so
    repair_node can detect which groups survived into the next round and
    promote them past the cascade-ranking prior. Warnings are excluded —
    they don't enter the repair prompt and shouldn't pollute the survival
    set. Phase 3(a): the message component is normalised via
    ``_normalize_diagnostic_message`` so a partial fix that swaps one
    concrete type for another doesn't break the survival match.
    """
    out: set[str] = set()
    for err in errors:
        if str(err.get("severity", "error")).lower() == "warning":
            continue
        code = str(err.get("error_code", "UNKNOWN"))
        msg = _normalize_diagnostic_message(
            str(err.get("message", "No message"))
        )
        out.add(f"{code}::{msg}")
    return sorted(out)


def _rotate_diag_fingerprints_delta(
    state: AgentState,
    diagnostics: list[Any],
) -> dict[str, Any]:
    """Return the state-fragment keys any node must merge into its return
    dict whenever it populates ``compiler_errors`` and routes toward
    ``repair_node``. Rotates ``last_diag_fingerprints`` →
    ``prior_diag_fingerprints`` and derives the new "current" from
    ``diagnostics``.

    Without this rotation the reflection judge in repair_node sees a
    stale "current" (typically ``[]`` from the last green build) and
    hallucinates a PROGRESS verdict, resetting the
    ``consecutive_distraction_rounds`` circuit-breaker and letting the
    repair loop run forever. Regression fix for session 7e4cba32 where
    the prod-smoke short-circuit hit exactly this class.

    Any new short-circuit / gate that emits ``compiler_errors`` should
    merge this helper's result into its return; the alternative
    (silently letting the judge see stale state) is a Class-2 bug
    waiting to be filed.
    """
    return {
        "prior_diag_fingerprints": list(
            state.get("last_diag_fingerprints") or []
        ),
        "last_diag_fingerprints": _fingerprint_diagnostics(diagnostics),
        "prior_diag_count": int(state.get("last_diag_count") or 0),
        "last_diag_count": sum(
            1 for e in diagnostics
            if str(e.get("severity", "error")).lower() != "warning"
        ),
    }


def _format_structured_diagnostic_payload(
    errors: list[DiagnosticObjectDict],
    *,
    max_total: int = 25,
) -> str:
    """Phase 4 — produce a structured JSON block of every diagnostic (up
    to ``max_total``) so the LLM can sort/filter/group itself.

    The cascade-defense layers expose the harness's *suggested* ordering
    via the markdown summary; this block gives the LLM the *raw* data
    to override that ordering if it disagrees. Caps at ``max_total`` to
    bound token cost (250-error builds would otherwise pay an extra
    50-100k tokens of JSON). When the cap is hit, an ``"_truncated"``
    field announces it so the LLM knows there's more not shown.

    Returns the empty string when there's nothing to serialise — the
    caller will skip appending an empty section.
    """
    if not errors:
        return ""
    payload_items: list[dict[str, Any]] = []
    for err in errors[:max_total]:
        item = {
            "file": str(err.get("file", "?")),
            "line": int(err.get("line", 0) or 0),
            "column": int(err.get("column", 0) or 0),
            "code": str(err.get("error_code", "?")),
            "severity": str(err.get("severity", "error")),
            "message": str(err.get("message", "")),
        }
        ctx = (err.get("semantic_context") or "").strip()
        if ctx:
            item["semantic_context"] = ctx[:1000]
        sym = (err.get("missing_symbol") or "").strip()
        if sym:
            item["missing_symbol"] = sym
        payload_items.append(item)
    payload: dict[str, Any] = {"diagnostics": payload_items}
    if len(errors) > max_total:
        payload["_truncated"] = {
            "shown": max_total,
            "total": len(errors),
            "reason": (
                f"structured payload capped at {max_total} items to bound "
                "token cost; see the markdown summary above for the "
                "harness's prioritised view of the rest."
            ),
        }
    body = json.dumps(payload, indent=2, default=str)
    return (
        "\n\n### Structured payload (for programmatic processing)\n"
        "_The same diagnostics in machine-readable form. If you want to "
        "sort or filter differently than the harness's cascade ranking "
        "above, use this view. Both views show the same underlying data._\n"
        f"```json\n{body}\n```"
    )


def _format_diagnostics_for_repair(
    errors: list[DiagnosticObjectDict],
    *,
    prior_fingerprints: Optional[set[str]] = None,
    promoted_codes: Optional[set[str]] = None,
    emit_structured_payload: bool = True,
) -> str:
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

    # Tag each group with its survival status from the prior round so the
    # ranker can promote persisted groups past the cascade prior. Layer 3
    # of the cascade-defense hierarchy — see the function-level note above.
    # Phase 3(a): the lookup uses the SAME normalisation as
    # ``_fingerprint_diagnostics`` so a partial fix that swapped one
    # concrete type for another still matches the prior round's group.
    _prior: set[str] = prior_fingerprints or set()
    # LLM-requested promotion (Phase 1.2). The model emits
    # ``<<<PROMOTE_DEFERRED>>>`` with codes; repair_node captures them and
    # passes them here. These rank ABOVE survival promotion: an explicit
    # request from the model that just saw the prompt is the strongest
    # signal possible. Normalised to upper-case so ``ts2769`` matches
    # ``TS2769``.
    _promoted: set[str] = {c.upper() for c in (promoted_codes or set())}
    for group in groups.values():
        normalized_fp = (
            f"{group['code']}::"
            f"{_normalize_diagnostic_message(str(group['message']))}"
        )
        group["persisted"] = normalized_fp in _prior
        group["llm_promoted"] = (
            str(group["code"]).upper() in _promoted
        )

    # Rank groups by likely cascade impact so the top-N the LLM sees are the
    # ones whose fix is most likely to make other errors disappear.
    # Heuristic order (lower rank = higher priority, sorted ascending):
    #   1. Survival rank — groups that persisted from the previous repair
    #      round are unconditionally promoted. Empirical evidence beats the
    #      cascade prior: if a group survived the last round, the prior was
    #      wrong about it cascading away. This is the fix for the failure
    #      mode where TS2769 / TS2353 / any non-upstream error was deferred
    #      for 3 rounds, never got top-N context, and burned the budget.
    #   2. Severity error > warning (warnings don't break the build).
    #   3. "Upstream" kinds first — undefined names, missing imports, missing
    #      deps. F821/F401 are pyflakes-shaped; TS2304/TS2305/TS2307 are
    #      the TypeScript equivalents (cannot-find-name / missing-export /
    #      cannot-find-module); ImportError / ModuleNotFoundError surface
    #      from pytest. Fix one of these and multiple downstream diagnostics
    #      often vanish at once.
    #   4. Original first-seen position breaks ties so the LLM still gets a
    #      stable order across iterations.
    _UPSTREAM_PREFIXES = (
        # Python — pyflakes / pylint / pytest collection
        "F821", "F401", "E0001", "E0401", "E0602", "E1101",
        # TypeScript — tsc absence-shaped diagnostics
        "TS2304", "TS2305", "TS2307", "TS2552", "TS2459",
        # Phase 3(c) — per-language absence-shaped codes from supported
        # parsers. Each language's "X cannot be found" / "no such Y" /
        # "module not declared" errors get the cascade-prior bump that
        # F821 et al. already enjoy. Add new prefixes here as you add
        # parser support; sourced from observed compiler/linter output
        # rather than allowlist guesses.
        # Java — javac
        "JAVA:CANNOT_FIND_SYMBOL", "JAVA:PACKAGE_DOES_NOT_EXIST",
        # Generic miss markers from the harness's own parsers / runners
        "MISSING_DEP", "MISSING_IMPORT", "IMPORTERROR", "MODULENOTFOUND",
        "SYNTAXERR", "TEST_FAILURE:IMPORTERROR",
    )

    def _llm_promoted_rank(g: dict[str, Any]) -> int:
        return 0 if g.get("llm_promoted") else 1

    def _survival_rank(g: dict[str, Any]) -> int:
        return 0 if g.get("persisted") else 1

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
        key=lambda pair: (
            _llm_promoted_rank(pair[1]),
            _survival_rank(pair[1]),
            _severity_rank(pair[1]),
            _kind_rank(pair[1]),
            pair[0],
        ),
    )
    # Layer 2 — small-N short-circuit. Deferring a group strips its full
    # source context from the prompt; that's only a worthwhile token saving
    # when there are enough groups that the context bloat would matter. At
    # ≤ 5 distinct shapes, show all of them with full context. This handles
    # the late-iteration "small surviving tail" case where the LLM has
    # cleared most diagnostics and is being asked to finish off a handful,
    # without the deferral mechanism stealing context from any of them.
    TOP_N = 3
    _SMALL_N_THRESHOLD = 5
    if len(group_list) <= _SMALL_N_THRESHOLD:
        shown = [g for _, g in ranked]
        hidden: list[dict[str, Any]] = []
    else:
        shown = [g for _, g in ranked[:TOP_N]]
        hidden = [g for _, g in ranked[TOP_N:]]

    lines: list[str] = [
        f"## Compiler Diagnostics ({len(errors)} total, "
        f"{len(groups)} distinct shape{'s' if len(groups) != 1 else ''})\n"
    ]
    if hidden:
        # Layer 0 — honest wording. The previous text claimed deferred
        # groups "may resolve on their own", which the LLM took at face
        # value and consistently ignored them. They didn't resolve; the
        # repair budget exhausted; HITL fired. The replacement framing
        # tells the truth: deferred items aren't sentenced, just lower
        # priority for THIS round's context budget. If they survive into
        # the next round, Layer 3 will promote them.
        lines.append(
            f"_Showing the top {len(shown)} of {len(groups)} groups ranked by "
            f"likely cascade impact (fixing an upstream undefined-name / "
            f"missing-import often resolves multiple downstream errors). "
            f"Full source context is shown for these. The {len(hidden)} "
            f"deferred group(s) are listed for awareness — address them "
            f"too in the same response if the fix is unambiguous from "
            f"the message alone; otherwise focus on the top groups and "
            f"the deferred items will be promoted to top context next "
            f"iteration if they persist._\n"
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
        # Layer 0 — mark persisted groups in the header so the LLM knows
        # exactly which item the harness had to bump past the cascade
        # prior. Signals "this is the one we missed last round, don't
        # miss it again." LLM-promoted groups (Phase 1.2 escape hatch)
        # get a distinct tag so the model knows its request was honored.
        tags: list[str] = []
        if group.get("llm_promoted"):
            tags.append("**[promoted at your request]**")
        if group.get("persisted"):
            tags.append("**[persisted from previous round]**")
        tag_block = (" " + " ".join(tags)) if tags else ""
        lines.append(
            f"**Error {i}:** `{group['code']}` × {count} "
            f"[{group['severity']}]{tag_block}"
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
        # Phase 2.1 — decision-point logging. Emit a structured event
        # naming exactly which groups got deferred from full context, so
        # post-mortem investigations can grep one event name and see
        # whether the harness's prompt-shaping starved the model of
        # signal. Would have caught session 6cf20a5d's TS2769 deferral
        # in one grep instead of an hour of debugging.
        try:
            from harness.observability import emit_event as _emit_drop
            _emit_drop(
                "dropped_from_prompt",
                site="deferred_diagnostics",
                dropped_count=len(hidden),
                kept_count=len(shown),
                reason="cascade_rank_topN_cutoff",
                examples=[
                    {
                        "code": g["code"],
                        "message_excerpt": (g["message"][:80] + "...") if len(g["message"]) > 80 else g["message"],
                        "occurrences": len(g.get("locations", [])),
                    }
                    for g in hidden[:5]
                ],
            )
        except Exception:  # noqa: BLE001 — telemetry must not block
            pass
        # One-line summary of the deferred groups so the LLM knows they exist
        # without bloating the prompt with full context blocks. Each entry
        # lists code + a short message excerpt + count.
        tail_lines = [
            "",
            f"### Deferred ({len(hidden)} group(s) — top {len(shown)} have priority for this round):",
        ]
        for g in hidden:
            msg = g["message"]
            if len(msg) > 80:
                msg = msg[:77] + "..."
            tail_lines.append(
                f"- `{g['code']}` × {len(g['locations'])}: {msg}"
            )
        # Phase 1.2 — advertise the escape hatch. The LLM can override
        # this round's cascade ranking by emitting a PROMOTE_DEFERRED
        # block; the codes named get full ``semantic_context`` next round.
        tail_lines.extend([
            "",
            "**If you believe a deferred group above is actually the "
            "blocker** (e.g. it's a type-mismatch the cascade prior "
            "underestimates, not a downstream cascade victim), emit "
            "this block alongside your patches to force full source "
            "context on the next round:",
            "```",
            "<<<PROMOTE_DEFERRED>>>",
            "codes: TS2769, F401",
            "<<<END_PROMOTE_DEFERRED>>>",
            "```",
            "The named codes get top-N treatment regardless of cascade "
            "rank. Use sparingly — promotion costs context budget on "
            "the next round.",
        ])
        lines.extend(tail_lines)
    rendered = "\n".join(lines)
    # Phase 4 — append a structured JSON block of every diagnostic so
    # the LLM can sort/filter/group on its own if it disagrees with the
    # harness's cascade ranking. Capped + opt-out via config so the
    # token cost is bounded.
    if emit_structured_payload:
        rendered += _format_structured_diagnostic_payload(errors)
    return rendered


# ---------------------------------------------------------------------------
# 6. Conditional Edge: Compiler → Next Node
# ---------------------------------------------------------------------------

def route_after_compiler(state: AgentState) -> Literal["repair_node", "human_intervention_node", "security_scan_node", "test_generation_node"]:
    """
    Conditional edge router executed after compiler_node completes.

    Decision matrix (in priority order; the first matching row fires):
        budget_remaining_usd <= 0                            → HITL
        exit_code == 0                                       → security_scan / code_review
                                                               (dict aliases ``security_scan_node`` to ``code_review_node``)
        env_misconfig flag set                               → HITL
        llm_silent flag set                                  → HITL
        no_tests_collected AND has_source                    → test_generation_node
        no_tests_collected AND empty repo                    → HITL
        consecutive_zero_patch_rounds >= 2 (non-autofixable) → HITL
        consecutive_zero_patch_rounds >= 5 (generic)         → HITL
        total_repairs >= max_iterations (non-autofixable)    → HITL
        same MISSING_DEP symbol >= 3 in a row                → HITL
        total_repairs >= max_iterations (autofixable)        → repair_node (autofix bypass)
        otherwise (build failed, within budget)              → repair_node

    Counter semantics (Phase E.3+):
        - In batch-mode (``current_batch_id > 0``), ``total_repairs``
          tallies repair cycles for the CURRENT BATCH's verification
          chain. ``batch_commit_node`` resets it between batches.
        - In monolithic mode (no batch), ``total_repairs`` is per-session.
        - ``consecutive_zero_patch_rounds`` is per-compile-cycle in
          both modes — it tracks rounds where the patcher landed
          nothing, regardless of how stories/batches are organized.

    ``max_iterations`` is read from
    ``gateway.config.max_patch_repair_iterations`` (config-driven,
    default 5; the historical default of 3 only applies when the router
    is invoked in a test that bypasses gateway initialization).
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

    # Layer 3 — global no-progress failsafe. Comes before the success
    # path so that a stuck repair loop can't camouflage itself with an
    # eventual clean compile; but yields to the budget check above which
    # is the harder financial gate.
    from harness.no_progress import tripped as _np_tripped
    if _np_tripped(loop_counter):
        logger.error(
            "[router] no-progress failsafe tripped — budget bleeding "
            "without successful patches. Routing to HITL."
        )
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

    # Sandbox CommandValidator refused the build command (e.g. ``cd`` or
    # ``bash`` not in security.allowed_commands). The validator's config is
    # global (~/.harness/...) and the patcher allowlist cannot write
    # there — three rounds of repair would burn the budget producing
    # patches that get rejected at the allowlist (session db6bfcbe).
    # Route straight to HITL with the matched rule so the operator can
    # adjust the policy and resume.
    if state.get("node_state", {}).get("build_command_blocked"):
        rule = state.get("node_state", {}).get("build_command_blocked_rule", "")
        logger.warning(
            "[router] Build command blocked by sandbox security validator "
            "(rule=%s). Repair loop cannot reach the global validator config. "
            "Routing to HITL.",
            rule,
        )
        return _transition("human_intervention_node")

    # Build command references `cd <dir>` but <dir> doesn't exist and the
    # compiler_node auto-rewire (Fixes 2 & 3) couldn't repoint it at a
    # spec-approved root. Every repair round the LLM emits will either
    # target the missing dir (allowlist rejects → 0 patches applied) or
    # give up. The rewire is a config-level fix — operator must edit the
    # build_command or the workspace layout, not the code. Escalate to
    # HITL immediately.
    if state.get("node_state", {}).get("build_command_cd_missing"):
        missing = state.get("node_state", {}).get("build_command_cd_missing_dir", "")
        logger.warning(
            "[router] Build command's `cd %s` failed and no auto-rewire "
            "target is available. Repair cannot fix build wiring. "
            "Routing to HITL.", missing,
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

    # Same-file REPLACE_BLOCK stuck-target tripwire. When any single file
    # racks up >=3 consecutive REPLACE_BLOCK misses — the LLM emitting
    # search blocks the patcher can't match against the on-disk content,
    # round after round — the directive at >=2 (_format_replace_block_miss_
    # directive) has demonstrably not unstuck the LLM. Burning further
    # iterations on the same file only piles up identical rejections. Go
    # to HITL with a specific reason: the operator can manually fix the
    # one file or accept that codegen of that file should be retried.
    #
    # Sized to fire AFTER the >=2 directive has had one full round to
    # work (LLM saw "use a different operation" prompt + the actual file
    # bytes, and still missed). Three is the smallest count where giving
    # up is more useful than retrying.
    _STUCK_TARGET_LIMIT = 3
    _rb_per_file = loop_counter.get("replace_block_misses_per_file") or {}
    if isinstance(_rb_per_file, dict):
        stuck_files = sorted(
            f for f, n in _rb_per_file.items()
            if isinstance(n, int) and n >= _STUCK_TARGET_LIMIT
        )
        if stuck_files:
            logger.warning(
                "[router] REPLACE_BLOCK stuck on file(s) %s (>=%d misses each). "
                "The LLM keeps emitting search blocks the patcher rejects against "
                "the on-disk content. Routing to HITL: the operator can edit the "
                "file by hand or restart codegen for it.",
                stuck_files, _STUCK_TARGET_LIMIT,
            )
            return _transition("human_intervention_node")

    # All-allowlist-rejected tripwire. This is a much sharper signal than
    # the generic zero-patch counter: the LLM emitted patches, all of them
    # targeted paths outside the patcher allowlist, and nothing landed.
    # Another round won't change that — the model is picking targets from
    # signals in its system prompt (build_command, stale examples) that
    # disagree with the allowlist, and only the operator can rewire that.
    # Escalate after 1 round instead of the generic 2- / 5-round waits so
    # we don't burn budget on a self-repeating rejection.
    consecutive_all_rejected = int(
        loop_counter.get("consecutive_all_allowlist_rejected_rounds", 0) or 0
    )
    if consecutive_all_rejected >= 1:
        logger.warning(
            "[router] %d consecutive round(s) where every patch was rejected "
            "by the allowlist. The LLM is targeting paths outside the "
            "configured layout; another round won't change that. Routing to "
            "HITL so the operator can widen the allowlist or rewire the "
            "signals the LLM is picking up.",
            consecutive_all_rejected,
        )
        return _transition("human_intervention_node")

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

    # Consecutive-DISTRACTION circuit breaker. ``repair_node`` ticks
    # ``consecutive_distraction_rounds`` for every reflection verdict in
    # {DISTRACTION, REGRESSION} and resets it on PROGRESS. When the count
    # saturates the configured cap, escalate to HITL — the judgment LLM
    # has told us N rounds running that the repair LLM isn't addressing
    # the real blocker, and the existing fingerprint-shrinkage gate
    # below can't catch this because shrinkage oscillates across rounds.
    # Gate is NOT bypassed by has_autofixable: a DISTRACTION verdict
    # already means the autofix path hasn't been advancing the blocker.
    max_distraction = (
        int(getattr(gw.config, "max_consecutive_distraction_rounds", 3))
        if gw is not None else 3
    )
    consecutive_distraction = int(
        loop_counter.get("consecutive_distraction_rounds", 0) or 0
    )
    if consecutive_distraction >= max_distraction:
        logger.warning(
            "[router] %d consecutive reflection verdict(s) flagged "
            "DISTRACTION/REGRESSION (cap=%d). The judgment LLM has "
            "repeatedly said the repair LLM isn't touching the real "
            "blocker; further rounds will keep cycling on the same "
            "guidance. Routing to HITL.",
            consecutive_distraction, max_distraction,
        )
        return _transition("human_intervention_node")

    # Phase 1.1 — progress-based budget gate. The HITL escalation now reads
    # ``no_progress_repairs`` (rounds that failed to shrink the fingerprint
    # set; see repair_node for the counter) instead of raw ``total_repairs``.
    # A session that goes 251 → 9 → 3 → 1 → 0 over 4 rounds no longer
    # escalates at round 3 even though it spent 3 iterations — those rounds
    # all made progress and don't count against the budget. The safety
    # ceiling at ``2 * max_iterations`` total prevents runaway loops where
    # the LLM keeps producing different patches that never converge but
    # appear to make per-round progress (fingerprint churn).
    no_progress_repairs = int(loop_counter.get("no_progress_repairs", 0))
    if no_progress_repairs >= max_iterations and not has_autofixable:
        logger.warning(
            "[router] No-progress repair limit reached (%d non-progress / "
            "%d cap; %d total rounds). Routing to HITL.",
            no_progress_repairs, max_iterations, total_repairs,
        )
        return _transition("human_intervention_node")

    # Hard safety ceiling on total iterations regardless of per-round
    # progress signal. Catches the fingerprint-churn case where each round
    # technically resolves one fingerprint but introduces a new one of
    # equal weight, so no_progress_repairs never trips.
    _TOTAL_HARD_CAP_MULTIPLIER = 2
    total_hard_cap = max_iterations * _TOTAL_HARD_CAP_MULTIPLIER
    if total_repairs >= total_hard_cap and not has_autofixable:
        logger.warning(
            "[router] Hard total-iteration ceiling reached (%d/%d). "
            "Loop is making per-round progress but not converging. "
            "Routing to HITL.",
            total_repairs, total_hard_cap,
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
# Extension alternation matches harness.spec_files.SPEC_FILE_EXTS — keep
# the two in lockstep when adding a new spec extension.
_CR_FILENAME_PREFIX = re.compile(
    r"^CR-(\d+)(?:[-_].*)?\.(?:txt|md|pdf)$", re.IGNORECASE,
)


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

    - A filename matching ``CR-<N>-*`` or ``CR-<N>.{txt,md,pdf}`` keeps
      its operator-supplied ``N``.
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
    "Main.java",
    "pyproject.toml", "package.json", "pom.xml", "build.gradle",
    "Makefile", "README.md",
)
_REVERSE_ENGINEER_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".java",
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
    pending = list_spec_files(cr_dir, exclude=frozenset({"applied"}))

    if not pending:
        logger.error(
            "[change_requests] No pending spec files (.txt / .md / .pdf) "
            "under %s — cmd_run should have rejected this earlier.", cr_dir,
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
            content = read_spec_file(rec["abs_path"])
        except (OSError, ValueError) as exc:
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

    # CR → STORY bridge. When ``decomposition_enabled``, every CR becomes
    # one ``stories`` row with ``external_ref='CR-N'`` so the downstream
    # planner can treat CRs as first-class stories and the patcher's
    # marker contract emits BOTH `# STORY-M: …` and `# CR-N: …`. Skipped
    # silently when story-mode is off — the existing CR-only flow keeps
    # working byte-for-byte.
    extra_state: dict[str, Any] = {}
    if state.get("decomposition_enabled", False) and records:
        try:
            from harness import story_state as _sst
            workspace = state.get("workspace_path", "")
            app_name = _sst.app_name_for_workspace(workspace)
            conn = _sst.open_story_db()
            try:
                # Skip CRs already mirrored (resume safety — running ingest
                # twice for the same CR set must not create duplicate rows).
                # Scope by workspace so two apps with their own CR-1 don't
                # cross-contaminate.
                existing_refs = {
                    r[0] for r in conn.execute(
                        "SELECT external_ref FROM stories "
                        "WHERE workspace = ? AND external_ref IS NOT NULL",
                        (app_name,),
                    )
                }
                # Each bridged story corresponds to one CR — tag it as a
                # CR-kind row stamped with that CR's id so traceability
                # can answer "which stories did CR-2 produce?".
                #
                # Every story needs a feature_id under the v4 schema, so
                # seed the synthetic ``change-request`` feature lazily
                # (idempotent — one row per workspace, created on first
                # CR bridge).
                created_keys: list[str] = []
                cr_feature_seeded = False
                for rec in records:
                    ref = f"CR-{rec['cr_id']}"
                    if ref in existing_refs:
                        continue
                    if not cr_feature_seeded:
                        _sst.ensure_feature(
                            conn, app_name, _sst.CR_FEATURE_KEY,
                            name="Change requests",
                            description=(
                                "Stories auto-bridged from change-request "
                                "text files ingested via teane patch."
                            ),
                        )
                        cr_feature_seeded = True
                    # v5 traceability uniformity: every story (CR-derived
                    # or spec-derived) MUST link to at least one
                    # requirements row. Seed a synthetic ``CR-N``
                    # requirement so the audit gate can join CR work
                    # the same way it joins spec work, without an
                    # ``OR build_kind='cr'`` special case.
                    req_id = _sst.ensure_requirement(
                        conn, app_name, ref,
                        kind="cr_synthetic",
                        title=rec.get("original_name", ref),
                        body=f"Synthetic requirement for change request {ref}.",
                    )
                    keys = _sst.create_stories(
                        conn, app_name,
                        [{
                            "title": rec.get("original_name", ref),
                            "feature": _sst.CR_FEATURE_KEY,
                            "description": f"Auto-bridged from {ref}.",
                            # Acceptance criteria for CR-derived stories is a
                            # single line — the operator can refine via the
                            # STORIES gatekeeper before the batch loop fires.
                            "acceptance_criteria": [
                                f"All edits demanded by {ref} land cleanly."
                            ],
                            "depends_on": [],
                            "scope_files": [],
                            "external_ref": ref,
                        }],
                        build_kind=_sst.BUILD_KIND_CR,
                        cr_ids=[int(rec["cr_id"])],
                    )
                    created_keys.extend(keys)
                    # Link this bridged story to its synthetic CR
                    # requirement. Lookup-by-key (rather than
                    # threading the id through create_stories) keeps
                    # the public signature unchanged — same trade-off
                    # decomposition_node makes for spec-derived stories.
                    for key in keys:
                        row = _sst.get_story(conn, app_name, key)
                        if row is not None:
                            _sst.link_story_to_requirements(
                                conn, app_name, row["id"], [ref],
                            )
                    # Suppress an unused-variable warning while still
                    # documenting the contract: req_id exists so the
                    # synthetic row is materialised here even if the
                    # link below silently no-ops.
                    del req_id
                if created_keys:
                    _sst.regenerate_markdown_views(conn, workspace)
                    logger.info(
                        "[change_requests] bridged %d CR(s) into stories: %s",
                        len(created_keys), ", ".join(created_keys),
                    )
                    extra_state["stories_db_path"] = _sst.state_db_path()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            # Bridging failure is non-fatal — the CR session still runs
            # in monolithic mode if story-mode plumbing chokes.
            logger.exception("[change_requests] CR → STORY bridge failed; proceeding without stories")

    return {
        "messages": new_messages,
        "change_request_files": records,
        # Discovery pipeline runs in delta mode for change-request sessions
        # regardless of the operator's --spec-discovery flag — the gatekeeper is
        # the whole point of the folder convention. Clear skip_discovery
        # so route_after_spec_review honors interview follow-ups instead
        # of short-circuiting to the gatekeeper.
        "skip_discovery": False,
        **extra_state,
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


def _build_story_preamble(state: AgentState, phase: str) -> str:
    """Return the per-story scoping preamble for ``phase`` (one of
    ``"patching"``, ``"tests"``). Empty string when no story is active.

    The preamble names the current ``STORY-N`` key, reproduces the
    acceptance criteria the patcher must satisfy, advises the LLM to
    stay within ``story_scope_files`` (advisory — the allowlist is
    NOT tightened so cross-story coupling is possible; it just
    becomes a tracked file-link in ``TRACEABILITY.md``), and lays
    out the marker contract:

      - Production code: ``# STORY-N: …`` (Python) / ``// STORY-N: …``
        (JS/TS/Go/Rust/Java) — one terse comment per new function /
        class / region, naming the originating story.
      - When the story was derived from a CR
        (``external_ref="CR-N"``), emit BOTH markers (``# STORY-N: …``
        AND ``# CR-N: …``) so ``grep`` queries against either ID find
        the same artifacts.
    """
    story_key = state.get("current_story_id", "") or ""
    if not story_key:
        return ""

    workspace = state.get("workspace_path") or ""
    story = None
    ac_rows: list[dict] = []
    if workspace:
        try:
            from harness import story_state as _sst
            app_name = _sst.app_name_for_workspace(workspace)
            conn = _sst.open_story_db()
            try:
                story = _sst.get_story(conn, app_name, story_key)
                # v5: fetch the AC rows separately so the preamble can
                # render each criterion with its stable ac_key. The
                # test-gen @verifies marker contract references these
                # keys verbatim (e.g. ``# @verifies: STORY-3.AC-2``),
                # so the LLM must see them in the source material.
                if story is not None:
                    ac_rows = _sst.list_acceptance_criteria(
                        conn, app_name, story["id"],
                    )
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            story = None
    if story is None:
        return ""

    ac = story.get("acceptance_criteria") or []
    scope = state.get("story_scope_files") or story.get("scope_files") or []
    external = story.get("external_ref") or ""

    # Prefer the row-with-keys rendering when ac_rows came back; fall
    # back to plain bullets when the side-table is empty (legacy data
    # or a peek-failure path) so the preamble stays useful.
    if ac_rows:
        ac_block = "\n".join(
            f"  - {r['ac_key']}: {r['text']}" for r in ac_rows
        )
    else:
        ac_block = (
            "\n".join(f"  - {c}" for c in ac) if ac else "  - (none recorded)"
        )
    scope_block = (
        "\n".join(f"  - {p}" for p in scope) if scope else
        "  - (unscoped — touch only what the acceptance criteria require)"
    )

    marker_lines = [
        f"Tag the code that satisfies this story with `# {story_key}: …` "
        "(Python) / `// " + story_key + ": …` (JS/TS/Go/Rust/Java) — "
        "one terse comment per new function/class/region. New files get "
        "the marker on the line below the module docstring or imports.",
    ]
    if external:
        marker_lines.append(
            f"This story was derived from {external}; emit BOTH "
            f"`# {story_key}: …` AND `# {external}: …` so grep on either ID "
            "finds the same artifacts."
        )

    if phase == "tests":
        marker_lines.append(
            f"Name new test functions `test_{story_key.lower().replace('-', '_')}"
            "_<descriptive>` and reference the story in each test docstring "
            f"(e.g. `\"\"\"Verifies {story_key}: …\"\"\"`)."
        )

    rules = "\n".join(marker_lines)

    return (
        f"## Story Scope: {story_key} — {story.get('title', '')}\n\n"
        "This patching turn is scoped to ONE story. Focus on the "
        "acceptance criteria below; do not attempt to deliver the "
        "whole specification in a single pass.\n\n"
        "### Acceptance criteria\n"
        f"{ac_block}\n\n"
        "### File scope (advisory)\n"
        f"{scope_block}\n\n"
        "Editing files outside this list is allowed when an "
        "acceptance criterion demands it (e.g. a shared util that "
        "needs one extra parameter), but those edits will be recorded "
        "as cross-story coupling in `TRACEABILITY.md`. Keep the blast "
        "radius small.\n\n"
        "### Marker contract\n"
        f"{rules}\n\n"
        "---\n\n"
    )


def _build_batch_scope_preamble(state: AgentState) -> str:
    """Multi-story preamble for the per-batch verification phase.

    ``story_loop_node`` clears ``current_story_id=""`` when a batch is
    fully patched and hands control to the verification chain
    (``speculative → test_generation → lintgate → compiler →
    code_review``). At that point :func:`_build_story_preamble`
    returns empty — but the test-generation LLM still gets RULE 5
    requiring it to cite AC keys via ``# @verifies:`` markers. With
    no preamble, the LLM has no AC keys to cite and fabricates them;
    the marker gate then loops until ``max_iterations`` and
    eventually escalates to HITL.

    This helper renders a "Batch Scope" block listing every story
    just patched in the current batch alongside its acceptance
    criteria (with stable ``STORY-N.AC-N`` keys). The test-gen LLM
    can then cite real keys; the marker gate validates them; the
    link writer persists ``test_verifies_ac`` edges to real ACs.

    Returns ``""`` when there's no active batch context — either
    monolithic / non-agile mode, or a single-story phase where
    :func:`_build_story_preamble` already rendered the per-story
    preamble (callers should prefer the single-story form when
    ``current_story_id`` is set; this helper is the fallback).
    """
    if state.get("current_story_id"):
        # Single story is active — caller should use _build_story_preamble.
        return ""
    patched_keys = list(state.get("batch_patched_story_keys") or [])
    if not patched_keys:
        return ""
    workspace = state.get("workspace_path") or ""
    if not workspace:
        return ""
    try:
        from harness import story_state as _sst
        app_name = _sst.app_name_for_workspace(workspace)
        conn = _sst.open_story_db()
        try:
            stories: list[dict[str, Any]] = []
            for key in patched_keys:
                row = _sst.get_story(conn, app_name, key)
                if row is None:
                    continue
                acs = _sst.list_acceptance_criteria(conn, app_name, row["id"])
                stories.append({"row": row, "acs": acs})
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return ""
    if not stories:
        return ""

    lines: list[str] = [
        f"## Batch Scope: {len(stories)} story / stories patched in this batch",
        "",
        "The patcher just landed the stories below as a unit. Generate "
        "tests that verify their acceptance criteria. Cite the AC keys "
        "EXACTLY as shown below in your ``@verifies:`` markers — every "
        "test file MUST cite at least one of these keys.",
        "",
    ]
    for s in stories:
        row = s["row"]
        title = row.get("title", "")
        lines.append(f"### {row['story_key']} — {title}")
        lines.append("")
        if s["acs"]:
            lines.append("**Acceptance criteria:**")
            lines.append("")
            for ac in s["acs"]:
                lines.append(f"  - {ac['ac_key']}: {ac['text']}")
        else:
            lines.append("  _(no acceptance criteria recorded — skip this story)_")
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _build_arch_summary_preamble(
    state: AgentState, *, consumer: str = "patcher",
) -> tuple[str, dict[str, Any]]:
    """Return ``(preamble_text, resolved_summary)`` for a downstream node.

    Reads ``state["arch_summary"]`` if present; otherwise lazy-loads
    from ``<workspace>/docs/SPEC_ARCHITECTURE.md`` (monolithic / non-
    decomposition flows skip ``decomposition_node`` entirely, so the
    summary never gets populated by that path). The resolved summary
    is returned alongside the preamble so the caller can stash it
    back onto state as part of its delta — a one-time cost per
    session.

    ``consumer`` picks which one-paragraph guidance block prefaces
    the tables: ``"patcher"`` (emit NO_PROGRESS on gap), ``"reviewer"``
    (flag drift as a finding), or ``"test_generator"`` (treat the
    tables as a coverage target). The endpoint / component tables
    are identical across consumers; only the guidance differs.

    Returns ``("", {})`` when the arch doc has no §11 jsonc block,
    schema_version is unrecognised, or the file is missing. Callers
    MUST treat this as "fall back to the prose handoff" and never
    branch on the structured fields being present.
    """
    summary = state.get("arch_summary") or {}
    if not summary:
        from harness.arch_summary import load_arch_summary
        loaded = load_arch_summary(state.get("workspace_path") or "")
        summary = loaded or {}

    if not summary:
        return "", {}

    from harness.arch_summary import render_arch_preamble
    return render_arch_preamble(summary, consumer=consumer), summary


async def reverse_spec_node(state: AgentState) -> dict[str, Any]:
    """Reverse-engineer SPEC_REQUIREMENTS.md / SPEC_ARCHITECTURE.md drafts
    from an existing codebase.

    Only fires when ``flow=patch`` and ``generate_specs`` resolved active
    (see ``route_after_start``). Runs a deterministic workspace-telemetry
    scan, reads any ``product_spec/*.txt`` files, then synthesises an
    operator-facing starting draft via the planning LLM. The draft is
    appended to ``messages`` as a system note so the downstream
    ``requirements_discovery_node`` / ``architecture_discovery_node``
    interview chain seeds its questions from the draft instead of
    starting from zero.

    The draft is NOT written to disk here — ``spec_review_node`` and
    ``write_spec_node`` own the eventual persistence after the operator
    has approved the refined version.
    """
    workspace_path = state.get("workspace_path", "")
    if not workspace_path:
        logger.warning("[reverse_spec] No workspace_path on state — skipping.")
        return {}

    from harness.deploy import scan_workspace_telemetry

    try:
        telemetry = scan_workspace_telemetry(workspace_path)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning(
            "[reverse_spec] workspace telemetry scan failed: %s — "
            "synthesising spec draft from product_spec/ alone.", exc,
        )
        telemetry = {"app_name": os.path.basename(workspace_path.rstrip(os.sep))}

    # product_spec/*.txt is the operator's narrative starting point. When
    # the folder is missing or empty we fall through to telemetry only.
    product_spec_text = ""
    try:
        spec_dir = os.path.join(workspace_path, "product_spec")
        if os.path.isdir(spec_dir):
            chunks: list[str] = []
            for name in sorted(os.listdir(spec_dir)):
                if not name.endswith(".txt"):
                    continue
                path = os.path.join(spec_dir, name)
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    chunks.append(f"# === {name} ===\n{fh.read().rstrip()}")
            product_spec_text = "\n\n".join(chunks)
    except OSError as exc:
        logger.info(
            "[reverse_spec] product_spec/ scan skipped: %s "
            "(telemetry-only synthesis).", exc,
        )

    gateway = get_gateway()
    if gateway is None:
        logger.error("[reverse_spec] No gateway configured.")
        return {"node_state": {"error": "no gateway"}}

    from harness.gateway import NodeRole

    try:
        telemetry_json = json.dumps(telemetry, indent=2, default=str)
    except (TypeError, ValueError) as exc:
        logger.warning("[reverse_spec] telemetry JSON encode failed: %s", exc)
        telemetry_json = str(telemetry)

    # Phase 8c: branch the SPEC_REQUIREMENTS format on the agile flag
    # so the reverse-engineered draft matches whatever shape the
    # downstream discovery interview will produce. The architecture
    # half is unchanged either way — only the requirements vocabulary
    # differs.
    if state.get("decomposition_enabled"):
        requirements_shape = (
            "  - SPEC_REQUIREMENTS.md: the WHAT in **SAFe agile** shape. "
            "Group capabilities as `## Epic: EPIC-NNN — <title>`, decompose "
            "each epic into `### Feature: FEAT-NNN — <title>` sections, "
            "and decompose each feature into `#### Story: STORY-NNN — "
            "<title>` sections with Given/When/Then acceptance criteria. "
            "Non-functional concerns go in `#### Enabler Story: "
            "STORY-NFR-NNN — <title>` sections under the relevant feature. "
            "Use the agile vocabulary throughout.\n"
        )
    else:
        requirements_shape = (
            "  - SPEC_REQUIREMENTS.md: the WHAT in **flat waterfall** "
            "shape (ISO 29148 style). List functional requirements as "
            "`### FR-NNN: <one-line title>` sections, each stating what "
            "the system **shall** do plus acceptance criteria. "
            "Non-functional requirements use `### NFR-CATEGORY-NNN: "
            "<title>` (e.g. `NFR-SEC-001`, `NFR-PERF-014`). Do NOT use "
            "agile vocabulary (no Epic / Feature / Story sections, no "
            "INVEST framing, no user-story \"as a … I want …\" wording).\n"
        )
    prompt = (
        "You are reverse-engineering a project specification from an "
        "existing codebase. Produce TWO markdown documents in one reply, "
        "delimited by the markers `<SPEC_REQUIREMENTS>` and "
        "`<SPEC_ARCHITECTURE>` (one each, no nesting):\n\n"
        f"{requirements_shape}"
        "  - SPEC_ARCHITECTURE.md: the HOW — workspace layout, modules, "
        "data flow, external dependencies, build / run commands, "
        "deployment topology when discernible.\n\n"
        "Ground every claim in what the code already does. When a "
        "section can't be derived from the codebase, write a single "
        "`(unspecified — needs operator review)` placeholder rather "
        "than inventing detail. The drafts will be fed into a "
        "human-in-the-loop discovery + review loop that refines them, "
        "so a thin honest draft beats a thick speculative one.\n\n"
        "## Workspace telemetry (deterministic scan)\n\n"
        f"```json\n{telemetry_json}\n```\n\n"
        "## Operator-supplied narrative (product_spec/*.txt)\n\n"
        f"{product_spec_text or '(none provided)'}\n\n"
        "Emit only the two delimited markdown documents in your reply."
    )

    budget = state.get("budget_remaining_usd", 0.0)
    response, new_budget = await gateway.dispatch(
        messages=[
            {"role": "system", "content":
                "Reverse-engineer specs from a codebase. Be honest about gaps."},
            {"role": "user", "content": prompt},
        ],
        role=NodeRole.PLANNING,
        budget_remaining_usd=budget,
    )

    draft = (response.content or "").strip()
    if not draft:
        logger.warning("[reverse_spec] Empty draft from LLM — falling through.")
        return {"budget_remaining_usd": new_budget}

    # The draft rides into the messages so requirements_discovery_node
    # picks it up as starting context. We add it as a user message
    # (rather than mutating the cached system prompt) so the discovery
    # node sees it as a normal turn in the conversation.
    messages = list(state.get("messages", []))
    messages.append({
        "role": "user",
        "content": (
            "[reverse-engineered spec draft — refine and approve via the "
            "discovery + review loop, do not treat as final]\n\n" + draft
        ),
    })

    logger.info(
        "[reverse_spec] Synthesised %d-char draft from telemetry "
        "(%d product_spec chars). Remaining budget: $%.4f",
        len(draft), len(product_spec_text), new_budget,
    )

    return {
        "messages": messages,
        "budget_remaining_usd": new_budget,
    }


async def story_reopen_node(state: AgentState) -> dict[str, Any]:
    """Re-classify existing DONE stories after a patch-flow spec revision.

    Fires in agile-patch when SPEC_REQUIREMENTS.md / SPEC_ARCHITECTURE.md
    have been revised this run (via ``--spec-discovery`` or
    ``--generate-specs``) and at least one DONE story already exists in
    ``state.db`` for the workspace. LLM-judged pass produces one verdict
    per story:

      - ``unaffected``: leave DONE (acceptance criteria still hold).
      - ``reopen``: acceptance criteria drifted → flip to ``reopened``
        so the story loop runs the story again.
      - ``new``: an entirely new story the spec now requires; appended
        to the planner queue (the decomposition node's augment mode
        picks it up on the next pass).

    Verdicts are written into ``node_state.story_reopen_verdicts`` for
    operator review and the actual DB transitions are committed before
    the node returns.
    """
    workspace_path = state.get("workspace_path", "")
    if not workspace_path:
        return {}

    from harness import story_state

    try:
        app = story_state.app_name_for_workspace(workspace_path)
    except ValueError as exc:
        logger.warning("[story_reopen] invalid workspace: %s", exc)
        return {}

    db_path = story_state.state_db_path()
    if not os.path.isfile(db_path):
        return {}
    try:
        conn = story_state.open_story_db(workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[story_reopen] open_story_db failed: %s", exc)
        return {}

    try:
        existing = [
            s for s in story_state.list_stories(conn, app)
            if s.get("status") == "done"
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[story_reopen] list_stories failed: %s", exc)
        conn.close()
        return {}

    if not existing:
        conn.close()
        logger.info("[story_reopen] no DONE stories; skipping verdict pass.")
        return {}

    spec_req_path = os.path.join(workspace_path, "docs", "SPEC_REQUIREMENTS.md")
    spec_arch_path = os.path.join(workspace_path, "docs", "SPEC_ARCHITECTURE.md")
    try:
        spec_req = open(spec_req_path, "r", encoding="utf-8").read() if os.path.isfile(spec_req_path) else ""
        spec_arch = open(spec_arch_path, "r", encoding="utf-8").read() if os.path.isfile(spec_arch_path) else ""
    except OSError as exc:
        logger.warning("[story_reopen] spec read failed: %s", exc)
        conn.close()
        return {}

    gateway = get_gateway()
    if gateway is None:
        logger.error("[story_reopen] No gateway configured.")
        conn.close()
        return {}

    from harness.gateway import NodeRole

    stories_payload = [
        {
            "story_key": s.get("story_key"),
            "title": s.get("title", ""),
            "acceptance_criteria": s.get("acceptance_criteria") or [],
        }
        for s in existing
    ]

    prompt = (
        "The project spec was just revised in a patch run. For each "
        "story below, judge whether its acceptance criteria still hold "
        "under the NEW spec. Reply with strict JSON of the form\n\n"
        '  {"verdicts": [{"story_key": "STORY-3", "verdict": '
        '"unaffected"|"reopen", "reason": "..."}]}\n\n'
        "`unaffected` = the story\'s criteria are still correct under "
        "the new spec; leave DONE.\n"
        "`reopen` = at least one acceptance criterion drifted under "
        "the new spec; the story must run again. Use `reason` to name "
        "the criterion that drifted.\n\n"
        "## New SPEC_REQUIREMENTS.md\n\n"
        f"{spec_req}\n\n## New SPEC_ARCHITECTURE.md\n\n{spec_arch}\n\n"
        "## DONE stories\n\n"
        f"```json\n{json.dumps(stories_payload, indent=2)}\n```\n\n"
        "Be conservative: only mark `reopen` when you can point at the "
        "specific criterion that drifted. Marking everything `reopen` "
        "wastes the operator's tokens."
    )

    budget = state.get("budget_remaining_usd", 0.0)
    try:
        response, new_budget = await gateway.dispatch(
            messages=[
                {"role": "system", "content":
                    "You are classifying whether existing stories still match a revised spec."},
                {"role": "user", "content": prompt},
            ],
            role=NodeRole.PLANNING,
            budget_remaining_usd=budget,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[story_reopen] dispatch failed: %s", exc)
        conn.close()
        return {"budget_remaining_usd": budget}

    raw = (response.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    verdicts: list[dict[str, Any]] = []
    try:
        data = json.loads(raw)
        verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
    except json.JSONDecodeError as exc:
        logger.warning("[story_reopen] verdict JSON parse failed: %s", exc)

    reopened: list[str] = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        key = v.get("story_key")
        verdict = v.get("verdict")
        if verdict == "reopen" and isinstance(key, str):
            try:
                changed = story_state.mark_reopened(conn, app, key)
                if changed:
                    reopened.append(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[story_reopen] mark_reopened(%s) failed: %s", key, exc,
                )

    conn.close()
    logger.info(
        "[story_reopen] verdicts=%d reopened=%s remaining_budget=$%.4f",
        len(verdicts), reopened, new_budget,
    )

    node_state = dict(state.get("node_state", {}))
    node_state["story_reopen_verdicts"] = verdicts
    node_state["story_reopen_reopened"] = reopened
    return {
        "node_state": node_state,
        "budget_remaining_usd": new_budget,
    }


async def patch_reconcile_node(state: AgentState) -> dict[str, Any]:
    """Append the ``patch`` reconcile preamble to the conversation, then
    hand off to ``planning_node``.

    Only fires when ``flow=patch`` and no ``change_requests/*.txt`` are
    present and ``generate_specs`` was not resolved active — i.e. the
    operator wants an incremental change against an existing codebase
    that already has approved specs. The preamble pins the planner to
    "reconcile what's drifted; leave conformant code untouched" rather
    than re-emitting a from-scratch implementation blueprint.

    Reads SPEC_REQUIREMENTS.md / SPEC_ARCHITECTURE.md from ``docs/`` if
    they exist (fall-through is graceful — the user prompt + telemetry
    is still enough for the planner to do something useful).
    """
    workspace_path = state.get("workspace_path", "")
    if not workspace_path:
        return {}

    spec_excerpts: list[str] = []
    for name, label in (
        ("SPEC_REQUIREMENTS.md", "## Current SPEC_REQUIREMENTS.md"),
        ("SPEC_ARCHITECTURE.md", "## Current SPEC_ARCHITECTURE.md"),
    ):
        path = os.path.join(workspace_path, "docs", name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    spec_excerpts.append(f"{label}\n\n{fh.read().rstrip()}")
            except OSError as exc:
                logger.info(
                    "[patch_reconcile] failed to read %s: %s", path, exc,
                )

    # Lightweight tree summary so the planner sees what already exists.
    # Capped at a depth + entry count so a sprawling workspace doesn't
    # blow the system prompt.
    tree_lines: list[str] = []
    try:
        from harness.deploy import scan_workspace_telemetry
        telemetry = scan_workspace_telemetry(workspace_path)
        src_dirs = telemetry.get("source_dirs") or []
        manifests = telemetry.get("manifests_found") or {}
        if src_dirs:
            tree_lines.append("Top-level source directories: " + ", ".join(src_dirs[:10]))
        if manifests:
            for lang, files in list(manifests.items())[:6]:
                tree_lines.append(f"{lang}: {', '.join(files[:4])}")
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
        logger.info("[patch_reconcile] telemetry skipped: %s", exc)

    tree_block = "\n".join(tree_lines) if tree_lines else "(workspace telemetry unavailable)"
    spec_block = "\n\n".join(spec_excerpts) if spec_excerpts else "(no SPEC_* docs found in docs/)"

    preamble = (
        "## Patch-mode reconcile preamble\n\n"
        "An existing implementation lives at the workspace root. The "
        "approved spec documents below describe the intended behaviour. "
        "Reconcile the implementation against the spec and the user's "
        "request: change only what has drifted, what the user is asking "
        "for, or what the spec now requires. Leave conformant code "
        "untouched. Justify each modification in your plan.\n\n"
        "## Workspace shape\n\n"
        f"{tree_block}\n\n"
        f"{spec_block}\n"
    )

    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": preamble})
    logger.info(
        "[patch_reconcile] Reconcile preamble appended "
        "(%d spec chars, %d tree lines).",
        sum(len(x) for x in spec_excerpts), len(tree_lines),
    )
    return {"messages": messages}


def route_after_start(state: AgentState) -> Literal[
    "requirements_discovery_node",
    "patching_node",
    "decomposition_node",
    "story_reopen_node",
    "ingest_change_requests_node",
    "reverse_spec_node",
    "patch_reconcile_node",
    "deployment_discovery_node",
    "generate_deployment_spec_node",
    "test_node",
]:
    """START edge router. Module-level so tests can call it directly.

    Precedence:
      1. ``flow == "deploy"`` → straight into the deployment chain. With
         ``cd_discovery=True`` enter at deployment_discovery_node so the
         LLM-driven interview synthesises DEPLOYMENT_BLUEPRINT.md; else
         skip straight to generate_deployment_spec_node which builds
         the blueprint from workspace telemetry alone.
      2. ``flow == "patch"`` with ``generate_specs`` resolved active →
         reverse_spec_node first, then through the discovery / spec_review
         pipeline as usual.
      3. ``change_request_mode`` → ingest_change_requests_node (a populated
         change_requests/ folder always overrides skip_discovery so a
         misconfigured run still goes through the gatekeeper pipeline).
      4. ``flow == "patch"`` without CRs and without generate_specs →
         patch_reconcile_node injects the "reconcile against existing
         tree" preamble before handing off to planning_node.
      5. ``skip_discovery`` + ``decomposition_enabled`` → enter the agile
         pipeline at decomposition_node (or story_reopen_node first when
         a brownfield workspace already has DONE stories). Mirrors the
         ARCHITECTURE-gate branch in ``route_after_gatekeeper`` —
         needed because ``--agile=true`` without ``--spec-discovery=true``
         (the common case: specs were synthesised pre-graph in cmd_run)
         would otherwise fall through to monolithic patching, silently
         negating the operator's choice of agile mode.
      6. ``skip_discovery`` → patching_node (the bare existing-project path
         from before change-request mode existed; legacy callers / tests).
      7. Default → requirements_discovery_node (greenfield discovery).
    """
    flow = state.get("flow", FLOW_BUILD)
    if flow == FLOW_TEST:
        logger.info("[router] flow=test. Routing START → test_node.")
        return "test_node"
    if flow == FLOW_DEPLOY:
        if state.get("cd_discovery", False):
            logger.info(
                "[router] flow=deploy + cd_discovery. "
                "Routing START → deployment_discovery_node."
            )
            return "deployment_discovery_node"
        logger.info(
            "[router] flow=deploy. Routing START → generate_deployment_spec_node "
            "(telemetry-only blueprint)."
        )
        return "generate_deployment_spec_node"
    if flow == FLOW_PATCH and state.get("generate_specs", False):
        logger.info(
            "[router] flow=patch + generate_specs. "
            "Routing START → reverse_spec_node."
        )
        return "reverse_spec_node"
    if state.get("change_request_mode", False):
        logger.info(
            "[router] change_request_mode active. "
            "Routing START → ingest_change_requests_node."
        )
        return "ingest_change_requests_node"
    if flow == FLOW_PATCH:
        logger.info(
            "[router] flow=patch (no CRs, no generate_specs). "
            "Routing START → patch_reconcile_node."
        )
        return "patch_reconcile_node"
    if state.get("skip_discovery", False):
        # Agile mode opt-in: when the operator passed --agile=true, the
        # graph MUST enter the decomposition → per-batch patching loop
        # even when discovery is skipped. Without this branch the same
        # ARCHITECTURE-gate routing that exists in
        # route_after_gatekeeper is silently bypassed and the run
        # collapses to monolithic patching despite --agile=true. The
        # brownfield case (FLOW_PATCH on a workspace that already has
        # DONE stories) reopens drifted stories first; greenfield goes
        # straight into decomposition.
        if state.get("decomposition_enabled", False):
            if (
                flow == FLOW_PATCH
                and _workspace_has_done_stories(state)
            ):
                logger.info(
                    "[router] spec discovery skipped + agile mode + existing "
                    "DONE stories. Routing START → story_reopen_node."
                )
                return "story_reopen_node"
            logger.info(
                "[router] spec discovery skipped + agile mode. "
                "Routing START → decomposition_node."
            )
            return "decomposition_node"
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

    # Prompt body lives in harness/skills/docgen/*.md so it can be iterated
    # (sectors added, threat-model coverage tightened) without touching code.
    # Per-project overrides at {workspace_path}/skills/docgen/*.md win.
    from harness import docgen_prompts
    workspace_for_overrides = state.get("workspace_path") or None
    current_budget_for_focus = state.get("budget_remaining_usd", 0.0)
    focus_sectors: Optional[list[str]] = None
    if is_followup:
        # LLM-judgment #6: ask which 3-5 sectors most need re-auditing
        # this round and splice them in. Disabled / failed → None → empty
        # block → behaves exactly like the pre-focus follow-up.
        focus_sectors, current_budget_for_focus = await _maybe_discovery_followup_focus(
            gate="REQUIREMENTS",
            question_count=question_count,
            sectors=_REQUIREMENTS_SECTORS,
            messages=messages,
            budget=current_budget_for_focus,
        )
        prompt = docgen_prompts.load(
            "requirements_discovery_followup", workspace_for_overrides
        ).replace("{ROUND_NUMBER}", str(question_count + 1))
        prompt = prompt.replace(
            "{FOCUS_SECTORS_BLOCK}", _render_focus_block(focus_sectors or []),
        )
    else:
        # Phase 8a: branch the discovery prompt on the agile flag.
        # Agile mode uses ``requirements_discovery.md`` (current
        # canonical content: INVEST stories, Given/When/Then ACs)
        # which downstream produces a SAFe-shaped spec. Waterfall
        # mode uses ``requirements_discovery_waterfall.md`` which
        # frames Sector 2 around flat FR-NNN "shall" statements.
        # Both files live in ``harness/skills/docgen/`` and per-
        # workspace overrides at ``{workspace}/skills/docgen/*.md``
        # still win for either name.
        skill_name = (
            "requirements_discovery"
            if state.get("decomposition_enabled")
            else "requirements_discovery_waterfall"
        )
        prompt = docgen_prompts.load(
            skill_name, workspace_for_overrides
        )

    # Delta-mode preamble: when change_request_mode is active, prepend the
    # CR-N attribution rules and the "ask delta-shaped questions only"
    # instruction so the LLM doesn't re-elicit baseline requirements on
    # an existing-project run. No-op (empty string) when not in CR mode.
    prompt = _build_change_request_preamble(state, "requirements") + prompt
    messages.append({"role": "user", "content": prompt})

    from harness.gateway import NodeRole

    # current_budget_for_focus already reflects whatever the focus picker
    # spent on its sub-call (a few cents at most). Carry that forward
    # so the planning dispatch isn't handed a stale "as if focus never
    # ran" budget.
    current_budget = current_budget_for_focus
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
            logger.info(
                "[reqs_disc] Discovery response failed trust validation "
                "(%s) — asking planner to re-emit.", trust_errors,
            )
            # One-shot repair: re-prompt the planner with the schema so
            # a formatting slip doesn't terminate the interview and leave
            # the operator staring at an empty screen.
            async def _reqs_repair_dispatch(msgs, bud):
                return await gateway.dispatch(
                    messages=list(msgs),
                    role=NodeRole.PLANNING,
                    budget_remaining_usd=bud,
                )
            _discovery_schema = (
                "A JSON object with exactly two top-level keys: "
                "'modules' and 'complete'. 'modules' is an array of "
                "objects, each with 'name' (string) and 'questions' "
                "(array of objects with fields 'id' (string), 'text' "
                "(string), 'critical' (bool)). 'complete' is a boolean "
                "signalling whether discovery is finished. Emit no "
                "other top-level keys."
            )
            repaired, budget = await _repair_malformed_json(
                raw_text=response.content or "",
                schema_hint=_discovery_schema,
                dispatch=_reqs_repair_dispatch,
                budget_remaining_usd=budget,
                purpose="requirements_discovery",
            )
            if not isinstance(repaired, dict):
                logger.warning(
                    "[reqs_disc] Repair pass did not yield a JSON "
                    "object — terminating discovery.",
                )
                return {
                    "messages": messages,
                    "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                    "budget_remaining_usd": budget,
                }
            discovery_data = repaired

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

        # LLM-judgment saturation check (#3). After at least one prior
        # round, ask whether further questions would meaningfully refine
        # the spec. When yes, override complete=True so route_after_discovery
        # advances to write_spec_node without another interview pass.
        complete, budget = await _maybe_discovery_saturation_check(
            gate="REQUIREMENTS",
            question_count=question_count,
            complete=complete,
            discovery_data=discovery_data,
            messages=messages,
            budget=budget,
        )

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

    from harness import docgen_prompts
    workspace_for_overrides = state.get("workspace_path") or None
    current_budget_for_focus = state.get("budget_remaining_usd", 0.0)
    focus_sectors: Optional[list[str]] = None
    if is_followup:
        focus_sectors, current_budget_for_focus = await _maybe_discovery_followup_focus(
            gate="ARCHITECTURE",
            question_count=question_count,
            sectors=_ARCHITECTURE_SECTORS,
            messages=messages,
            budget=current_budget_for_focus,
        )
        prompt = docgen_prompts.load(
            "architecture_discovery_followup", workspace_for_overrides
        ).replace("{ROUND_NUMBER}", str(question_count + 1))
        prompt = prompt.replace(
            "{FOCUS_SECTORS_BLOCK}", _render_focus_block(focus_sectors or []),
        )
    else:
        prompt = docgen_prompts.load(
            "architecture_discovery", workspace_for_overrides
        )
    # Delta-mode preamble — see ``_build_change_request_preamble``. In
    # delta mode the LLM is told to short-circuit (modules=[], complete=
    # true) when no CR is architecture-significant, so light fixes don't
    # spin up the full architecture review cycle.
    prompt = _build_change_request_preamble(state, "architecture") + prompt
    messages.append({"role": "user", "content": prompt})

    from harness.gateway import NodeRole

    current_budget = current_budget_for_focus
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
            logger.info(
                "[arch_disc] Discovery response failed trust validation "
                "(%s) — asking planner to re-emit.", trust_errors,
            )
            async def _arch_repair_dispatch(msgs, bud):
                return await gateway.dispatch(
                    messages=list(msgs),
                    role=NodeRole.PLANNING,
                    budget_remaining_usd=bud,
                )
            _arch_schema = (
                "A JSON object with exactly two top-level keys: "
                "'modules' and 'complete'. 'modules' is an array of "
                "objects, each with 'name' (string) and 'questions' "
                "(array of objects with fields 'id' (string), 'text' "
                "(string), 'critical' (bool)). 'complete' is a boolean. "
                "Emit no other top-level keys."
            )
            repaired, budget = await _repair_malformed_json(
                raw_text=response.content or "",
                schema_hint=_arch_schema,
                dispatch=_arch_repair_dispatch,
                budget_remaining_usd=budget,
                purpose="architecture_discovery",
            )
            if not isinstance(repaired, dict):
                logger.warning(
                    "[arch_disc] Repair pass did not yield a JSON "
                    "object — terminating discovery.",
                )
                return {
                    "messages": messages,
                    "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                    "budget_remaining_usd": budget,
                }
            discovery_data = repaired

        complete = discovery_data.get("complete", False)
        modules = discovery_data.get("modules", [])
        total_q = sum(len(m.get("questions", [])) for m in modules)
        critical_count = sum(1 for m in modules for q in m.get("questions", []) if q.get("critical"))

        messages.append({"role": "assistant", "content": json.dumps(discovery_data, sort_keys=True)})

        # LLM-judgment saturation check (#3). See requirements_discovery_node
        # for the rationale; same helper, scoped to ARCHITECTURE.
        complete, budget = await _maybe_discovery_saturation_check(
            gate="ARCHITECTURE",
            question_count=question_count,
            complete=complete,
            discovery_data=discovery_data,
            messages=messages,
            budget=budget,
        )

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

    # Deterministic workspace scan — runs token-free and grounds the LLM's
    # suggested_answers in actual evidence (detected frameworks, databases,
    # port hints from .env / compose, presence of Dockerfile / Caddyfile,
    # …). Without this block the LLM is guessing industry defaults blind,
    # which is what causes the discovery loop to re-ask the same questions
    # ("do you want a .env for secrets?" on a project that already has one).
    telemetry_block = ""
    try:
        from harness.deploy import scan_workspace_telemetry
        workspace = state.get("workspace_path", os.getcwd())
        telemetry = scan_workspace_telemetry(workspace)
        telemetry_block = (
            "\n\n## Detected workspace facts (deterministic scan)\n\n"
            "The block below is a token-free scan of the actual workspace. "
            "Treat it as AUTHORITATIVE ground-truth — base every "
            "``suggested_answer`` on this evidence, not on generic "
            "industry defaults. Examples: if ``existing_infrastructure.dockerfile`` "
            "is true, suggest reusing it instead of asking whether to "
            "create one. If ``port_hints`` are non-empty, suggest those "
            "ports. If a database is listed under ``databases_detected``, "
            "the secrets question should presume that database. If a "
            "sector is fully answered by the evidence below, OMIT its "
            "questions and reflect the answer in ``summary`` so the "
            "operator doesn't get asked things the workspace already "
            "answers.\n\n"
            "```json\n"
            + json.dumps(telemetry, indent=2, sort_keys=True, default=str)
            + "\n```"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[deploy_disc] Workspace telemetry scan failed: %s", exc)

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
code fences.{telemetry_block}{resolved_block}"""

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
JSON. No markdown, no explanation, no code blocks.""" + telemetry_block + resolved_block

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
            logger.info(
                "[deploy_disc] Discovery response failed trust validation "
                "(%s) — asking planner to re-emit.", trust_errors,
            )
            async def _deploy_repair_dispatch(msgs, bud):
                return await gateway.dispatch(
                    messages=list(msgs),
                    role=NodeRole.PLANNING,
                    budget_remaining_usd=bud,
                )
            _deploy_schema = (
                "A JSON object with exactly two top-level keys: "
                "'modules' and 'complete'. 'modules' is an array of "
                "objects, each with 'name' (string) and 'questions' "
                "(array of objects with fields 'id' (string), 'text' "
                "(string), 'critical' (bool)). 'complete' is a boolean. "
                "Emit no other top-level keys."
            )
            repaired, budget = await _repair_malformed_json(
                raw_text=response.content or "",
                schema_hint=_deploy_schema,
                dispatch=_deploy_repair_dispatch,
                budget_remaining_usd=budget,
                purpose="deploy_discovery",
            )
            if not isinstance(repaired, dict):
                logger.warning(
                    "[deploy_disc] Repair pass did not yield a JSON "
                    "object — terminating discovery.",
                )
                return {
                    "messages": messages,
                    "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                    "budget_remaining_usd": budget,
                }
            discovery_data = repaired

        complete = discovery_data.get("complete", False)
        modules = discovery_data.get("modules", [])
        total_q = sum(len(m.get("questions", [])) for m in modules)
        critical_count = sum(1 for m in modules for q in m.get("questions", []) if q.get("critical"))

        messages.append({"role": "assistant", "content": json.dumps(discovery_data, sort_keys=True)})

        # LLM-judgment saturation check (#3). See requirements_discovery_node
        # for the rationale; same helper, scoped to DEPLOYMENT. Especially
        # high-leverage here because deployment_discovery already grounds
        # suggested_answers in workspace telemetry — when the scan answers
        # everything, the saturation check converts that signal into a
        # round-skip instead of another follow-up pass.
        complete, budget = await _maybe_discovery_saturation_check(
            gate="DEPLOYMENT",
            question_count=question_count,
            complete=complete,
            discovery_data=discovery_data,
            messages=messages,
            budget=budget,
        )

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
        max_cycles=_resolve_max_continuation_cycles(llm_dispatch_config or {}),
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
        logger.info(
            "[spec_review] Critique was not valid JSON (%s) — asking "
            "reviewer to re-emit.", exc,
        )
        # One-shot repair: re-prompt the reviewer with the schema and
        # the offending text before discarding its output. See
        # :func:`_repair_malformed_json`.
        async def _spec_repair_dispatch(msgs, bud):
            return await gateway.dispatch(
                messages=list(msgs),
                role=NodeRole.DOC_REVIEWER,
                budget_remaining_usd=bud,
            )
        _schema_hint = (
            "A JSON object with keys: 'issues' (array of strings), "
            "'gaps' (array of strings), 'clarity_items' (array of "
            "strings), 'followup_questions' (array of objects with "
            "fields id, text, critical). Any of the arrays may be "
            "empty; unknown keys are permitted."
        )
        repaired, new_budget = await _repair_malformed_json(
            raw_text=critique_response.content or "",
            schema_hint=_schema_hint,
            dispatch=_spec_repair_dispatch,
            budget_remaining_usd=new_budget,
            purpose="spec_review_critique",
        )
        result["new_budget_usd"] = new_budget
        if isinstance(repaired, dict):
            critique = repaired
        else:
            logger.warning(
                "[spec_review] Repair pass did not yield a JSON object — "
                "passing through.",
            )
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
            max_cycles=_resolve_max_continuation_cycles(llm_dispatch_config or {}),
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
    loop_counter = dict(state.get("loop_counter") or {})
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
    loop_counter = dict(state.get("loop_counter") or {})
    counter = loop_counter.get("review_code", 0)
    # In batch-mode, scope the reviewer to the current batch's files
    # rather than the cumulative session set — otherwise batch-N's review
    # re-reads files batch-(N-1) already reviewed. ``_scope_files_for_consumer``
    # falls back to ``modified_files`` for non-batch sessions and for the
    # very first invocation before patching has populated the batch list.
    modified_files = _scope_files_for_consumer(state)
    batch_id = int(state.get("current_batch_id") or 0)
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
        scope_label = f"batch {batch_id}" if batch_id else "session"
        logger.info(
            "[code_review] no modified_files in %s scope — skipping.", scope_label,
        )
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

    # Architecture-summary preamble — tells the reviewer to flag drift
    # between the code under review and the resolved §11 tables
    # (endpoint paths, schema names, contract location, component
    # paths). Empty string when the arch doc has no §11 block, in
    # which case the reviewer falls back to the prose document the
    # system prompt already carries.
    arch_preamble, _resolved_arch = _build_arch_summary_preamble(
        state, consumer="reviewer",
    )
    critique_user_prompt = (
        arch_preamble
        + "## Modified Files\n" + "\n".join(snapshot_chunks) +
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
        max_cycles=_resolve_max_continuation_cycles(state),
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
        logger.info(
            "[code_review] Critique was not valid JSON (%s) — asking "
            "reviewer to re-emit.", exc,
        )
        async def _code_repair_dispatch(msgs, bud):
            return await gateway.dispatch(
                messages=list(msgs),
                role=NodeRole.CODE_REVIEWER,
                budget_remaining_usd=bud,
            )
        _findings_schema = (
            "A JSON object with a top-level 'findings' key holding an "
            "array of finding objects. Each finding must include at "
            "least: 'file' (string), 'severity' (one of 'critical', "
            "'high', 'medium', 'low', 'info'), and 'message' (string). "
            "May also include 'line' (int), 'category' (string), "
            "'suggestion' (string). Emit an empty array when the code "
            "is clean."
        )
        repaired, new_budget = await _repair_malformed_json(
            raw_text=critique_response.content or "",
            schema_hint=_findings_schema,
            dispatch=_code_repair_dispatch,
            budget_remaining_usd=new_budget,
            purpose="code_review_critique",
        )
        if isinstance(repaired, dict) and isinstance(repaired.get("findings"), list):
            critique = repaired
            findings = repaired["findings"]
        else:
            logger.warning(
                "[code_review] Repair pass did not yield a valid "
                "findings JSON — passing through.",
            )
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
        out: dict[str, Any] = {
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
        # Phase K — mark batch-mode's review gate as passed.
        if int(state.get("current_batch_id") or 0):
            out["batch_gate_progress"] = _mark_batch_gate(
                state, "review_passed",
            )
        return out

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
            max_cycles=_resolve_max_continuation_cycles(state),
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
        "batch_modified_files": _extend_batch_scope(state, new_modified_files),
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
        install_path = None

    # v5 traceability audit — SQL-backed, surfaces BOTH untraced
    # requirements AND untested acceptance criteria from state.db.
    # When the audit reports any gap AND ``traceability.enforce`` is
    # true (default), the session blocks via a synthetic compiler
    # error so route_after_installation_doc reroutes to HITL instead
    # of the END terminator. Operators with broken specs can set
    # ``traceability.enforce = false`` in .harness_config.json to
    # disable the block during transition (the report still prints).
    traceability_blocked = False
    try:
        from harness.traceability import audit_workspace, format_report
        report = audit_workspace(workspace_path)
        if report is not None and report.has_failures():
            report_text = format_report(report)
            tr_cfg = (state.get("harness_config") or {}).get(
                "traceability", {},
            )
            enforce = bool(tr_cfg.get("enforce", True))
            if report_text:
                print()
                if enforce:
                    print("==================== TRACEABILITY BLOCK ====================")
                    print(report_text)
                    print("Set traceability.enforce=false in .harness_config.json")
                    print("to bypass and ship anyway (NOT RECOMMENDED).")
                    print("==========================================================")
                else:
                    print("[traceability advisory — enforce disabled]")
                    print(report_text)
                logger.info(
                    "[traceability] reqs %d/%d (%.0f%%), ACs %d/%d (%.0f%%); "
                    "untraced=%d, untested=%d; enforce=%s",
                    report.traced_reqs, report.total_reqs,
                    report.req_coverage_pct,
                    report.verified_acs, report.total_acs,
                    report.ac_coverage_pct,
                    len(report.untraced), len(report.untested_acs),
                    enforce,
                )
            if enforce:
                traceability_blocked = True
    except Exception as exc:  # noqa: BLE001 — audit must never break the run
        logger.debug("[traceability] Audit failed (%s); skipping report.", exc)

    # End-of-run "application usage guide" — prints a visible paragraph
    # to stdout so the operator doesn't have to open INSTALLATION.md to
    # learn what to do next. Always renders the deterministic backbone
    # (URLs from blueprint published ports + CLI commands for known
    # data-plane images); the LLM polish prepends a one-sentence app
    # summary when configured. Falls back to a single "see
    # INSTALLATION.md" pointer when there is no blueprint (no deploy or
    # --deploy-dev=false).
    new_budget = await _emit_application_usage_guide(
        state=state,
        blueprint=blueprint,
        install_path=install_path,
        budget=state.get("budget_remaining_usd", 0.0),
    )

    node_state_delta: dict[str, Any] = {"current_node": "installation_doc"}
    out: dict[str, Any] = {
        "budget_remaining_usd": new_budget,
    }
    if install_path:
        out["installation_doc_path"] = install_path
    if traceability_blocked:
        # Surface the failure to the router so it reroutes to HITL
        # instead of END. The diagnostic carries enough context for
        # the operator to find the offending FRs/ACs without re-running
        # the audit.
        node_state_delta["traceability_blocked"] = True
        out["exit_code"] = 1
    out["node_state"] = node_state_delta
    return out


def route_after_installation_doc(state: AgentState) -> str:
    """Conditional edge after ``installation_doc_node``.

    Returns ``"human_intervention_node"`` when the v5 traceability
    audit gated the session with ``enforce=true``; otherwise
    ``END`` (the standard terminator). Operators with broken
    specs can opt out by setting ``traceability.enforce = false``
    in ``.harness_config.json`` (the gate then logs but does not
    block — the audit print still fires).
    """
    from langgraph.graph import END
    node_state = state.get("node_state") or {}
    if node_state.get("traceability_blocked"):
        return "human_intervention_node"
    return END


async def _emit_application_usage_guide(
    *,
    state: AgentState,
    blueprint: Optional[dict[str, Any]],
    install_path: Optional[str],
    budget: float,
) -> float:
    """Print the end-of-run application usage guide to stdout.

    Returns the (possibly-updated) ``budget_remaining_usd`` so the
    caller can thread it back into state. Pure stdout side-effect plus
    one optional cheap LLM call for the summary polish.
    """
    from harness.deploy import render_access_hints

    # When deployment_node populated a blueprint (any non-skipped path),
    # render the full access-hint paragraph. The healthy list lives next
    # to the blueprint on node_state.deployment.
    deployment_state = (state.get("node_state") or {}).get("deployment") or {}
    healthy_services: list[str] = []
    if isinstance(deployment_state, dict):
        h = deployment_state.get("healthy") or []
        if isinstance(h, list):
            healthy_services = [str(x) for x in h if isinstance(x, str)]

    hints = render_access_hints(
        blueprint=blueprint, healthy=healthy_services,
    )

    if not hints:
        # No-deploy / no-blueprint fallback: a single pointer line so
        # greenfield runs without --deploy-dev still get something
        # visible, instead of an unannounced silent exit.
        if install_path:
            print()
            print(
                "Setup instructions written to "
                f"{install_path} — follow them to run the app."
            )
        return budget

    # LLM polish (#5 judgment touchpoint). One-sentence summary
    # prepended to the deterministic hints; entirely optional and
    # kill-switched via config.llm_judgment.app_usage_guide.
    summary = ""
    gw = get_gateway()
    if gw is not None and bool(getattr(
        gw.config, "llm_judgment_app_usage_guide", True,
    )):
        prompt = _build_app_usage_summary_prompt(
            blueprint=blueprint,
            healthy=healthy_services,
            workspace_path=state.get("workspace_path", ""),
        )
        polish, budget = await _maybe_judgment_llm(
            prompt=prompt,
            budget_remaining_usd=budget,
            purpose="app_usage_guide",
            enabled=True,
        )
        if polish:
            # Trim to one sentence at most — the LLM occasionally
            # wanders. Take everything up to the first sentence
            # terminator that's followed by whitespace or end-of-string.
            first_sentence = re.split(
                r"(?<=[.!?])\s+", polish.strip(), maxsplit=1,
            )[0].strip()
            if first_sentence:
                summary = first_sentence

    print()
    if summary:
        print(f" {summary}")
        print()
    print(hints)
    if install_path:
        print(f" Full setup guide: {install_path}")
    return budget


def _build_app_usage_summary_prompt(
    *,
    blueprint: Optional[dict[str, Any]],
    healthy: list[str],
    workspace_path: str,
) -> str:
    """One-sentence summary prompt for the app_usage_guide judgment
    call. Compact on purpose — the LLM gets the blueprint shape plus a
    couple of evidence anchors so it doesn't have to guess what the
    app is."""
    services_block = "(none)"
    if isinstance(blueprint, dict):
        svcs = blueprint.get("services") or {}
        if isinstance(svcs, dict) and svcs:
            services_block = "\n".join(
                f"  - {name}: image={str(svc.get('base_image', '?'))}, "
                f"ports={svc.get('ports') or []}"
                for name, svc in sorted(svcs.items())
                if isinstance(svc, dict)
            )
    healthy_block = ", ".join(sorted(healthy)) if healthy else "(unknown)"
    return (
        "Compose ONE short sentence (max ~20 words, no markdown, no "
        "lists) telling the operator WHAT the application that was "
        "just deployed actually is. Use the deployment blueprint below "
        "as the source of truth. Good examples:\n"
        "  - \"A React storefront with a Node/Express API and a "
        "PostgreSQL database.\"\n"
        "  - \"A Flask-based REST API backed by Redis cache and "
        "MongoDB.\"\n"
        "Bad examples (too vague): \"A web application.\" / "
        "\"A containerized stack.\"\n"
        "Reply with the sentence and NOTHING else.\n\n"
        f"Workspace: {workspace_path}\n"
        f"Healthy services: {healthy_block}\n"
        f"Services:\n{services_block}\n"
    )


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
        elif gate == "STORIES":
            # Refine STORIES by re-running decomposition. The existing
            # rows stay in the DB (so STORY-N numbering keeps growing
            # monotonically); the operator can prune via direct DB edit
            # before the second pass runs.
            return "decomposition_node"

    # approve or manual: proceed forward
    if gate == "REQUIREMENTS":
        return "architecture_discovery_node"
    elif gate == "ARCHITECTURE":
        # Story-mode opt-in: when ``decomposition_enabled``, the approved
        # architecture spec hands off to the decomposition LLM instead of
        # going straight into monolithic patching. The decomposition node
        # then routes back through the STORIES gatekeeper.
        if state.get("decomposition_enabled", False):
            # PATCH flow on an agile-managed workspace with at least one
            # DONE story: re-classify those DONE stories against the
            # revised spec FIRST (story_reopen_node), then let the
            # decomposition node propose any new stories (augment mode).
            # Build flow and first-pass patch (no existing stories) skip
            # straight to decomposition.
            if (
                state.get("flow") == FLOW_PATCH
                and _workspace_has_done_stories(state)
            ):
                return "story_reopen_node"
            return "decomposition_node"
        return "patching_node"
    elif gate == "STORIES":
        return "batch_planner_node"
    elif gate == "DEPLOYMENT":
        return "deployment_node"

    return "__end__"


def _workspace_has_done_stories(state: AgentState) -> bool:
    """Return True when the global state DB has at least one DONE story
    for this workspace. Best-effort: degrades to False on any error so
    the router falls back to today's behavior.
    """
    workspace_path = state.get("workspace_path", "")
    if not workspace_path:
        return False
    try:
        from harness import story_state
        app = story_state.app_name_for_workspace(workspace_path)
        db = story_state.state_db_path()
        if not os.path.isfile(db):
            return False
        import sqlite3
        conn = sqlite3.connect(db)
        try:
            cur = conn.execute(
                "SELECT 1 FROM stories WHERE workspace = ? "
                "AND status = 'done' LIMIT 1",
                (app,),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — router must never raise
        return False


# ---------------------------------------------------------------------------
# 7. Route After HITL: Always Back to Compiler
# ---------------------------------------------------------------------------

def route_after_security_scan(state: AgentState) -> Literal[
    "repair_node", "human_intervention_node", "deployment_discovery_node",
    "deployment_node", "installation_doc_node", "compiler_node",
    "end_of_session_regression_node", "__end__",
]:
    """
    Conditional edge router executed after security_scan_node completes.

    Decision matrix:
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

        # Phase G — explicit end-of-session regression gate. The first
        # time security_scan exits clean, run the full build/test pack
        # one more time against the post-security-repair tree to catch
        # any breakage introduced after the last per-batch test pass.
        # Skipped on a session marker (the regression counter being
        # non-zero means we already burned this gate; the repair loop
        # from a failed EoS regression brings us back via
        # repair → compiler → code_review → security_scan, and we MUST
        # NOT re-enter the EoS gate or the loop runs forever).
        if loop_counter.get("end_of_session_regression_repair", 0) == 0:
            logger.info(
                "[router] Security scan clean — entering end-of-session "
                "regression check before deployment."
            )
            return "end_of_session_regression_node"

        # Post-deployment clean scan: do NOT re-enter discovery. The deploy
        # pipeline is one-shot per session — ``deployment_node`` having
        # returned at least once means we are PAST that gate. Without this
        # guard, route_after_security_scan kept rotating clean scans back
        # into ``deployment_discovery_node → write_spec_node → gatekeeper →
        # deployment_node``, producing the infinite loop observed in session
        # 951f102f (three full discovery+deploy passes in 6 minutes before
        # the repair budget was exhausted). The marker is
        # ``node_state.deployment`` — set by every terminal return path in
        # ``deployment_node`` (success / skipped / errored).
        ns = state.get("node_state") or {}
        if isinstance(ns, dict) and isinstance(ns.get("deployment"), dict):
            dep_state = ns["deployment"]
            logger.info(
                "[router] Security scan clean AND deployment_node has "
                "already run this session (deployment=%s). Routing to "
                "installation_doc_node — NOT re-entering deployment "
                "discovery.",
                dep_state.get(
                    "success",
                    dep_state.get("skipped", dep_state.get("phase", "unknown")),
                ),
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
    # into ``docs/SPEC_ARCHITECTURE.md``). Route to HITL rather than
    # ``__end__`` so the operator can still rescue the work: ack the
    # finding manually, fix it themselves, or suspend & resume. The
    # earlier "terminate the run" behavior killed local edits in flight
    # because the LangGraph session was unrecoverable without a fresh
    # resume cycle. The HITL menu surfaces "ceiling reached" so the
    # operator sees they're at the wall and not in a normal loop.
    if sec_attempts >= hard_ceiling:
        logger.error(
            "[router] Security HITL ping-pong hard ceiling reached "
            "(attempts=%d, ceiling=%d). %d finding(s) survived %d HITL "
            "resume(s) without being fixed. Escalating to HITL — operator "
            "must inspect findings manually, ack/suppress, or suspend.",
            sec_attempts, hard_ceiling, len(compiler_errors),
            sec_attempts - max_sec_attempts,
        )
        return "human_intervention_node"

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
# 6d.5. End-of-session regression — final test pack run after security_scan
#       passes, before deployment (Phase G).
# ---------------------------------------------------------------------------

async def end_of_session_regression_node(state: AgentState) -> dict[str, Any]:
    """Run the full build/test pack one final time after security_scan
    passes, before deployment.

    The per-batch pipeline already runs the test pack inside each batch's
    verification chain. This node fires ONCE at end-of-session to catch
    breakage introduced by the security-scan repair loop — which patches
    code AFTER the last per-batch test pass, so without this gate a
    security fix could deploy a regression. Delegates to ``compiler_node``
    for the actual build execution; differs only in:

    - Increments ``loop_counter["end_of_session_regression_repair"]``
      with its own configurable cap
      (``max_end_of_session_regression_cycles``, default 3).
    - Stamps ``node_state.current_node = "end_of_session_regression"``
      and ``node_state.end_of_session_phase = True`` so the router knows
      we're in the post-security final-verify phase, not a per-batch
      compile cycle.
    """
    loop_counter = dict(state.get("loop_counter", {}) or {})
    counter = int(
        loop_counter.get("end_of_session_regression_repair", 0) or 0
    )
    loop_counter["end_of_session_regression_repair"] = counter + 1

    logger.info(
        "[end_of_session_regression] Final regression check (attempt %d) "
        "after security_scan, before deployment.", counter + 1,
    )

    result = await compiler_node(state)
    if not isinstance(result, dict):
        return result  # type: ignore[return-value]

    ns = dict(result.get("node_state", {}) or {})
    ns["current_node"] = "end_of_session_regression"
    ns["end_of_session_phase"] = True
    result["node_state"] = ns

    # compiler_node returned its own loop_counter; merge our incremented
    # counter back in so the router's cap check sees the right value.
    merged_counter = dict(result.get("loop_counter", {}) or {})
    merged_counter["end_of_session_regression_repair"] = loop_counter[
        "end_of_session_regression_repair"
    ]
    result["loop_counter"] = merged_counter
    return result


def _resolve_post_eos_destination(state: AgentState) -> str:
    """Where to go when both security_scan AND end-of-session regression
    pass clean.

    Replicates the clean-path decision tree from
    ``route_after_security_scan`` so the deployment-vs-doc routing is
    consistent regardless of which gate cleared last. Same precedence:

    1. ``deployment_node`` has already run this session → ``installation_doc_node``
       (guard against the 951f102f re-entry loop — kept for parity even
       though regression-then-deploy ordering makes a second deploy hop
       very unlikely).
    2. ``dev_deployment`` is False → ``installation_doc_node``.
    3. ``cd_discovery`` is False → ``deployment_node`` (telemetry-only blueprint).
    4. otherwise → ``deployment_discovery_node`` (LLM-driven blueprint).
    """
    ns = state.get("node_state") or {}
    if isinstance(ns, dict) and isinstance(ns.get("deployment"), dict):
        return "installation_doc_node"
    if not state.get("dev_deployment", False):
        return "installation_doc_node"
    if not state.get("cd_discovery", False):
        return "deployment_node"
    return "deployment_discovery_node"


def route_after_end_of_session_regression(state: AgentState) -> Literal[
    "deployment_discovery_node",
    "deployment_node",
    "installation_doc_node",
    "repair_node",
    "human_intervention_node",
]:
    """After the end-of-session regression run:

    - budget exhausted → HITL
    - clean exit → onward to the deployment / installation-doc tail
      via ``_resolve_post_eos_destination``
    - exit non-zero AND repair budget remains → repair_node
    - exit non-zero AND repair cap reached → HITL

    Repair cap is read from
    ``gateway.config.max_end_of_session_regression_cycles`` (default 3,
    configurable). When the gateway is unavailable (some tests), the
    in-code default of 3 applies.
    """
    exit_code: int = state.get("exit_code", -1)
    loop_counter: dict[str, Any] = state.get("loop_counter", {})
    counter: int = int(
        loop_counter.get("end_of_session_regression_repair", 0) or 0
    )
    budget_remaining: float = state.get("budget_remaining_usd", 0.0)

    gw = get_gateway()
    max_cycles: int = (
        int(getattr(gw.config, "max_end_of_session_regression_cycles", 3))
        if gw is not None else 3
    )

    if budget_remaining <= 0.0:
        logger.warning(
            "[router] Budget exhausted ($%.4f) at end_of_session_regression. "
            "Routing to HITL.", budget_remaining,
        )
        return "human_intervention_node"

    if exit_code == 0:
        dest = _resolve_post_eos_destination(state)
        logger.info(
            "[router] end_of_session_regression clean. Routing to %s.", dest,
        )
        return dest  # type: ignore[return-value]

    if counter >= max_cycles:
        logger.warning(
            "[router] end_of_session_regression repair cap reached (%d/%d). "
            "Routing to HITL.", counter, max_cycles,
        )
        return "human_intervention_node"

    logger.info(
        "[router] end_of_session_regression failed (exit %d). "
        "Routing to repair_node (attempt %d/%d).",
        exit_code, counter + 1, max_cycles,
    )
    return "repair_node"


# ---------------------------------------------------------------------------
# 6e. Route After Deployment: success → installation_doc, else compiler logic
# ---------------------------------------------------------------------------

def route_after_deployment(state: AgentState) -> Literal[
    "installation_doc_node",
    "repair_node",
    "human_intervention_node",
]:
    """Route after deployment_node terminates.

    Terminal by design — the deploy pipeline is one-shot per session so
    every post-deploy state ends in either docs, repair, or HITL. NEVER
    delegates back to ``route_after_compiler`` (the historic fall-through
    silently re-entered ``security_scan_node`` whenever
    ``deployment.success`` wasn't True, which made
    ``deployment_node → security_scan → deployment_discovery → deployment_node``
    a closed loop — see session 951f102f for the worked example). Each
    outcome is now explicit:

        deployment.success == True     → installation_doc_node (clean deploy)
        deployment.skipped == True     → installation_doc_node (user declined
                                          preview / deployment_config.enabled
                                          is false — emit the docs we have
                                          and end the run)
        compiler_errors populated      → repair_node (deployment_node emitted
                                          a real DEPLOYMENT_* diagnostic that
                                          the LLM can attempt to fix)
        neither set                    → human_intervention_node (the
                                          missing-success-with-no-errors trap
                                          that produced the loop above; surface
                                          to the operator with the deployment
                                          dict in node_state instead of
                                          burning more iterations)
    """
    ns = state.get("node_state") or {}
    deployment = ns.get("deployment") if isinstance(ns, dict) else None
    compiler_errors = state.get("compiler_errors") or []

    if isinstance(deployment, dict) and deployment.get("success") is True:
        logger.info(
            "[router] Deployment succeeded. Routing to installation_doc_node "
            "before END."
        )
        return "installation_doc_node"

    if isinstance(deployment, dict) and deployment.get("skipped") is True:
        logger.info(
            "[router] Deployment skipped (%s). Routing to "
            "installation_doc_node before END — NOT re-entering security "
            "scan.",
            deployment.get("reason", "unknown"),
        )
        return "installation_doc_node"

    if compiler_errors:
        loop_counter = state.get("loop_counter") or {}
        attempts = int(loop_counter.get("deployment", 0))
        dep_cfg = state.get("deployment_config") or {}
        max_attempts = int(dep_cfg.get("max_deployment_attempts", 3))
        if attempts >= max_attempts:
            logger.warning(
                "[router] Deployment loop limit reached (%d/%d). %d "
                "diagnostic(s) survived repair. Routing to HITL.",
                attempts, max_attempts, len(compiler_errors),
            )
            return "human_intervention_node"
        logger.info(
            "[router] Deployment node emitted %d diagnostic(s) (attempt %d/%d). "
            "Routing to repair_node.",
            len(compiler_errors), attempts, max_attempts,
        )
        return "repair_node"

    logger.warning(
        "[router] Deployment node returned neither success nor errors "
        "(deployment=%r). Routing to HITL rather than re-entering the "
        "compile/scan loop.",
        deployment,
    )
    return "human_intervention_node"


# ---------------------------------------------------------------------------
# 7. Route After HITL: Always Back to Compiler
# ---------------------------------------------------------------------------

def route_after_hitl(state: AgentState) -> Literal[
    "compiler_node", "decomposition_node", "traceability_node", "__end__"
]:
    """
    After human intervention, route back to the node that triggered the
    escalation so the developer's fix is re-evaluated by the right phase.

    Default is ``compiler_node`` — that handles the loop/repair/env-misconfig
    family that dominates real escalations. The exceptions branch on the
    ``hitl_trigger`` label that ``human_intervention_node`` stamps via
    ``_infer_hitl_trigger`` so pre-code escalations don't get routed into
    a build of an empty workspace.

    Exceptions:
        - Developer chose to abandon ([q]) → END with git rollback
        - Developer chose to suspend ([s]) → END without rollback
        - Trigger was decomposition_{validation_failed,missing} →
          ``decomposition_node`` (compiler_node would find no source
          files and bounce straight back to HITL — surfaced by the
          FinancialResearch run where the depends_on-cycle HITL was
          resumed and the resulting pytest exit=5 re-escalated 3
          seconds later).
        - Trigger was traceability_block → ``traceability_node``
          (the build is already green; re-running the compiler would
          just exit 0 → END while the coverage gap is still open).

    Note: Memory cleanse for HITL resolution is handled inside
    compiler_node when exit_code == 0 after the re-validation build
    passes. For the decomposition/traceability re-entry paths the
    cleanse fires the next time the build completes successfully.
    """
    node_state: dict[str, Any] = state.get("node_state", {})
    if node_state.get("hitl_suspend", False):
        logger.info("[router] HITL: Developer chose to suspend. Routing to END.")
        return "__end__"
    if node_state.get("hitl_abandon", False):
        logger.info("[router] HITL: Developer chose to abandon. Routing to END.")
        return "__end__"

    trigger = str(node_state.get("hitl_trigger", "") or "")
    if trigger in ("decomposition_validation_failed", "decomposition_missing"):
        logger.info(
            "[router] HITL resolved (trigger=%s). Routing to decomposition_node "
            "for re-validation — compiler_node would find no source files.",
            trigger,
        )
        return "decomposition_node"
    if trigger == "traceability_block":
        logger.info(
            "[router] HITL resolved (trigger=%s). Routing to traceability_node "
            "— the build was already green when the gate failed.",
            trigger,
        )
        return "traceability_node"

    logger.info(
        "[router] HITL resolved (trigger=%s). Routing to compiler_node "
        "for re-validation.",
        trigger or "unknown",
    )
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
    graph.add_node("reverse_spec_node", reverse_spec_node)
    graph.add_node("patch_reconcile_node", patch_reconcile_node)
    graph.add_node("story_reopen_node", story_reopen_node)
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

    # Story-mode nodes (Agile decomposition + per-story TDD). Inert when
    # ``decomposition_enabled`` stays False — every router that branches into
    # them gates on that flag, so today's monolithic flow keeps running
    # exactly as it does on main.
    from harness.decomposition import decomposition_node as _decomposition_node
    from harness.spec_reconciler import (
        spec_reconciler_node as _spec_reconciler_node,
    )
    from harness.story_loop import (
        batch_planner_node as _batch_planner_node,
        story_loop_node as _story_loop_node,
        story_complete_node as _story_complete_node,
        batch_commit_node as _batch_commit_node,
        traceability_node as _traceability_node,
        route_after_batch_planner as _route_after_batch_planner,
        route_after_story_loop as _route_after_story_loop,
        route_after_story_complete as _route_after_story_complete,
        route_after_batch_commit as _route_after_batch_commit,
    )
    graph.add_node("decomposition_node", _decomposition_node)  # type: ignore[type-var]
    # spec_reconciler_node runs between decomposition_node and the STORIES
    # gatekeeper. It parses SPEC_REQUIREMENTS.md deterministically and
    # rewrites the workspace's stories/features rows using spec-authored
    # IDs — LLM output becomes ENRICHMENT (scope_files) only. Guards
    # against LLM renumbering / silent story-drop / phantom-feature drift.
    graph.add_node("spec_reconciler_node", _spec_reconciler_node)  # type: ignore[type-var]
    graph.add_node("batch_planner_node", _batch_planner_node)  # type: ignore[type-var]
    graph.add_node("story_loop_node", _story_loop_node)  # type: ignore[type-var]
    # Phase F removed ``story_test_first_node``. Its xfail-stub
    # generation duplicated what the patching LLM already produces from
    # the story preamble (``_build_story_preamble``), and the per-batch
    # verification chain (compile → review → test) runs the real test
    # suite once per batch — strict-xfail markers were never wired into
    # the routing anyway.
    graph.add_node("story_complete_node", _story_complete_node)  # type: ignore[type-var]
    # Phase E: batch_commit_node fires when a batch is fully resolved —
    # marks every constituent story done, optionally commits the batch
    # under a ``BATCH-N: …`` message, and resets per-batch state and
    # loop counters. ``route_after_story_loop`` routes batch-exhausted
    # transitions through here on the way to ``batch_planner_node``.
    graph.add_node("batch_commit_node", _batch_commit_node)  # type: ignore[type-var]
    graph.add_node("traceability_node", _traceability_node)  # type: ignore[type-var]

    # Register deployment spec node
    graph.add_node("generate_deployment_spec_node", generate_deployment_spec_node)

    # `teane test` entry node — verifies prereqs (clean deploy + build|patch
    # markers via harness/flow_state.py) then exits straight to END. Phases
    # 2-6 will expand the body to generate scenarios + run them + emit CRs.
    from harness.test_target import test_node
    graph.add_node("test_node", test_node)

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
            # Agile mode + skip_discovery: enter the per-batch story
            # pipeline directly. Mirrors the ARCHITECTURE-gate fan-out
            # so `--agile=true` engages even when specs were synthesised
            # pre-graph (the common case in cmd_run).
            "decomposition_node": "decomposition_node",
            "story_reopen_node": "story_reopen_node",
            "ingest_change_requests_node": "ingest_change_requests_node",
            # flow=patch + generate_specs → reverse-engineer specs from
            # the existing codebase, then funnel into the regular
            # requirements/architecture discovery + review chain.
            "reverse_spec_node": "reverse_spec_node",
            "patch_reconcile_node": "patch_reconcile_node",
            # flow=deploy entry edges: skip discovery / planning / patching
            # entirely and enter the deployment chain directly.
            "deployment_discovery_node": "deployment_discovery_node",
            "generate_deployment_spec_node": "generate_deployment_spec_node",
            # flow=test entry edge: prereq-gated no-op (Phase 1); later
            # phases bolt the e2e pipeline behind this same edge.
            "test_node": "test_node",
        },
    )

    # `teane test` exits straight to END for now. Phase 5 may rewire this
    # to a conditional edge that loops back through compile/repair when
    # generated scenarios fail to parse — TBD.
    graph.add_edge("test_node", END)

    # reverse_spec_node feeds its synthesised drafts into the standard
    # requirements/architecture discovery chain so spec_review_node still
    # vets them and the operator HITL gates still fire.
    graph.add_edge("reverse_spec_node", "requirements_discovery_node")

    # patch_reconcile_node appends the reconcile preamble + spec excerpts
    # to messages, then hands off to planning_node for the actual blueprint
    # generation. The planner produces a plan that respects "only change
    # what's drifted", and the rest of the standard build chain follows.
    graph.add_edge("patch_reconcile_node", "planning_node")

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
            # Story-mode targets — inert when decomposition_enabled=False.
            "decomposition_node": "decomposition_node",
            "batch_planner_node": "batch_planner_node",
            # PATCH flow + agile + existing DONE stories: classify the
            # existing stories against the revised spec FIRST, then fall
            # through to decomposition_node in augment mode.
            "story_reopen_node": "story_reopen_node",
            "__end__": END,
        },
    )

    # story_reopen_node always hands off to decomposition_node next: any
    # `reopen` verdicts flipped DONE → REOPENED in the DB, and the
    # decomposition pass picks up brand-new stories from the revised spec
    # in augment mode.
    graph.add_edge("story_reopen_node", "decomposition_node")

    # =====================================================================
    # Story-mode pipeline (Agile decomposition + per-batch verification).
    #
    #   decomposition_node → human_gatekeeper(STORIES)
    #   gatekeeper STORIES approve → batch_planner_node
    #   batch_planner_node →┬─ story_loop_node (batch planned)
    #                       └─ traceability_node (all done / stalled)
    #   story_loop_node    →┬─ patching_node (next story picked; story
    #                       │   preamble carries acceptance criteria)
    #                       └─ speculative_node (batch exhausted → enter
    #                           per-batch verification chain)
    #   patching_node      →┬─ story_loop_node (advance to next story)
    #                       └─ speculative_node (monolithic / batch repair)
    #   code_review_node   →┬─ batch_commit_node (current_batch_id > 0)
    #                       │   then batch_planner_node for next batch
    #                       ├─ story_complete_node (legacy non-batch story-mode)
    #                       └─ {compiler_node | security_scan_node}
    #   traceability_node  → security_scan_node (rejoins existing tail)
    # =====================================================================
    def route_after_decomposition(state: AgentState) -> Literal[
        "spec_reconciler_node", "human_intervention_node"
    ]:
        """Failure routing for ``decomposition_node``.

        Validation / dispatch / JSON-decode failures set
        ``node_state.decomposition_failed`` so the pipeline can divert to
        HITL instead of presenting an empty STORIES gate (which the
        gatekeeper would happily let the developer "approve", and the
        batch planner would then see an empty DB and report
        ``all_complete=True`` — generating code with zero traceability).

        Success routes through ``spec_reconciler_node`` before the
        STORIES gate so spec-authored IDs override any LLM renumbering.
        """
        ns = state.get("node_state", {}) or {}
        if ns.get("decomposition_failed"):
            logger.warning(
                "[router] decomposition_node failed (%s); routing to HITL.",
                ns.get("error", "unknown"),
            )
            return "human_intervention_node"
        return "spec_reconciler_node"

    graph.add_conditional_edges(
        "decomposition_node",
        route_after_decomposition,
        {
            "spec_reconciler_node": "spec_reconciler_node",
            "human_intervention_node": "human_intervention_node",
        },
    )

    def route_after_spec_reconciler(state: AgentState) -> Literal[
        "human_gatekeeper_node", "human_intervention_node"
    ]:
        """Route reconciler → STORIES gate on success, HITL on failure.

        ``reconcile_failed`` covers parse / SQL / regen errors from the
        deterministic path. A missing spec file (``skipped_reason ==
        'spec_missing'``) or CR-mode skip both fall through to the
        STORIES gate — the LLM's output survives untouched in those
        paths, which matches pre-reconciler behavior.
        """
        ns = state.get("node_state", {}) or {}
        if ns.get("reconcile_failed"):
            logger.warning(
                "[router] spec_reconciler_node failed (%s); routing to HITL.",
                ns.get("error", "unknown"),
            )
            return "human_intervention_node"
        return "human_gatekeeper_node"

    graph.add_conditional_edges(
        "spec_reconciler_node",
        route_after_spec_reconciler,
        {
            "human_gatekeeper_node": "human_gatekeeper_node",
            "human_intervention_node": "human_intervention_node",
        },
    )
    graph.add_conditional_edges(
        "batch_planner_node",
        _route_after_batch_planner,
        {
            "story_loop_node": "story_loop_node",
            "traceability_node": "traceability_node",
            "human_intervention_node": "human_intervention_node",
        },
    )
    graph.add_conditional_edges(
        "story_loop_node",
        _route_after_story_loop,
        {
            # Phase F: route directly to patching_node — the story
            # acceptance criteria are injected into the patching LLM
            # preamble by _build_story_preamble.
            "patching_node": "patching_node",
            # Phase E.3: batch-exhausted (all stories patched) →
            # speculative_node to kick off the per-batch verification
            # chain. The chain ends in code_review, which routes to
            # batch_commit_node via route_after_code_review when
            # current_batch_id is set.
            "speculative_node": "speculative_node",
            # Phase K resume short-circuits: skip to the next un-passed
            # gate when the crash happened mid-verification.
            "code_review_node": "code_review_node",
            "batch_commit_node": "batch_commit_node",
        },
    )
    graph.add_conditional_edges(
        "story_complete_node",
        _route_after_story_complete,
        {
            "story_loop_node": "story_loop_node",
        },
    )
    graph.add_conditional_edges(
        "batch_commit_node",
        _route_after_batch_commit,
        {
            "batch_planner_node": "batch_planner_node",
        },
    )
    graph.add_edge("traceability_node", "security_scan_node")

    # Architecture discovery loop (entered from gatekeeper approve)
    graph.add_edge("architecture_discovery_node", "discovery_interview_loop")

    # =====================================================================
    # Code generation pipeline (after ARCHITECTURE gate approved).
    #
    # Phase E.3 — In batch-mode (``current_batch_id > 0``) with an
    # active story (``current_story_id`` set), patching_node loops back
    # to story_loop_node so the *next* story in the batch can patch
    # before the verification chain (speculative → compile → review)
    # fires *once* against the batch's combined patches. The old
    # per-story chain ran the whole verification path for every story.
    # In monolithic mode (no batch active) the edge is the original
    # patching → speculative.
    # =====================================================================
    def route_after_patching(state: AgentState) -> Literal[
        "story_loop_node", "speculative_node", "human_intervention_node"
    ]:
        """Advance to the next story in the batch, kick off the per-batch
        verification chain, or escalate to HITL when the patching turn
        is stuck producing zero patches.

        HITL guard (added after a story-mode session burned ~1h22m and
        $18 looping story_loop ↔ patching with patches=0 every iteration):
        when ``loop_counter['consecutive_zero_patch_rounds'] >= 2`` the
        LLM has failed to land any patch for two consecutive turns. Same
        threshold as ``route_after_compiler`` for the repair loop, so
        the two zero-patch tripwires behave symmetrically.

        In monolithic mode (no batch active) or when a story-mode batch
        has no live story cursor (e.g. a batch-level repair re-enters
        patching), this falls straight through to ``speculative_node``
        — the historical behavior.
        """
        loop_counter = state.get("loop_counter", {}) or {}
        # Layer 3 — global no-progress failsafe runs first so it can
        # short-circuit any other route decision when budget has been
        # bleeding without progress.
        from harness.no_progress import tripped as _np_tripped
        if _np_tripped(loop_counter):
            logger.error(
                "[route_after_patching] no-progress failsafe tripped — "
                "budget spent without producing patches exceeded the "
                "threshold. Escalating to human_intervention_node."
            )
            return "human_intervention_node"
        consecutive_zero = int(
            loop_counter.get("consecutive_zero_patch_rounds", 0) or 0
        )
        if consecutive_zero >= 2:
            logger.warning(
                "[route_after_patching] consecutive_zero_patch_rounds=%d ≥ 2; "
                "escalating to human_intervention_node — patching is stuck "
                "in a no-progress loop.",
                consecutive_zero,
            )
            return "human_intervention_node"
        if int(state.get("current_batch_id") or 0) and (
            state.get("current_story_id") or ""
        ):
            return "story_loop_node"
        return "speculative_node"

    graph.add_conditional_edges(
        "patching_node",
        route_after_patching,
        {
            "story_loop_node": "story_loop_node",
            "speculative_node": "speculative_node",
            "human_intervention_node": "human_intervention_node",
        },
    )
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
        "compiler_node", "security_scan_node",
        "story_complete_node", "batch_commit_node"
    ]:
        """Route after the reviewer finishes.

        Decision matrix:

        1. Reviewer re-patched → ``compiler_node`` (re-validate the
           re-patch against the build).
        2. Batch-mode (``current_batch_id > 0``) → ``batch_commit_node``
           (Phase E.3 — the per-batch verification chain just passed,
           now seal the batch). This fires regardless of whether
           ``current_story_id`` is set: ``story_loop_node`` clears it
           when batch_complete triggers verification, but if a node in
           the chain re-set it we still want batch sealing.
        3. Legacy per-story story-mode (no batch active but
           ``current_story_id`` set) → ``story_complete_node`` to keep
           the older single-story TDD path working for tests that
           bypass the batch planner.
        4. Monolithic → ``security_scan_node`` (today's behavior).
        """
        node_state = state.get("node_state", {}) or {}
        if node_state.get("repatched", False):
            logger.info("[router] code_review re-patched — re-running compiler.")
            return "compiler_node"
        if int(state.get("current_batch_id") or 0):
            logger.info("[router] code_review clean in batch-mode — sealing via batch_commit_node.")
            return "batch_commit_node"
        if state.get("current_story_id"):
            return "story_complete_node"
        return "security_scan_node"

    graph.add_conditional_edges(
        "code_review_node",
        route_after_code_review,
        {
            "compiler_node": "compiler_node",
            "security_scan_node": "security_scan_node",
            "story_complete_node": "story_complete_node",
            "batch_commit_node": "batch_commit_node",
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
    # v5 traceability gate — when audit_workspace reports gaps AND
    # ``traceability.enforce`` is true (default), installation_doc_node
    # sets node_state.traceability_blocked and this conditional edge
    # reroutes the session to HITL instead of END so the operator
    # can address the gaps before shipping. enforce=false makes the
    # gate advisory-only (report prints, edge falls through to END).
    graph.add_conditional_edges(
        "installation_doc_node",
        route_after_installation_doc,
        {
            "human_intervention_node": "human_intervention_node",
            END: END,
        },
    )

    # Route security_scan clean → deployment discovery (or installation_doc
    # for --deploy-dev=false success exits); findings → repair_node
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
            # Audit #18 pre_exit_verify re-compile target (rare opt-in).
            "compiler_node": "compiler_node",
            # Phase G: final regression check before deployment.
            "end_of_session_regression_node": "end_of_session_regression_node",
            "__end__": END,
        },
    )

    # Phase G — register the end-of-session regression node + its router.
    # The node delegates to compiler_node for the actual build run, but
    # owns its own repair-iteration counter and routes to deployment
    # destinations on clean instead of code_review.
    graph.add_node(
        "end_of_session_regression_node", end_of_session_regression_node,
    )  # type: ignore[type-var]
    graph.add_conditional_edges(
        "end_of_session_regression_node",
        route_after_end_of_session_regression,
        {
            "deployment_discovery_node": "deployment_discovery_node",
            "deployment_node": "deployment_node",
            "installation_doc_node": "installation_doc_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
        },
    )

    # Deployment discovery → interview loop
    graph.add_edge("deployment_discovery_node", "discovery_interview_loop")

    # After generating deployment spec, route back to gatekeeper for approval/refine
    graph.add_edge("generate_deployment_spec_node", "human_gatekeeper_node")

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
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
        },
    )

    # After HITL resolution, go back to the upstream phase (compiler /
    # decomposition / traceability) so the developer's fix is re-evaluated
    # by the right node. Compiler is the default; decomposition and
    # traceability are taken when hitl_trigger says the escalation came
    # from a pre-code phase. END is reached only on suspend/abandon.
    graph.add_conditional_edges(
        "human_intervention_node",
        route_after_hitl,
        {
            "compiler_node": "compiler_node",
            "decomposition_node": "decomposition_node",
            "traceability_node": "traceability_node",
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

async def _reset_stale_gate_counters_on_resume(
    compiled_graph: Any, config: dict[str, Any]
) -> None:
    """Zero the repair-loop gate counters on every resume entry.

    The repair-loop gates (``consecutive_zero_patch_rounds``,
    ``no_progress_repairs``, ``consecutive_distraction_rounds``,
    ``missing_dep_consecutive_same``) accumulate across rounds. When a
    session ends — clean exit, external kill, or [s] Save & Quit — and
    the operator later runs ``teane resume``, the checkpoint restores
    these counters at their last values. If any of them was sitting at
    or above the cap when the session ended, the FIRST call to
    ``route_after_compiler`` on the resumed run trips HITL with the
    same trigger as last time, before any new repair round runs. That
    failure mode looks like "resume immediately HITLs for the same
    reason" to the operator and is what motivated this helper.

    The operator's mental model on ``teane resume`` is "give me another
    chance" — matching that semantics, we zero the gate counters so the
    new run gets a fresh repair budget. ``total_repairs`` (telemetry +
    hard ceiling) is preserved so a session can't run away forever
    across many resumes; the hard cap at ``2 * max_iterations`` still
    bounds total work.

    The helper runs UNCONDITIONALLY for every resume — orthogonal to
    :func:`_rewind_suspended_checkpoint`, which only fires for [s]
    suspends and rewrites the entire ``loop_counter`` dict with a
    hard-coded 4-key reset. When the rewind ran just before this
    helper, ``loop_counter`` is already minimal and the gate-key
    snapshot collapses to all-zeros → this helper short-circuits to a
    no-op via the ``any(before.values())`` guard.
    """
    try:
        state = await compiled_graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001 — defensive; never block resume
        logger.debug(
            "[run_graph] Could not read state for gate-counter reset: %s",
            exc,
        )
        return
    if state is None:
        return
    values = getattr(state, "values", None) or {}
    loop_counter = dict(values.get("loop_counter", {}) or {})
    gate_keys = (
        "consecutive_zero_patch_rounds",
        "no_progress_repairs",
        "consecutive_distraction_rounds",
        "consecutive_low_signal_rounds",
        "missing_dep_consecutive_same",
    )
    before = {k: int(loop_counter.get(k, 0) or 0) for k in gate_keys}
    if not any(before.values()):
        return
    for k in gate_keys:
        loop_counter[k] = 0
    logger.info(
        "[run_graph] Resume: zeroing stale gate counters %s so the operator's "
        "explicit resume gets a fresh repair budget. total_repairs preserved.",
        before,
    )
    try:
        await compiled_graph.aupdate_state(
            config,
            {"loop_counter": loop_counter},
        )
    except Exception as exc:  # noqa: BLE001 — log but don't crash resume
        logger.warning(
            "[run_graph] Failed to reset stale gate counters on resume: %s. "
            "First repair round may HITL immediately.",
            exc,
        )


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
    suspended_from: Optional[str] = node_state.get("suspended_from")
    if not node_state.get("hitl_suspend"):
        # Back-compat: pre-fix gatekeeper [s] checkpoints never set
        # hitl_suspend — sniff gatekeeper_action="suspend" so an existing
        # stuck session (saved before the cli.py fix that stamps the
        # flag) can still be rescued on resume.
        if node_state.get("gatekeeper_action") != "suspend":
            return
        suspended_from = "gatekeeper"

    # Distinguish suspend source. Discovery-interview suspends must NOT
    # rewind through human_intervention_node — that route ends up at
    # compiler_node and re-runs the entire build/security pipeline,
    # throwing away work the user already completed (code, tests, security
    # scan) before the discovery phase even started.
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

    if suspended_from == "gatekeeper":
        # Save & Quit at human_gatekeeper_node ([s] at the post-spec
        # approval prompt). The naive rewind ("stamp gatekeeper just
        # returned cleared state") would re-fire route_after_gatekeeper
        # with gatekeeper_action cleared → silent auto-approve forward.
        # That's surprising: the user may have suspended specifically to
        # think about refining or to manually edit. Re-execute the
        # gatekeeper node so its menu re-renders, matching what pressing
        # [s] in hitl_menu_loop semantically means ("come back to this
        # decision").
        #
        # To force re-execution: stamp the predecessor as just-returned,
        # so its outgoing edge fires INTO human_gatekeeper_node. The
        # predecessor is gate-specific.
        gate = values.get("current_gate", "")
        gate_to_predecessor = {
            "REQUIREMENTS": "spec_review_node",
            "ARCHITECTURE": "spec_review_node",
            "DEPLOYMENT": "generate_deployment_spec_node",
            "STORIES": "decomposition_node",
        }
        predecessor = gate_to_predecessor.get(gate)
        if predecessor is None:
            logger.warning(
                "[run_graph] Resume rewind: gatekeeper suspend with unknown "
                "current_gate=%r — cannot determine predecessor node. "
                "Falling back to the default hitl_menu rewind path.",
                gate,
            )
        else:
            cleared = dict(node_state)
            cleared["hitl_suspend"] = False
            cleared.pop("suspended_from", None)
            cleared.pop("gatekeeper_action", None)
            update: dict[str, Any] = {"node_state": cleared}
            # spec_review_node has a CONDITIONAL outgoing edge. The
            # only branch we want is human_gatekeeper_node; force it by
            # clearing reviewer_followups so route_after_spec_review
            # falls through to gatekeeper instead of looping back to
            # discovery_interview_loop.
            if predecessor == "spec_review_node":
                update["reviewer_followups"] = []
            logger.info(
                "[run_graph] Resume rewind: gatekeeper [s] in %s phase. "
                "Re-firing %s → human_gatekeeper_node so the menu re-renders.",
                gate or "UNKNOWN", predecessor,
            )
            try:
                await compiled_graph.aupdate_state(
                    config, update, as_node=predecessor,
                )
            except Exception as exc:  # noqa: BLE001 — log but don't crash resume
                logger.warning(
                    "[run_graph] Failed to rewind gatekeeper-suspend checkpoint: %s. "
                    "Resume may no-op; user can start a fresh session.",
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
    decomposition_enabled: bool = False,
    commit_on_story: bool = False,
    story_batch_size: int = 5,
    story_repair_cap: int = 3,
    repo_memory_config: Optional[dict[str, Any]] = None,
    repo_index_config: Optional[dict[str, Any]] = None,
    llm_dispatch_config: Optional[dict[str, Any]] = None,
    flow: str = FLOW_BUILD,
    generate_specs: bool = False,
    full_config: Optional[dict[str, Any]] = None,
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
    from harness.story_state import state_db_path as _state_db_path

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
        decomposition_enabled=decomposition_enabled,
        # Global state.db now — same path regardless of workspace, with
        # rows scoped by the workspace folder's basename (app name).
        stories_db_path=_state_db_path() if decomposition_enabled else "",
        commit_on_story=commit_on_story,
        story_batch_size=story_batch_size,
        story_repair_cap=story_repair_cap,
        flow=flow,
        generate_specs=generate_specs,
        config=full_config,
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
        # Repair-loop gates accumulate across rounds, so the value
        # checkpointed at session-end is often at-or-near a cap. Without
        # this reset, the first call to route_after_compiler on a
        # resumed run trips HITL immediately on stale counters — same
        # trigger as the prior session, no new work executed. Reset
        # gate counters here so the operator's explicit resume gets a
        # fresh budget; total_repairs (the hard-ceiling counter) is
        # preserved so multi-resume sessions still bound out.
        await _reset_stale_gate_counters_on_resume(compiled_graph, config)
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
