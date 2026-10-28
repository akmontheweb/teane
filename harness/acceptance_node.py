"""In-build acceptance graph node (ADR-0006, Phase 1).

Spliced between a green compile and code review. Per batch, it:

  1. gathers the batch's stories + acceptance criteria,
  2. generates integration-altitude scenarios (Phase-0 generator; LLM or fallback),
  3. writes them + a ``client`` conftest into the workspace,
  4. runs them in the sandbox and triages each failure
     (:mod:`harness.acceptance_run`),
  5. records passes with ``build`` provenance, persists deferrals for re-queue,
     and — for *attributable* failures under a per-batch cap — feeds the repair
     loop; over-cap residue parks as a deferral and the batch proceeds.

Safety invariants (ADR-0006 + the finsearch-156032347 lesson):
  * ``acceptance.enabled=false`` → the node is a pure pass-through no-op; the
    graph behaves exactly as before this node existed.
  * The node NEVER hard-fails the batch. Attributable failures drive repair only
    within ``max_repair_rounds_per_batch``; everything else defers. Any harness
    error (no gateway, sandbox crash, no app discovered) defers, never fails.

``route_after_acceptance`` reads the flags this node sets in ``node_state``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from harness import acceptance_gen as ag
from harness import acceptance_run as ar
from harness import story_state

logger = logging.getLogger(__name__)


def _passthrough(
    state: dict, *, reason: str,
    unverifiable: Optional[dict[str, tuple[str, str]]] = None,
) -> dict[str, Any]:
    """Skip acceptance for this batch, reporting whatever was in scope.

    ``unverifiable`` names criteria no in-build test can decide. Passing
    through without them was how a batch whose whole scope was ui criteria
    reported nothing at all — see :func:`_unverifiable_in_build`.
    """
    logger.info("[acceptance] %s — passing through.", reason)
    _ns: dict[str, Any] = {
        **(state.get("node_state") or {}),
        "current_node": "acceptance",
        "skipped": True,
        "acceptance_attributable": False,
    }
    if unverifiable:
        _result = ar.AcceptanceRunResult(
            ran=False, outcomes=ar.unverifiable_outcomes(unverifiable),
        )
        _ns["acceptance_summary"] = ar.summarize(_result)
        logger.info(
            "[acceptance] %d criterion/criteria were in scope and cannot be "
            "verified in-build (%s) — reported unverified, not omitted.",
            len(unverifiable), ", ".join(sorted(unverifiable)),
        )
    return {"node_state": _ns}


async def acceptance_node(state: "dict") -> dict[str, Any]:  # AgentState at runtime
    cfg = state.get("acceptance_config", {}) or {}
    if not cfg.get("enabled", False):
        return _passthrough(state, reason="acceptance.enabled is false")

    workspace = state.get("workspace_path", os.getcwd())
    batch_id = int(state.get("current_batch_id") or 0)
    budget = float(state.get("budget_remaining_usd", 0.0) or 0.0)
    loop_counter = dict(state.get("loop_counter") or {})
    max_repair = int(cfg.get("max_repair_rounds_per_batch", 2))
    # The budget is PER BATCH, as the config key and this module's docstring
    # both say — so the counter has to be scoped to the batch that spent it.
    # It was a bare session-wide scalar that nothing ever reset, so the first
    # batch to use its two rounds exhausted the budget for the whole run and
    # every later batch parked its attributable failures without a single
    # repair attempt. lumina 01a15406 (2026-10-19): batch 71 used rounds 1/2
    # and 2/2, then batch 72 parked 9 verified, code-attributed failures with
    # zero attempts and batch 73 would have done the same.
    #
    # Scoped by remembering which batch the count belongs to rather than by
    # making the counter a dict: the scalar survives checkpoint JSON
    # round-trips unchanged, and a batch revisit correctly continues its own
    # count instead of starting over.
    _counted_batch = loop_counter.get("acceptance_repair_batch")
    try:
        _counted_batch = int(_counted_batch) if _counted_batch is not None else None
    except (TypeError, ValueError):
        _counted_batch = None
    repair_rounds = (
        int(loop_counter.get("acceptance_repair", 0))
        if _counted_batch == batch_id else 0
    )
    scenario_source = str(cfg.get("scenario_source", "fallback"))
    integ_dir_rel = str(cfg.get("integration_dir", "tests/acceptance"))

    # --- 1. resolve the batch's stories + their integration scenarios ----------
    try:
        app_name = story_state.app_name_for_workspace(workspace)
        conn = story_state.open_story_db(workspace_path=workspace)
    except Exception as exc:  # noqa: BLE001
        return _passthrough(state, reason=f"story DB unavailable ({exc})")

    try:
        if batch_id:
            story_keys = story_state.story_keys_for_batch(conn, app_name, batch_id)
        else:
            story_keys = list(state.get("batch_patched_story_keys") or [])
        already_passed = story_state.acs_verified_with_provenance(conn, app_name, "build")
        # Re-queue: previously dependency-blocked ACs — and ACs whose suite
        # merely failed to collect/provision last time (Fix B) — get another
        # attempt now that more of the app (and its deps) may be in place.
        deferred_before = {
            d["ac_key"]: d["ac_id"]
            for d in story_state.list_acceptance_deferrals(conn, app_name)
            if d["status"] in (
                ar.STATUS_DEFERRED_DEP, ar.STATUS_DEFERRED_COLLECTION,
            )
        }
    finally:
        _close(conn)

    if not story_keys:
        return _passthrough(state, reason="no stories in batch scope")

    gateway = None
    if scenario_source == "llm":
        from harness.graph import get_gateway  # lazy
        gateway = get_gateway()

    # Resolve DB isolation ONCE, before generation. The scenario prompt and the
    # rendered conftest must agree: a generator told it has a fresh database
    # writes collection-total assertions, and if the conftest then hands it a
    # suite-wide shared DB those tests can never pass no matter what the repair
    # loop does to the production code (lumina session 01a079dc → zero-patch HITL).
    db_override = _resolve_db_override(workspace, cfg)
    # An injectable clock makes date-dependent criteria provable instead of
    # calendar-dependent (lumina-run17-20260924-2351: a leap-day scenario
    # asserted a February birthday was visible through a 30-day window, in
    # September). Resolved BEFORE generation so the prompt and the rendered
    # conftest agree about whether freeze_today exists — the same contract
    # the db_isolated flag has.
    clock_override = ag.discover_clock_override(workspace)
    # NEVER withdraw a capability the already-written tests depend on. The
    # conftest is re-rendered on every acceptance pass while the per-story
    # test files persist, so a discovery that fails once would delete
    # `freeze_today` from under tests generated in an earlier batch — which
    # is how lumina-run18-20260925-0748 ended: repair was asked to add a
    # method back to a generated file it would lose again on the next pass.
    if clock_override is None:
        clock_override = _remembered_clock(state, workspace, integ_dir_rel)
    if clock_override:
        _remember_clock(state, clock_override)
        logger.info(
            "[acceptance] clock freeze available via %s.%s — scenarios may "
            "pin today's date.",
            clock_override["module"], clock_override["symbol"],
        )

    scenarios, not_runtime, budget = await _generate_batch_scenarios(
        workspace, story_keys, cfg, gateway=gateway, budget=budget,
        db_isolated=db_override is not None,
        clock_available=clock_override is not None,
    )
    # ui-only ACs (no integration scenario) are recorded as needs-browser deferrals.
    _persist_ui_only_deferrals(workspace, app_name, scenarios)
    _persist_not_runtime_deferrals(workspace, app_name, not_runtime)
    _unverifiable = _unverifiable_in_build(scenarios, not_runtime)

    runnable = ar.select_runnable(scenarios, already_passed=already_passed)
    # Include re-queued dependency-blocked ACs whose scenario is in this set.
    if not runnable:
        # A batch of nothing but ui criteria is not a batch with nothing in
        # it, and this path used to report exactly that.
        return _passthrough(
            state, reason="no runnable integration ACs this batch",
            unverifiable=_unverifiable,
        )

    # Optional schema-typed baseline seed (applied to the isolated DB in conftest).
    seed_dict, budget = await _generate_seed(workspace, cfg, gateway=gateway, budget=budget)

    # --- 2. write conftest + integration files ---------------------------------
    written = _write_integration_suite(
        workspace, integ_dir_rel, runnable, cfg, seed=seed_dict,
        db_override=db_override, clock_override=clock_override,
    )
    if not written:
        return _passthrough(state, reason="could not write integration suite / no app client")

    # --- 3. run in the sandbox + triage ---------------------------------------
    runner = _make_sandbox_runner(state, integ_dir_rel)
    result = ar.run_acceptance(
        runnable, written, workspace, runner=runner, story_keys=story_keys,
        already_passed=already_passed, unverifiable=_unverifiable,
    )
    logger.info("[acceptance] batch %s: %s", batch_id, ar.summarize(result))

    # --- 4. record outcomes ----------------------------------------------------
    _record_outcomes(workspace, app_name, result, integ_dir_rel, runnable,
                     deferred_before=deferred_before)
    _unquarantine_passed(result, written)

    # --- 5. route --------------------------------------------------------------
    attributable = result.attributable()
    node_state_base = {
        **(state.get("node_state") or {}),
        "current_node": "acceptance",
        "skipped": False,
        "acceptance_summary": ar.summarize(result),
    }

    if attributable and repair_rounds < max_repair:
        diags = list(state.get("compiler_errors") or [])
        diags.extend(_acceptance_diagnostics(attributable, written))
        loop_counter["acceptance_repair"] = repair_rounds + 1
        loop_counter["acceptance_repair_batch"] = batch_id
        logger.info(
            "[acceptance] %d attributable failure(s) → repair (round %d/%d)",
            len(attributable), repair_rounds + 1, max_repair,
        )
        return {
            "compiler_errors": diags,
            "loop_counter": loop_counter,
            "budget_remaining_usd": budget,
            "node_state": {**node_state_base, "acceptance_attributable": True},
        }

    if attributable:
        # Over cap → park as a defect and proceed (complete_with_blocks territory).
        logger.warning(
            "[acceptance] %d attributable failure(s) remain after %d/%d repair "
            "round(s) for batch %s — parking as deferred defects and "
            "proceeding.",
            len(attributable), repair_rounds, max_repair, batch_id,
        )
        _park_over_cap(workspace, app_name, attributable, written)

    return {
        "loop_counter": loop_counter,
        "budget_remaining_usd": budget,
        "node_state": {**node_state_base, "acceptance_attributable": False},
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _remember_clock(state: dict, clock: dict[str, Any]) -> None:
    """Keep this run's clock discovery so a later pass cannot lose it."""
    try:
        ns = state.setdefault("node_state", {})
        ns["acceptance_clock_override"] = dict(clock)
    except Exception:  # noqa: BLE001 — memory of a discovery is best-effort
        pass


def _remembered_clock(
    state: dict, workspace: str, integ_dir_rel: str,
) -> Optional[dict[str, Any]]:
    """The run's earlier clock discovery, when tests still depend on it.

    Reused ONLY when both halves still hold: some generated test calls
    ``freeze_today``, and the remembered module still declares the symbol.
    Rendering an import for a symbol the app has deleted would turn every
    acceptance test into a collection error — a worse failure than the one
    this guards against.
    """
    prev = (state.get("node_state") or {}).get("acceptance_clock_override")
    if not isinstance(prev, dict) or not prev.get("module"):
        return None
    out_dir = os.path.join(workspace, integ_dir_rel)
    try:
        used = any(
            "freeze_today" in open(os.path.join(out_dir, f), encoding="utf-8",
                                   errors="replace").read()
            for f in os.listdir(out_dir)
            if f.startswith("test_") and f.endswith(".py")
        )
    except OSError:
        return None
    if not used:
        return None
    mod_path = os.path.join(
        workspace, *str(prev["module"]).split(".")) + ".py"
    try:
        with open(mod_path, "r", encoding="utf-8", errors="replace") as fh:
            if str(prev.get("symbol", "")) not in fh.read():
                logger.warning(
                    "[acceptance] remembered clock %s is gone from %s — not "
                    "reusing it; tests calling freeze_today will fail "
                    "honestly rather than on an import error.",
                    prev.get("symbol"), prev["module"],
                )
                return None
    except OSError:
        return None
    logger.info(
        "[acceptance] clock discovery found nothing this pass; reusing this "
        "run's earlier %s.%s so the conftest does not withdraw freeze_today "
        "from tests already written against it.",
        prev["module"], prev.get("symbol"),
    )
    return prev


def _resolve_db_override(
    workspace: str, cfg: dict[str, Any],
) -> Optional[ag.DbOverride]:
    """Decide how (or whether) each acceptance test gets its own database.

    Operator override (``acceptance.db_path_env``) wins; otherwise auto-discover
    from the app's pydantic-settings class. Returns None when isolation cannot be
    arranged — and says so at WARNING, because that silently degrades every
    acceptance criterion that depends on a known-empty database.
    """
    forced = str(cfg.get("db_path_env", "") or "")
    if forced:
        return ag.DbOverride(forced, ag._override_kind_for_name(forced))
    found = ag.discover_db_override(workspace)
    if found is None:
        logger.warning(
            "[acceptance] no DB override env var discovered — acceptance tests "
            "will SHARE one database across the whole suite. Scenarios will be "
            "generated under the shared-DB contract (no collection-total "
            "assertions). Set `acceptance.db_path_env` to the env var that "
            "re-points this app's database to restore per-test isolation.",
        )
    else:
        logger.info(
            "[acceptance] per-test DB isolation via %s (%s-valued).",
            found.env_var, found.kind,
        )
    return found


async def _generate_batch_scenarios(
    workspace: str,
    story_keys: list[str],
    cfg: dict[str, Any],
    *,
    gateway: Any,
    budget: float,
    db_isolated: bool = False,
    clock_available: bool = False,
) -> tuple[list[ag.AcceptanceScenario], dict[str, str], float]:
    """Generate scenarios for every story in the batch (LLM or fallback).

    Also returns the not-runtime criteria (AC key → rationale), which have
    no scenarios to carry their classification.
    """
    all_scen: list[ag.AcceptanceScenario] = []
    not_runtime: dict[str, str] = {}
    use_llm = str(cfg.get("scenario_source", "fallback")) == "llm" and gateway is not None
    for sk in story_keys:
        ctx = ag.gather_story_acceptance_context(
            workspace, sk, stack={"backend": "fastapi"},
            suite_dir=os.path.join(
                workspace, str(cfg.get("integration_dir", "tests/acceptance"))),
        )
        if ctx is None:
            continue
        if use_llm:
            res = await ag.generate_acceptance_scenarios(
                ctx, gateway=gateway, budget_remaining_usd=budget, config=cfg,
                db_isolated=db_isolated, clock_available=clock_available,
            )
            budget = res.budget_remaining_usd
        else:
            res = ag.fallback_acceptance_scenarios(ctx)
        all_scen.extend(res.scenarios)
        not_runtime.update(res.not_runtime)
    return all_scen, not_runtime, budget


async def _generate_seed(
    workspace: str, cfg: dict[str, Any], *, gateway: Any, budget: float,
) -> tuple[dict[str, Any], float]:
    """Generate schema-typed baseline seed rows (LLM). Empty when not enabled.

    Only the LLM seed generator produces useful rows; the ``test_data_gen``
    fallback is a stub, so ``seed_source != 'llm'`` (or no gateway) yields no
    seed and the conftest relies on self-contained scenarios.
    """
    if str(cfg.get("seed_source", "fallback")) != "llm" or gateway is None:
        return {"tables": {}}, budget
    try:
        ctx = ag.gather_seed_context(workspace)
        res = await ag.generate_seed_data_llm(
            ctx, gateway=gateway, budget_remaining_usd=budget, config=cfg)
        return res.seed, res.budget_remaining_usd
    except Exception as exc:  # noqa: BLE001 — seeding is best-effort
        logger.warning("[acceptance] seed generation failed: %s — no seed.", exc)
        return {"tables": {}}, budget


# ---------------------------------------------------------------------------
# Writing the suite
# ---------------------------------------------------------------------------


def _write_integration_suite(
    workspace: str, integ_dir_rel: str, runnable: list[ag.AcceptanceScenario],
    cfg: dict[str, Any], *, seed: Optional[dict[str, Any]] = None,
    db_override: Optional[ag.DbOverride] = None,
    clock_override: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Write one pytest module per story + a shared conftest. Return abs paths.

    Returns [] when no app client can be built (no ``client`` fixture ⇒ nothing
    can run ⇒ the caller safely passes through rather than emitting failures).
    ``db_override`` is the isolation decision made before generation (see
    :func:`_resolve_db_override`) so the conftest matches the contract the
    scenarios were written against; it is re-resolved here only when the caller
    passed nothing. When ``seed`` carries rows AND the DB can be isolated,
    ``seed.json`` is written and the conftest applies it to each fresh database.
    """
    discovery = ag.discover_app_factory(workspace)
    if discovery is None:
        logger.info("[acceptance] no FastAPI app discovered — cannot build client fixture.")
        return []

    if db_override is None:
        db_override = _resolve_db_override(workspace, cfg)
    db_env_var = db_override.env_var if db_override else None
    db_value_kind = db_override.kind if db_override else ag.DB_VALUE_PATH
    seed_rows = bool((seed or {}).get("tables"))
    apply_seed = seed_rows and bool(db_env_var)

    out_dir = os.path.join(workspace, integ_dir_rel)
    try:
        os.makedirs(out_dir, exist_ok=True)
        conftest = ag.render_acceptance_conftest(
            discovery, db_env_var=db_env_var, db_value_kind=db_value_kind,
            seed=apply_seed, clock=clock_override)
        with open(os.path.join(out_dir, "conftest.py"), "w", encoding="utf-8") as fh:
            fh.write(conftest)
        if apply_seed:
            import json as _json
            with open(os.path.join(out_dir, "seed.json"), "w", encoding="utf-8") as fh:
                _json.dump(seed, fh, indent=2, sort_keys=True)
        elif seed_rows:
            logger.info("[acceptance] seed generated but DB not isolatable — skipping seed apply.")
    except OSError as exc:
        logger.warning("[acceptance] could not write conftest: %s", exc)
        return []

    by_story: dict[str, list[ag.AcceptanceScenario]] = {}
    for sc in runnable:
        story = sc.verifies.split(".")[0] if "." in sc.verifies else sc.verifies
        by_story.setdefault(story, []).append(sc)

    written: list[str] = []
    for story, scen in by_story.items():
        content = ag.render_integration_file(story, scen)
        fname = f"test_{story.lower().replace('-', '_')}_acceptance.py"
        path = os.path.join(out_dir, fname)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            written.append(path)
        except OSError as exc:
            logger.warning("[acceptance] could not write %s: %s", path, exc)
    return written


# ---------------------------------------------------------------------------
# Sandbox runner
# ---------------------------------------------------------------------------


# Shell markers echoed between the provisioning steps so a single sandbox
# run can tell WHICH step failed: install → app-import preflight → pytest.
# Unique tokens that never appear in pytest's own output and never parse as
# a test-summary line.
_INSTALL_OK_MARKER = "__ACCEPTANCE_INSTALL_OK__"
_IMPORT_OK_MARKER = "__ACCEPTANCE_APP_IMPORT_OK__"


def _make_sandbox_runner(state: dict, integ_dir_rel: str) -> ar.AcceptanceRunner:
    """Build the sandbox runner closure the engine calls to execute the suite.

    The composed command runs, in one ephemeral sandbox:
      1. (B1) install the workspace's Python deps + a marker, so the run has
         the same environment the compiler proved importable;
      2. (B2/B4) an import preflight that replicates the conftest's app
         import (``from <module> import <symbol>``) + a marker, so a bad or
         unbuilt entrypoint is pinpointed and the expensive seeded run is
         skipped when the app can't even import;
      3. the pytest suite itself.
    Missing markers in the output localise the failure to the install step
    or the app import; either raises :class:`ar.AcceptanceCollectionError`
    with a precise message, keeping a "the suite could not run" outcome
    honestly distinct from "a prerequisite isn't built yet".
    """

    def _run(paths: list[str], workspace: str) -> list[ar.TestOutcome]:
        from harness.sandbox import SandboxExecutor

        sandbox_cfg = dict(state.get("sandbox_config", {}) or {})
        pytest_cmd = (
            f"python -m pytest {integ_dir_rel} "
            "-rA --tb=short -p no:cacheprovider -q"
        )

        # Fix B1 — provision the acceptance sandbox to match the environment
        # the compiler proved importable. Each sandbox invocation is an
        # ephemeral container, so an install activated during compile does
        # NOT persist here: without re-installing, pytest collection-errors
        # on the conftest's ``from fastapi.testclient import TestClient`` /
        # ``from <app> import create_app`` even though the identical import
        # was green seconds earlier. Reuse the prod-import smoke check's
        # install composer (root + first-level Python manifests) and force
        # network on for that install. Falls back to the bare pytest command
        # if the composer is unavailable or there's no Python manifest.
        install_step: Optional[str] = None
        try:
            from harness.graph import _compose_prod_smoke_install_step
            install_step = _compose_prod_smoke_install_step(workspace)
        except Exception as exc:  # noqa: BLE001 — provisioning is best-effort
            logger.debug("[acceptance] install-step compose skipped: %s", exc)

        # Fix B2/B4 — import preflight. Replicate the conftest's app import
        # EXACTLY (``from <module> import <symbol>``, from the same regex
        # discovery the conftest was rendered from), so a probe failure ⟺ a
        # real collection failure — no false negatives — and a wrong/unbuilt
        # entrypoint is reported as such rather than as a generic collection
        # error. ``&&`` short-circuits the (seeded) pytest run when the app
        # can't import, so the expensive path is skipped on the failure case.
        discovery = ag.discover_app_factory(workspace)
        app_module = discovery.get("module") if discovery else None
        app_symbol = discovery.get("symbol") if discovery else None

        parts: list[str] = []
        if install_step:
            parts.append(install_step)
            parts.append(f"echo {_INSTALL_OK_MARKER}")
            allow_network = True  # the composed `uv pip install` needs PyPI
        else:
            allow_network = bool(state.get("allow_network", False))
        if app_module and app_symbol:
            parts.append(f'python -c "from {app_module} import {app_symbol}"')
            parts.append(f"echo {_IMPORT_OK_MARKER}")
        parts.append(pytest_cmd)
        cmd = " && ".join(parts)

        executor = SandboxExecutor(
            workspace_path=workspace,
            allow_network=allow_network,
            sandbox_config=sandbox_cfg,
        )
        try:
            build_result = _await(executor.run(cmd))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[acceptance] sandbox run failed: %s — deferring.", exc)
            return []
        raw = getattr(build_result, "full_output", "") or getattr(build_result, "raw_output", "")
        tail = (raw or "").strip()[-400:]

        # B1 — the install step failed and short-circuited before anything ran.
        if install_step and _INSTALL_OK_MARKER not in raw:
            logger.warning(
                "[acceptance] dependency provisioning failed in the "
                "acceptance sandbox — recording as collection-error, not "
                "blocked-by-dependency. Tail:\n%s", tail,
            )
            raise ar.AcceptanceCollectionError(
                f"acceptance dependency install failed: {tail}"
            )
        # B4 — the discovered app entrypoint (or a dep it needs) did not
        # import; the conftest would fail identically at collection.
        if app_module and app_symbol and _IMPORT_OK_MARKER not in raw:
            logger.warning(
                "[acceptance] discovered app entrypoint '%s:%s' did not "
                "import in the acceptance sandbox — recording as "
                "collection-error. Tail:\n%s", app_module, app_symbol, tail,
            )
            raise ar.AcceptanceCollectionError(
                f"discovered app entrypoint '{app_module}:{app_symbol}' did "
                f"not import in the acceptance sandbox: {tail}"
            )
        # B3 — the app imported cleanly but the suite still didn't run (a
        # test-file or conftest-only import problem, or a collection banner).
        outcomes = ar.parse_pytest_outcomes(raw)
        if ar.is_collection_error(raw) or not outcomes:
            logger.warning(
                "[acceptance] suite did not run (collection error) despite a "
                "clean app import — likely a generated test-file import "
                "issue. Recording as collection-error. Tail:\n%s", tail,
            )
            raise ar.AcceptanceCollectionError(
                tail or "acceptance suite produced no test outcomes"
            )
        return outcomes

    return _run


def _await(coro):
    """Run ``coro`` to completion from sync code, whether or not a loop is live.

    ``acceptance_node`` is async (already inside the graph's loop), but the
    engine's runner seam is sync. We run the sandbox coroutine on a dedicated
    thread's event loop to avoid ``asyncio.run`` inside a running loop.
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


# ---------------------------------------------------------------------------
# Recording outcomes
# ---------------------------------------------------------------------------


def _record_outcomes(
    workspace: str,
    app_name: str,
    result: ar.AcceptanceRunResult,
    integ_dir_rel: str,
    runnable: list[ag.AcceptanceScenario],
    *,
    deferred_before: dict[str, int],
) -> None:
    """Persist passes (build provenance) + deferrals; clear resolved deferrals."""
    func_by_ac = {s.verifies: ag.integration_function_name(s) for s in runnable}
    try:
        conn = story_state.open_story_db(workspace_path=workspace)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[acceptance] cannot open DB to record outcomes: %s", exc)
        return
    try:
        for o in result.outcomes:
            ac = story_state.get_ac_by_key(conn, app_name, o.ac_key)
            if ac is None:
                continue
            ac_id = int(ac["id"])
            if o.status == ar.STATUS_PASSED:
                story = o.ac_key.split(".")[0]
                test_path = os.path.join(
                    integ_dir_rel, f"test_{story.lower().replace('-', '_')}_acceptance.py")
                story_state.link_test_to_ac(
                    conn, app_name, test_path, ac_id,
                    test_function_name=func_by_ac.get(o.ac_key, ""),
                    provenance="build",
                )
                story_state.clear_acceptance_deferral(conn, app_name, ac_id)
            elif o.is_deferred:
                story_state.record_acceptance_deferral(
                    conn, app_name, ac_id, o.status, o.detail)
    finally:
        _close(conn)


def _ui_only_acs(scenarios: list[ag.AcceptanceScenario]) -> set[str]:
    """ACs with an e2e/ui scenario and no integration one.

    Extracted so the reporting path and the deferral-recording path cannot
    disagree about which criteria these are — they did, silently, for as long
    as only the latter existed.
    """
    ui_acs = {
        s.verifies for s in scenarios
        if s.classification == ag.CLASS_UI and s.altitude == ag.ALTITUDE_E2E
    }
    has_integration = {
        s.verifies for s in scenarios
        if s.altitude == ag.ALTITUDE_INTEGRATION
    }
    return ui_acs - has_integration


def _unverifiable_in_build(
    scenarios: list[ag.AcceptanceScenario], not_runtime: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """``{ac_key: (status, detail)}`` for every criterion in this batch's
    scope that no in-build test can decide.

    ui-only criteria need a browser against the app running on Docker, which
    is ``teane test``'s job; not-runtime criteria are about the repository or
    process. Deferring both is right. Leaving them out of the run result —
    which is where ``summarize`` and node_state's ``acceptance_summary`` come
    from — is what made batch 88 report ``total:10`` for 17 criteria and let
    STORY-003 seal ``done`` with 0 of 4 verified, silently.
    """
    out: dict[str, tuple[str, str]] = {}
    for ac_key in sorted(_ui_only_acs(scenarios)):
        out[ac_key] = (
            ar.STATUS_DEFERRED_UI,
            "ui-only: deferred to post-deploy teane test",
        )
    for ac_key, why in sorted(not_runtime.items()):
        out[ac_key] = (
            ar.STATUS_DEFERRED_NOT_RUNTIME,
            f"not-runtime: {why}" if why else "not-runtime",
        )
    return out


def _persist_ui_only_deferrals(
    workspace: str, app_name: str, scenarios: list[ag.AcceptanceScenario],
) -> None:
    """Record ui-only ACs (e2e only, no integration) as needs-browser deferrals."""
    ui_only = _ui_only_acs(scenarios)
    if not ui_only:
        return
    try:
        conn = story_state.open_story_db(workspace_path=workspace)
    except Exception:  # noqa: BLE001
        return
    try:
        for ac_key in ui_only:
            ac = story_state.get_ac_by_key(conn, app_name, ac_key)
            if ac is not None:
                story_state.record_acceptance_deferral(
                    conn, app_name, int(ac["id"]),
                    ar.STATUS_DEFERRED_UI, "ui-only: deferred to post-deploy teane test")
    finally:
        _close(conn)


def _persist_not_runtime_deferrals(
    workspace: str, app_name: str, not_runtime: dict[str, str],
) -> None:
    """Record not-runtime ACs as deferrals carrying what would verify them."""
    if not not_runtime:
        return
    logger.info(
        "[acceptance] %d criterion/criteria are about the repository or "
        "process, not the running app — deferred, no runtime test: %s",
        len(not_runtime), sorted(not_runtime),
    )
    try:
        conn = story_state.open_story_db(workspace_path=workspace)
    except Exception:  # noqa: BLE001
        return
    try:
        for ac_key, why in not_runtime.items():
            ac = story_state.get_ac_by_key(conn, app_name, ac_key)
            if ac is not None:
                story_state.record_acceptance_deferral(
                    conn, app_name, int(ac["id"]),
                    ar.STATUS_DEFERRED_NOT_RUNTIME,
                    f"not-runtime: {why}" if why else "not-runtime")
    finally:
        _close(conn)


def _park_over_cap(
    workspace: str, app_name: str, attributable: list[ar.ACOutcome],
    written: Optional[list[str]] = None,
) -> None:
    """Park over-cap attributable failures as deferrals (complete_with_blocks),
    and QUARANTINE their tests so the next batch does not re-pay for them.

    Recording the deferral was only half the job. The failing test stayed
    live, so the next batch collected it, failed it, and spent its own repair
    rounds on a defect already written down: batch 89 burnt 6 empty rounds
    and 2 resumes on batch 88's ``STORY-004.AC-1``. Sealing on an
    attributable failure is ADR-0006 Phase 1 working as designed; inheriting
    it is not.

    The quarantine is self-clearing — see
    :func:`acceptance_gen.quarantine_criterion` and the XPASS handling in
    ``parse_pytest_outcomes`` — so a criterion that later starts passing
    reports a real pass and :func:`_unquarantine_passed` strips the marker.
    """
    try:
        conn = story_state.open_story_db(workspace_path=workspace)
    except Exception:  # noqa: BLE001
        return
    try:
        for o in attributable:
            ac = story_state.get_ac_by_key(conn, app_name, o.ac_key)
            if ac is not None:
                story_state.record_acceptance_deferral(
                    conn, app_name, int(ac["id"]), ar.STATUS_ATTRIBUTABLE,
                    f"over repair cap: {o.detail}")
    finally:
        _close(conn)
    for o in attributable:
        _rewrite_suite(
            o.ac_key, written or [],
            lambda src, key=o.ac_key, detail=o.detail: ag.quarantine_criterion(
                src, key, detail,
            ),
            action="quarantined",
        )


def _unquarantine_passed(
    result: ar.AcceptanceRunResult, written: Optional[list[str]],
) -> None:
    """Strip the quarantine marker from every criterion that now passes.

    A stale marker is worse than none: a later regression would report XFAIL
    — "quarantined, not verified" — instead of the attributable failure it
    is, so the marker meant to stop a batch inheriting a defect would start
    hiding one.
    """
    for o in result.outcomes:
        if o.status != ar.STATUS_PASSED:
            continue
        _rewrite_suite(
            o.ac_key, written or [],
            lambda src, key=o.ac_key: ag.unquarantine_criterion(src, key),
            action="un-quarantined",
        )


def _rewrite_suite(
    ac_key: str, written: list[str], edit: Any, *, action: str,
) -> None:
    """Apply ``edit`` to the suite file holding ``ac_key``, if it changes it.

    ``edit`` returns None for "nothing to do", which is the common case on
    both paths (already quarantined / never was), so the file is left
    untouched rather than rewritten with identical bytes.
    """
    path, _line = _locate_ac_test(ac_key, written)
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        new = edit(src)
        if not new or new == src:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
    except OSError as exc:
        logger.warning(
            "[acceptance] could not %s %s in %s: %s",
            action, ac_key, path, exc,
        )
        return
    logger.info("[acceptance] %s %s in %s", action, ac_key, path)


def _acceptance_diagnostics(
    attributable: list[ar.ACOutcome], written: list[str],
) -> list[dict[str, Any]]:
    """Synthesise repair-loop diagnostics for attributable acceptance failures.

    Points at the acceptance test file for location context; the message directs
    repair at the PRODUCTION behaviour (repair is barred from editing tests, so it
    fixes the code, not the oracle — exactly the intended flow).
    """
    # Locate each criterion's OWN test. This used to be ``written[0]``, line
    # 0, for every criterion — the first suite file in the batch, whichever
    # story it belonged to — so the diagnostic pointed at the wrong file and
    # no test, and a proved contradiction built on it could not name its side.
    out: list[dict[str, Any]] = []
    for o in attributable:
        test_path, line = _locate_ac_test(o.ac_key, written)
        out.append({
            "file": test_path,
            "line": line,
            "severity": "error",
            "error_code": "ACCEPTANCE_GAP",
            "message": (
                f"Acceptance criterion {o.ac_key} is not satisfied by the current "
                f"production code (in-process acceptance test failed): {o.detail}. "
                f"Fix the application code so the criterion holds; do not weaken the test."
            ),
        })
    return out


def _locate_ac_test(ac_key: str, written: list[str]) -> tuple[str, int]:
    """``(file, def_line)`` of the generated test verifying ``ac_key``."""
    story = ac_key.rsplit(".AC", 1)[0]
    fname = f"test_{story.lower().replace('-', '_')}_acceptance.py"
    candidates = [p for p in written if os.path.basename(p) == fname] or list(written)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        for key, first, _last in ag.verifies_by_function(src).values():
            if key == ac_key:
                # The def sits on the line after its @verifies marker.
                return path, first + 1
    return (candidates[0] if candidates else "tests/acceptance"), 0


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_acceptance(state: dict) -> str:
    """Attributable-under-cap → repair; otherwise → code review (unchanged tail)."""
    ns = state.get("node_state", {}) or {}
    if ns.get("acceptance_attributable"):
        return "repair_node"
    return "code_review_node"


def _close(conn: Any) -> None:
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


__all__ = ["acceptance_node", "route_after_acceptance"]
