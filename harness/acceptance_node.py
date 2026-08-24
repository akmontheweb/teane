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


def _passthrough(state: dict, *, reason: str) -> dict[str, Any]:
    logger.info("[acceptance] %s — passing through.", reason)
    return {
        "node_state": {
            **(state.get("node_state") or {}),
            "current_node": "acceptance",
            "skipped": True,
            "acceptance_attributable": False,
        }
    }


async def acceptance_node(state: "dict") -> dict[str, Any]:  # AgentState at runtime
    cfg = state.get("acceptance_config", {}) or {}
    if not cfg.get("enabled", False):
        return _passthrough(state, reason="acceptance.enabled is false")

    workspace = state.get("workspace_path", os.getcwd())
    batch_id = int(state.get("current_batch_id") or 0)
    budget = float(state.get("budget_remaining_usd", 0.0) or 0.0)
    loop_counter = dict(state.get("loop_counter") or {})
    repair_rounds = int(loop_counter.get("acceptance_repair", 0))
    max_repair = int(cfg.get("max_repair_rounds_per_batch", 2))
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

    scenarios, budget = await _generate_batch_scenarios(
        workspace, story_keys, cfg, gateway=gateway, budget=budget,
    )
    # ui-only ACs (no integration scenario) are recorded as needs-browser deferrals.
    _persist_ui_only_deferrals(workspace, app_name, scenarios)

    runnable = ar.select_runnable(scenarios, already_passed=already_passed)
    # Include re-queued dependency-blocked ACs whose scenario is in this set.
    if not runnable:
        return _passthrough(state, reason="no runnable integration ACs this batch")

    # Optional schema-typed baseline seed (applied to the isolated DB in conftest).
    seed_dict, budget = await _generate_seed(workspace, cfg, gateway=gateway, budget=budget)

    # --- 2. write conftest + integration files ---------------------------------
    written = _write_integration_suite(workspace, integ_dir_rel, runnable, cfg, seed=seed_dict)
    if not written:
        return _passthrough(state, reason="could not write integration suite / no app client")

    # --- 3. run in the sandbox + triage ---------------------------------------
    runner = _make_sandbox_runner(state, integ_dir_rel)
    result = ar.run_acceptance(
        runnable, written, workspace, runner=runner, story_keys=story_keys,
        already_passed=already_passed,
    )
    logger.info("[acceptance] batch %s: %s", batch_id, ar.summarize(result))

    # --- 4. record outcomes ----------------------------------------------------
    _record_outcomes(workspace, app_name, result, integ_dir_rel, runnable,
                     deferred_before=deferred_before)

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
            "[acceptance] %d attributable failure(s) exceed cap %d — parking as "
            "deferred defects and proceeding.", len(attributable), max_repair,
        )
        _park_over_cap(workspace, app_name, attributable)

    return {
        "loop_counter": loop_counter,
        "budget_remaining_usd": budget,
        "node_state": {**node_state_base, "acceptance_attributable": False},
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def _generate_batch_scenarios(
    workspace: str,
    story_keys: list[str],
    cfg: dict[str, Any],
    *,
    gateway: Any,
    budget: float,
) -> tuple[list[ag.AcceptanceScenario], float]:
    """Generate scenarios for every story in the batch (LLM or fallback)."""
    all_scen: list[ag.AcceptanceScenario] = []
    use_llm = str(cfg.get("scenario_source", "fallback")) == "llm" and gateway is not None
    for sk in story_keys:
        ctx = ag.gather_story_acceptance_context(workspace, sk, stack={"backend": "fastapi"})
        if ctx is None:
            continue
        if use_llm:
            res = await ag.generate_acceptance_scenarios(
                ctx, gateway=gateway, budget_remaining_usd=budget, config=cfg,
            )
            budget = res.budget_remaining_usd
        else:
            res = ag.fallback_acceptance_scenarios(ctx)
        all_scen.extend(res.scenarios)
    return all_scen, budget


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
) -> list[str]:
    """Write one pytest module per story + a shared conftest. Return abs paths.

    Returns [] when no app client can be built (no ``client`` fixture ⇒ nothing
    can run ⇒ the caller safely passes through rather than emitting failures).
    When ``seed`` carries rows AND the DB can be isolated, ``seed.json`` is written
    and the conftest applies it to each test's fresh database.
    """
    discovery = ag.discover_app_factory(workspace)
    if discovery is None:
        logger.info("[acceptance] no FastAPI app discovered — cannot build client fixture.")
        return []

    # DB isolation env var: operator override wins; else auto-discover (pydantic
    # settings env_prefix + db-path field). None → conftest skips isolation.
    db_env_var = str(cfg.get("db_path_env", "") or "") or ag.discover_db_env_var(workspace)
    seed_rows = bool((seed or {}).get("tables"))
    apply_seed = seed_rows and bool(db_env_var)

    out_dir = os.path.join(workspace, integ_dir_rel)
    try:
        os.makedirs(out_dir, exist_ok=True)
        conftest = ag.render_acceptance_conftest(
            discovery, db_env_var=db_env_var, seed=apply_seed)
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


def _persist_ui_only_deferrals(
    workspace: str, app_name: str, scenarios: list[ag.AcceptanceScenario],
) -> None:
    """Record ui-only ACs (e2e only, no integration) as needs-browser deferrals."""
    ui_acs = {
        s.verifies for s in scenarios
        if s.classification == ag.CLASS_UI and s.altitude == ag.ALTITUDE_E2E
    }
    has_integration = {s.verifies for s in scenarios if s.altitude == ag.ALTITUDE_INTEGRATION}
    ui_only = ui_acs - has_integration
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


def _park_over_cap(
    workspace: str, app_name: str, attributable: list[ar.ACOutcome],
) -> None:
    """Park over-cap attributable failures as deferrals (complete_with_blocks)."""
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


def _acceptance_diagnostics(
    attributable: list[ar.ACOutcome], written: list[str],
) -> list[dict[str, Any]]:
    """Synthesise repair-loop diagnostics for attributable acceptance failures.

    Points at the acceptance test file for location context; the message directs
    repair at the PRODUCTION behaviour (repair is barred from editing tests, so it
    fixes the code, not the oracle — exactly the intended flow).
    """
    test_path = written[0] if written else "tests/acceptance"
    out: list[dict[str, Any]] = []
    for o in attributable:
        out.append({
            "file": test_path,
            "line": 0,
            "severity": "error",
            "error_code": "ACCEPTANCE_GAP",
            "message": (
                f"Acceptance criterion {o.ac_key} is not satisfied by the current "
                f"production code (in-process acceptance test failed): {o.detail}. "
                f"Fix the application code so the criterion holds; do not weaken the test."
            ),
        })
    return out


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
