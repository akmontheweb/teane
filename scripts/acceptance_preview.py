"""Preview the ADR-0006 dual-altitude acceptance generator on a real story.

The Phase-0 "measure before wiring" vehicle: gather one story's acceptance
context from a workspace, run the generator (LLM or offline fallback), and print
(and optionally write) the generated integration + e2e scenarios so their quality
can be eyeballed before the generator is wired into the test pipeline / the
Phase-1 in-build ``acceptance_node``.

Usage:
    python scripts/acceptance_preview.py --workspace /path/to/app --story STORY-001
    python scripts/acceptance_preview.py -w /path/to/app -s STORY-001 --source fallback
    python scripts/acceptance_preview.py -w /path/to/app -s STORY-001 --write

``--source llm`` (default) uses the configured gateway (needs the model's API key
in the environment — ``source ~/.zshrc`` first). ``--source fallback`` is offline.

Exit codes: 0 = scenarios produced; 2 = no scenarios (empty/all-dropped);
1 = setup error (story not found, no gateway for --source llm).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow running as a plain script (python scripts/acceptance_preview.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import acceptance_gen as ag  # noqa: E402


def _load_config() -> dict:
    """Best-effort load of the harness config (for the acceptance.* knobs)."""
    import json
    from harness import cli

    try:
        path = cli.default_config_path() if hasattr(cli, "default_config_path") else "config/config.json"
    except Exception:  # noqa: BLE001
        path = "config/config.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        cfg.pop("_help", None)
        return cfg.get("acceptance", {}) or {}
    except Exception:  # noqa: BLE001
        return {}


def _print_result(res: ag.AcceptanceGenResult, *, write: bool, workspace: str, integration_dir: str) -> None:
    integ = res.integration()
    e2e = res.e2e()
    print(f"\n=== {res.story_key} — source={res.source} ===")
    print(f"  scenarios kept: {len(res.scenarios)} "
          f"(integration={len(integ)}, e2e={len(e2e)}); dropped={len(res.dropped)}")
    for d in res.dropped:
        print(f"  DROPPED {d['verifies']} [{d['altitude']}]: {d['reason']}")

    if integ:
        rendered = ag.render_integration_file(res.story_key, integ)
        print("\n----- integration (pytest, in-process) -----\n")
        print(rendered)
        if write:
            out_dir = os.path.join(workspace, integration_dir)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"test_{res.story_key.lower().replace('-', '_')}_acceptance.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(rendered)
            print(f"[wrote] {path}")

    if e2e:
        rendered = ag.render_e2e_spec(res.story_key, e2e)
        print("\n----- e2e (Playwright) -----\n")
        print(rendered)
        if write:
            out_dir = os.path.join(workspace, "tests", "e2e")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{res.story_key.lower().replace('-', '_')}.spec.ts")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(rendered)
            print(f"[wrote] {path}")


async def _run_llm(ctx: ag.StoryAcceptanceContext, cfg: dict) -> ag.AcceptanceGenResult:
    from harness.graph import get_gateway

    gateway = get_gateway()
    if gateway is None:
        # The gateway is injected by the CLI run path; when running this script
        # standalone we build one from config the same way the CLI does.
        gateway = _build_gateway_from_config()
    if gateway is None:
        print("ERROR: no gateway available for --source llm. Ensure model API keys "
              "are exported (source ~/.zshrc) or use --source fallback.", file=sys.stderr)
        raise SystemExit(1)
    return await ag.generate_acceptance_scenarios(
        ctx, gateway=gateway, budget_remaining_usd=5.0, config=cfg,
    )


def _build_gateway_from_config():
    """Construct a Gateway from config/config.json for standalone runs.

    Mirrors the CLI run path: register the configured models, then build the
    gateway from the same dict. Model API keys must be in the environment.
    """
    try:
        import json
        from harness.gateway import create_gateway_from_config, register_models_from_config

        with open("config/config.json", "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        raw.pop("_help", None)
        register_models_from_config(raw)
        return create_gateway_from_config(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not build gateway: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preview ADR-0006 acceptance scenarios for one story.")
    ap.add_argument("-w", "--workspace", required=True, help="workspace path")
    ap.add_argument("-s", "--story", required=True, help="story key, e.g. STORY-001")
    ap.add_argument("--source", choices=["llm", "fallback"], default="llm")
    ap.add_argument("--write", action="store_true", help="write rendered files into the workspace")
    ap.add_argument("--seed", action="store_true", help="also generate + print seed data")
    args = ap.parse_args(argv)

    cfg = _load_config()
    integration_dir = cfg.get("integration_dir", "tests/acceptance")

    ctx = ag.gather_story_acceptance_context(
        args.workspace, args.story, stack={"note": "preview"},
    )
    if ctx is None:
        print(f"ERROR: story {args.story} not found in {args.workspace}", file=sys.stderr)
        return 1
    print(f"[context] {args.story}: {len(ctx.acceptance_criteria)} ACs, "
          f"{len(ctx.routes)} routes discovered")

    if args.source == "fallback":
        res = ag.fallback_acceptance_scenarios(ctx)
    else:
        res = asyncio.run(_run_llm(ctx, cfg))

    _print_result(res, write=args.write, workspace=args.workspace, integration_dir=integration_dir)

    if args.seed:
        _preview_seed(args.workspace, args.source, cfg, write=args.write)

    return 0 if res.scenarios else 2


def _preview_seed(workspace: str, source: str, cfg: dict, *, write: bool) -> None:
    """Generate + print seed data for the workspace (schema-grounded)."""
    import json

    seed_ctx = ag.gather_seed_context(workspace)
    print(f"\n[seed context] tables discovered: {sorted(seed_ctx.table_names())}; "
          f"stories: {len(seed_ctx.stories)}")
    if source == "fallback":
        from harness.test_data_gen import fallback_seed, SchemaContext
        seed = fallback_seed(SchemaContext(workspace_path=workspace, flow_kind=seed_ctx.flow_kind))
        dropped: list = []
    else:
        from harness.graph import get_gateway
        gateway = get_gateway() or _build_gateway_from_config()
        if gateway is None:
            print("  (no gateway — skipping LLM seed)", file=sys.stderr)
            return
        res = asyncio.run(ag.generate_seed_data_llm(
            seed_ctx, gateway=gateway, budget_remaining_usd=2.0, config=cfg))
        seed, dropped = res.seed, res.dropped
    print("\n----- seed.json -----\n")
    print(json.dumps(seed, indent=2, sort_keys=True))
    for d in dropped:
        print(f"  DROPPED table {d.get('table')}: {d.get('reason')}")
    if write:
        out_dir = os.path.join(workspace, "tests", "e2e", "fixtures")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "seed.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(seed, fh, indent=2, sort_keys=True)
        print(f"[wrote] {path}")


if __name__ == "__main__":
    raise SystemExit(main())
