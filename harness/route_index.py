"""Workspace-wide HTTP route → source file index (AST only).

Why this exists
---------------
The repair judge is grounded: it may name the files the diagnostics name,
plus the ones ``graph._related_files_evidence`` proves are related. That
rule is what stops invented paths, and it must stay. The related-files walk
that widens it is a *static import* walk, which is exactly the wrong
instrument for an integration test:

An ADR-0006 acceptance test reaches the application through a pytest
fixture. ``tests/acceptance/test_story_006_acceptance.py`` imports nothing
but ``pytest``; the ``client`` fixture lives in a generated ``conftest.py``
and the app is constructed from a module named in a *string*. So the import
walk returns no production file for it, the grounding rule then makes every
production file unnameable, and the judge can only name whatever the rest of
the failing set happened to drag in.

lumina 01a21929 (run 20) died of precisely that. Three failing diagnostics:
a generated contract test, and two acceptance tests whose real defects were
``server/app/api/birthdays.py`` (a GET-by-id with no owner filter answering
200 where the criterion requires 404) and ``server/app/main.py`` (no
``RequestValidationError`` handler, so FastAPI's raw 422 leaked where the
spec wants a 400 envelope). Neither file was reachable: the only production
file in the grounded set was ``server/app/schemas/birthday.py``, dragged in
by the contract test's import line. The judge named that schema for eleven
consecutive verdicts because it was the only legal answer, repair oscillated
on it, and the run spent 30% of its budget there.

The two edges this module adds back
-----------------------------------
1. **The route table.** A test that failed asserting on ``GET
   /api/v1/birthdays/<id>`` is asserting on whatever module registers that
   path. :func:`route_modules_for_paths` resolves it, including the prefix
   contributed by ``include_router`` in a *different* file.
2. **Dotted module strings.** :func:`module_refs_in` resolves
   ``_APP_MODULE = 'server.app.main'`` and ``_CLOCK_BINDINGS = [...]`` —
   real dependency edges that carry no ``import`` statement. The generated
   acceptance conftest declares its entire wiring this way.

Pure AST + filesystem reads. Never imports the workspace's code.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Iterable, Optional

__all__ = [
    "build_route_index",
    "route_modules_for_paths",
    "request_paths_in_text",
    "module_refs_in",
    "conftest_chain_for",
]

logger = logging.getLogger(__name__)

#: Directories never worth descending into.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "out", "site-packages", ".next", "target", "coverage",
})

#: A quoted path-looking literal: ``"/api/v1/birthdays/{record_id}"``. Also
#: matches the body of an f-string, which is what a test's failing source
#: line usually is.
_PATH_LITERAL_RE = re.compile(r"""["'](/[A-Za-z0-9_\-./{}:]*)["']""")

#: ``"server.app.main"`` / ``'server.app.api.birthdays'`` — a dotted name in
#: a string literal. Two or more segments, so a bare word is not a candidate.
_DOTTED_STRING_RE = re.compile(
    r"""["']([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)["']"""
)

#: Paths that carry no routing information worth matching on.
_UNINTERESTING_PATHS: frozenset[str] = frozenset({"/", ""})


def _read(path: str, limit: int = 200_000) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return None


def _iter_python_files(
    workspace_path: str, *, max_files: int,
) -> Iterable[str]:
    """Workspace-relative ``.py`` paths, vendor and cache dirs pruned."""
    seen = 0
    for dirpath, dirnames, filenames in os.walk(workspace_path):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            yield os.path.relpath(
                os.path.join(dirpath, name), workspace_path,
            ).replace("\\", "/")
            seen += 1
            if seen >= max_files:
                return


def _module_to_rel(workspace_path: str, dotted: str) -> Optional[str]:
    """``server.app.main`` → ``server/app/main.py`` when that file exists in
    the workspace, else None (third-party, stdlib, or a namespace package)."""
    if not dotted or dotted.startswith("."):
        return None
    base = os.path.join(*dotted.split("."))
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.exists(os.path.join(workspace_path, candidate)):
            return candidate.replace("\\", "/")
    return None


def module_refs_in(
    workspace_path: str, rel_files: Iterable[str],
) -> list[str]:
    """Workspace files named by a dotted STRING literal inside ``rel_files``.

    ``_APP_MODULE = 'server.app.main'`` is a dependency edge with no import
    statement behind it; so is every entry of
    ``_CLOCK_BINDINGS = ['server.app.api.birthdays', ...]`` and every
    ``importlib.import_module("...")``. A static import walk sees none of
    them, which is why a generated acceptance conftest looks like it depends
    on nothing but pytest.

    Order is deterministic (sorted) and self-references are dropped.
    """
    skip = {str(r).replace("\\", "/") for r in rel_files}
    out: list[str] = []
    seen: set[str] = set()
    for rel in rel_files:
        src = _read(os.path.join(workspace_path, str(rel)))
        if src is None:
            continue
        for m in _DOTTED_STRING_RE.finditer(src):
            cand = _module_to_rel(workspace_path, m.group(1))
            if cand is None or cand in skip or cand in seen:
                continue
            seen.add(cand)
            out.append(cand)
    return sorted(out)


def conftest_chain_for(
    workspace_path: str, rel_files: Iterable[str],
) -> list[str]:
    """The ``conftest.py`` files pytest would load for ``rel_files``.

    pytest collects conftest from the test's own directory and every ancestor
    up to the rootdir, so each one is a real dependency of the test — the
    fixtures it runs on are defined there. No ``import`` statement records
    that edge, so the import walk misses it entirely.
    """
    out: list[str] = []
    seen: set[str] = set()
    for rel in rel_files:
        parts = str(rel).replace("\\", "/").split("/")[:-1]
        while True:
            cand = "/".join([*parts, "conftest.py"]) if parts else "conftest.py"
            if (
                cand not in seen
                and os.path.exists(os.path.join(workspace_path, cand))
            ):
                seen.add(cand)
                out.append(cand)
            if not parts:
                break
            parts = parts[:-1]
    return out


def _include_router_edges(
    workspace_path: str, rel: str, tree: ast.Module,
) -> list[tuple[str, str]]:
    """``(included_rel_file, prefix)`` for each ``include_router`` call in
    ``rel``.

    Handles the two shapes that actually occur:

    * ``app.include_router(birthdays.router, prefix="/x")`` — the module
      alias is resolved through this file's own imports.
    * ``app.include_router(router, prefix="/x")`` after
      ``from server.app.api.birthdays import router`` — the bare name is
      resolved through the ``from ... import`` that bound it.
    """
    # name → dotted module it refers to, from this file's imports.
    alias_module: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                alias_module[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue  # relative import — not resolvable from the root
            for a in node.names:
                alias_module[a.asname or a.name] = f"{node.module}.{a.name}"
                # ``from pkg import mod`` also binds ``mod`` as that module.
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr != "include_router":
            continue
        prefix = ""
        for kw in node.keywords:
            if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    prefix = kw.value.value
        if not node.args:
            continue
        arg = node.args[0]
        dotted = ""
        if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
            # ``birthdays.router`` → the module ``birthdays`` resolves to.
            dotted = alias_module.get(arg.value.id, "")
        elif isinstance(arg, ast.Name):
            # ``router``, bound by ``from a.b.c import router``: strip the
            # trailing symbol to get the module.
            bound = alias_module.get(arg.id, "")
            dotted = bound.rsplit(".", 1)[0] if "." in bound else ""
        target = _module_to_rel(workspace_path, dotted) if dotted else None
        if target and target != rel:
            out.append((target, prefix))
    return out


def build_route_index(
    workspace_path: str, *, max_files: int = 4000,
) -> list[tuple[str, str]]:
    """``(full_path_template, rel_file)`` for every FastAPI route found.

    The template is the route as a *client* would request it: the decorator's
    path, the router's own ``APIRouter(prefix=...)``, and any prefix added by
    an ``include_router`` call in another file, concatenated. Resolving that
    last part is the whole point — a module declaring ``@router.get(
    "/{record_id}")`` serves ``/api/v1/birthdays/{record_id}``, and only the
    file that mounted it knows so.
    """
    try:
        from harness.contract_tests import parse_fastapi_routes
    except Exception:  # noqa: BLE001 — index is best-effort
        return []
    own: dict[str, list[str]] = {}
    edges: dict[str, list[tuple[str, str]]] = {}
    for rel in _iter_python_files(workspace_path, max_files=max_files):
        src = _read(os.path.join(workspace_path, rel))
        if src is None or (
            "include_router" not in src
            and "APIRouter" not in src
            and "FastAPI" not in src
        ):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        try:
            specs = parse_fastapi_routes(src, rel_path=rel)
        except Exception:  # noqa: BLE001
            specs = []
        if specs:
            own[rel] = [s.path for s in specs]
        found = _include_router_edges(workspace_path, rel, tree)
        if found:
            edges[rel] = found

    # Accumulate mount prefixes down the include graph. Depth is bounded by
    # the number of files, and ``visited`` makes a cycle terminate.
    mounted: dict[str, set[str]] = {}
    for root in list(edges):
        stack: list[tuple[str, str]] = [(root, "")]
        visited: set[tuple[str, str]] = set()
        while stack:
            cur, prefix = stack.pop()
            if (cur, prefix) in visited:
                continue
            visited.add((cur, prefix))
            for child, child_prefix in edges.get(cur, []):
                full = prefix + (child_prefix or "")
                mounted.setdefault(child, set()).add(full)
                stack.append((child, full))

    out: list[tuple[str, str]] = []
    for rel, paths in own.items():
        prefixes = mounted.get(rel) or {""}
        for prefix in prefixes:
            for p in paths:
                full = "/" + "/".join(
                    seg for seg in (prefix + p).split("/") if seg
                )
                out.append((full or "/", rel))
    return out


def _template_matches(template: str, observed: str) -> bool:
    """True when ``observed`` is a request against ``template``.

    ``{...}`` is a wildcard on BOTH sides: the template's path params, and
    the f-string placeholders that survive in a test's failing-source line
    (``client.get(f"/api/v1/birthdays/{record_id}")``).
    """
    t = [s for s in template.split("/") if s]
    o = [s for s in observed.split("/") if s]
    if len(t) != len(o) or not t:
        return False
    for ts, os_ in zip(t, o):
        if ts.startswith("{") or os_.startswith("{"):
            continue
        if ts != os_:
            return False
    return True


def request_paths_in_text(text: str) -> list[str]:
    """HTTP-looking path literals in ``text``, longest first.

    Fed the diagnostic's message and semantic context, which is where the
    failing source line and the response body live.
    """
    found: list[str] = []
    seen: set[str] = set()
    for m in _PATH_LITERAL_RE.finditer(text or ""):
        p = m.group(1)
        if p in _UNINTERESTING_PATHS or p in seen:
            continue
        if "." in p.rsplit("/", 1)[-1]:
            continue  # a file path, not a route (``/tmp/x.py``)
        seen.add(p)
        found.append(p)
    found.sort(key=lambda s: (-len(s.split("/")), s))
    return found


def route_modules_for_paths(
    workspace_path: str,
    paths: Iterable[str],
    *,
    index: Optional[list[tuple[str, str]]] = None,
) -> list[str]:
    """Workspace files registering a route that ``paths`` would reach.

    Deterministic order: the file for the longest matching template first, so
    the most specific route wins when several match.
    """
    table = index if index is not None else build_route_index(workspace_path)
    if not table:
        return []
    hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    for observed in paths:
        for template, rel in table:
            if not _template_matches(template, observed):
                continue
            if rel in seen:
                continue
            seen.add(rel)
            hits.append((len(template.split("/")), rel))
    hits.sort(key=lambda pair: (-pair[0], pair[1]))
    return [rel for _, rel in hits]
