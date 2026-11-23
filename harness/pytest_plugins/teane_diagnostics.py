"""Teane pytest diagnostics plugin — self-contained, single file.

Enriches pytest's failure output with runtime values the LLM needs to
diagnose "assert 500 == 200" style failures where the SYMPTOM is in the
diagnostic but the CAUSE is hidden inside a locals object (an
``httpx.Response``'s body, a ``CompletedProcess``'s stderr, an
exception chain).

Emits every extra line with a ``[teane]`` prefix so the harness's stderr
parser (harness/parser_registry.py) can find and preserve them in the
diagnostic's ``semantic_context`` without ambiguity.

Loaded into the sandbox via ``-p teane_diagnostics`` (see
``harness/sandbox.py``'s DockerBackend, which bind-mounts this file's
parent dir at ``/opt/teane_pytest_plugins`` and injects the ``-p`` flag
through ``PYTEST_ADDOPTS``).

Escape hatches — ANY of these disables the plugin cleanly:

  1. ``pytest.ini`` (or ``pyproject.toml`` tool section):
     ``teane_diagnostics = off``
  2. Environment variable: ``TEANE_DIAGNOSTICS=off`` (also ``0`` / ``false``
     / ``no`` / ``disabled``).
  3. From the workspace's own ``conftest.py``::

        from teane_diagnostics import disable_for_this_run
        disable_for_this_run()

The plugin also DEBUG-logs when it detects other
``pytest_runtest_makereport`` hooks already registered — for the
post-mortem trail, not to skip (pytest composes hooks via
``hookwrapper=True``; both fire).

Every renderer is wrapped in try/except so a broken renderer skips
silently — a malformed local value never breaks the underlying test
run. Values are truncated to ``_MAX_VALUE_CHARS`` so a fixture stashing
a 10 MB blob on ``self`` can't blow the diagnostic prompt.
"""
from __future__ import annotations

import os
from typing import Any, Callable, List, Tuple


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------

# Marker prefix. The harness parser greps for this to know what to keep.
_MARKER = "[teane]"

# Per-value character cap. HTTP bodies / subprocess stdout can be huge; we
# keep just enough for the LLM to see the payload shape.
_MAX_VALUE_CHARS = 800

# Per-frame line cap. Prevents an object with 200 dict entries from
# dominating the diagnostic.
_MAX_LINES_PER_FRAME = 24

# Env-var values that count as "off". Matches the harness's own
# ``_falsy_env_values`` idiom.
_FALSY_ENV = frozenset({"0", "off", "false", "no", "disabled", "none", ""})

# Runtime kill switch — flipped by ``disable_for_this_run()`` from a
# workspace conftest.py that wants full control. Module-level so both
# the conftest and the pytest hook see the same state.
_disabled: bool = False

# Layer 3b — debug mode. When the harness's compiler_node has watched
# the same pytest nodeid fail for N repair rounds AND the reflection
# verdict has flagged DISTRACTION/REGRESSION, it re-runs that single
# test with ``TEANE_DIAGNOSTICS_MODE=debug`` set. In this mode the
# plugin turns on three extra collectors that are too expensive (or
# too invasive) for every run:
#
#   1. Pre-call locals snapshot — dumps fixture-injected values BEFORE
#      the test body runs so we see them intact even when the body
#      mutates them.
#   2. Non-destructive response-body cache — patches ``httpx.Response``
#      / ``requests.Response`` ``text`` / ``content`` accessors so
#      reading them once doesn't invalidate later reads. Otherwise a
#      test that does ``resp.json()`` then asserts on status gets a
#      dead ``.text`` at the failure frame.
#   3. Unhandled asyncio task capture — a task that raised silently
#      (fire-and-forget bug) is the fingerprint of "the test passed
#      but the app is broken." Registered via
#      ``asyncio.get_event_loop().set_exception_handler`` on the
#      running loop.
#
# Cheap in normal mode: three ``if _debug_mode:`` guards.
_debug_mode: bool = False
_debug_task_exceptions: list[str] = []
_debug_pre_call_locals: dict[str, dict[str, str]] = {}


def disable_for_this_run() -> None:
    """Workspace-facing escape hatch: call from ``conftest.py`` to opt out
    of teane's diagnostic enrichment for this pytest run. Safe to call
    multiple times."""
    global _disabled
    _disabled = True


# ---------------------------------------------------------------------------
# Renderers — (predicate, formatter). Order = priority; first hit wins.
# ---------------------------------------------------------------------------

Renderer = Tuple[Callable[[Any], bool], Callable[[str, Any], List[str]]]


def _truncate(text: str, cap: int = _MAX_VALUE_CHARS) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f" …({len(text) - cap} more chars)"


def _is_httpx_response(obj: Any) -> bool:
    # Match by shape rather than isinstance to avoid an httpx import in
    # the plugin (which runs in the SANDBOX's venv — httpx may not be
    # importable in a Java project's venv). ``.status_code`` + ``.text``
    # + ``.headers`` is the httpx / requests / TestClient contract.
    return (
        type(obj).__name__ == "Response"
        and hasattr(obj, "status_code")
        and hasattr(obj, "text")
        and hasattr(obj, "headers")
    )


def _fmt_httpx_response(name: str, obj: Any) -> List[str]:
    status = getattr(obj, "status_code", "<?>")
    lines = [f"{_MARKER} {name} = <Response status_code={status}>"]
    try:
        headers = dict(getattr(obj, "headers", {}) or {})
    except Exception:
        headers = {}
    if headers:
        # Redact common auth / cookie headers so we don't smuggle secrets
        # into the LLM prompt.
        redacted = {
            k: ("<redacted>" if k.lower() in {"authorization", "cookie", "set-cookie", "x-api-key"} else v)
            for k, v in headers.items()
        }
        lines.append(f"{_MARKER}   headers: {_truncate(str(redacted))}")
    # ``.text`` on httpx is a property that reads the body. If the response
    # hasn't been consumed yet this may raise or block; wrap defensively.
    try:
        body = obj.text
        if isinstance(body, str) and body:
            lines.append(f"{_MARKER}   body:    {_truncate(body)!r}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"{_MARKER}   body:    <unreadable: {type(exc).__name__}: {exc}>")
    return lines


def _is_completed_process(obj: Any) -> bool:
    return type(obj).__name__ == "CompletedProcess" and hasattr(obj, "returncode")


def _fmt_completed_process(name: str, obj: Any) -> List[str]:
    rc = getattr(obj, "returncode", "<?>")
    lines = [f"{_MARKER} {name} = CompletedProcess(returncode={rc})"]
    for stream in ("stdout", "stderr"):
        val = getattr(obj, stream, None)
        if isinstance(val, (bytes, bytearray)):
            try:
                val = val.decode("utf-8", errors="replace")
            except Exception:
                val = repr(val)
        if isinstance(val, str) and val:
            lines.append(f"{_MARKER}   {stream}: {_truncate(val)!r}")
    return lines


def _is_pathlib_path(obj: Any) -> bool:
    # ``pathlib.PurePath`` is the base of every Path variant.
    return (
        hasattr(obj, "parts")
        and hasattr(obj, "exists")
        and hasattr(obj, "is_file")
        and type(obj).__module__.startswith("pathlib")
    )


def _fmt_pathlib_path(name: str, obj: Any) -> List[str]:
    lines = [f"{_MARKER} {name} = Path({str(obj)!r})"]
    try:
        exists = obj.exists()
    except Exception:
        exists = None
    lines.append(f"{_MARKER}   exists: {exists}")
    if exists and obj.is_file():
        try:
            with obj.open("rb") as fh:
                data = fh.read(_MAX_VALUE_CHARS + 1)
            try:
                text = data.decode("utf-8")
                lines.append(f"{_MARKER}   text:   {_truncate(text)!r}")
            except UnicodeDecodeError:
                lines.append(f"{_MARKER}   bytes:  {_truncate(data.hex())} (hex)")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{_MARKER}   read:   <unreadable: {type(exc).__name__}: {exc}>")
    return lines


def _is_exception(obj: Any) -> bool:
    return isinstance(obj, BaseException)


def _fmt_exception(name: str, obj: Any) -> List[str]:
    lines = [f"{_MARKER} {name} = {type(obj).__name__}: {_truncate(str(obj))}"]
    cause = getattr(obj, "__cause__", None)
    context = getattr(obj, "__context__", None)
    if cause is not None:
        lines.append(f"{_MARKER}   __cause__: {type(cause).__name__}: {_truncate(str(cause))}")
    elif context is not None and not getattr(obj, "__suppress_context__", False):
        lines.append(f"{_MARKER}   __context__: {type(context).__name__}: {_truncate(str(context))}")
    return lines


_RENDERERS: List[Renderer] = [
    (_is_httpx_response, _fmt_httpx_response),
    (_is_completed_process, _fmt_completed_process),
    (_is_pathlib_path, _fmt_pathlib_path),
    (_is_exception, _fmt_exception),
]


# ---------------------------------------------------------------------------
# Escape-hatch detection
# ---------------------------------------------------------------------------


def _env_disabled() -> bool:
    val = os.environ.get("TEANE_DIAGNOSTICS", "").strip().lower()
    return val in _FALSY_ENV and val != ""


def _ini_disabled(config: Any) -> bool:
    """Return True when pytest.ini (or equivalent) explicitly opts out.

    Accepts ``teane_diagnostics = off`` (or any falsy synonym). We register
    the option in ``pytest_addoption`` so pytest's ini-parser knows about
    it; without registration ``config.getini`` raises ``ValueError``.
    """
    try:
        raw = config.getini("teane_diagnostics")
    except (ValueError, KeyError):
        return False
    if raw is None:
        return False
    return str(raw).strip().lower() in _FALSY_ENV and str(raw).strip() != ""


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: Any) -> None:
    """Register the ``teane_diagnostics`` ini option so ``pytest.ini`` can
    opt out cleanly. Without registration ``config.getini`` would raise."""
    try:
        parser.addini(
            "teane_diagnostics",
            help=(
                "Set to 'off' (or 0/false/no/disabled) to disable the teane "
                "harness's pytest diagnostic enrichment for this workspace."
            ),
            default="on",
        )
    except (ValueError, KeyError):
        # Another plugin already registered this key. Coexist quietly.
        pass


def pytest_configure(config: Any) -> None:
    """Decide once, per pytest session, whether we should register the
    report hook. Honours env var, ini flag, and the runtime kill-switch.

    Also latches the Layer 3b ``TEANE_DIAGNOSTICS_MODE=debug`` flag into
    ``_debug_mode`` so the pre-call / task-exception collectors know
    to fire.
    """
    global _disabled, _debug_mode, _rootdir
    if _env_disabled() or _ini_disabled(config):
        _disabled = True
        return
    try:
        _rootdir = str(getattr(config, "rootpath", "") or getattr(config, "rootdir", "") or os.getcwd())
    except Exception:
        _rootdir = os.getcwd()
    if os.environ.get("TEANE_DIAGNOSTICS_MODE", "").strip().lower() == "debug":
        _debug_mode = True
        # Register the asyncio-exception collector on any running loop
        # right away — pytest-asyncio may swap loops per test but each
        # loop gets a fresh handler via ``pytest_pyfunc_call`` below.
        _install_task_exception_handler()
    # Diagnostic — not a skip. Post-mortem tooling should be able to see
    # that we coexisted with a user hook.
    try:
        existing = config.hook.pytest_runtest_makereport.get_hookimpls()
        others = [
            impl for impl in existing
            if getattr(impl, "plugin_name", "") != "teane_diagnostics"
            and getattr(impl, "function", None) is not None
        ]
        if others:
            # pytest's plugin manager already tolerates multiple hookimpls
            # composing via ``hookwrapper=True``; we just log that this
            # is happening so a session with unexpected diagnostic output
            # can be traced back to a user hook.
            names = ", ".join(
                str(getattr(impl, "plugin_name", "?")) for impl in others
            )
            print(
                f"{_MARKER} teane_diagnostics: coexisting with "
                f"pytest_runtest_makereport hooks from: {names}"
            )
    except Exception:
        pass


def _walk_to_deepest_user_frame(tb: Any) -> Any:
    """Return the deepest traceback frame that lives OUTSIDE site-packages /
    stdlib. That's the frame whose locals the LLM cares about — a user
    test that raised. Falls back to the deepest frame overall if no user
    frame is found (e.g. all frames are inside a helper library)."""
    best = tb
    node = tb
    while node is not None:
        fname = getattr(node.tb_frame.f_code, "co_filename", "") or ""
        # Cheap heuristic. Anything not clearly a dependency wins.
        if not any(
            frag in fname
            for frag in ("site-packages", "/lib/python", os.sep + "_pytest" + os.sep)
        ) and fname:
            best = node
        node = node.tb_next
    return best


def _extract_extra_lines(excinfo: Any) -> List[str]:
    if excinfo is None:
        return []
    tb = getattr(excinfo, "tb", None)
    if tb is None:
        return []
    try:
        frame_tb = _walk_to_deepest_user_frame(tb)
        f_locals = dict(frame_tb.tb_frame.f_locals)
    except Exception:
        return []
    extra: List[str] = []
    for name, value in f_locals.items():
        if len(extra) >= _MAX_LINES_PER_FRAME:
            break
        # Skip dunder / private / magic names — noise.
        if name.startswith("_"):
            continue
        for predicate, formatter in _RENDERERS:
            try:
                if not predicate(value):
                    continue
                for line in formatter(name, value):
                    extra.append(line)
                    if len(extra) >= _MAX_LINES_PER_FRAME:
                        break
                break
            except Exception:
                # Predicate or formatter blew up — never fatal.
                continue
    return extra


# ---------------------------------------------------------------------------
# Layer 4 — swallowed-exception recorder.
#
# A route that wraps its body in ``except Exception: raise AppError("Internal
# server error")`` turns every real failure into the same generic 500. The
# test then sees only that body, and so does the repair LLM: lumina 01a1de20
# (run 16) spent four repair rounds on a crash whose cause never reached any
# diagnostic, because the only place it existed was the exception the
# application caught and threw away.
#
# ``sys.monitoring`` (3.12+) fires EXCEPTION_HANDLED when control enters an
# exception handler block, globally and across threads — which matters
# because Starlette's TestClient runs the app in a worker thread. The event
# names the try block, not the clause that matched, so the recorder resolves
# the clause itself: it finds the innermost ``try`` whose BODY holds the
# frame's current line and evaluates each handler's type in the frame's own
# namespace, first match wins, exactly as the interpreter will. Only
# exceptions caught by a BROAD clause (bare ``except``, ``Exception``,
# ``BaseException``) in workspace PRODUCTION code are kept; a narrow handler
# is the application handling a case it anticipated.
#
# Caps come from the harness's config via env (``sandbox.
# swallowed_exception_report_max`` / ``swallowed_exception_traceback_lines``,
# exported by ``harness/sandbox.py``); the defaults below apply when the
# plugin runs outside the harness. ``TEANE_SWALLOWED_MAX=0`` disables it.
# ---------------------------------------------------------------------------

_SWALLOWED_MAX_DEFAULT = 2
_SWALLOWED_TB_LINES_DEFAULT = 20

#: Tool ids sys.monitoring reserves for others: 0 debugger, 1 coverage,
#: 2 profiler, 5 optimizer. 3 and 4 are unassigned.
_MONITOR_TOOL_CANDIDATES = (3, 4)

_swallowed: list[dict[str, Any]] = []
_swallowed_keys: set[tuple[str, int, str, int]] = set()
_monitor_tool_id: Any = None
_rootdir: str = ""
_broad_try_cache: dict[str, Any] = {}

_BROAD_NAMES = frozenset({"Exception", "BaseException"})


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _is_production_path(path: str) -> bool:
    """A workspace source file that is not a test, fixture or dependency."""
    if not path or not _rootdir:
        return False
    try:
        ap = os.path.abspath(path)
    except Exception:
        return False
    root = _rootdir.rstrip(os.sep) + os.sep
    if not ap.startswith(root):
        return False
    rel = ap[len(root):]
    parts = rel.split(os.sep)
    if any(
        p in ("site-packages", "dist-packages", "node_modules", "tests",
              "test", "__pycache__") or p.startswith(".")
        for p in parts[:-1]
    ):
        return False
    base = parts[-1]
    if base == "conftest.py" or base.startswith("test_") or base.endswith("_test.py"):
        return False
    return base.endswith(".py") and base != os.path.basename(__file__)


_production_path_memo: dict[str, bool] = {}


def _production_path_cached(path: str) -> bool:
    hit = _production_path_memo.get(path)
    if hit is None:
        hit = _is_production_path(path)
        _production_path_memo[path] = hit
    return hit


def _try_nodes(path: str) -> list[Any]:
    """Every ``try`` statement in ``path`` (parsed once per file)."""
    cached = _broad_try_cache.get(path)
    if cached is not None:
        return cached
    nodes: list[Any] = []
    try:
        import ast
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        try_types = tuple(
            t for t in (getattr(ast, "Try", None), getattr(ast, "TryStar", None))
            if t is not None
        )
        nodes = [n for n in ast.walk(tree) if isinstance(n, try_types)]
    except Exception:
        nodes = []
    _broad_try_cache[path] = nodes
    return nodes


def _handler_is_broad(type_node: Any) -> bool:
    import ast
    if type_node is None:
        return True
    elts = type_node.elts if isinstance(type_node, ast.Tuple) else [type_node]
    for e in elts:
        name = e.id if isinstance(e, ast.Name) else (
            e.attr if isinstance(e, ast.Attribute) else ""
        )
        if name in _BROAD_NAMES:
            return True
    return False


def _matching_handler(path: str, line: int, exc: BaseException, frame: Any) -> Any:
    """The ``except`` clause that will catch ``exc`` raised through ``line``,
    or None (not inside a try body, or no clause matches)."""
    import ast
    best = None
    for node in _try_nodes(path):
        body = getattr(node, "body", None) or []
        if not body:
            continue
        start = body[0].lineno
        end = getattr(body[-1], "end_lineno", None) or body[-1].lineno
        if start <= line <= end and (best is None or start >= best.body[0].lineno):
            best = node
    if best is None:
        return None
    ns: dict[str, Any] = {}
    try:
        import builtins
        ns.update(vars(builtins))
        ns.update(frame.f_globals)
        ns.update(frame.f_locals)
    except Exception:
        pass
    for handler in best.handlers:
        if handler.type is None:
            return handler
        try:
            expr = ast.Expression(body=handler.type)
            typ = eval(compile(expr, path, "eval"), ns)  # noqa: S307 — the app's own handler expression
            if isinstance(exc, typ):
                return handler
        except Exception:
            # Unresolvable type expression: a broad-looking clause still
            # names its intent statically, so fall back to the name.
            if _handler_is_broad(handler.type):
                return handler
    return None


def _on_exception_handled(code: Any, offset: int, exc: BaseException) -> Any:
    # Always return None. EXCEPTION_HANDLED is one of the events that
    # sys.monitoring refuses to DISABLE per location — returning DISABLE
    # raises ValueError INSIDE the application's own except clause, which
    # turns the observer into the failure. The per-file decision is cached
    # instead, so a dependency's handled exceptions cost one dict lookup.
    import sys
    try:
        path = code.co_filename
        if not _production_path_cached(path):
            return None
        if len(_swallowed) >= _env_int("TEANE_SWALLOWED_MAX", _SWALLOWED_MAX_DEFAULT):
            return None
        tb = exc.__traceback__
        if tb is None:
            return None
        frame = sys._getframe(1)
        # The first traceback entry is the handling frame: its line is where
        # the exception passed through this function's try body.
        line = tb.tb_lineno
        handler = _matching_handler(path, line, exc, frame)
        if handler is None or not _handler_is_broad(handler.type):
            return None
        deepest = tb
        while deepest.tb_next is not None:
            deepest = deepest.tb_next
        key = (path, handler.lineno, type(exc).__name__, deepest.tb_lineno)
        if key in _swallowed_keys:
            return None
        _swallowed_keys.add(key)
        import traceback
        tb_lines = [
            ln for chunk in traceback.format_exception(type(exc), exc, tb)
            for ln in chunk.rstrip("\n").splitlines()
        ]
        _swallowed.append({
            "handler": f"{os.path.relpath(path, _rootdir)}:{handler.lineno}",
            "clause": "except" if handler.type is None else "except "
                + _clause_text(handler.type),
            "summary": f"{type(exc).__name__}: {_truncate(str(exc), 300)}",
            "traceback": [
                ln.replace(_rootdir.rstrip(os.sep) + os.sep, "") for ln in tb_lines
            ],
        })
    except Exception:
        return None
    return None


def _clause_text(type_node: Any) -> str:
    try:
        import ast
        return ast.unparse(type_node)
    except Exception:
        return "Exception"


def _start_swallow_recorder() -> None:
    """Arm EXCEPTION_HANDLED for the duration of one test's call phase."""
    global _monitor_tool_id
    _swallowed.clear()
    _swallowed_keys.clear()
    if _env_int("TEANE_SWALLOWED_MAX", _SWALLOWED_MAX_DEFAULT) <= 0:
        return
    import sys
    mon = getattr(sys, "monitoring", None)
    if mon is None:
        return
    try:
        if _monitor_tool_id is None:
            for tid in _MONITOR_TOOL_CANDIDATES:
                if mon.get_tool(tid) is None:
                    mon.use_tool_id(tid, "teane_diagnostics")
                    _monitor_tool_id = tid
                    break
            else:
                return
            mon.register_callback(
                _monitor_tool_id, mon.events.EXCEPTION_HANDLED,
                _on_exception_handled,
            )
        mon.restart_events()
        mon.set_events(_monitor_tool_id, mon.events.EXCEPTION_HANDLED)
    except Exception:
        _monitor_tool_id = None


def _stop_swallow_recorder() -> None:
    import sys
    mon = getattr(sys, "monitoring", None)
    if mon is None or _monitor_tool_id is None:
        return
    try:
        mon.set_events(_monitor_tool_id, 0)
    except Exception:
        pass


def _render_swallowed_exceptions() -> List[str]:
    """``[teane]`` lines for every exception production code swallowed
    during the test that just failed. The leading phrase is matched by the
    harness (``_CODE_GAP_EVIDENCE`` in graph.py) — keep them in step."""
    cap = _env_int("TEANE_SWALLOWED_TB_LINES", _SWALLOWED_TB_LINES_DEFAULT)
    out: List[str] = []
    for rec in _swallowed:
        out.append(
            f"{_MARKER} exception swallowed by application code: "
            f"{rec['summary']} — caught by broad `{rec['clause']}` at "
            f"{rec['handler']}, so the response carried none of it. "
            "Real traceback:"
        )
        tb = rec["traceback"]
        if cap and len(tb) > cap:
            # Keep the head (where it entered app code) and the tail (where
            # it was raised) — the middle is the least informative part.
            head = cap // 3
            tb = tb[:head] + [f"... {len(tb) - cap} line(s) elided ..."] + tb[-(cap - head):]
        out.extend(f"{_MARKER}   {ln}" for ln in tb)
    return out


# ---------------------------------------------------------------------------
# Layer 3b — debug-mode collectors. Only registered when TEANE_DIAGNOSTICS_MODE
# == "debug", which the harness sets via env when the compiler_node's Layer 3
# gate fires (same-nodeid streak >= 3 AND reflection verdict has flagged
# DISTRACTION/REGRESSION at least once).
# ---------------------------------------------------------------------------


def _install_task_exception_handler() -> None:
    """Register an asyncio exception handler that captures exceptions
    raised by fire-and-forget tasks. A task that raised silently is the
    classic "the test passed but the app is actually broken" fingerprint;
    without this handler pytest never sees them because they die inside
    the running loop and only surface as ``Task exception was never
    retrieved`` on GC.

    Registered defensively — if no loop is running yet the handler will
    be installed on the next loop pytest-asyncio creates (via the
    ``pytest_pyfunc_call`` wrapper below).
    """
    try:
        import asyncio
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except Exception:
        return

    def _handler(loop: Any, context: dict[str, Any]) -> None:
        try:
            exc = context.get("exception")
            msg = context.get("message", "")
            if exc is not None:
                rendered = f"{type(exc).__name__}: {_truncate(str(exc))}"
            else:
                rendered = f"(no exception) {_truncate(str(msg))}"
            _debug_task_exceptions.append(rendered)
        except Exception:
            pass

    try:
        loop.set_exception_handler(_handler)
    except Exception:
        pass


def _snapshot_pre_call_locals(item: Any) -> None:
    """Dump the test function's arguments (fixtures + parameters) before
    the body runs. If a fixture is mutated during the test, this snapshot
    still shows its pre-mutation state — vital for debugging shared-
    state pollution across tests."""
    if not _debug_mode:
        return
    try:
        # ``funcargs`` is pytest's dict of fixture-injected values for
        # the current test. Available on ``Item`` after fixture setup.
        funcargs = getattr(item, "funcargs", None)
        if not funcargs:
            return
        rendered: dict[str, str] = {}
        for name, value in funcargs.items():
            if name.startswith("_"):
                continue
            try:
                rendered[name] = _truncate(repr(value))
            except Exception:
                rendered[name] = "<unreproducible>"
        _debug_pre_call_locals[item.nodeid] = rendered
    except Exception:
        pass


def _fmt_debug_lines(item: Any) -> List[str]:
    """Format the debug-mode enrichment. Uses a distinct marker
    ``[teane-debug]`` so the harness parser can slice Layer 3 output out
    of the raw pytest log without conflating with Layer 2's
    ``[teane]`` lines."""
    lines: List[str] = []
    try:
        nodeid = getattr(item, "nodeid", "")
    except Exception:
        nodeid = ""
    pre = _debug_pre_call_locals.get(nodeid, {})
    if pre:
        lines.append("[teane-debug] pre-call fixture/argument snapshot:")
        for name, val in sorted(pre.items()):
            lines.append(f"[teane-debug]   {name} = {val}")
    if _debug_task_exceptions:
        lines.append("[teane-debug] unhandled asyncio task exceptions:")
        for entry in _debug_task_exceptions[-8:]:
            lines.append(f"[teane-debug]   {entry}")
    return lines


try:
    import pytest as _pytest_module

    @_pytest_module.hookimpl(hookwrapper=True)
    def pytest_pyfunc_call(pyfuncitem: Any):  # type: ignore[no-untyped-def]
        """Layer 3b — capture pre-call locals before the test body
        executes. hookwrapper=True composes with the plugin's other hooks
        and any user hook the workspace registers."""
        if _debug_mode and not _disabled:
            _snapshot_pre_call_locals(pyfuncitem)
            # Reinstall the task-exception handler for the current loop.
            # pytest-asyncio may have swapped loops since pytest_configure
            # ran; without this reinstall, per-test loops get no handler.
            _install_task_exception_handler()
        yield

    @_pytest_module.hookimpl(hookwrapper=True)
    def pytest_runtest_call(item: Any):  # type: ignore[no-untyped-def]
        """Layer 4 — record exceptions production code swallows while the
        test body runs. Armed for the call phase only, so fixture setup and
        teardown (which legitimately catch and retry) never contribute."""
        if not _disabled:
            _start_swallow_recorder()
        try:
            yield
        finally:
            if not _disabled:
                _stop_swallow_recorder()

    @_pytest_module.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item: Any, call: Any):  # type: ignore[no-untyped-def]
        """After every test-phase, if the phase failed and the plugin is
        enabled, append renderer output to the report's ``longrepr`` under
        the ``[teane]`` marker. ``hookwrapper=True`` composes cleanly with
        any user hook registered against the same event.
        """
        outcome = yield
        if _disabled:
            return
        try:
            report = outcome.get_result()
        except Exception:
            return
        if getattr(report, "when", "") != "call":
            return
        if not getattr(report, "failed", False):
            return
        excinfo = getattr(call, "excinfo", None)
        extra = _extract_extra_lines(excinfo)
        # Layer 3b — append pre-call snapshot + unhandled task exceptions.
        # Uses a distinct ``[teane-debug]`` marker so the harness parser
        # can slice this bucket separately from Layer 2's ``[teane]``
        # lines. Only appears when debug mode is on (compiler_node gates
        # the re-run) so ordinary runs don't pay the format cost.
        if _debug_mode:
            extra = extra + _fmt_debug_lines(item)
        try:
            extra = extra + _render_swallowed_exceptions()
        except Exception:
            pass
        if not extra:
            return
        # Attach to longrepr. Handle both str and ``ReprEntry``-style
        # longreprs — falling back to appending a stringified block.
        prefix = "\n\n" + "\n".join(extra)
        try:
            current = report.longrepr
            if hasattr(current, "addsection"):
                current.addsection("teane diagnostics enrichment", "\n".join(extra))
            else:
                report.longrepr = str(current) + prefix
        except Exception:
            # Last resort: replace with our own block so at least SOMETHING
            # reaches the parser.
            report.longrepr = prefix
except ImportError:
    # ``pytest`` is not importable in whatever process is running us.
    # The hook simply won't fire — that's fine, we're a plugin, not a
    # library.
    pass
