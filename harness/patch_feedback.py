"""Post-patcher LLM feedback composition.

Shared by :func:`harness.graph.patching_node` and
:func:`harness.graph.repair_node` so both give the LLM the same per-file
rejection tag + corrective directive. Prior to this module the two nodes
kept their own copies of the status-message construction, and
patching_node's copy emitted only ``"Failed to apply N patch(es)."`` —
38 chars, no file names, no reasons — leaving the LLM with no signal
whether it had hit a "file missing" (switch to CREATE_FILE), a
"search miss" (re-READ_FILE), or an "ambiguous match" (add context)
error. finsearch session 156032347 stalled 9 consecutive stories on
this loop before the gap was identified.

The compose helper returns three values so the caller can drop each
one into its own place in state:

    status_msg, patch_failures, allowlist_rejections = compose_patch_feedback(...)

    messages.append(MessageDict(role="system", content=status_msg))
    node_state["patch_failures"] = patch_failures
    node_state["allowlist_rejections"] = allowlist_rejections
"""

from __future__ import annotations

from typing import Any

from harness.patcher import PatchResult, _classify_patch_failure


# The wider-context marker the patcher embeds in ``PatchResult.error`` when
# it can show the file lines around the failing search block. Definition
# lives here so :func:`_store_patch_failure_error` can preserve the whole
# marker window when a failure carries it. Re-exported from
# :mod:`harness.graph` for backwards compat with existing tests.
_PATCH_ERROR_WIDER_CONTEXT_MARKER = (
    "Current file content (around closest match):"
)


def _store_patch_failure_error(error_text: str | None) -> str:
    """Prepare a patcher error message for storage in ``node_state``.

    When the error includes the wider-context window marker, return the
    full text — the repair round's ``error_summary`` needs those bytes to
    write a correct SEARCH block. Otherwise cap at 3000 chars so a
    runaway log line can't blow the state.
    """
    err = error_text or ""
    if _PATCH_ERROR_WIDER_CONTEXT_MARKER in err:
        return err
    return err[:3000]


# Per-classification directive shown to the LLM alongside the file name and
# operation type. Keys are the tags produced by
# :func:`harness.patcher._classify_patch_failure`. Missing tags fall back to
# ``_DEFAULT_DIRECTIVE``.
_DIRECTIVE_BY_TAG: dict[str, str] = {
    "file missing": (
        "the target file does not exist on disk yet. Use CREATE_FILE — do "
        "NOT re-emit REPLACE_BLOCK / INSERT_AT_BLOCK / INSERT_AT_LINE "
        "against this path. If it's a package's `__init__.py`, "
        "CREATE_FILE the `__init__.py` first, then edit its siblings."
    ),
    "search miss": (
        "the search block did not match the file bytes. Re-READ_FILE the "
        "target and copy an exact unique block verbatim — do not "
        "reformat, do not paraphrase, do not merge non-adjacent lines."
    ),
    "ambiguous match": (
        "the search block matched more than one location. Add surrounding "
        "context lines above and/or below so exactly one region matches."
    ),
    "rejected: file already exists": (
        "the target file already exists on disk. Use REPLACE_BLOCK / "
        "INSERT_AT_BLOCK / INSERT_AT_LINE to modify it — do not "
        "CREATE_FILE."
    ),
    "path denied": (
        "the target path is outside the workspace or contains '..'. Use "
        "a relative path under the workspace root."
    ),
    "allowlist denied": (
        "the target path is not in the configured allowlist. Move the "
        "file under one of the ALLOWED PATHS listed at the start of this "
        "turn."
    ),
    "no blocks parsed": (
        "the parser could not extract any patch blocks — likely a format "
        "error in the block markers. Re-emit using the exact canonical "
        "`<<<OP>>> ... <<<END_OP>>>` shape."
    ),
}
_DEFAULT_DIRECTIVE = (
    "read the error text above and adjust the block accordingly."
)


def _format_per_file_directives(patch_failures: list[dict]) -> str:
    """Return the per-file "rejection details" section appended to
    ``status_msg`` so the LLM's next round sees a concrete corrective
    directive per failure. Empty when there are no non-allowlist failures.
    """
    if not patch_failures:
        return ""
    lines = []
    for pf in patch_failures:
        if not isinstance(pf, dict):
            continue
        file_path = pf.get("file") or "?"
        op = pf.get("operation") or "?"
        err = pf.get("error") or ""
        tag = _classify_patch_failure(
            err if isinstance(err, str) else str(err)
        )
        directive = _DIRECTIVE_BY_TAG.get(tag, _DEFAULT_DIRECTIVE)
        lines.append(f"  - {file_path} ({op}, {tag}): {directive}")
    return (
        "\n[Per-file rejection details — read carefully, do not re-emit "
        "the same block:]\n" + "\n".join(lines)
    )


def compose_patch_feedback(
    patch_results: list[PatchResult],
    allowed_paths: list[str],
    parse_miss_diag: str,
    *,
    prefix: str,
    success_count: int,
    fail_count: int,
    no_op_count: int,
    max_failures_stored: int = 5,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Compose the post-patcher LLM status message and the two state
    buckets (``patch_failures`` and ``allowlist_rejections``) the router
    and next repair round both read.

    ``prefix`` absorbs the two callers' cosmetic differences:

      - :func:`patching_node` → ``"[System]:"``
      - :func:`repair_node`  → ``"[System]: Repair attempt {N}:"``

    ``success_count`` and ``fail_count`` are passed in (not recomputed from
    ``patch_results``) because :func:`patching_node` may demote
    ``success_count`` to 0 when every "success" was a stuck-file
    re-read-required rejection.
    """
    fail_results = [r for r in patch_results if not r.success]

    allowlist_rejections: list[dict[str, Any]] = [
        {"file": r.file, "operation": r.operation, "reason": r.error}
        for r in fail_results
        if isinstance(r.error, str) and "not in skill allowlist" in r.error
    ]
    patch_failures: list[dict[str, Any]] = [
        {
            "file": r.file,
            "operation": (
                r.operation.value
                if hasattr(r.operation, "value") else str(r.operation)
            ),
            "error": _store_patch_failure_error(r.error or ""),
        }
        for r in fail_results
        if isinstance(r.error, str) and "not in skill allowlist" not in r.error
    ][:max_failures_stored]

    if success_count > 0:
        status_msg = (
            f"{prefix} Applied {success_count}/{len(patch_results)} "
            f"patches successfully."
        )
        if no_op_count > 0:
            status_msg += (
                f" {no_op_count} were idempotency no-ops (target file "
                f"already at expected state — no actual change made)."
            )
        if fail_count > 0:
            failed_files = ", ".join(r.file for r in fail_results)
            status_msg += f" Failed on: {failed_files}."
    else:
        status_msg = f"{prefix} Failed to apply {fail_count} patch(es)."
        if fail_count == 0 and parse_miss_diag:
            # LLM emitted marker openers but nothing parsed. Overwrite the
            # vacuous default with the parser's concrete diagnosis (the
            # cc9ab6a path).
            status_msg = (
                f"{prefix} Emitted patch marker(s) but zero blocks "
                f"landed — the parser could not extract any. "
                f"{parse_miss_diag}"
            )

    status_msg += _format_per_file_directives(patch_failures)

    if allowlist_rejections:
        rejected_paths = ", ".join(
            sorted({str(r["file"]) for r in allowlist_rejections})
        )
        status_msg += (
            f"\n[Allowlist] Rejected paths outside the configured layout: "
            f"{rejected_paths}. Allowed roots: {allowed_paths}."
        )

    return status_msg, patch_failures, allowlist_rejections
