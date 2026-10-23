"""One patch dialect per call, chosen by provider capability — never both.

The system prompt used to carry the native-tool bridge AND the full text-DSL
reference at the same time: the bridge said "DO NOT reproduce the `<<<>>>`
markers" while 10,696 chars of `## Patch Syntax` / `## Edit Invariants`
taught that exact syntax below it. A model shown a syntax uses it, which is
the whole explanation for

  * `<<<EDIT_FILE>>>` emissions the patcher parsed as zero blocks (1e08364),
  * non-canonical `REWRITE_FILE` output needing salvage (5cb41d7),
  * the zero-patch rounds that cascade into the distraction loop.

Native mode now gets the tool list plus the invariants that still apply to a
tool call; the per-operation FIELD semantics live in each tool's JSON-Schema
`description`, where the provider renders them beside the call site. DSL mode
— a provider that cannot do tool use — gets the full text reference and no
tool talk. Tool calls are still folded to DSL text and parsed by the shared
`PatchBlock` backend, so the parser is NOT dead code: it is the convergence
point for both front-ends.
"""

import harness.tool_schemas as ts
from harness.graph import (
    _DSL_PATCH_SYNTAX,
    _NATIVE_EDIT_INVARIANTS,
    _NATIVE_TOOL_MODE_BRIDGE,
    _build_patch_interface_block,
)

#: Patch operations that exist as both a tool and a DSL marker. Control
#: markers with no tool form (READ_FILE is a tool, but PROMOTE_DEFERRED and
#: the web/MCP blocks are text-only by design) are excluded.
_PATCH_OPS = (
    "REPLACE_BLOCK", "CREATE_FILE", "REWRITE_FILE", "DELETE_BLOCK",
    "INSERT_AT_BLOCK", "INSERT_AT_LINE", "REPLACE_LINE_RANGE",
)


class TestExactlyOneDialectIsTaught:
    def test_native_mode_teaches_no_dsl_markers(self):
        block = _build_patch_interface_block(True)
        assert "<<<" not in block

    def test_native_mode_names_no_dsl_operations(self):
        """Even without the markers, naming REPLACE_BLOCK invites the
        syntax — the old reminder said "call the tool of the same meaning"
        precisely because the prompt still named blocks."""
        block = _build_patch_interface_block(True)
        for op in _PATCH_OPS:
            assert op not in block, f"native prompt still names {op}"

    def test_native_mode_names_the_tools_it_offers(self):
        block = _build_patch_interface_block(True)
        for tool in ("edit_file", "create_file", "rewrite_file", "read_file"):
            assert f"`{tool}`" in block, f"native prompt omits {tool}"

    def test_native_mode_does_not_name_the_internal_only_ops(self):
        """The four line/anchor ops totalled 17 of 1,229 observed calls and
        ``delete_block`` was never called once. Their schemas and apply
        paths stay (autofix and security_fix_rules build them
        programmatically), but offering them to the model is surface it
        can only get wrong — every one is an ``edit_file`` once the bytes
        are in hand, which read-before-edit guarantees."""
        block = _build_patch_interface_block(True)
        for tool in ("delete_block", "insert_at_block", "insert_at_line",
                     "replace_line_range"):
            assert f"`{tool}`" not in block, f"native prompt still offers {tool}"

    def test_dsl_mode_keeps_every_operation_it_documented(self):
        block = _build_patch_interface_block(False)
        # NOTE: the DSL reference documents 5 of the 7 operations — it never
        # taught INSERT_AT_LINE or REPLACE_LINE_RANGE, even though the parser
        # accepts both and the native tool list offers them. That gap predates
        # this change; asserted as-is so the split is provably lossless rather
        # than quietly "fixed" here.
        for op in ("REPLACE_BLOCK", "CREATE_FILE", "REWRITE_FILE",
                   "DELETE_BLOCK", "INSERT_AT_BLOCK"):
            assert f"<<<{op}>>>" in block, f"DSL prompt lost {op}"
        assert "<<<READ_FILE>>>" in block
        assert "Edit Invariants" in block

    def test_dsl_mode_does_not_offer_tools(self):
        """A provider that cannot call tools must not be told to."""
        block = _build_patch_interface_block(False)
        assert "NATIVE TOOL-USE" not in block
        assert "CALLING a tool" not in block

    def test_the_two_modes_are_disjoint_not_additive(self):
        native = _build_patch_interface_block(True)
        dsl = _build_patch_interface_block(False)
        assert native != dsl
        # The old bug was concatenation: bridge + DSL in one prompt.
        assert _NATIVE_TOOL_MODE_BRIDGE not in dsl
        assert _DSL_PATCH_SYNTAX not in native

    def test_dsl_text_is_preserved_byte_for_byte(self):
        """A DSL-mode run must behave exactly as it did before the split."""
        assert _DSL_PATCH_SYNTAX.lstrip("\n").startswith("## Edit Invariants")
        assert "## Patch Syntax" in _DSL_PATCH_SYNTAX
        # Size guard: the extracted reference was 10,696 chars.
        assert len(_DSL_PATCH_SYNTAX) > 10_000


class TestNeitherDialectIsTheRealHazard:
    """Teaching one dialect is only safe if it is the one the gateway will
    honour. The prompt reads ``patcher.use_structured_tools``; the gateway
    requires that AND ``spec.supports_tools`` for the routed model,
    suppressing the tools array otherwise (gateway.py:3605). While the
    prompt carried BOTH dialects that mismatch was survivable — the model
    wrote DSL. With native mode teaching no DSL, a tool-incapable model
    would have no tools to call and no syntax to write, and every patching
    turn would produce nothing. ``config.json`` ships
    ``ollama:gemma4:12b`` with ``supports_tools: false``.
    """

    def _cfg(self, routing, models):
        return {"model_routing": routing, "models": models}

    def test_tool_less_primary_falls_back_to_the_dsl(self):
        block = _build_patch_interface_block(True, self._cfg(
            {"patching_primary": "ollama:gemma4:12b"},
            {"ollama:gemma4:12b": {"supports_tools": False}},
        ))
        assert "<<<REPLACE_BLOCK>>>" in block

    def test_tool_less_fallback_also_falls_back(self):
        """The fallback is dispatched on the primary's empty-content
        exhaustion, so it needs a dialect it can actually use too."""
        block = _build_patch_interface_block(True, self._cfg(
            {"patching_primary": "good", "patching_fallback": "bad"},
            {"good": {"supports_tools": True}, "bad": {"supports_tools": False}},
        ))
        assert "<<<REPLACE_BLOCK>>>" in block

    def test_repair_role_is_checked_too_not_just_patching(self):
        block = _build_patch_interface_block(True, self._cfg(
            {"repair_primary": "bad"}, {"bad": {"supports_tools": False}},
        ))
        assert "<<<REPLACE_BLOCK>>>" in block

    def test_missing_key_is_read_as_FALSE_like_the_gateway(self):
        """``ModelSpec`` parses a missing ``supports_tools`` as False
        (gateway.py:320). Reading it as True here would let an entry that
        merely omits the key reopen the mismatch."""
        block = _build_patch_interface_block(True, self._cfg(
            {"patching_primary": "m"}, {"m": {}},
        ))
        assert "<<<REPLACE_BLOCK>>>" in block

    def test_capable_models_still_get_native(self):
        block = _build_patch_interface_block(True, self._cfg(
            {"patching_primary": "ok", "repair_primary": "ok"},
            {"ok": {"supports_tools": True}},
        ))
        assert "NATIVE TOOL-USE" in block
        assert "<<<" not in block

    def test_the_shipped_config_resolves_to_native(self):
        """Guard against a routing edit silently downgrading every prompt."""
        import json
        import os

        import harness.graph as g

        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(g.__file__))),
            "config", "config.json",
        )
        with open(cfg_path, encoding="utf-8") as fh:
            shipped = json.load(fh)
        assert "NATIVE TOOL-USE" in _build_patch_interface_block(True, shipped)

    def test_a_model_in_neither_config_nor_catalogue_gets_the_dsl(self):
        """Capability now resolves config-first, then the bundled
        ``model_prices.json`` registry. A key in NEITHER is tool-less as
        far as dispatch is concerned (``ModelSpec`` would be absent), so
        the safe dialect is the DSL."""
        block = _build_patch_interface_block(True, self._cfg(
            {"patching_primary": "mystery:not-a-real-model"}, {},
        ))
        assert "<<<REPLACE_BLOCK>>>" in block

    def test_a_catalogue_model_absent_from_config_still_gets_native(self):
        """The bundled catalogue is consulted, so an operator who routes to
        a shipped model without re-declaring it in ``models`` keeps the
        native interface."""
        from harness.gateway import _MODEL_REGISTRY

        capable = next(
            (k for k, s in _MODEL_REGISTRY.items() if s.supports_tools), None,
        )
        if capable is None:
            import pytest as _pytest
            _pytest.skip("bundled catalogue declares no tool-capable model")
        block = _build_patch_interface_block(True, self._cfg(
            {"patching_primary": capable, "repair_primary": capable}, {},
        ))
        assert "NATIVE TOOL-USE" in block

    def test_force_local_only_is_honoured(self):
        """``select_model`` short-circuits every role to the local model, so
        the dialect must follow that model's capability — a path the
        graph-local duplicate missed entirely."""
        block = _build_patch_interface_block(True, {
            "model_routing": {
                "force_local_only": True,
                "ollama_local_model": "ollama:gemma4:12b",
                "patching_primary": "capable",
            },
            "models": {
                "capable": {"supports_tools": True},
                "ollama:gemma4:12b": {"supports_tools": False},
            },
        })
        assert "<<<REPLACE_BLOCK>>>" in block

    def test_bare_and_absent_config_behave_as_before(self):
        for cfg in (None, {}, {"patcher": {"use_structured_tools": True}}):
            assert "NATIVE TOOL-USE" in _build_patch_interface_block(True, cfg)

    def test_malformed_config_does_not_raise(self):
        for cfg in ({"models": "oops", "model_routing": {}},
                    {"models": {}, "model_routing": "oops"},
                    {"models": {"m": "notadict"},
                     "model_routing": {"patching_primary": "m"}}):
            assert _build_patch_interface_block(True, cfg)

    def test_the_flag_off_still_wins_regardless_of_capability(self):
        block = _build_patch_interface_block(False, self._cfg(
            {"patching_primary": "ok"}, {"ok": {"supports_tools": True}},
        ))
        assert "<<<REPLACE_BLOCK>>>" in block


class TestNativeInvariantsKeepTheSemantics:
    """The field FORMATS moved to the schemas; the RULES stay in the prompt
    because they are semantics, not syntax."""

    def test_exact_byte_matching_survives(self):
        assert "EXACT-BYTE" in _NATIVE_EDIT_INVARIANTS

    def test_line_number_prefix_rule_survives(self):
        assert "N| " in _NATIVE_EDIT_INVARIANTS

    def test_read_before_edit_survives(self):
        low = _NATIVE_EDIT_INVARIANTS.lower()
        assert "read before you edit" in low
        assert "read_file" in _NATIVE_EDIT_INVARIANTS

    def test_count_rule_survives_without_dsl_syntax(self):
        assert "count" in _NATIVE_EDIT_INVARIANTS
        assert "count:" not in _NATIVE_EDIT_INVARIANTS.replace('count: "', "")

    def test_rewrite_is_framed_as_an_escape_hatch(self):
        assert "escape hatch" in _NATIVE_EDIT_INVARIANTS

    def test_it_is_far_smaller_than_the_dsl_reference(self):
        assert len(_NATIVE_EDIT_INVARIANTS) < len(_DSL_PATCH_SYNTAX) / 3


class TestToolSchemasDoNotTeachTheDsl:
    """The schemas are model-facing: a description saying "Equivalent to
    <<<REPLACE_BLOCK>>> in the text DSL" teaches the DSL at the call site,
    which defeats removing it from the prompt."""

    def _schemas(self):
        out = []
        for name in dir(ts):
            if not name.endswith("_SCHEMA"):
                continue
            val = getattr(ts, name)
            if isinstance(val, dict) and val.get("name"):
                out.append(val)
        assert out, "no schemas discovered"
        return out

    def test_no_schema_mentions_dsl_markers(self):
        for s in self._schemas():
            assert "<<<" not in repr(s), f"{s['name']} still shows a marker"

    def test_no_schema_cross_references_the_text_dsl(self):
        for s in self._schemas():
            assert "text DSL" not in repr(s), f"{s['name']} cites the DSL"

    def test_edit_file_carries_the_invariants_itself(self):
        desc = ts.EDIT_FILE_SCHEMA["description"]
        # The rules that used to be reachable only via the prompt's DSL block.
        assert "verbatim substring" in desc
        assert "N| " in desc
        assert "read_file" in desc

    def test_edit_file_still_documents_count(self):
        props = ts.EDIT_FILE_SCHEMA["input_schema"]["properties"]
        assert set(props["count"]["enum"]) == {"unique", "all", "first"}

    def test_rewrite_file_still_warns_about_partial_content(self):
        desc = ts.REWRITE_FILE_SCHEMA["description"]
        assert "COMPLETE" in desc
        assert "escape hatch" in desc


class TestInternalOnlyOpsStillWork:
    """Narrowing the model-facing menu must not break the harness's own
    deterministic fixers, which construct these operations directly."""

    def test_they_are_not_advertised(self):
        offered = {t["name"] for t in ts.PATCH_TOOLS}
        for name in ("delete_block", "insert_at_block", "insert_at_line",
                     "replace_line_range"):
            assert name not in offered

    def test_the_offered_set_is_exactly_the_industry_shape(self):
        """Read, three ways to change a file, two whole-file lifecycle ops —
        and nothing else."""
        offered = {t["name"] for t in ts.PATCH_TOOLS}
        assert offered == {
            "read_file", "edit_file", "create_file", "rewrite_file",
            "delete_file", "move_file",
        }

    def test_but_they_are_still_declared_and_dispatchable(self):
        internal = {t["name"] for t in ts.INTERNAL_ONLY_PATCH_TOOLS}
        assert internal == {
            "delete_block", "insert_at_block", "insert_at_line",
            "replace_line_range",
        }
        # A stored call from an older checkpoint must still apply.
        block = ts.tool_call_to_patch_block({
            "name": "delete_block",
            "input": {"file_path": "a.py", "search": "x = 1"},
        })
        assert block is not None and block.file == "a.py"

    def test_the_operation_types_autofix_depends_on_still_exist(self):
        from harness.patcher import OperationType

        for name in ("DELETE_BLOCK", "INSERT_AT_BLOCK", "INSERT_AT_LINE",
                     "REPLACE_LINE_RANGE"):
            assert hasattr(OperationType, name), (
                f"{name} removed, but autofix.py / security_fix_rules.py "
                f"build it programmatically"
            )

    def test_advertised_and_internal_sets_are_disjoint(self):
        a = {t["name"] for t in ts.PATCH_TOOLS}
        b = {t["name"] for t in ts.INTERNAL_ONLY_PATCH_TOOLS}
        assert not (a & b)


class TestTheSharedBackendIsStillReachable:
    """Tool calls are rendered to DSL and parsed by the same backend, so the
    parser is the convergence point for both dialects — not dead code."""

    def test_tool_calls_still_render_to_dsl(self):
        dsl = ts.tool_call_to_dsl({
            "name": "edit_file",
            "input": {
                "file_path": "a/b.py",
                "old_string": "x = 1",
                "new_string": "x = 2",
            },
        })
        assert dsl and "<<<REPLACE_BLOCK>>>" in dsl
        assert "file: a/b.py" in dsl

    def test_rendered_dsl_parses_back_into_a_patch_block(self):
        from harness.patcher import parse_patch_blocks

        dsl = ts.tool_call_to_dsl({
            "name": "edit_file",
            "input": {
                "file_path": "a/b.py",
                "old_string": "x = 1",
                "new_string": "x = 2",
            },
        })
        blocks = parse_patch_blocks(dsl)
        assert len(blocks) == 1
        assert blocks[0].file == "a/b.py"


class TestWholeFileLifecycleTools:
    """Until ``delete_file`` / ``move_file`` existed the patcher could
    create, edit and overwrite a file but never remove or rename one, so a
    refactor that split or renamed a module left the original behind with
    no operation capable of cleaning it up. Every industry editor covers
    this, usually through a shell the harness does not expose.
    """

    def test_both_are_offered_to_the_model(self):
        offered = {t["name"] for t in ts.PATCH_TOOLS}
        assert {"delete_file", "move_file"} <= offered

    def test_the_prompt_names_them_and_says_to_use_them(self):
        block = _build_patch_interface_block(True)
        assert "`delete_file`" in block
        assert "`move_file`" in block
        # The capability is useless if the model never thinks to reach for
        # it; the prompt has to say obsolete files must go.
        assert "obsolete" in block.lower()

    def test_tool_calls_map_to_patch_blocks(self):
        from harness.patcher import OperationType

        d = ts.tool_call_to_patch_block({
            "name": "delete_file", "input": {"file_path": "a/old.py"},
        })
        assert d.operation is OperationType.DELETE_FILE and d.file == "a/old.py"
        m = ts.tool_call_to_patch_block({
            "name": "move_file",
            "input": {"file_path": "a/old.py", "dest_path": "b/new.py"},
        })
        assert m.operation is OperationType.MOVE_FILE
        assert m.file == "a/old.py" and m.dest == "b/new.py"

    def test_they_round_trip_through_the_dsl(self):
        """``render_patch_block`` is what lets a native tool call reach the
        shared apply backend in repair. An op the DSL cannot carry is an op
        the repair fold silently DROPS, so the round trip is load-bearing —
        not cosmetic."""
        from harness.patcher import (
            OperationType, PatchBlock, parse_patch_blocks, render_patch_block,
        )

        for block in (
            PatchBlock(operation=OperationType.DELETE_FILE, file="a/old.py"),
            PatchBlock(operation=OperationType.MOVE_FILE, file="a/old.py",
                       dest="b/new.py"),
        ):
            dsl = render_patch_block(block)
            assert dsl, f"{block.operation.name} has no DSL form"
            back = parse_patch_blocks(dsl)
            assert len(back) == 1
            assert back[0].operation is block.operation
            assert back[0].file == block.file
            assert back[0].dest == block.dest

    def test_the_repair_fold_does_not_drop_them(self):
        for call in (
            {"name": "delete_file", "input": {"file_path": "a/old.py"}},
            {"name": "move_file",
             "input": {"file_path": "a/old.py", "dest_path": "b/new.py"}},
        ):
            assert ts.tool_call_to_dsl(call), f"{call['name']} folds to None"


class TestLifecycleSafety:
    def _patcher(self, tmp_path):
        from harness.patcher import HybridPatcher
        return HybridPatcher(workspace_root=str(tmp_path))

    def _apply(self, p, block):
        import asyncio
        return asyncio.run(p.apply_patch(block))

    def _block(self, op, **kw):
        from harness.patcher import OperationType, PatchBlock
        return PatchBlock(operation=getattr(OperationType, op), **kw)

    def test_move_refuses_to_clobber(self, tmp_path):
        """A rename that eats another file is damage a repair loop cannot
        see it has done."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        r = self._apply(self._patcher(tmp_path),
                        self._block("MOVE_FILE", file="a.py", dest="b.py"))
        assert not r.success
        assert "already exists" in (r.error or "")
        assert (tmp_path / "b.py").read_text() == "y = 2\n"

    def test_move_destination_cannot_escape_the_workspace(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        r = self._apply(self._patcher(tmp_path),
                        self._block("MOVE_FILE", file="a.py",
                                    dest="../escaped.py"))
        assert not r.success
        assert (tmp_path / "a.py").exists()

    def test_move_needs_a_destination(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        r = self._apply(self._patcher(tmp_path),
                        self._block("MOVE_FILE", file="a.py", dest=""))
        assert not r.success

    def test_move_of_a_missing_source_fails(self, tmp_path):
        r = self._apply(self._patcher(tmp_path),
                        self._block("MOVE_FILE", file="nope.py", dest="b.py"))
        assert not r.success

    def test_move_onto_itself_is_a_no_op(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        r = self._apply(self._patcher(tmp_path),
                        self._block("MOVE_FILE", file="a.py", dest="a.py"))
        assert r.success
        assert (tmp_path / "a.py").read_text() == "x = 1\n"

    def test_delete_is_idempotent(self, tmp_path):
        """Resume-safe, matching create_file's semantics: re-running a
        batch must not fail on a file the first pass already removed."""
        (tmp_path / "a.py").write_text("x = 1\n")
        p = self._patcher(tmp_path)
        assert self._apply(p, self._block("DELETE_FILE", file="a.py")).success
        assert self._apply(p, self._block("DELETE_FILE", file="a.py")).success

    def test_delete_refuses_a_directory(self, tmp_path):
        """A recursive delete is not something a generation loop should be
        able to reach for."""
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "keep.py").write_text("x = 1\n")
        r = self._apply(self._patcher(tmp_path),
                        self._block("DELETE_FILE", file="pkg"))
        assert not r.success
        assert (tmp_path / "pkg" / "keep.py").exists()

    def test_delete_cannot_escape_the_workspace(self, tmp_path):
        outside = tmp_path.parent / "victim.py"
        outside.write_text("precious\n")
        try:
            r = self._apply(self._patcher(tmp_path),
                            self._block("DELETE_FILE", file="../victim.py"))
            assert not r.success
            assert outside.exists(), "deleted a file outside the workspace"
        finally:
            if outside.exists():
                outside.unlink()


class TestLifecycleOpsRespectTheExistingGuards:
    """Adding a tool that WRITES to a second path reopens every guard that
    only ever inspected one. ``move_file`` has a destination the allowlist
    and the test-tamper guard had no reason to look at, and ``delete_file``
    produces a path that must not flow downstream as "modified"."""

    def _apply(self, blocks, ws, allowed=None, existing=None):
        import asyncio

        from harness.patcher import apply_patch_blocks
        return asyncio.run(apply_patch_blocks(
            blocks, str(ws), list(existing or []), allowed_paths=allowed,
        ))

    def _mv(self, src, dest):
        from harness.patcher import OperationType, PatchBlock
        return PatchBlock(
            operation=OperationType.MOVE_FILE, file=src, dest=dest,
        )

    def _rm(self, path):
        from harness.patcher import OperationType, PatchBlock
        return PatchBlock(operation=OperationType.DELETE_FILE, file=path)

    def test_move_destination_must_be_in_the_allowlist(self, tmp_path):
        """``_resolve_safe`` stops a workspace escape but knows nothing
        about the round's allowed scope, so a move could carry a permitted
        file to an off-limits path."""
        (tmp_path / "allowed.py").write_text("x = 1\n")
        results, _ = self._apply(
            [self._mv("allowed.py", "forbidden/x.py")], tmp_path,
            allowed=["allowed.py"],
        )
        assert not results[0].success
        assert "destination" in (results[0].error or "")
        assert (tmp_path / "allowed.py").exists(), "source was moved anyway"

    def test_move_within_the_allowlist_still_applies(self, tmp_path):
        (tmp_path / "ok").mkdir()
        (tmp_path / "ok" / "a.py").write_text("x = 1\n")
        results, modified = self._apply(
            [self._mv("ok/a.py", "ok/b.py")], tmp_path, allowed=["ok/"],
        )
        assert results[0].success
        assert modified == ["ok/b.py"]

    def test_a_deleted_path_never_enters_modified_files(self, tmp_path):
        """``modified_files`` is what lintgate, the diagnostics gate and
        traceability iterate over. A removed path there hands them a file
        to format, type-check and attribute that no longer exists."""
        (tmp_path / "doomed.py").write_text("y = 2\n")
        results, modified = self._apply(
            [self._rm("doomed.py")], tmp_path,
            allowed=["doomed.py"], existing=["doomed.py"],
        )
        assert results[0].success
        assert "doomed.py" not in modified

    def test_a_move_retires_the_source_it_emptied(self, tmp_path):
        (tmp_path / "ok").mkdir()
        (tmp_path / "ok" / "a.py").write_text("z = 3\n")
        results, modified = self._apply(
            [self._mv("ok/a.py", "ok/b.py")], tmp_path,
            allowed=["ok/"], existing=["ok/a.py"],
        )
        assert results[0].success
        assert "ok/a.py" not in modified, "stale source left in modified_files"
        assert "ok/b.py" in modified

    def test_repair_cannot_author_a_test_by_moving_one_in(self):
        """Repair may not write test files. The guard only ever inspected
        the SOURCE, so moving a non-test file to a test path would have
        authored one through a path it never looked at."""
        from harness.graph import _reject_test_patch_blocks

        kept, rejected = _reject_test_patch_blocks([
            self._mv("src/helper.py", "server/tests/test_sneaky.py"),
        ])
        assert kept == []
        assert len(rejected) == 1
        assert "test file" in (rejected[0].error or "")

    def test_repair_cannot_delete_a_protected_test(self):
        """Deleting a failing test is the purest form of the reward hack
        the guard exists to stop."""
        from harness.graph import _reject_test_patch_blocks

        kept, rejected = _reject_test_patch_blocks([
            self._rm("server/tests/test_failing.py"),
        ])
        assert kept == []
        assert len(rejected) == 1

    def test_ordinary_lifecycle_changes_are_still_allowed(self):
        """The guards must not become a blanket ban — deleting an obsolete
        module and renaming a source file are the point of these tools."""
        from harness.graph import _reject_test_patch_blocks

        kept, rejected = _reject_test_patch_blocks([
            self._mv("src/a.py", "src/b.py"),
            self._rm("src/obsolete.py"),
        ])
        assert len(kept) == 2
        assert rejected == []


class TestImportPruningRule:
    """Observed in all three lumina runs: the model prunes an import it
    believes is unused while making a functional edit, the post-patch
    validator rejects the WHOLE patch over the resulting NameError, and
    the change the round was for is rolled back with it.

    Run 01a167cd, one single edit_file call — not two, so nothing about
    validation SCOPE is implicated:

        old_string: from fastapi import APIRouter, Depends, Query, Request
        new_string: from fastapi import APIRouter, Depends, Request

    ``Query`` was still called in the body. Earlier runs lost `Generator`,
    `ZoneInfoNotFoundError`, `re`, `get_settings` and `date` the same way.
    """

    def test_the_rule_is_in_the_native_invariants(self):
        from harness.graph import _NATIVE_EDIT_INVARIANTS

        assert "Never tidy imports" in _NATIVE_EDIT_INVARIANTS

    def test_it_reaches_the_assembled_prompt(self):
        block = _build_patch_interface_block(True)
        assert "Never tidy imports" in block

    def test_it_states_the_consequence_not_just_the_prohibition(self):
        """A bare "don't" teaches nothing. The model needs to know the
        whole patch is rejected, because that is what makes the rule worth
        obeying over its own tidiness instinct."""
        from harness.graph import _NATIVE_EDIT_INVARIANTS

        low = _NATIVE_EDIT_INVARIANTS.lower()
        assert "whole" in low and "rolled back" in low

    def test_it_offers_the_sanctioned_alternative(self):
        """Rules with no permitted path get worked around."""
        from harness.graph import _NATIVE_EDIT_INVARIANTS

        assert "its own separate edit" in _NATIVE_EDIT_INVARIANTS
        assert "grep" in _NATIVE_EDIT_INVARIANTS

    def test_the_invariants_stay_short(self):
        """The whole point of splitting the dialects was a compact set of
        semantics; an invariants block that grows without bound recreates
        the 10,696-char reference it replaced."""
        from harness.graph import _DSL_PATCH_SYNTAX, _NATIVE_EDIT_INVARIANTS

        assert len(_NATIVE_EDIT_INVARIANTS) < len(_DSL_PATCH_SYNTAX) / 3
