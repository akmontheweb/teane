"""Tests for Phase 7 — skills, patcher, speculative refinements."""

import tempfile
from pathlib import Path

import pytest

from harness.patcher import PatchResult, OperationType
from harness.skills import (
    SubAgentSkill, SkillSchema, SkillType, SkillRegistry, get_skill
)


class TestPatchResult:
    """Test PatchResult dataclass."""

    def test_success_result(self):
        """Successful patch result."""
        result = PatchResult(
            success=True,
            file="test.py",
            operation=OperationType.CREATE_FILE,
            lines_changed=10,
        )
        assert result.success is True
        assert result.operation == OperationType.CREATE_FILE

    def test_failure_result(self):
        """Failed patch result with error."""
        result = PatchResult(
            success=False,
            file="app.py",
            operation=OperationType.REPLACE_BLOCK,
            error="Search block not found",
        )
        assert result.success is False
        assert "not found" in result.error

    def test_idempotent_no_op(self):
        """No-op result from idempotent operation."""
        result = PatchResult(
            success=True,
            file="already.py",
            operation=OperationType.CREATE_FILE,
            lines_changed=0,
            message="already at target state",
        )
        assert result.lines_changed == 0
        assert "target state" in result.message


class TestOperationType:
    """Test OperationType enum."""

    def test_all_operation_types_exist(self):
        """Should have all operation types."""
        assert hasattr(OperationType, "CREATE_FILE")
        assert hasattr(OperationType, "REPLACE_BLOCK")
        assert hasattr(OperationType, "DELETE_BLOCK")
        assert hasattr(OperationType, "INSERT_AT_BLOCK")

    def test_operation_type_values(self):
        """Operation types should have string values."""
        assert OperationType.CREATE_FILE.value == "create_file"
        assert OperationType.REPLACE_BLOCK.value == "replace_block"


class TestSubAgentSkill:
    """Test SubAgentSkill construction."""

    def test_skill_construction(self):
        """Should construct SubAgentSkill with schema and prompt."""
        schema = SkillSchema(
            name="analyze",
            description="Analyze code",
            skill_type=SkillType.SUBAGENT,
        )
        skill = SubAgentSkill(
            schema=schema,
            system_prompt="You are a code analyzer",
            model_override="claude-opus",
            max_iterations=5,
        )
        assert skill.name == "analyze"
        assert skill.system_prompt == "You are a code analyzer"
        assert skill.model_override == "claude-opus"
        assert skill.max_iterations == 5

    def test_skill_with_allowed_paths(self):
        """Should construct with allowed paths restriction."""
        schema = SkillSchema(
            name="refactor",
            description="Refactor code",
            skill_type=SkillType.SUBAGENT,
        )
        allowed = ["src/", "tests/"]
        skill = SubAgentSkill(
            schema=schema,
            system_prompt="Refactor the code",
            allowed_paths=allowed,
        )
        assert skill.allowed_paths == allowed


class TestPatcherEdgeCases:
    """Test patcher edge cases."""

    def test_create_file_operation(self):
        """CREATE_FILE operation type."""
        op = OperationType.CREATE_FILE
        assert op == OperationType.CREATE_FILE

    def test_replace_block_operation(self):
        """REPLACE_BLOCK operation type."""
        op = OperationType.REPLACE_BLOCK
        assert op == OperationType.REPLACE_BLOCK

    def test_patch_result_with_message(self):
        """PatchResult with message."""
        result = PatchResult(
            success=True,
            file="app.py",
            operation=OperationType.REPLACE_BLOCK,
            lines_changed=5,
            message="function signature updated",
        )
        assert result.message == "function signature updated"


class TestSpeculativeRefinements:
    """Test speculative build refinements."""

    def test_variant_comparison(self):
        """Compare variant results."""
        from harness.speculative import VariantResult

        v1 = VariantResult(
            index=0,
            variant_id="v1",
            worktree_path="/tmp/v1",
            exit_code=1,
        )
        v2 = VariantResult(
            index=1,
            variant_id="v2",
            worktree_path="/tmp/v2",
            exit_code=0,
        )
        # v2 passed, v1 failed
        assert v2.passed is True
        assert v1.passed is False

    def test_cache_env_isolation(self):
        """Cache environment should isolate variants."""
        from harness.speculative import _build_variant_cache_env

        with tempfile.TemporaryDirectory() as tmpdir:
            env = _build_variant_cache_env(tmpdir)
            # Should have environment variables
            assert isinstance(env, dict)
            assert len(env) > 0


class TestSkillsDiscovery:
    """Test skill registry and discovery."""

    def test_get_skill_returns_none_for_missing(self):
        """Should return None for unregistered skill."""
        result = get_skill("nonexistent_skill_xyz_12345")
        assert result is None

    def test_skill_schema_construction(self):
        """Should construct SkillSchema with metadata."""
        schema = SkillSchema(
            name="formatter",
            description="Format code",
            skill_type=SkillType.TOOL,
            tags=["code-formatting", "automation"],
            module="harness.skills",
        )
        assert schema.name == "formatter"
        assert "format" in schema.description.lower()
        assert SkillType.TOOL in [SkillType.TOOL]

    def test_skill_types(self):
        """All SkillType enum values should be valid."""
        assert SkillType.TOOL.value == "tool"
        assert SkillType.PIPELINE.value == "pipeline"
        assert SkillType.SUBAGENT.value == "subagent"
        assert SkillType.DOCGEN.value == "docgen"


class TestPatcherErrorMessages:
    """Test patcher error reporting."""

    def test_error_with_file_context(self):
        """Error result with file context."""
        result = PatchResult(
            success=False,
            file="src/main.py",
            operation=OperationType.DELETE_BLOCK,
            error="Block not found at line 42",
        )
        assert "main.py" in result.file
        assert "line 42" in result.error

    def test_success_message(self):
        """Success result with informative message."""
        result = PatchResult(
            success=True,
            file="utils.py",
            operation=OperationType.INSERT_AT_BLOCK,
            lines_changed=3,
            message="Added helper function after imports",
        )
        assert result.success is True
        assert "helper" in result.message.lower()


class TestSkillParameter:
    """Test SkillParameter dataclass."""

    def test_construct_minimal(self):
        """Should construct with minimal fields."""
        from harness.skills import SkillParameter

        param = SkillParameter(name="input", description="Input text")
        assert param.name == "input"
        assert param.type == "string"
        assert param.required is True

    def test_construct_with_options(self):
        """Should construct with enum options."""
        from harness.skills import SkillParameter

        param = SkillParameter(
            name="mode",
            type="string",
            description="Execution mode",
            enum=["fast", "accurate"],
        )
        assert param.enum == ["fast", "accurate"]

    def test_construct_optional(self):
        """Should support optional parameters."""
        from harness.skills import SkillParameter

        param = SkillParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds",
            required=False,
        )
        assert param.required is False
        assert param.type == "integer"


class TestToolSkill:
    """Test ToolSkill construction."""

    def test_tool_skill_schema_conversion(self):
        """Should convert to OpenAI tool schema."""
        from harness.skills import ToolSkill, SkillParameter

        schema = SkillSchema(
            name="add_numbers",
            description="Add two numbers",
            skill_type=SkillType.TOOL,
            parameters=[
                SkillParameter(name="a", type="number", description="First number"),
                SkillParameter(name="b", type="number", description="Second number"),
            ],
        )

        async def dummy_fn(a: float, b: float) -> float:
            return a + b

        skill = ToolSkill(schema=schema, fn=dummy_fn)
        tool_schema = skill.to_tool_schema()

        assert tool_schema["type"] == "function"
        assert tool_schema["function"]["name"] == "add_numbers"
        assert "a" in tool_schema["function"]["parameters"]["properties"]


class TestPipelineSkill:
    """Test PipelineSkill construction."""

    def test_pipeline_skill_node_fn(self):
        """Should wrap and expose node function."""
        from harness.skills import PipelineSkill

        async def dummy_node(state: dict) -> dict:
            return state

        schema = SkillSchema(
            name="my_node",
            description="A pipeline node",
            skill_type=SkillType.PIPELINE,
        )

        skill = PipelineSkill(schema=schema, node_fn=dummy_node)
        node_fn = skill.to_node_fn()

        assert node_fn is dummy_node
