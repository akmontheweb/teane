"""Tests for the technology-specific style-guide loader (harness/style_guides.py).

The loader mirrors the two-tier skills system: harness-shipped defaults plus
per-project overrides, both filtered by `applies_to:` frontmatter against
the workspace's detected stack tags.
"""

from __future__ import annotations

import os
from textwrap import dedent

import pytest

from harness.style_guides import (
    HARNESS_STYLE_GUIDES_DIR,
    _load_style_guides_markdown,
    load_style_guides,
)


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture
def workspace(tmp_path) -> str:
    return str(tmp_path)


class TestLoadStyleGuidesMarkdown:
    """Direct tests against the directory-scanning helper."""

    def test_missing_directory_returns_empty(self, workspace):
        body, loaded = _load_style_guides_markdown(
            os.path.join(workspace, "does-not-exist"),
            workspace_tags={"python"},
        )
        assert body == ""
        assert loaded == set()

    def test_filters_by_workspace_tags_intersection(self, workspace):
        sg_dir = os.path.join(workspace, "style_guides")
        _write(os.path.join(sg_dir, "python.md"), dedent("""\
            ---
            applies_to: [python]
            ---
            ## Python Style
            Indent with 4 spaces.
        """))
        _write(os.path.join(sg_dir, "react.md"), dedent("""\
            ---
            applies_to: [react]
            ---
            ## React Style
            PascalCase components.
        """))

        body, loaded = _load_style_guides_markdown(
            sg_dir, workspace_tags={"python"},
        )
        assert "Python Style" in body
        assert "React Style" not in body
        assert loaded == {"python.md"}

    def test_no_frontmatter_loads_unconditionally(self, workspace):
        # A style guide file without an `applies_to:` block is treated as
        # universal — matches the skills loader semantics. This is the
        # opt-out for guides that have no language target (e.g. a shared
        # "always document your changes" rule).
        sg_dir = os.path.join(workspace, "style_guides")
        _write(os.path.join(sg_dir, "universal.md"), "Always be kind to your reader.")
        body, loaded = _load_style_guides_markdown(
            sg_dir, workspace_tags={"python"},
        )
        assert "kind to your reader" in body
        assert "universal.md" in loaded

    def test_frontmatter_multi_tag_or_semantics(self, workspace):
        # `applies_to: [react, vue]` should load when EITHER tag is
        # present. The current css.md in harness defaults uses this
        # pattern.
        sg_dir = os.path.join(workspace, "style_guides")
        _write(os.path.join(sg_dir, "css.md"), dedent("""\
            ---
            applies_to: [react, vue, css]
            ---
            ## CSS
            Mobile-first.
        """))
        body_vue, _ = _load_style_guides_markdown(sg_dir, workspace_tags={"vue"})
        assert "Mobile-first" in body_vue
        body_react, _ = _load_style_guides_markdown(sg_dir, workspace_tags={"react"})
        assert "Mobile-first" in body_react
        body_none, _ = _load_style_guides_markdown(sg_dir, workspace_tags={"java"})
        assert body_none == ""

    def test_respects_max_file_chars_cap(self, workspace):
        # Big files must not blow the prompt budget.
        sg_dir = os.path.join(workspace, "style_guides")
        big_body = "X" * 50_000
        _write(os.path.join(sg_dir, "big.md"), big_body)
        body, _ = _load_style_guides_markdown(
            sg_dir, workspace_tags=None, max_file_chars=1024,
        )
        # 1024 is the read cap; some rstrip is fine.
        assert len(body) <= 1024

    def test_skip_filenames_excludes_already_loaded(self, workspace):
        # Project tier loads first; harness tier must skip filenames the
        # project already covered, so the project file wins outright.
        sg_dir = os.path.join(workspace, "style_guides")
        _write(os.path.join(sg_dir, "python.md"), dedent("""\
            ---
            applies_to: [python]
            ---
            Project python rules.
        """))
        body, loaded = _load_style_guides_markdown(
            sg_dir, workspace_tags={"python"},
            skip_filenames={"python.md"},
        )
        assert body == ""
        assert loaded == set()


class TestLoadStyleGuides:
    """Tests against the public two-tier entry point."""

    def test_empty_workspace_no_tags_returns_empty(self, workspace):
        # No project overrides + tags = empty set means no shipped guide
        # matches, so the loader returns "" and the system prompt skips
        # the section entirely. This keeps non-covered stacks free of
        # prompt growth.
        assert load_style_guides(workspace, workspace_tags=set()) == ""

    def test_loads_shipped_python_guide_for_python_workspace(self, workspace):
        # Sanity: the shipped python.md must surface for a python workspace.
        body = load_style_guides(workspace, workspace_tags={"python"})
        assert "Python Style Guide" in body
        assert "PEP 8" in body

    def test_project_override_replaces_harness_default(self, workspace):
        # When a project ships its own python.md, it must REPLACE the
        # harness default rather than concatenate with it. Same precedence
        # rule the skills system uses.
        sg_dir = os.path.join(workspace, "style_guides")
        _write(os.path.join(sg_dir, "python.md"), dedent("""\
            ---
            applies_to: [python]
            ---
            ## House Python Style
            Tabs only. Don't @ us.
        """))
        body = load_style_guides(workspace, workspace_tags={"python"})
        assert "House Python Style" in body
        # The shipped one must be gone — its PEP 8 sentence must NOT appear.
        assert "PEP 8" not in body

    def test_project_overrides_render_before_harness_defaults(self, workspace):
        # Project tier first so house style appears ahead of shipped
        # defaults when both contribute (different filenames).
        sg_dir = os.path.join(workspace, "style_guides")
        _write(os.path.join(sg_dir, "house-rules.md"), dedent("""\
            ---
            applies_to: [python]
            ---
            HOUSE_RULES_MARKER
        """))
        body = load_style_guides(workspace, workspace_tags={"python"})
        # HOUSE_RULES_MARKER should appear BEFORE the shipped PEP 8 text.
        house_idx = body.find("HOUSE_RULES_MARKER")
        pep_idx = body.find("PEP 8")
        assert house_idx >= 0
        assert pep_idx >= 0
        assert house_idx < pep_idx

    def test_returns_empty_when_no_matching_tags(self, workspace):
        # An entirely-unrecognised tag set should produce empty output —
        # nothing in the shipped guides matches and the loader returns "".
        body = load_style_guides(workspace, workspace_tags={"erlang"})
        assert body == ""

    def test_max_total_chars_truncates(self, workspace):
        # Even when many guides match (e.g. a polyglot frontend project
        # picks up react + vue + angular + html + css + javascript +
        # typescript), the total must stay bounded.
        body = load_style_guides(
            workspace,
            workspace_tags={"react", "vue", "angular", "html", "css", "typescript", "node"},
            max_total_chars=2048,
        )
        assert len(body) <= 2048


class TestShippedStyleGuides:
    """Validate the harness-shipped guides themselves — names, frontmatter,
    and that the user-requested set is present."""

    # Focused single-source guides: each distills one authoritative source
    # into ~3 KB of bullets. Cap at 4 KB to keep budgets predictable.
    FOCUSED_FILES = {
        "python.md", "java.md", "nodejs.md", "javascript.md", "typescript.md",
        "react.md", "vue.md", "angular.md", "html.md", "css.md", "sql.md",
        "flutter.md",
    }
    # Extended platform guides: distill a comprehensive platform standard
    # (Apple HIG, Material 3) — these are larger than focused guides
    # because the source itself is platform-wide, not a single style
    # treatise. Cap at 8 KB.
    EXTENDED_FILES = {
        "mobile-ios.md", "mobile-android.md",
    }
    # Composite multi-source design-system specs: one file synthesizes
    # several authoritative sources into an in-depth spec (palette,
    # typography, components, states, framework config). Larger cap is
    # justified because the file replaces what would otherwise be 5–10
    # smaller files, and gets prefix-cached across calls.
    COMPOSITE_FILES = {
        "web-design-system.md",
    }
    EXPECTED_FILES = FOCUSED_FILES | EXTENDED_FILES | COMPOSITE_FILES

    def test_all_expected_guides_shipped(self):
        present = {
            f for f in os.listdir(HARNESS_STYLE_GUIDES_DIR)
            if f.endswith(".md")
        }
        missing = self.EXPECTED_FILES - present
        assert not missing, f"Missing shipped style guides: {sorted(missing)}"

    def test_every_shipped_guide_has_source_citation(self):
        # Each shipped guide must cite the authoritative source it draws
        # from — both a licensing courtesy and a load-bearing UX cue for
        # the LLM to weight rules.
        for fname in self.EXPECTED_FILES:
            path = os.path.join(HARNESS_STYLE_GUIDES_DIR, fname)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "### Source" in content, (
                f"{fname} must include a '### Source' section citing the authoritative guide"
            )

    def test_every_shipped_guide_has_applies_to_frontmatter(self):
        # The frontmatter is the filtering mechanism — without it, the
        # guide would load unconditionally and bloat every prompt.
        for fname in self.EXPECTED_FILES:
            path = os.path.join(HARNESS_STYLE_GUIDES_DIR, fname)
            with open(path, encoding="utf-8") as f:
                head = f.read(200)
            assert "applies_to:" in head, (
                f"{fname} must declare an applies_to: frontmatter list"
            )

    def test_focused_guides_under_4kb(self):
        # Focused single-source guides must stay tight — anything over
        # 4 KB has probably absorbed material that belongs in its own
        # separate guide.
        for fname in self.FOCUSED_FILES:
            path = os.path.join(HARNESS_STYLE_GUIDES_DIR, fname)
            size = os.path.getsize(path)
            assert size <= 4096, (
                f"{fname} is {size} bytes — focused single-source guides must stay under 4 KB"
            )

    def test_extended_guides_under_8kb(self):
        # Extended platform guides synthesize a comprehensive platform
        # standard (HIG, M3). 8 KB cap leaves room for the platform's
        # surface area without bloating the prompt.
        for fname in self.EXTENDED_FILES:
            path = os.path.join(HARNESS_STYLE_GUIDES_DIR, fname)
            size = os.path.getsize(path)
            assert size <= 8 * 1024, (
                f"{fname} is {size} bytes — extended platform guides must stay under 8 KB"
            )

    def test_composite_guides_under_24kb(self):
        # Composite specs run larger (palette + typography + component
        # CSS + state matrix + framework config). 24 KB is the per-file
        # read cap; anything over that gets silently truncated by the
        # loader.
        for fname in self.COMPOSITE_FILES:
            path = os.path.join(HARNESS_STYLE_GUIDES_DIR, fname)
            size = os.path.getsize(path)
            assert size <= 24 * 1024, (
                f"{fname} is {size} bytes — composite specs must stay under "
                f"the 24 KB per-file read cap or the loader will truncate them"
            )


class TestStackDetectorNewTags:
    """Regression tests for the new typescript/tailwind/html/css tags."""

    def test_typescript_detected_via_tsconfig(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "tsconfig.json"), "w") as f:
            f.write("{}")
        tags = _detect_workspace_stack(workspace)
        assert "typescript" in tags

    def test_typescript_detected_via_package_json_dep(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"typescript": "^5"}}')
        tags = _detect_workspace_stack(workspace)
        assert "typescript" in tags

    def test_tailwind_detected_via_config_file(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "tailwind.config.js"), "w") as f:
            f.write("module.exports = {};")
        tags = _detect_workspace_stack(workspace)
        assert "tailwind" in tags

    def test_tailwind_detected_via_dep(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"tailwindcss": "^3"}}')
        tags = _detect_workspace_stack(workspace)
        assert "tailwind" in tags

    def test_html_css_implied_by_frontend_framework(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"react": "^18"}}')
        tags = _detect_workspace_stack(workspace)
        assert "react" in tags
        assert "html" in tags
        assert "css" in tags

    def test_html_detected_via_root_html_file(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "index.html"), "w") as f:
            f.write("<!doctype html><title>x</title>")
        tags = _detect_workspace_stack(workspace)
        assert "html" in tags

    def test_plain_python_workspace_skips_new_frontend_tags(self, workspace):
        # A pure Python workspace must not gain html/css/typescript tags
        # accidentally — that would inject unrelated style content.
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "requirements.txt"), "w") as f:
            f.write("flask==2.0\n")
        tags = _detect_workspace_stack(workspace)
        assert "python" in tags
        assert "html" not in tags
        assert "css" not in tags
        assert "typescript" not in tags
        assert "tailwind" not in tags
        assert "ios" not in tags
        assert "android" not in tags


class TestMobilePlatformDetection:
    """Regression tests for the ios / android target-platform tags."""

    def test_flutter_with_both_platform_dirs_tags_both(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "pubspec.yaml"), "w") as f:
            f.write("name: app\n")
        os.makedirs(os.path.join(workspace, "lib"))
        os.makedirs(os.path.join(workspace, "ios"))
        os.makedirs(os.path.join(workspace, "android"))
        tags = _detect_workspace_stack(workspace)
        assert "flutter" in tags
        assert "ios" in tags
        assert "android" in tags

    def test_flutter_with_only_ios_dir_skips_android_tag(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "pubspec.yaml"), "w") as f:
            f.write("name: app\n")
        os.makedirs(os.path.join(workspace, "lib"))
        os.makedirs(os.path.join(workspace, "ios"))
        tags = _detect_workspace_stack(workspace)
        assert "ios" in tags
        assert "android" not in tags

    def test_native_ios_via_podfile_and_xcodeproj(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "Podfile"), "w") as f:
            f.write("platform :ios, '15.0'\n")
        os.makedirs(os.path.join(workspace, "MyApp.xcodeproj"))
        tags = _detect_workspace_stack(workspace)
        assert "ios" in tags
        assert "android" not in tags

    def test_native_android_via_gradle_plugin(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "build.gradle"), "w") as f:
            f.write("plugins {\n  id 'com.android.application'\n}\n")
        tags = _detect_workspace_stack(workspace)
        assert "android" in tags
        assert "ios" not in tags

    def test_native_android_via_app_build_gradle(self, workspace):
        from harness.impact import _detect_workspace_stack
        os.makedirs(os.path.join(workspace, "app"))
        with open(os.path.join(workspace, "app/build.gradle"), "w") as f:
            f.write("apply plugin: 'com.android.application'\n")
        tags = _detect_workspace_stack(workspace)
        assert "android" in tags

    def test_react_native_tags_both_platforms(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"react-native": "^0.74"}}')
        tags = _detect_workspace_stack(workspace)
        assert "ios" in tags
        assert "android" in tags

    def test_expo_tags_both_platforms(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"expo": "~50.0.0"}}')
        tags = _detect_workspace_stack(workspace)
        assert "ios" in tags
        assert "android" in tags

    def test_pure_web_react_does_not_tag_mobile(self, workspace):
        from harness.impact import _detect_workspace_stack
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"react": "^18"}}')
        tags = _detect_workspace_stack(workspace)
        assert "react" in tags
        assert "ios" not in tags
        assert "android" not in tags


class TestArchitectureSpecAugmentation:
    """Regression tests for the SPEC_ARCHITECTURE.md tag-augmentation path.

    Greenfield ``--new-build`` runs reset the workspace to empty before
    the planner fires, so filesystem detection alone returns no stack
    tags — and the web-app file-manifest contract (graph.py:1049) keys
    off ``"html" in tags``, so it never injects and the planner emits a
    backend-only blueprint. Mining the architecture spec for stack hints
    bridges that gap. Augmentation is purely additive.
    """

    def _write_spec(self, workspace, body):
        docs = os.path.join(workspace, "docs")
        os.makedirs(docs, exist_ok=True)
        with open(os.path.join(docs, "SPEC_ARCHITECTURE.md"), "w") as f:
            f.write(body)

    def test_empty_workspace_no_spec_stays_empty(self, workspace):
        from harness.impact import _detect_workspace_stack
        assert _detect_workspace_stack(workspace) == set()

    def test_spec_workspace_layout_block_seeds_frontend_tags(self, workspace):
        from harness.impact import _detect_workspace_stack
        self._write_spec(workspace, '\n'.join([
            '# Arch',
            '```json',
            '{"workspace_layout": {"roots": ['
            '{"path": "server", "purpose": "api", "stack": "fastapi"},'
            '{"path": "client", "purpose": "web", "stack": "vue"}'
            '], "test_placement": "co-located", "root_files": []}}',
            '```',
        ]))
        tags = _detect_workspace_stack(workspace)
        # Frontend root → vue + transitive html/css/node
        assert "vue" in tags
        assert "html" in tags
        assert "css" in tags
        # Backend root → fastapi + python
        assert "fastapi" in tags
        assert "python" in tags

    def test_spec_freetext_react_fastapi_seeds_full_stack(self, workspace):
        from harness.impact import _detect_workspace_stack
        self._write_spec(
            workspace,
            "Frontend: React 18 SPA with TypeScript.\n"
            "Backend: FastAPI on PostgreSQL.\n",
        )
        tags = _detect_workspace_stack(workspace)
        assert {"react", "html", "css", "fastapi", "python", "postgres", "typescript"} <= tags

    def test_spec_flask_only_does_not_imply_frontend(self, workspace):
        from harness.impact import _detect_workspace_stack
        self._write_spec(workspace, "A Flask service returning JSON. No SPA.")
        tags = _detect_workspace_stack(workspace)
        assert "python" in tags
        # No frontend tags — Flask alone must not trigger the web-app
        # file-manifest contract.
        assert "html" not in tags
        assert "css" not in tags
        assert "react" not in tags
        assert "vue" not in tags

    def test_spec_augmentation_is_additive_does_not_override_manifest(
        self, workspace,
    ):
        from harness.impact import _detect_workspace_stack
        # Manifest says React; spec mentions Vue. Both should appear —
        # augmentation never strips the manifest-derived tag.
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"react": "^18"}}')
        self._write_spec(workspace, "We will use Vue 3 for the SPA.")
        tags = _detect_workspace_stack(workspace)
        assert "react" in tags
        assert "vue" in tags

    def test_spec_passing_mention_without_context_does_not_fire(
        self, workspace,
    ):
        from harness.impact import _detect_workspace_stack
        # "React" alone is too ambiguous; the regex requires a context
        # word (version number, "SPA", "frontend", etc.).
        self._write_spec(workspace, "The system has a reactive design.")
        tags = _detect_workspace_stack(workspace)
        assert "react" not in tags
        assert "html" not in tags


class TestSystemPromptInjection:
    """Integration: _build_system_prompt assembles the Coding Style Guides
    section only when relevant tags are present."""

    def test_python_workspace_gets_python_guide_only(self, workspace):
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "requirements.txt"), "w") as f:
            f.write("flask==2.0\npsycopg==3.0\n")
        prompt = _build_system_prompt(workspace, "python3 -m pytest")
        assert "## Coding Style Guides" in prompt
        assert "Python Style Guide" in prompt
        assert "PEP 8" in prompt
        assert "SQL Style Guide" in prompt  # psycopg → postgres → sql.md
        # No frontend leakage.
        assert "React Style" not in prompt
        assert "Vue Style" not in prompt
        assert "Angular Style" not in prompt

    def test_react_typescript_workspace_loads_relevant_guides(self, workspace):
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write(
                '{"dependencies": {"react": "^18", "typescript": "^5", '
                '"tailwindcss": "^3"}}'
            )
        with open(os.path.join(workspace, "tsconfig.json"), "w") as f:
            f.write("{}")
        prompt = _build_system_prompt(workspace, "npm test")
        assert "React Style Guide" in prompt
        assert "TypeScript Style Guide" in prompt
        assert "JavaScript Style Guide" in prompt  # node tag
        assert "HTML Style Guide" in prompt
        assert "CSS Style Guide" in prompt
        # Tailwind reference appears via css.md whose applies_to includes tailwind.
        assert "Tailwind" in prompt
        # No Python or SQL leakage.
        assert "Python Style Guide" not in prompt
        assert "SQL Style Guide" not in prompt

    def test_unknown_stack_omits_style_section_entirely(self, workspace):
        # When no shipped guide matches, the loader returns "" and the
        # system prompt skips the header line so prompt size stays flat
        # for stacks we don't cover.
        from harness.graph import _build_system_prompt
        # Empty workspace -> no tags -> empty style guides block.
        prompt = _build_system_prompt(workspace, "make build")
        assert "## Coding Style Guides" not in prompt

    def test_flutter_workspace_gets_flutter_guide(self, workspace):
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "pubspec.yaml"), "w") as f:
            f.write("name: app\nenvironment:\n  sdk: '>=3.0.0'\n")
        os.makedirs(os.path.join(workspace, "lib"))
        prompt = _build_system_prompt(workspace, "flutter test")
        assert "Flutter Style Guide" in prompt

    def test_web_workspace_gets_composite_design_system(self, workspace):
        # The composite web-design-system.md is the harness default
        # whenever the workspace is identified as web frontend work.
        # A bare React project must surface it alongside the focused
        # React guide.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"react": "^18"}}')
        prompt = _build_system_prompt(workspace, "npm test")
        assert "Composite Web Design System" in prompt
        # Module signatures — surface a token from each module so a
        # silent truncation would be caught here.
        assert "MODULE 1" in prompt and "Premium Indigo" in prompt
        assert "MODULE 2" in prompt and "Type Scale" in prompt
        assert "MODULE 3" in prompt and ".btn-primary" in prompt
        assert "MODULE 4" in prompt  # Interaction matrix
        assert "MODULE 5" in prompt and "tailwind.config.js" in prompt

    def test_pure_backend_workspace_skips_composite_design_system(self, workspace):
        # A Python-only / Java-only / Go-only workspace must not load
        # the web composite — that'd be hundreds of irrelevant tokens.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "requirements.txt"), "w") as f:
            f.write("flask==2.0\n")
        prompt = _build_system_prompt(workspace, "pytest")
        assert "Composite Web Design System" not in prompt

    def test_flutter_with_both_platforms_gets_both_mobile_guides(self, workspace):
        # A Flutter project that retains both ios/ and android/ platform
        # folders (the default) must surface both HIG and M3 guides.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "pubspec.yaml"), "w") as f:
            f.write("name: app\nenvironment:\n  sdk: '>=3.0.0'\n")
        os.makedirs(os.path.join(workspace, "lib"))
        os.makedirs(os.path.join(workspace, "ios"))
        os.makedirs(os.path.join(workspace, "android"))
        prompt = _build_system_prompt(workspace, "flutter test")
        assert "iOS Style Guide" in prompt
        assert "Apple Human Interface Guidelines" in prompt
        assert "Android Style Guide" in prompt
        assert "Material Design 3" in prompt
        # Flutter base guidance still loads alongside.
        assert "Flutter Style Guide" in prompt

    def test_flutter_ios_only_skips_android_guide(self, workspace):
        # A Flutter project with android/ removed targets iOS only; the
        # M3 guide is irrelevant there.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "pubspec.yaml"), "w") as f:
            f.write("name: app\nenvironment:\n  sdk: '>=3.0.0'\n")
        os.makedirs(os.path.join(workspace, "lib"))
        os.makedirs(os.path.join(workspace, "ios"))
        prompt = _build_system_prompt(workspace, "flutter test")
        assert "iOS Style Guide" in prompt
        assert "Android Style Guide" not in prompt
        assert "Material Design 3" not in prompt

    def test_native_ios_only_workspace_gets_hig_guide(self, workspace):
        # A native Swift/UIKit/SwiftUI project (no Flutter) — detected via
        # Podfile + *.xcodeproj — must still get the HIG guide.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "Podfile"), "w") as f:
            f.write("platform :ios, '15.0'\n")
        os.makedirs(os.path.join(workspace, "MyApp.xcodeproj"))
        prompt = _build_system_prompt(workspace, "xcodebuild test")
        assert "iOS Style Guide" in prompt
        assert "Apple Human Interface Guidelines" in prompt
        assert "Android Style Guide" not in prompt

    def test_native_android_only_workspace_gets_m3_guide(self, workspace):
        # A native Kotlin/Android project — detected via build.gradle with
        # com.android.application plugin — must get the M3 guide and not
        # the iOS one.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "build.gradle"), "w") as f:
            f.write("plugins {\n  id 'com.android.application'\n}\n")
        prompt = _build_system_prompt(workspace, "./gradlew test")
        assert "Android Style Guide" in prompt
        assert "Material Design 3" in prompt
        assert "iOS Style Guide" not in prompt

    def test_react_native_workspace_gets_both_mobile_guides(self, workspace):
        # React Native targets both iOS and Android by default — both
        # platform guides must load. Expo apps too.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"react": "^18", "react-native": "^0.74"}}')
        prompt = _build_system_prompt(workspace, "npm test")
        assert "iOS Style Guide" in prompt
        assert "Android Style Guide" in prompt

    def test_pure_web_workspace_skips_mobile_guides(self, workspace):
        # A standard React-web project (no react-native, no platform
        # folders) must NOT load mobile platform guides — they're
        # mobile-specific noise for a web app.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"react": "^18"}}')
        prompt = _build_system_prompt(workspace, "npm test")
        assert "iOS Style Guide" not in prompt
        assert "Android Style Guide" not in prompt

    def test_pure_backend_workspace_skips_mobile_guides(self, workspace):
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "requirements.txt"), "w") as f:
            f.write("flask==2.0\n")
        prompt = _build_system_prompt(workspace, "pytest")
        assert "iOS Style Guide" not in prompt
        assert "Android Style Guide" not in prompt

    def test_project_can_override_composite_design_system(self, workspace):
        # The two-tier loader must let a project replace the composite
        # default with its own house design system — this is the whole
        # point of the override mechanic when teams have their own
        # design tokens.
        from harness.graph import _build_system_prompt
        with open(os.path.join(workspace, "package.json"), "w") as f:
            f.write('{"dependencies": {"react": "^18"}}')
        sg_dir = os.path.join(workspace, "style_guides")
        os.makedirs(sg_dir)
        with open(os.path.join(sg_dir, "web-design-system.md"), "w") as f:
            f.write(dedent("""\
                ---
                applies_to: [react, vue, angular, html, css, tailwind]
                ---
                ## House Design System
                Brand primary: #FF0066.
            """))
        prompt = _build_system_prompt(workspace, "npm test")
        assert "House Design System" in prompt
        assert "#FF0066" in prompt
        # The shipped Premium Indigo composite must NOT also appear —
        # project overrides REPLACE, they don't concatenate.
        assert "Premium Indigo" not in prompt
        assert "Composite Web Design System (Premium Light Mode)" not in prompt
