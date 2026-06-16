"""Form-schema derivation from the strict config validator.

The dashboard's config editor is form-based. Hand-curating a form per
config section would drift the moment a new key landed in
``harness/cli.py``'s validator tables, so we **derive** the form
schema from the same source the validator uses:

- ``_KNOWN_TOP_LEVEL_KEYS`` — the section names.
- ``_KNOWN_NESTED_KEYS[section]`` — the per-section field names.
- ``_TYPE_SCHEMA[\"section.field\"]`` — the runtime type tuple.

A round-trip test asserts every nested key in ``_TYPE_SCHEMA`` is
renderable + persistable through this layer; if someone lands a new
key without a type entry the form refuses to render the field (and
the round-trip test fails CI), so the validator stays the source of
truth.

The HTML renderer in :mod:`harness.dashboard` consumes the descriptors
this module produces. No HTML lives here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Form field descriptor
# ---------------------------------------------------------------------------

FORM_KIND_CHECKBOX = "checkbox"   # bool
FORM_KIND_NUMBER_INT = "number_int"
FORM_KIND_NUMBER_FLOAT = "number_float"
FORM_KIND_TEXT = "text"
FORM_KIND_TEXTAREA = "textarea"
FORM_KIND_JSON_LIST = "json_list"
FORM_KIND_JSON_DICT = "json_dict"
FORM_KIND_SELECT = "select"       # str with a finite choice set


# Enum tables: dotted-key → tuple of valid string values. When a field's
# dotted key is in this map, the form renders a <select> and the parser
# rejects any value not in the list. Seeded from cli.py's existing enum
# frozensets — extend here as new bounded-choice fields land.
def _field_choices() -> dict[str, tuple[str, ...]]:
    # Late import — cli.py imports web_forms transitively, so deferring
    # avoids a circular import at module load.
    from harness.cli import _VALID_SANDBOX_BACKENDS, _VALID_SELECTION_STRATEGIES
    return {
        "sandbox.backend": tuple(sorted(_VALID_SANDBOX_BACKENDS)),
        "speculative.selection_strategy": tuple(sorted(_VALID_SELECTION_STRATEGIES)),
    }


# Operator-facing descriptions per dotted key. The form's middle column
# ("Meaning") reads from this map. Empty string when a key isn't listed —
# the renderer falls back to "no description".
_FIELD_DESCRIPTIONS: dict[str, str] = {
    # Top-level scalars.
    "build_command": "Shell command the harness runs after applying patches to verify the build.",
    "allow_network": "Whether the sandbox may reach the network. Off by default for security.",
    "product_spec_dir": "Folder name (relative to the workspace) containing the product spec the planner reads.",
    "change_requests_dir": "Folder name (relative to the workspace) where the harness drops change-request markdown files.",
    # Sandbox.
    "sandbox.backend": "Sandbox engine. 'auto' picks the best available on this host.",
    "sandbox.docker_image": "Docker image used when backend is 'docker'.",
    "sandbox.docker_memory_limit": "Memory limit for the docker sandbox (e.g. '2g').",
    "sandbox.docker_cpu_limit": "CPU limit for the docker sandbox (number of cores, may be fractional).",
    # Token budget.
    "token_budget.hard_cap_usd": "Hard ceiling on cumulative LLM spend per session. Exits when crossed.",
    "token_budget.context_window_threshold_pct": "Trigger speculative compaction when context fills past this %.",
    # Logging.
    "logging.log_dir": "Directory where per-session JSONL logs are written.",
    "logging.level": "Python logging level (DEBUG, INFO, WARNING, ERROR).",
    # Schedule.
    "schedule.enabled": "Whether the cron-driven scheduled-job daemon is on.",
    "schedule.tick_seconds": "How often the daemon checks for due jobs.",
    "schedule.harness_binary": "Path to the harness CLI (used to spawn scheduled runs).",
    # Dashboard.
    "dashboard.enabled": "Whether the web UI is allowed to start (the subcommand itself overrides this).",
    "dashboard.host": "Bind host for the web UI. Default 127.0.0.1 for localhost-only.",
    "dashboard.port": "Bind port for the web UI.",
    "dashboard.token_env": "Env var name holding the bearer token. Empty disables auth.",
    "dashboard.writes_enabled": "Allow the editing UI + Run-from-web. Off = read-only.",
    "dashboard.docs_dir": "Folder of .md / .txt docs surfaced under View Documents.",
    "dashboard.carbon_css_url": "URL to the Carbon Design System CSS. Override for air-gapped deploys.",
    # Speculative.
    "speculative.selection_strategy": "How parallel speculative attempts are picked.",
    # Repo index.
    "repo_index.enabled": "Inject semantic retrieval results into the planner context.",
    "repo_index.backend": "Retrieval backend: 'tfidf' (default) or 'openai_embeddings'.",
    "repo_index.top_k": "Number of chunks injected into the planner context.",
    # Web tools.
    "web_tools.enabled": "Register web_fetch / web_search skills with the gateway.",
    "web_tools.max_bytes": "Max bytes returned by a single web_fetch call.",
    "web_tools.timeout_seconds": "Timeout for web_fetch and web_search calls.",
    # Persistence.
    "persistence.db_path": "Path to the LangGraph checkpoint SQLite database.",
}


@dataclass
class FormField:
    """One renderable field in a config section.

    ``kind`` is one of the ``FORM_KIND_*`` constants. ``type_tuple`` is
    the validator's runtime-type tuple (preserved so the form's POST
    handler can route the parsed value through the same gate the
    validator uses).
    """

    section: str
    name: str
    kind: str
    type_tuple: tuple[type, ...]
    current_value: Any = None
    required: bool = False
    secret: bool = False  # render as <input type=password>; never echo on errors
    placeholder: str = ""
    choices: Optional[tuple[str, ...]] = None  # for FORM_KIND_SELECT
    description: str = ""

    @property
    def dotted_key(self) -> str:
        return f"{self.section}.{self.name}" if self.section else self.name


@dataclass
class FormSection:
    """All renderable fields in one section.

    ``section`` is the top-level config key. ``fields`` is in
    sorted-name order. Sections whose every type entry is missing fall
    out as empty, which is fine — the renderer skips them.
    """

    section: str
    fields: list[FormField] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Type → form kind mapping
# ---------------------------------------------------------------------------

def kind_for_type_tuple(type_tuple: tuple[type, ...]) -> str:
    """Pick the render kind from the validator's type tuple.

    Order of preference (longest-match first so ``(int, float)`` picks
    float, not int):

    - ``(bool,)`` → checkbox
    - ``(int, float)`` or ``(float, ...)`` → number with step=any
    - ``(int,)`` → number with step=1
    - ``(list,)`` → JSON list textarea
    - ``(dict,)`` → JSON dict textarea
    - ``(str,)`` → text input
    - anything mixed or unknown → text input as the safest fallback
    """
    if not type_tuple:
        return FORM_KIND_TEXT
    s = set(type_tuple)
    if bool in s:
        return FORM_KIND_CHECKBOX
    if float in s:
        return FORM_KIND_NUMBER_FLOAT
    if int in s and float not in s:
        return FORM_KIND_NUMBER_INT
    if list in s:
        return FORM_KIND_JSON_LIST
    if dict in s:
        return FORM_KIND_JSON_DICT
    return FORM_KIND_TEXT


# ---------------------------------------------------------------------------
# 3. Building descriptors from the live validator tables
# ---------------------------------------------------------------------------

def build_section(
    section: str,
    *,
    current_config: Optional[dict[str, Any]] = None,
) -> FormSection:
    """Build a :class:`FormSection` for one top-level config section.

    Pulls field names from :data:`harness.cli._KNOWN_NESTED_KEYS` and
    types from :data:`harness.cli._TYPE_SCHEMA`. Fields without a type
    entry are quietly dropped (the round-trip test guards against this).
    """
    from harness.cli import _KNOWN_NESTED_KEYS, _TYPE_SCHEMA
    field_names = sorted(_KNOWN_NESTED_KEYS.get(section, frozenset()))
    out = FormSection(section=section)
    section_data = (current_config or {}).get(section) or {}
    choices_map = _field_choices()
    for name in field_names:
        dotted = f"{section}.{name}"
        type_tuple = _TYPE_SCHEMA.get(dotted)
        if type_tuple is None:
            logger.debug("[web_forms] no type entry for %s; skipping", dotted)
            continue
        choices = choices_map.get(dotted)
        kind = FORM_KIND_SELECT if choices else kind_for_type_tuple(type_tuple)
        out.fields.append(FormField(
            section=section, name=name,
            kind=kind,
            type_tuple=type_tuple,
            current_value=section_data.get(name),
            choices=choices,
            description=_FIELD_DESCRIPTIONS.get(dotted, ""),
        ))
    return out


def all_sections(
    *, current_config: Optional[dict[str, Any]] = None,
) -> list[FormSection]:
    """Build form schemas for every known top-level section. Sections
    without any renderable fields are still returned (empty) so the UI
    can render a placeholder.
    """
    from harness.cli import _KNOWN_TOP_LEVEL_KEYS, _KNOWN_NESTED_KEYS
    out: list[FormSection] = []
    for section in sorted(_KNOWN_TOP_LEVEL_KEYS):
        # Top-level scalar keys (e.g. "build_command") aren't in
        # _KNOWN_NESTED_KEYS; render them as a single-field section
        # carrying the section name as the field name.
        if section not in _KNOWN_NESTED_KEYS:
            from harness.cli import _TYPE_SCHEMA
            type_tuple = _TYPE_SCHEMA.get(section)
            if type_tuple is None:
                # Scalar section without a typed entry (e.g. "models",
                # "model_routing" — render via per-section editors,
                # not the generic form). Skip from the generic editor.
                out.append(FormSection(section=section, fields=[]))
                continue
            section_data = (current_config or {})
            out.append(FormSection(
                section=section,
                fields=[FormField(
                    section="", name=section,
                    kind=kind_for_type_tuple(type_tuple),
                    type_tuple=type_tuple,
                    current_value=section_data.get(section),
                    description=_FIELD_DESCRIPTIONS.get(section, ""),
                )],
            ))
            continue
        out.append(build_section(section, current_config=current_config))
    return out


# ---------------------------------------------------------------------------
# 4. Parsing form POST data back into typed Python values
# ---------------------------------------------------------------------------

class FormParseError(ValueError):
    """Raised when a form field value can't be parsed into its declared
    type. Carries the offending dotted key + the operator-facing error
    message so the renderer can show a per-field error."""

    def __init__(self, dotted_key: str, message: str):
        super().__init__(f"{dotted_key}: {message}")
        self.dotted_key = dotted_key
        self.message = message


def parse_value(field_: FormField, raw: Any) -> Any:
    """Parse a single field's raw POST value into its declared type.

    HTML form values arrive as strings (or lists for multi-valued
    fields); checkboxes are present-or-absent. We coerce to the
    validator's expected type and raise :class:`FormParseError` on
    failure so the renderer can surface a per-field error.
    """
    kind = field_.kind
    if kind == FORM_KIND_SELECT:
        choice = "" if raw is None else str(raw).strip()
        if not choice:
            raise FormParseError(field_.dotted_key, "value required (pick one)")
        choices = field_.choices or ()
        if choice not in choices:
            raise FormParseError(
                field_.dotted_key,
                f"value {choice!r} not in {list(choices)}",
            )
        return choice
    if kind == FORM_KIND_CHECKBOX:
        if isinstance(raw, bool):
            return raw
        if raw is None or raw == "":
            return False
        if isinstance(raw, str):
            return raw.lower() in ("1", "true", "on", "yes")
        return bool(raw)
    if kind == FORM_KIND_NUMBER_INT:
        if raw is None or raw == "":
            raise FormParseError(field_.dotted_key, "value required (integer)")
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise FormParseError(field_.dotted_key, f"not a valid integer: {exc}") from exc
    if kind == FORM_KIND_NUMBER_FLOAT:
        if raw is None or raw == "":
            raise FormParseError(field_.dotted_key, "value required (number)")
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise FormParseError(field_.dotted_key, f"not a valid number: {exc}") from exc
    if kind == FORM_KIND_JSON_LIST:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return []
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise FormParseError(field_.dotted_key, f"not valid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise FormParseError(field_.dotted_key, "JSON must be a list")
        return parsed
    if kind == FORM_KIND_JSON_DICT:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return {}
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise FormParseError(field_.dotted_key, f"not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise FormParseError(field_.dotted_key, "JSON must be an object")
        return parsed
    # text / unknown — pass through stringified.
    if raw is None:
        return ""
    return str(raw)


def parse_section_post(
    section: FormSection, post_data: dict[str, Any],
) -> tuple[dict[str, Any], list[FormParseError]]:
    """Parse a posted form payload back into a section dict.

    ``post_data`` is a mapping of dotted-key → raw form value (string
    or list, as multipart/x-www-form-urlencoded delivers). Missing
    checkboxes are interpreted as ``False`` (HTML omits absent
    checkboxes from the POST body).

    Returns ``(section_dict, errors)`` so the caller can re-render the
    form with per-field errors when validation fails. The ``section_dict``
    is **always** present even when ``errors`` is non-empty so the
    operator's other typed values aren't lost.
    """
    out: dict[str, Any] = {}
    errors: list[FormParseError] = []
    for f in section.fields:
        raw = post_data.get(f.dotted_key, None)
        # Lists from form parsing arrive as the first/last; pick last
        # to honour "value last write wins".
        if isinstance(raw, list):
            raw = raw[-1] if raw else None
        try:
            value = parse_value(f, raw)
        except FormParseError as exc:
            errors.append(exc)
            continue
        out[f.name] = value
    return out, errors


# ---------------------------------------------------------------------------
# 5. Coverage check — round-trip validator vs. form schema
# ---------------------------------------------------------------------------

def renderable_dotted_keys() -> set[str]:
    """Every dotted key the form schema can render. The round-trip test
    asserts this equals (or is a near-equal of) the validator's
    ``_TYPE_SCHEMA`` keys. New config keys land in both lists or fail
    the test."""
    sections = all_sections(current_config=None)
    keys: set[str] = set()
    for s in sections:
        for f in s.fields:
            keys.add(f.dotted_key)
    return keys


# ---------------------------------------------------------------------------
# 6. Run-harness CLI flag schema (per-flag inputs on the Run page)
# ---------------------------------------------------------------------------

# Yes/No select kind — operator-friendly form rendering for boolean CLI
# flags. The form layer maps "yes" → emit the flag, "no" → omit; this
# is more discoverable than a single text box where operators have to
# remember --allow-network vs --allow_network.
FORM_KIND_YES_NO = "yes_no"


@dataclass
class RunFlag:
    """One CLI flag the operator can toggle from the Run Harness page.

    ``kind`` is one of the ``FORM_KIND_*`` constants. ``flag`` is the
    canonical long-form (e.g. ``--allow-network``); ``flag_off`` is the
    explicit-off form for tri-valued flags (e.g. ``--git=disable``).
    ``yes_emits_flag`` controls how ``FORM_KIND_YES_NO`` collapses to an
    argv list: True means "yes" emits the flag and "no" omits it
    (store_true semantics); False is the opposite.
    """

    name: str                 # form field id ("allow_network")
    label: str                # human-facing label ("Allow network")
    description: str          # operator-facing helper text
    kind: str                 # FORM_KIND_*
    flag: str = ""            # e.g. "--allow-network"; for SELECT/TEXT/NUMBER, emitted as `<flag> <value>`
    default: Any = None       # default form value
    yes_emits_flag: bool = True   # YES_NO: "yes" → emit flag, "no" → omit
    choices: tuple[str, ...] = ()  # FORM_KIND_SELECT
    min_value: Optional[int] = None  # FORM_KIND_NUMBER_INT bounds
    max_value: Optional[int] = None

    @property
    def field_id(self) -> str:
        return f"flag.{self.name}"


# Operator-facing schema of every flag the Run page surfaces. Mirrors
# the interactive `harness run` wizard (see :func:`harness.wizard.run_setup_wizard`)
# which asks for exactly five things: workspace, prompt, git mode, new
# build, discover. Workspace + prompt have dedicated inputs at the top
# of the form; the remaining three live in this list. Operators reach
# the other CLI flags (build-cmd, allow-network, verbose, etc.) from the
# terminal — keeping the web page in sync with the wizard avoids
# burying the common path under a wall of advanced toggles.
_RUN_FLAGS: tuple[RunFlag, ...] = (
    RunFlag(
        name="git",
        label="Git mode",
        description="'enable' (default) uses git for stash/patch-branch/rollback. 'disable' skips every git-aware step — pick this when the target repo isn't under git.",
        kind=FORM_KIND_SELECT,
        flag="--git",
        default="enable",
        choices=("enable", "disable"),
    ),
    RunFlag(
        name="new_build",
        label="New build",
        description="When true, delete every file at the workspace root except product_spec/ and .git/, then start fresh on a clean baseline. Defaults to false (steady-state).",
        kind=FORM_KIND_SELECT,
        flag="--new-build",
        default="false",
        choices=("false", "true"),
    ),
    RunFlag(
        name="discover",
        label="Run discovery",
        description="Run the full requirements/architecture/deployment discovery interview before code generation. Recommended for greenfield projects; skipped by default for incremental patching.",
        kind=FORM_KIND_YES_NO,
        flag="--discover",
        default="no",
    ),
)


def run_flags() -> tuple[RunFlag, ...]:
    """Operator-facing CLI flag schema for the Run Harness page."""
    return _RUN_FLAGS


def build_run_argv_from_form(
    post_data: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Translate a POSTed Run Harness form into a list of CLI argv tokens.

    Returns ``(argv, errors)`` — errors is a list of operator-facing
    messages keyed by flag name (e.g. "spec_review_cycles: must be 0-5"),
    suitable for surfacing back to the form.

    Unset / blank fields collapse to "use the CLI default" (i.e. they
    don't add anything to argv). The yes/no kinds emit the flag only
    when the operator says yes.
    """
    argv: list[str] = []
    errors: list[str] = []
    for f in _RUN_FLAGS:
        raw = post_data.get(f.field_id, None)
        if isinstance(raw, list):
            raw = raw[-1] if raw else None
        if raw is None:
            value: str = ""
        else:
            value = str(raw).strip()

        if f.kind == FORM_KIND_TEXT:
            if not value:
                continue
            argv.extend([f.flag, value])
            continue
        if f.kind == FORM_KIND_YES_NO:
            picked = value.lower() if value else str(f.default).lower()
            if picked not in ("yes", "no"):
                errors.append(f"{f.name}: must be yes or no, got {value!r}")
                continue
            if (picked == "yes") == f.yes_emits_flag:
                argv.append(f.flag)
            continue
        if f.kind == FORM_KIND_SELECT:
            picked = value or str(f.default)
            if picked not in f.choices:
                errors.append(
                    f"{f.name}: value {picked!r} not in {list(f.choices)}"
                )
                continue
            # Skip when the operator hasn't moved off the default — the CLI
            # already applies the same default, so an explicit
            # `--flag=default` token is noise.
            if picked == str(f.default):
                continue
            # --new-build / --git both take the form `--flag=value` in the
            # CLI, so emit a single equals-joined token for consistency.
            argv.append(f"{f.flag}={picked}")
            # --new-build=true triggers a confirmation prompt in cmd_run;
            # the dashboard spawns the subprocess without a TTY, so the
            # prompt would hang forever. Mirror what the interactive
            # wizard does (wizard.py sets args.assume_yes=True on the
            # same branch) and emit --yes.
            if f.name == "new_build" and picked == "true":
                argv.append("--yes")
            continue
        if f.kind == FORM_KIND_NUMBER_INT:
            if not value:
                continue
            try:
                num = int(value)
            except ValueError:
                errors.append(f"{f.name}: not a valid integer ({value!r})")
                continue
            if f.min_value is not None and num < f.min_value:
                errors.append(f"{f.name}: must be >= {f.min_value}")
                continue
            if f.max_value is not None and num > f.max_value:
                errors.append(f"{f.name}: must be <= {f.max_value}")
                continue
            argv.extend([f.flag, str(num)])
            continue
        # Unknown kind — skip silently.
    return argv, errors


# ---------------------------------------------------------------------------
# 7. Configure Harness page — semantic grouping of config sections
# ---------------------------------------------------------------------------

# Map every top-level config section to a human-facing group. The
# Configure Harness page renders one collapsible group per entry; the
# accordion of section editors sits inside the group body. When a new
# top-level section lands in :data:`harness.cli._KNOWN_TOP_LEVEL_KEYS`,
# add it to one of these groups so it's discoverable in the UI.
#
# Order in this tuple is the render order. Sections within each group
# stay in alphabetical order (consistent with ``all_sections``).
_CONFIG_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "general",
        "General",
        (
            "build_command", "allow_network",
            "product_spec_dir", "change_requests_dir", "change_requests",
            "patcher", "compiler", "languages",
        ),
    ),
    (
        "llm_registry",
        "LLM Registry",
        ("models",),
    ),
    (
        "llm_routing",
        "LLM Routing",
        ("model_routing", "llm_dispatch"),
    ),
    (
        "sandbox_security",
        "Sandbox & Security",
        ("sandbox", "security", "redaction"),
    ),
    (
        "budget_throttling",
        "Budget & Throttling",
        ("token_budget", "node_throttle", "metrics"),
    ),
    (
        "logging_debug",
        "Logging & Debug",
        ("logging", "debug"),
    ),
    (
        "skills_tools",
        "Skills & Tools",
        ("skills", "web_tools", "mcp"),
    ),
    (
        "patching_speculation",
        "Patching & Speculation",
        ("speculative", "impact", "lintgate", "test_generation"),
    ),
    (
        "storage_memory",
        "Storage & Memory",
        ("persistence", "memory", "repo_index"),
    ),
    (
        "deployment",
        "Deployment",
        ("deployment",),
    ),
    (
        "scheduling",
        "Scheduling",
        ("schedule",),
    ),
    (
        "dashboard",
        "Harness Web",
        ("dashboard",),
    ),
    (
        "github",
        "GitHub",
        ("github",),
    ),
)


@dataclass
class ConfigGroup:
    """One collapsible group of related config sections on the Configure
    Harness page.

    ``sections`` is in the order ``_CONFIG_GROUPS`` declares; the renderer
    iterates that order directly.
    """

    slug: str
    title: str
    sections: list[FormSection] = field(default_factory=list)


def grouped_sections(
    *, current_config: Optional[dict[str, Any]] = None,
) -> list[ConfigGroup]:
    """Build the grouped form schema the Configure Harness page renders.

    Every section in :func:`all_sections` lands in exactly one group;
    sections not yet listed in :data:`_CONFIG_GROUPS` fall into a
    catch-all "Other" group so they remain reachable. The catch-all is a
    safety net — new sections should be moved into a semantic group as
    soon as they land, not left in Other.
    """
    sections = all_sections(current_config=current_config)
    by_name = {s.section: s for s in sections}
    groups: list[ConfigGroup] = []
    placed: set[str] = set()
    for slug, title, section_names in _CONFIG_GROUPS:
        group = ConfigGroup(slug=slug, title=title)
        for name in section_names:
            sec = by_name.get(name)
            if sec is None:
                continue
            group.sections.append(sec)
            placed.add(name)
        groups.append(group)
    unplaced = [s for s in sections if s.section not in placed]
    if unplaced:
        groups.append(ConfigGroup(slug="other", title="Other", sections=unplaced))
    return groups
