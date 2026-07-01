"""Tests for ``harness/req_ids.py`` — the shared regex + parser
behind the v5 requirements ingest.

Two heading families coexist:

- Waterfall / ISO 29148: ``FR-NNN``, ``NFR-XXX-NNN``, ``US-NN-NN``
- Agile / SAFe (Phase 8): ``EPIC-NNN``, ``FEAT-NNN``, ``STORY-NNN``,
  ``STORY-NFR-NNN``

Phase 8 added the SAFe family so the agile-mode spec produced by
``requirements_doc.md`` Path A is parseable by the same ingest the
waterfall flow uses.
"""

from __future__ import annotations

from harness.req_ids import (
    EPIC_ID_RE,
    FEAT_ID_RE,
    FR_ID_RE,
    NFR_ID_RE,
    STORY_ID_RE,
    STORY_NFR_ID_RE,
    US_ID_RE,
    kind_for,
    parse_spec_requirements,
)


# ---------------------------------------------------------------------------
# kind_for — every family round-trips through the dispatch table
# ---------------------------------------------------------------------------

class TestKindFor:

    def test_waterfall_families(self):
        assert kind_for("FR-001") == "fr"
        assert kind_for("FR-9999") == "fr"
        assert kind_for("NFR-SEC-001") == "nfr"
        assert kind_for("NFR-PERF-014") == "nfr"
        assert kind_for("US-01-02") == "us"

    def test_agile_safe_families(self):
        assert kind_for("EPIC-001") == "epic"
        assert kind_for("FEAT-014") == "feat"
        assert kind_for("STORY-101") == "safe_story"
        assert kind_for("STORY-0001") == "safe_story"

    def test_safe_nfr_story_wins_over_story_prefix(self):
        """STORY-NFR-NNN is a strict prefix superset of STORY-NNN —
        the NFR family must be checked first."""
        assert kind_for("STORY-NFR-001") == "safe_nfr_story"
        assert kind_for("STORY-NFR-014") == "safe_nfr_story"

    def test_short_digit_story_ids_now_match_after_canonicalisation(self):
        """The former ``STORY_ID_RE`` "3+ digits" rule was retired: a
        spec author writing ``STORY-1`` gets the same req_key as
        ``STORY-001`` via :func:`canonicalize_req_key`. Kind-lookup
        therefore reports ``safe_story`` for either width; the two
        namespaces (spec req_key vs v5 work-unit story_key) are kept
        separate by call-site, not by digit count."""
        assert kind_for("STORY-1") == "safe_story"
        assert kind_for("STORY-9") == "safe_story"
        assert kind_for("STORY-99") == "safe_story"
        assert kind_for("STORY-001") == "safe_story"

    def test_v5_ac_keys_not_matched(self):
        """``STORY-3.AC-2`` is the v5 AC marker form — the SAFe
        regex must not accept it as a story."""
        assert kind_for("STORY-3.AC-2") is None
        assert kind_for("STORY-001.AC-1") is None

    def test_unknown_returns_none(self):
        assert kind_for("CR-7") is None
        assert kind_for("BOGUS-123") is None
        assert kind_for("FR-") is None
        assert kind_for("") is None


# ---------------------------------------------------------------------------
# parse_spec_requirements — both heading shapes
# ---------------------------------------------------------------------------

class TestParseWaterfall:
    """Flat ``### FR-NNN: Title`` heading shape — the form
    ``docs/SPEC_REQUIREMENTS.md`` in this repo uses today."""

    SPEC = (
        "# Product spec\n\n"
        "Some preamble.\n\n"
        "### FR-001: Login\n"
        "User can log in.\n\n"
        "### FR-002: Logout\n"
        "User can log out and the session ends.\n\n"
        "#### NFR-SEC-001: Token storage\n"
        "Session tokens MUST be hashed at rest.\n\n"
        "### US-03-02: Reset confirmation screen\n"
        "User sees a confirmation page after reset.\n"
    )

    def test_parses_all_four_headings(self):
        rows = parse_spec_requirements(self.SPEC)
        assert [r.req_key for r in rows] == [
            "FR-001", "FR-002", "NFR-SEC-001", "US-03-02",
        ]

    def test_kinds_assigned_correctly(self):
        rows = parse_spec_requirements(self.SPEC)
        kinds = {r.req_key: r.kind for r in rows}
        assert kinds["FR-001"] == "fr"
        assert kinds["NFR-SEC-001"] == "nfr"
        assert kinds["US-03-02"] == "us"

    def test_body_captured_until_next_heading(self):
        rows = parse_spec_requirements(self.SPEC)
        by_key = {r.req_key: r for r in rows}
        assert by_key["FR-001"].body == "User can log in."
        assert by_key["NFR-SEC-001"].body == (
            "Session tokens MUST be hashed at rest."
        )

    def test_source_line_one_indexed(self):
        rows = parse_spec_requirements(self.SPEC)
        # FR-001 is on line 5 (1-indexed) of self.SPEC.
        assert rows[0].source_line == 5


class TestParseAgileSAFe:
    """SAFe ``## Epic: EPIC-NNN — Title`` heading shape — emitted by
    Path A of ``harness/skills/docgen/requirements_doc.md``."""

    SPEC = (
        "# Product spec\n\n"
        "## Epic: EPIC-001 — Authentication\n"
        "All user-identity capabilities.\n\n"
        "### Feature: FEAT-014 — Password reset\n"
        "Operator can reset password via email.\n\n"
        "#### Story: STORY-101 — Operator clicks reset link\n"
        "Confirms via email link, sets new password.\n\n"
        "#### Enabler Story: STORY-NFR-001 — TLS 1.3 minimum\n"
        "All endpoints terminate TLS 1.3+.\n"
    )

    def test_parses_all_safe_headings(self):
        rows = parse_spec_requirements(self.SPEC)
        assert [r.req_key for r in rows] == [
            "EPIC-001", "FEAT-014", "STORY-101", "STORY-NFR-001",
        ]

    def test_kinds_assigned_correctly(self):
        rows = parse_spec_requirements(self.SPEC)
        kinds = {r.req_key: r.kind for r in rows}
        assert kinds["EPIC-001"] == "epic"
        assert kinds["FEAT-014"] == "feat"
        assert kinds["STORY-101"] == "safe_story"
        assert kinds["STORY-NFR-001"] == "safe_nfr_story"

    def test_titles_captured_without_label_prefix(self):
        rows = parse_spec_requirements(self.SPEC)
        by_key = {r.req_key: r for r in rows}
        # The "Epic: " / "Feature: " / "Story: " / "Enabler Story: "
        # label prefix must be stripped from the captured title.
        assert by_key["EPIC-001"].title == "Authentication"
        assert by_key["FEAT-014"].title == "Password reset"
        assert by_key["STORY-101"].title == "Operator clicks reset link"
        assert by_key["STORY-NFR-001"].title == "TLS 1.3 minimum"

    def test_body_captured(self):
        rows = parse_spec_requirements(self.SPEC)
        by_key = {r.req_key: r for r in rows}
        assert by_key["FEAT-014"].body == (
            "Operator can reset password via email."
        )


class TestMixedSpec:
    """Specs in the wild may mix shapes — operator manually edited a
    SAFe spec to add a flat FR row, etc. The parser must accept the
    union."""

    def test_mixed_safe_and_waterfall(self):
        spec = (
            "## Epic: EPIC-001 — Auth\n"
            "Epic body.\n\n"
            "### Feature: FEAT-001 — Login\n"
            "Feature body.\n\n"
            "### FR-001: Legacy flat FR\n"
            "Operator added this by hand.\n"
        )
        rows = parse_spec_requirements(spec)
        assert [r.req_key for r in rows] == [
            "EPIC-001", "FEAT-001", "FR-001",
        ]
        assert [r.kind for r in rows] == ["epic", "feat", "fr"]


# ---------------------------------------------------------------------------
# Individual regex sanity (used by traceability.py and ingest separately)
# ---------------------------------------------------------------------------

class TestRegexSanity:

    def test_story_id_re_accepts_1_to_4_digits(self):
        # The "3+ digits" rule was retired — canonicalisation now
        # collapses any width into the padded form, and STORY_ID_RE
        # matches every valid width. Namespace separation between
        # spec req_keys and v5 work-unit story_keys is enforced by
        # the call-site, not by the regex.
        assert STORY_ID_RE.search("STORY-1")
        assert STORY_ID_RE.search("STORY-9")
        assert STORY_ID_RE.search("STORY-99")
        assert STORY_ID_RE.search("STORY-001")
        assert STORY_ID_RE.search("STORY-100")
        assert STORY_ID_RE.search("STORY-1000")

    def test_epic_and_feat_accept_1_4_digits(self):
        assert EPIC_ID_RE.search("EPIC-1")
        assert EPIC_ID_RE.search("EPIC-9999")
        assert FEAT_ID_RE.search("FEAT-1")
        assert FEAT_ID_RE.search("FEAT-9999")

    def test_safe_nfr_story_matches(self):
        assert STORY_NFR_ID_RE.search("STORY-NFR-001")
        assert STORY_NFR_ID_RE.search("STORY-NFR-14")

    def test_word_boundaries_avoid_substring_matches(self):
        # ``USER-1234`` shouldn't be mistaken for ``FR-1234`` or similar.
        assert not FR_ID_RE.search("USER-1234")
        assert not NFR_ID_RE.search("ANFR-SEC-001")
        # ``US-`` is a real prefix, so this one IS expected to match.
        assert US_ID_RE.search("US-01-02")


# ---------------------------------------------------------------------------
# normalize_dashes — LLM-emitted Unicode hyphen/dash variants
# ---------------------------------------------------------------------------

class TestUnicodeHyphenNormalization:
    """Regression for the FinancialResearch HITL incident (session
    ``5aef5fc7``, 2026-07-01): the spec-synthesis LLM emitted every
    requirement heading with U+2011 NON-BREAKING HYPHEN, so
    ``parse_spec_requirements`` matched zero rows, ``known_req_keys``
    was empty, decomposition's auto-repair had nothing to fall back
    to, and the graph routed to human_intervention. Normalising at
    ingest closes the whole class."""

    AGILE_SPEC_WITH_NB_HYPHEN = (
        "# SRS\n"
        "\n"
        "## Epic: EPIC‑001 — Auth\n"
        "\n"
        "### Feature: FEAT‑014 — Password reset\n"
        "\n"
        "#### Story: STORY‑101 — Operator can reset own password\n"
        "Body prose.\n"
        "\n"
        "#### Enabler Story: STORY‑NFR‑001 — TLS ≥ 1.3\n"
    )

    def test_non_breaking_hyphen_in_agile_headings_still_parses(self):
        rows = parse_spec_requirements(self.AGILE_SPEC_WITH_NB_HYPHEN)
        keys = {r.req_key for r in rows}
        # Canonical ASCII keys — comparison-safe against DB / LLM output.
        assert keys == {
            "EPIC-001", "FEAT-014", "STORY-101", "STORY-NFR-001",
        }

    def test_en_dash_and_em_dash_variants_also_normalize(self):
        # LLM sometimes reaches for EN DASH (U+2013) or EM DASH (U+2014)
        # inside an ID position when auto-formatting compound tokens.
        # Zero-padded 3-digit canonical form after dash normalisation.
        spec = (
            "### FR–2: Login endpoint\n"
            "\n"
            "### NFR–SEC–1: Session encryption\n"
            "\n"
            "### FR—3: Logout endpoint\n"
        )
        keys = {r.req_key for r in parse_spec_requirements(spec)}
        assert keys == {"FR-002", "NFR-SEC-001", "FR-003"}

    def test_ascii_input_is_unchanged(self):
        from harness.req_ids import normalize_dashes
        s = "STORY-001: Plain ASCII heading — no drift"
        # Idempotent on pure-ASCII hyphens; em-dash in title normalises
        # (harmless, since titles are display-only downstream).
        got = normalize_dashes(s)
        assert "STORY-001" in got
        assert "‑" not in got and "—" not in got

    def test_normalize_dashes_is_idempotent(self):
        from harness.req_ids import normalize_dashes
        once = normalize_dashes("STORY‑001 — title")
        twice = normalize_dashes(once)
        assert once == twice == "STORY-001 - title"


# ---------------------------------------------------------------------------
# canonicalize_req_key — collapse STORY-1 / STORY-01 / STORY-001 etc.
# ---------------------------------------------------------------------------

class TestCanonicalizeReqKey:
    """The parser and validator both fold requirement keys to a
    canonical zero-padded form so a spec author's ``STORY-1`` and an
    LLM's ``STORY-001`` compare equal downstream."""

    def test_short_forms_zero_pad_to_three_digits(self):
        from harness.req_ids import canonicalize_req_key
        assert canonicalize_req_key("STORY-1") == "STORY-001"
        assert canonicalize_req_key("STORY-01") == "STORY-001"
        assert canonicalize_req_key("FR-7") == "FR-007"
        assert canonicalize_req_key("EPIC-2") == "EPIC-002"
        assert canonicalize_req_key("FEAT-14") == "FEAT-014"
        assert canonicalize_req_key("STORY-NFR-3") == "STORY-NFR-003"
        assert canonicalize_req_key("NFR-SEC-1") == "NFR-SEC-001"

    def test_already_canonical_forms_pass_through(self):
        from harness.req_ids import canonicalize_req_key
        for k in ("STORY-001", "FR-014", "EPIC-999", "STORY-NFR-042",
                  "NFR-PERF-001", "STORY-1000"):
            assert canonicalize_req_key(k) == k

    def test_us_pair_uses_two_digit_segments(self):
        from harness.req_ids import canonicalize_req_key
        assert canonicalize_req_key("US-3-2") == "US-03-02"
        assert canonicalize_req_key("US-03-02") == "US-03-02"

    def test_unknown_shapes_returned_unchanged(self):
        from harness.req_ids import canonicalize_req_key
        # Placeholder work-unit key from augment mode — must not be
        # mistaken for a canonicalisable req_key.
        assert canonicalize_req_key("STORY-NEW-1") == "STORY-NEW-1"
        assert canonicalize_req_key("random-token") == "random-token"
        assert canonicalize_req_key("") == ""

    def test_unicode_hyphen_input_is_canonicalised(self):
        from harness.req_ids import canonicalize_req_key
        # LLM ships STORY‑1 with U+2011 — one call handles both
        # normalisation and zero-padding.
        assert canonicalize_req_key("STORY‑1") == "STORY-001"
        assert canonicalize_req_key("FR‑7") == "FR-007"

    def test_canonicalize_is_idempotent(self):
        from harness.req_ids import canonicalize_req_key
        once = canonicalize_req_key("STORY-1")
        twice = canonicalize_req_key(once)
        assert once == twice == "STORY-001"

    def test_spec_parser_canonicalises_headings(self):
        # A spec author who writes ``STORY-1`` (no padding) gets the
        # same DB row as the canonical ``STORY-001`` form — the parse
        # no longer silently skips short-digit headings.
        spec = (
            "## Epic: EPIC-1 — Auth\n"
            "### Feature: FEAT-2 — Login\n"
            "#### Story: STORY-1 — Sign in with email\n"
            "#### Enabler Story: STORY-NFR-1 — TLS 1.3\n"
            "### FR-3: Retry failed requests\n"
        )
        keys = {r.req_key for r in parse_spec_requirements(spec)}
        assert keys == {
            "EPIC-001", "FEAT-002", "STORY-001",
            "STORY-NFR-001", "FR-003",
        }
