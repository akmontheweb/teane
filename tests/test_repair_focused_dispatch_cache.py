"""Regression tests for the repair-node focused-dispatch cache prefix.

Ciod session 523e86a7 logged 29 ``cache_prefix_drift`` observability
events on the repair role — 55% of all drift on that build. Root
cause: the DISTRACTION/REGRESSION "focused dispatch" path in
``repair_node`` replaced ``messages`` with a fresh list whose
``messages[0]`` was the reflection banner (``Real blocker: <…>``,
``REQUIRED ACTION: <…>``, judge-named files — all round-specific).
The gateway's stable-prefix hash covers ``messages[0..1]``, so the
banner content invalidated the cache on every focused dispatch.

Fix (2026-07-04): pin ``messages[0]`` = SRS system prompt (session-
static) and ``messages[1]`` = the static repair-role framer, then
put the volatile reflection banner at ``messages[2]``. Provider
prompt caches hit consistently; the LLM still sees the banner
front-and-centre because it's the third message right after two
system framings.

These tests verify the rebuilt message shape mimicking the inline
``_focused`` assembly in ``repair_node``. The production loop is
inlined for readability; keep the two aligned.
"""

from __future__ import annotations


def _rebuild_focused(messages, reflection_msg, static_framer):
    """Replica of the inline ``_focused`` assembly from ``repair_node``.
    Extracted here so future refactors that split the production loop
    into a callable can reuse this suite."""
    focused = [
        messages[0],
        {"role": "system", "content": static_framer},
        {"role": "system", "content": reflection_msg},
    ]
    last_assistant = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m
            break
    for i, m in enumerate(messages):
        if i == 0:
            continue
        if m.get("role") != "system":
            continue
        content = str(m.get("content", "") or "")
        if len(content) > 8000:
            continue
        focused.append(m)
    if last_assistant is not None:
        focused.append(last_assistant)
    return focused


class TestFocusedDispatchCachePrefix:
    """The (session, role) stable-prefix hash covers the first 2
    messages. Both must be byte-stable across every focused dispatch
    within a session for the provider cache to hit."""

    def _hash_prefix(self, messages):
        """Cheap in-test replica of ``gateway.hash_stable_prefix``
        semantics: hash roles + contents of the first 2 messages. We
        don't import the real fn — the test verifies the shape, not
        the specific hash bytes."""
        import hashlib
        h = hashlib.sha256()
        for m in messages[:2]:
            h.update((m.get("role") or "").encode("utf-8"))
            h.update(b"|")
            content = m.get("content", "")
            h.update(content.encode("utf-8") if isinstance(content, str) else str(content).encode("utf-8"))
            h.update(b"\n---\n")
        return h.hexdigest()

    def test_stable_prefix_survives_reflection_content_change(self):
        # Two focused dispatches within the same session — the
        # reflection banner content changes (as it does every round),
        # but the stable prefix hash MUST stay the same.
        session_messages = [
            {"role": "system", "content": "You are an expert software engineer..."},
            {"role": "user", "content": "Build application"},
        ]
        static_framer = (
            "You are the repair LLM. Your only job this turn is to fix "
            "the failing diagnostic the judge names in the banner below."
        )
        round1 = _rebuild_focused(
            session_messages,
            reflection_msg=(
                "=== JUDGE'S VERDICT ===\n"
                "Real blocker: server/auth.py raises NameError: timezone\n"
                "REQUIRED ACTION: Import timezone from datetime."
            ),
            static_framer=static_framer,
        )
        round2 = _rebuild_focused(
            session_messages,
            reflection_msg=(
                "=== JUDGE'S VERDICT ===\n"
                "Real blocker: server/models/__init__.py:23 offset-naive vs aware\n"
                "REQUIRED ACTION: Normalize to UTC-aware datetimes."
            ),
            static_framer=static_framer,
        )
        assert self._hash_prefix(round1) == self._hash_prefix(round2), (
            "The stable-prefix hash MUST be identical across focused "
            "dispatches in the same session. Ciod 523e86a7 regression: "
            "prior to the fix, ``messages[0]`` was the reflection "
            "banner — content changed each round → cache invalidated → "
            "29 drift events per build."
        )

    def test_reflection_banner_still_present_in_focused_list(self):
        # The banner must appear in the message list (the LLM must
        # still see it) — just not at index 0. Verify it's at index 2.
        session_messages = [
            {"role": "system", "content": "SRS content..."},
            {"role": "user", "content": "Build app"},
        ]
        banner = "=== JUDGE'S VERDICT ===\nReal blocker: X"
        focused = _rebuild_focused(
            session_messages,
            reflection_msg=banner,
            static_framer="Static framer text.",
        )
        assert focused[0]["content"] == "SRS content..."
        assert focused[1]["content"] == "Static framer text."
        assert focused[2]["content"] == banner
        assert focused[2]["role"] == "system"

    def test_srs_at_index_0_when_present_in_input(self):
        # The SRS from the input session lands at index 0 of the
        # focused list. Confirms the "restore SRS at index 0" branch.
        session_messages = [
            {"role": "system", "content": "SRS anchor content"},
            {"role": "user", "content": "planning"},
            {"role": "assistant", "content": "prior patch attempt"},
        ]
        focused = _rebuild_focused(
            session_messages,
            reflection_msg="banner",
            static_framer="framer",
        )
        assert focused[0]["content"] == "SRS anchor content"

    def test_last_assistant_message_preserved(self):
        # The LLM's previous patch attempt must survive the rebuild so
        # the model sees its own last delta. Order: it comes AFTER the
        # framing + banner, not before.
        session_messages = [
            {"role": "system", "content": "SRS"},
            {"role": "user", "content": "planning"},
            {"role": "assistant", "content": "prior patch attempt"},
            {"role": "user", "content": "compiler failure feedback"},
        ]
        focused = _rebuild_focused(
            session_messages,
            reflection_msg="banner",
            static_framer="framer",
        )
        assistant_msgs = [m for m in focused if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "prior patch attempt"

    def test_small_system_notes_preserved(self):
        # Autofix / budget-warning system notes (short content) are
        # useful context and should survive. The >8000-char filter
        # skips full planning blobs but keeps short notes.
        session_messages = [
            {"role": "system", "content": "SRS"},
            {"role": "system", "content": "autofix note: pinned pytest"},
            {"role": "system", "content": "X" * 10000},  # planning-blob-sized
            {"role": "user", "content": "planning"},
        ]
        focused = _rebuild_focused(
            session_messages,
            reflection_msg="banner",
            static_framer="framer",
        )
        contents = [m["content"] for m in focused]
        assert "autofix note: pinned pytest" in contents
        assert "X" * 10000 not in contents
