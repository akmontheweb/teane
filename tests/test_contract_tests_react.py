"""ADR-0003 React tier — deterministic component render smoke tests.

Parses component prop types with the tsx tree-sitter grammar and emits a
render test only when every required prop is safely synthesisable and the
component is a props-driven presentational one (no provider hooks, not a
paramless container). Same correct-or-skip bias as the Python tiers.
"""

import pytest

from harness.contract_tests_react import (
    emit_react_contract_tests,
    parse_react_component,
    render_smoke_test,
)

# Skip the whole module if the tsx grammar isn't importable in this env.
pytest.importorskip("tree_sitter_language_pack")


_LEAF = """
import type { Contact } from '../types';

interface ContactListProps {
  contacts: Contact[];
  onEdit: (contact: Contact) => void;
  onDelete: (contact: Contact) => void;
}

export default function ContactList({ contacts, onEdit, onDelete }: ContactListProps) {
  return <ul>{contacts.map((c) => <li key={c.id}>{c.first_name}</li>)}</ul>;
}
"""

_PRIMITIVES = """
interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({ open, title, message, onConfirm, onCancel }: ConfirmDialogProps) {
  return open ? <div>{title}{message}</div> : null;
}
"""

_CUSTOM_OBJECT_PROP = """
import type { Contact } from '../types';

interface ContactCardProps {
  contact: Contact;
  onEdit: (contact: Contact) => void;
}

export default function ContactCard({ contact, onEdit }: ContactCardProps) {
  return <div>{contact.first_name}</div>;
}
"""

_CONTAINER_HOOK = """
import { useQuery } from '@tanstack/react-query';

interface Props { id: string; }

export default function Widget({ id }: Props) {
  const { data } = useQuery(['w', id], fetchW);
  return <div>{data}</div>;
}
"""

_PARAMLESS = """
import { useEffect, useState } from 'react';

export default function Dashboard() {
  const [x, setX] = useState(0);
  useEffect(() => { fetch('/api'); }, []);
  return <div>{x}</div>;
}
"""


class TestParse:
    def test_leaf_component_props(self):
        spec = parse_react_component(_LEAF, rel_path="c/ContactList.tsx")
        assert spec is not None
        assert spec.name == "ContactList"
        by = {p.name: p for p in spec.props}
        assert by["contacts"].ts_type == "Contact[]"
        assert by["onEdit"].ts_type == "(contact: Contact) => void"

    def test_optional_prop_flagged(self):
        spec = parse_react_component(_PRIMITIVES, rel_path="c/ConfirmDialog.tsx")
        by = {p.name: p for p in spec.props}
        assert by["confirmLabel"].optional is True
        assert by["open"].optional is False

    def test_provider_hook_skipped(self):
        assert parse_react_component(_CONTAINER_HOOK, rel_path="c/W.tsx") is None

    def test_paramless_container_skipped(self):
        assert parse_react_component(_PARAMLESS, rel_path="c/Dashboard.tsx") is None

    def test_malformed_source_skipped(self):
        assert parse_react_component("export default function (", rel_path="c/x.tsx") is None


class TestRender:
    def test_leaf_emits_correct_render(self):
        spec = parse_react_component(_LEAF, rel_path="client/src/components/ContactList.tsx")
        body = render_smoke_test(spec)
        assert body is not None
        assert "// @tests: client/src/components/ContactList.tsx" in body
        assert "import { render } from '@testing-library/react';" in body
        assert "import ContactList from '../ContactList';" in body
        # array → [], function → () => {}
        assert "render(<ContactList contacts={[]} onEdit={() => {}} onDelete={() => {}} />)" in body
        assert "expect(container).toBeTruthy()" in body

    def test_primitives_render(self):
        spec = parse_react_component(_PRIMITIVES, rel_path="c/ConfirmDialog.tsx")
        body = render_smoke_test(spec)
        # required primitives synthesised; optional confirmLabel omitted
        assert "open={false}" in body and "title={''}" in body and "message={''}" in body
        assert "confirmLabel" not in body

    def test_custom_object_prop_skips(self):
        # `contact: Contact` is a custom object → not synthesisable → skip.
        spec = parse_react_component(_CUSTOM_OBJECT_PROP, rel_path="c/ContactCard.tsx")
        assert render_smoke_test(spec) is None


class TestEmit:
    def _w(self, tmp_path, rel, body):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def test_writes_to_tests_dir_and_idempotent(self, tmp_path):
        self._w(tmp_path, "client/src/components/ContactList.tsx", _LEAF)
        written, markers = emit_react_contract_tests(
            str(tmp_path), ["client/src/components/ContactList.tsx"], "typescript",
        )
        out = "client/src/components/__tests__/ContactList.contract.test.tsx"
        assert written == [out]
        assert markers[out] == ["client/src/components/ContactList.tsx"]
        assert (tmp_path / out).is_file()
        # idempotent
        again, _ = emit_react_contract_tests(
            str(tmp_path), ["client/src/components/ContactList.tsx"], "typescript",
        )
        assert again == []

    def test_non_tsx_is_noop(self, tmp_path):
        self._w(tmp_path, "app/x.py", "x = 1\n")
        assert emit_react_contract_tests(str(tmp_path), ["app/x.py"], "python") == ([], {})

    def test_skips_existing_test_files(self, tmp_path):
        self._w(tmp_path, "client/src/components/__tests__/Foo.test.tsx", "x")
        assert emit_react_contract_tests(
            str(tmp_path),
            ["client/src/components/__tests__/Foo.test.tsx"],
            "typescript",
        ) == ([], {})
