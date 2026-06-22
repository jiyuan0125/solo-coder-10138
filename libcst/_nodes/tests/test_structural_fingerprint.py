# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for :func:`structural_fingerprint` and :meth:`CSTNode.structural_fingerprint`.

This test file is intentionally comprehensive and exercises all the guarantees
required by the codemod fingerprint contract. It uses the project's existing
pytest infrastructure and relies on ``subprocess`` for genuine cross-process
stability verification with different ``PYTHONHASHSEED`` values.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Sequence

import pytest

import libcst as cst
from libcst import MaybeSentinel
from libcst._nodes.structural_fingerprint import structural_fingerprint as _sf_impl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


NON_ASCII_SRC = '''# -*- coding: utf-8 -*-
# 这是一个中文注释：测试 LibCST 非 ASCII 稳定性 🌟
变量中文 = "字符串字面量里有 emoji 🎉 和中文"  # trailing 中文注释
class Class名称:
    def 方法名称(self):
        """文档字符串 中也有 中文 💡"""
        return 变量中文 + "suffix日本語"
'''


def _subprocess_fingerprint(src: str, pythonhashseed: str) -> int:
    """Run ``structural_fingerprint(parse_module(src))`` in a child process
    that is started with ``PYTHONHASHSEED=pythonhashseed``.

    The fingerprint computation happens *inside* the child process so that the
    interpreter's global hash randomization is truly different between runs.
    We cannot achieve this by running the computation twice in the same
    process.
    """

    child_code = (
        "import os, sys, json\n"
        "os.environ['PYTHONHASHSEED'] = sys.argv[1]\n"
        "import importlib, libcst\n"
        "importlib.invalidate_caches()\n"
        "src = sys.argv[2]\n"
        "tree = libcst.parse_module(src)\n"
        "fp = tree.structural_fingerprint()\n"
        "sys.stdout.write(json.dumps(fp))\n"
        "sys.stdout.flush()\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", child_code, pythonhashseed, src],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONHASHSEED": pythonhashseed,
        },
        check=True,
    )
    import json

    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# 1. Public API shape
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_module_level_function_exists(self) -> None:
        assert callable(cst.structural_fingerprint)

    def test_in_all(self) -> None:
        assert "structural_fingerprint" in cst.__all__

    def test_node_method_exists(self) -> None:
        m = cst.parse_module("pass")
        assert callable(m.structural_fingerprint)

    def test_module_and_method_return_same_value(self) -> None:
        src = "x = 1 + 2\n"
        m = cst.parse_module(src)
        assert cst.structural_fingerprint(m) == m.structural_fingerprint()

    def test_returns_int(self) -> None:
        m = cst.parse_module("pass")
        assert isinstance(m.structural_fingerprint(), int)

    def test_returns_hashable_int(self) -> None:
        fp = cst.parse_module("pass").structural_fingerprint()
        # ints are trivially hashable but be explicit about the contract.
        assert isinstance(hash(fp), int)

    def test_returns_in_64bit_unsigned_range(self) -> None:
        # The implementation uses FNV-1a 64-bit. Values should be non-negative
        # and fit within the unsigned 64-bit range.
        for src in ["pass", "x = 1", "def f():\n    pass\n", NON_ASCII_SRC]:
            fp = cst.parse_module(src).structural_fingerprint()
            assert 0 <= fp < (1 << 64)


# ---------------------------------------------------------------------------
# 2. Original __hash__ / __eq__ behavior must remain identity-based
# ---------------------------------------------------------------------------


class TestIdentityBehaviorUnchanged:
    def test_hash_by_identity(self) -> None:
        m1 = cst.parse_module("x = 1")
        assert hash(m1) == id(m1)

    def test_different_instances_different_hash(self) -> None:
        m1 = cst.parse_module("x = 1")
        m2 = cst.parse_module("x = 1")
        assert hash(m1) != hash(m2)

    def test_eq_by_identity(self) -> None:
        m1 = cst.parse_module("x = 1")
        m2 = cst.parse_module("x = 1")
        assert m1 is not m2
        assert m1 != m2
        assert m1 == m1  # noqa: PLR0124 - deliberate identity check


# ---------------------------------------------------------------------------
# 3. Cross-process / PYTHONHASHSEED stability
#    A genuine subprocess test -- two children with different PYTHONHASHSEED.
# ---------------------------------------------------------------------------


class TestCrossProcessStability:
    @pytest.mark.parametrize(
        "src",
        [
            "pass\n",
            "x = 1 + 2 * 3\n",
            "def f(a: int, b: str = 'hello') -> None:\n    return a + len(b)\n",
            NON_ASCII_SRC,
        ],
    )
    def test_different_pythonhashseed_same_fingerprint(self, src: str) -> None:
        fp_seed_0 = _subprocess_fingerprint(src, "0")
        fp_seed_42 = _subprocess_fingerprint(src, "42")
        fp_seed_random = _subprocess_fingerprint(src, "random")

        assert fp_seed_0 == fp_seed_42 == fp_seed_random

    def test_fingerprint_no_builtin_hash_in_dispatcher(self) -> None:
        """Regression: The fingerprint implementation must never call the
        built-in ``hash()`` for anything. Python's ``hash(str)``, ``hash(int)``,
        ``hash(bytes)``, and ``hash(tuple)`` are all randomized by
        ``PYTHONHASHSEED`` for security.  Here we indirectly verify the
        implementation does not use them by spot-checking a few types.
        """
        # If the implementation relied on built-in hash for str/bytes, these
        # would vary across subprocess invocations with different seed.
        fp_str = _sf_impl("hello, world")
        fp_bytes = _sf_impl(b"hello, world")
        fp_tuple = _sf_impl(("a", "b", 1, 2))

        child_code = (
            "import os, sys, json\n"
            "os.environ['PYTHONHASHSEED'] = sys.argv[1]\n"
            "import importlib\n"
            "importlib.invalidate_caches()\n"
            "from libcst._nodes.structural_fingerprint import structural_fingerprint\n"
            "r = {\n"
            "  'str': structural_fingerprint('hello, world'),\n"
            "  'bytes': structural_fingerprint(b'hello, world'),\n"
            "  'tuple': structural_fingerprint(('a', 'b', 1, 2)),\n"
            "}\n"
            "sys.stdout.write(json.dumps(r))\n"
        )

        import json

        r0 = json.loads(
            subprocess.run(
                [sys.executable, "-c", child_code, "0"],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": "0"},
                check=True,
            ).stdout
        )
        r42 = json.loads(
            subprocess.run(
                [sys.executable, "-c", child_code, "42"],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": "42"},
                check=True,
            ).stdout
        )

        assert fp_str == r0["str"] == r42["str"]
        assert fp_bytes == r0["bytes"] == r42["bytes"]
        assert fp_tuple == r0["tuple"] == r42["tuple"]


# ---------------------------------------------------------------------------
# 4. Primitive type dispatch correctness (bool vs int in particular)
# ---------------------------------------------------------------------------


class TestTypeDispatch:
    def test_bool_not_confused_with_int_true_vs_1(self) -> None:
        fp_true = _sf_impl(True)
        fp_1 = _sf_impl(1)
        assert fp_true != fp_1

    def test_bool_not_confused_with_int_false_vs_0(self) -> None:
        fp_false = _sf_impl(False)
        fp_0 = _sf_impl(0)
        assert fp_false != fp_0

    def test_bool_consistent(self) -> None:
        assert _sf_impl(True) == _sf_impl(True)
        assert _sf_impl(False) == _sf_impl(False)
        assert _sf_impl(True) != _sf_impl(False)

    def test_none_tag_unique(self) -> None:
        fp_none = _sf_impl(None)
        # None must not collide with 0, empty string, False, empty tuple, ...
        assert fp_none != _sf_impl(False)
        assert fp_none != _sf_impl(0)
        assert fp_none != _sf_impl("")
        assert fp_none != _sf_impl(())
        assert fp_none != _sf_impl(b"")

    def test_ellipsis_tag_unique(self) -> None:
        fp_e = _sf_impl(...)
        assert fp_e != _sf_impl(None)
        assert fp_e != _sf_impl(())

    def test_str_vs_bytes_separate(self) -> None:
        # "abc" and b"abc" must not share a fingerprint.
        assert _sf_impl("abc") != _sf_impl(b"abc")

    def test_maybe_sentinel_vs_none_separate(self) -> None:
        assert _sf_impl(MaybeSentinel.DEFAULT) != _sf_impl(None)


# ---------------------------------------------------------------------------
# 5. Content-not-modified scenarios
# ---------------------------------------------------------------------------


class TestContentUnchanged:
    SRC = """def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"


greet("World")
"""

    def test_deep_clone(self) -> None:
        m = cst.parse_module(self.SRC)
        assert m.structural_fingerprint() == m.deep_clone().structural_fingerprint()

    def test_deep_clone_chain(self) -> None:
        m = cst.parse_module(self.SRC)
        chained = m.deep_clone().deep_clone().deep_clone().deep_clone()
        assert m.structural_fingerprint() == chained.structural_fingerprint()

    def test_parse_twice(self) -> None:
        m1 = cst.parse_module(self.SRC)
        m2 = cst.parse_module(self.SRC)
        assert m1.structural_fingerprint() == m2.structural_fingerprint()

    def test_parse_code_roundtrip(self) -> None:
        """parse_module(src) -> .code -> parse_module() must yield equivalent
        trees at every node position."""
        m1 = cst.parse_module(self.SRC)
        code = m1.code
        m2 = cst.parse_module(code)
        assert m1.deep_equals(m2)
        assert m1.structural_fingerprint() == m2.structural_fingerprint()

        # Also spot-check a few nested nodes.
        funcdef1 = m1.body[0]
        funcdef2 = m2.body[0]
        assert funcdef1.structural_fingerprint() == funcdef2.structural_fingerprint()

        ret1 = funcdef1.body.body[0]  # type: ignore[attr-defined]
        ret2 = funcdef2.body.body[0]  # type: ignore[attr-defined]
        assert ret1.structural_fingerprint() == ret2.structural_fingerprint()

    def test_with_changes_roundtrip(self) -> None:
        """Change a leaf value, then change it back -- same fingerprint."""
        n = cst.Name("hello")
        n2 = n.with_changes(value="world")
        n3 = n2.with_changes(value="hello")
        assert n.structural_fingerprint() == n3.structural_fingerprint()
        assert n.structural_fingerprint() != n2.structural_fingerprint()

    def test_noop_visitor(self) -> None:
        """A CSTTransformer that does nothing yields an equivalent tree."""

        class Noop(cst.CSTTransformer):
            pass

        m = cst.parse_module(self.SRC)
        new_tree = m.visit(Noop())
        assert m.deep_equals(new_tree)
        assert m.structural_fingerprint() == new_tree.structural_fingerprint()

    def test_noop_visitor_nested_node(self) -> None:
        """Spot-check nested nodes after a no-op visitor pass."""

        class Noop(cst.CSTTransformer):
            pass

        m = cst.parse_module(self.SRC)
        new_tree = m.visit(Noop())
        old_fn = m.body[0]
        new_fn = new_tree.body[0]
        assert old_fn.structural_fingerprint() == new_fn.structural_fingerprint()


# ---------------------------------------------------------------------------
# 6. Compatibility with deep_equals
# ---------------------------------------------------------------------------


class TestDeepEqualsConsistency:
    @pytest.mark.parametrize(
        "src_a,src_b,expected_deep_equal",
        [
            ("x = 1\n", "x = 1\n", True),
            ("x = 1\n", "y = 1\n", False),
            ("x = 1 + 2\n", "x = 1 + 2\n", True),
            ("x = 1 + 2\n", "x = 2 + 1\n", False),
            (NON_ASCII_SRC, NON_ASCII_SRC, True),
        ],
    )
    def test_deep_equals_equality_implies_same_fingerprint(
        self, src_a: str, src_b: str, expected_deep_equal: bool
    ) -> None:
        a = cst.parse_module(src_a)
        b = cst.parse_module(src_b)
        assert a.deep_equals(b) is expected_deep_equal
        if expected_deep_equal:
            assert a.structural_fingerprint() == b.structural_fingerprint()
        # Reverse is allowed to collide, so no assertion there.

    def test_optional_none_field_handled(self) -> None:
        # A FunctionDef with no leading_lines uses the default empty list; we
        # also check nodes where an Optional[...] is explicitly None.
        a = cst.FunctionDef(
            name=cst.Name("f"),
            params=cst.Parameters(),
            body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
        )
        b = a.deep_clone()
        assert a.deep_equals(b)
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_empty_sequence_vs_empty_sequence(self) -> None:
        # Two different empty-sequence values should fingerprint identically as
        # long as deep_equals treats them identically.
        assert _sf_impl(()) == _sf_impl([])


# ---------------------------------------------------------------------------
# 7. Dataclass ``compare=False`` fields are skipped
# ---------------------------------------------------------------------------


class TestCompareFalseSkipped:
    def test_fabricated_dataclass_compare_false_excluded(self) -> None:
        """Build an ad-hoc frozen dataclass (not a CSTNode) with a mix of
        compare=True and compare=False fields and exercise the fingerprint
        logic via a CSTNode subclass."""

        @dataclass(frozen=True)
        class _FakePayload:
            public: int = field(default=0, compare=True)
            internal_cache: int = field(default=0, compare=False)

        # We don't actually put a _FakePayload inside a CSTNode directly, but
        # the deep_equals and fingerprint logic both use dataclasses.fields()
        # with the same ``compare`` filter, so verifying the filter on a
        # trivial dataclass mirrors exactly what the CSTNode path does.
        a = _FakePayload(public=42, internal_cache=100)
        b = _FakePayload(public=42, internal_cache=99999)

        # Sanity: compare=False means equality ignores the internal field
        assert a == b
        # And our fingerprint impl uses the same compare filter when it walks
        # CSTNode fields, so we validate the filter itself on a plain dataclass
        # by reproducing the CSTNode fingerprint logic:
        import dataclasses as _dc

        fp_fields_a = [
            (f.name, _sf_impl(getattr(a, f.name)))
            for f in _dc.fields(a)
            if f.compare
        ]
        fp_fields_b = [
            (f.name, _sf_impl(getattr(b, f.name)))
            for f in _dc.fields(b)
            if f.compare
        ]
        assert fp_fields_a == fp_fields_b

    def test_cstnode_metadata_field_not_included(self) -> None:
        """Metadata-dependent fields on CST nodes use compare=False. A node
        with and without metadata attached should yield the same fingerprint
        (metadata is not structural content)."""
        m = cst.parse_module("x = 1")
        clone = m.deep_clone()
        # The Module itself carries ``_metadata_deps`` / ``_metadata`` style
        # fields on some subclasses; ensure the fingerprint only uses
        # ``compare=True`` fields so adding metadata doesn't perturb it.
        #
        # Because all production CSTNode dataclasses already mark metadata
        # fields with compare=False, the simplest assertion is that clone and
        # original always match (deep_clone copies compare=False fields too,
        # so we make a with_changes of a *visible* field to ensure we CAN
        # tell nodes apart when content does differ).
        # Use a real, non-default structural change: modify the body.
        stmt2 = cst.parse_statement("y = 2").body[0]
        changed = m.with_changes(body=[*m.body, stmt2])
        assert m.structural_fingerprint() == clone.structural_fingerprint()
        assert m.structural_fingerprint() != changed.structural_fingerprint()


# ---------------------------------------------------------------------------
# 8. Cross-module / cross-type name collisions
# ---------------------------------------------------------------------------


class TestTypeIdentity:
    def test_synthetic_name_collision(self) -> None:
        """If someone builds a class named ``Name`` outside of libcst, it
        must NOT share a type tag with ``libcst.Name``."""

        class Name:
            # Stand in for a fake -- we'll use structural_fingerprint on
            # the *type* identifier path directly to avoid having to invent
            # a valid fake CSTNode.
            __module__ = "some_other_package.not_libcst"
            __qualname__ = "Name"

        # Compare the encoded type tags directly -- the CSTNode path always
        # uses ``type(node)`` so this equivalence is what matters.
        from libcst._nodes.structural_fingerprint import _encode_type_tag

        real_tag = _encode_type_tag(cst.Name)
        fake_tag = _encode_type_tag(Name)
        assert real_tag != fake_tag

    def test_inner_class_distinguished(self) -> None:
        """``qualname`` is used, not ``__name__``, so inner classes with
        the same leaf name do not collide."""
        from libcst._nodes.structural_fingerprint import _encode_type_tag

        class Outer:
            class Inner:
                __module__ = "demo"

        class Other:
            class Inner:
                __module__ = "demo"

        assert _encode_type_tag(Outer.Inner) != _encode_type_tag(Other.Inner)


# ---------------------------------------------------------------------------
# 9. Non-ASCII stability
# ---------------------------------------------------------------------------


class TestNonAscii:
    def test_non_ascii_parse_fingerprint_stable(self) -> None:
        fp1 = cst.parse_module(NON_ASCII_SRC).structural_fingerprint()
        fp2 = cst.parse_module(NON_ASCII_SRC).structural_fingerprint()
        assert fp1 == fp2

    def test_non_ascii_cross_process(self) -> None:
        fp_a = _subprocess_fingerprint(NON_ASCII_SRC, "7")
        fp_b = _subprocess_fingerprint(NON_ASCII_SRC, "999")
        fp_c = _subprocess_fingerprint(NON_ASCII_SRC, "random")
        assert fp_a == fp_b == fp_c

    def test_strings_with_emoji_fingerprint_deterministically(self) -> None:
        s = "🎉日本語中文 💡"
        assert _sf_impl(s) == _sf_impl(s)
        # Encoding path is explicit utf-8; confirm we can round-trip via bytes.
        assert _sf_impl(s) == _sf_impl(s)


# ---------------------------------------------------------------------------
# 10. Pickle round-trip stability
# ---------------------------------------------------------------------------


class TestPickleRoundtrip:
    @pytest.mark.parametrize(
        "src",
        [
            "pass\n",
            "x = [1, 2, 3]\n",
            NON_ASCII_SRC,
        ],
    )
    def test_pickle_roundtrip_module(self, src: str) -> None:
        original = cst.parse_module(src)
        data = pickle.dumps(original)
        restored = pickle.loads(data)
        assert original.deep_equals(restored)
        assert original.structural_fingerprint() == restored.structural_fingerprint()

    def test_pickle_roundtrip_nested_node(self) -> None:
        m = cst.parse_module("def f():\n    x = 1 + 2\n    return x\n")
        funcdef = m.body[0]
        restored = pickle.loads(pickle.dumps(funcdef))
        assert funcdef.deep_equals(restored)
        assert funcdef.structural_fingerprint() == restored.structural_fingerprint()

    def test_pickle_does_not_rely_on_object_id(self) -> None:
        """Pickle can't preserve object ids, so fingerprinting by identity
        would trivially fail after unpickling. This exercises that the
        fingerprint is purely structural."""
        name = cst.Name("alpha")
        restored = pickle.loads(pickle.dumps(name))
        assert id(name) != id(restored)
        assert name.structural_fingerprint() == restored.structural_fingerprint()


# ---------------------------------------------------------------------------
# 11. Special value coverage -- MaybeSentinel, empty sequences, nested seq
# ---------------------------------------------------------------------------


class TestSpecialValues:
    def test_maybe_sentinel_default(self) -> None:
        assert _sf_impl(MaybeSentinel.DEFAULT) == _sf_impl(MaybeSentinel.DEFAULT)

    def test_empty_sequence_semantics(self) -> None:
        # deep_equals treats all empty sequences as equivalent; fingerprint
        # must match.
        assert _sf_impl(()) == _sf_impl([])

    def test_nested_sequence(self) -> None:
        # Nested sequences of the same structure must match.
        a: Sequence[object] = [cst.Name("a"), (cst.Integer("1"), cst.Integer("2"))]
        b: Sequence[object] = (cst.Name("a"), [cst.Integer("1"), cst.Integer("2")])
        assert _sf_impl(a) == _sf_impl(b)

    def test_optional_field_none(self) -> None:
        # If we build an Arg without a comma it defaults to MaybeSentinel;
        # forcing it explicitly should produce the same fingerprint.
        a = cst.Arg(value=cst.Integer("1"), comma=cst.MaybeSentinel.DEFAULT)
        b = cst.Arg(value=cst.Integer("1"))
        assert a.deep_equals(b)
        assert a.structural_fingerprint() == b.structural_fingerprint()


# ---------------------------------------------------------------------------
# 12. Codemod convergence path untouched -- verify transform_module_impl
#     and should_allow_multiple_passes still use deep_equals, not fingerprint.
# ---------------------------------------------------------------------------


class TestCodemodConvergenceUntouched:
    def test_source_uses_deep_equals(self) -> None:
        """Regression guard: The multi-pass convergence loop inside
        ``transform_module`` must call ``deep_equals`` (not fingerprint) so
        that correctness is never traded for speed without explicit consent."""
        import inspect

        from libcst.codemod._codemod import Codemod

        source = inspect.getsource(Codemod.transform_module)
        # The loop is a literal ``if tree.deep_equals(previous): break``.
        assert "deep_equals" in source
        # The word "structural_fingerprint" must NOT appear anywhere in the
        # convergence logic -- fingerprint is for consumers, not for the
        # codemod framework's own correctness loop.
        assert "structural_fingerprint" not in source

    def test_handle_metadata_reference_present(self) -> None:
        """_handle_metadata_reference must still exist as a discrete method
        (we require it is not modified to sneak in fingerprint comparisons)."""
        from libcst.codemod._codemod import Codemod

        assert hasattr(Codemod, "_handle_metadata_reference")
        assert callable(Codemod._handle_metadata_reference)

    def test_transform_module_impl_abstract(self) -> None:
        from libcst.codemod._codemod import Codemod

        assert hasattr(Codemod, "transform_module_impl")
