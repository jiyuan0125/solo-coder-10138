# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import libcst as cst
from libcst._maybe_sentinel import MaybeSentinel
from libcst._nodes.structural_fingerprint import StructuralFingerprint


class TestStructuralFingerprintSameStructure:
    def test_leaf_same_value(self):
        a = cst.SimpleWhitespace("  ")
        b = cst.SimpleWhitespace("  ")
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_leaf_different_value(self):
        a = cst.SimpleWhitespace("  ")
        b = cst.SimpleWhitespace("   ")
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_nested_same(self):
        a = cst.EmptyLine(whitespace=cst.SimpleWhitespace(""))
        b = cst.EmptyLine(whitespace=cst.SimpleWhitespace(""))
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_nested_different(self):
        a = cst.EmptyLine(whitespace=cst.SimpleWhitespace(""))
        b = cst.EmptyLine(whitespace=cst.SimpleWhitespace(" "))
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_different_type_same_field_values(self):
        a = cst.Newline()
        b = cst.SimpleWhitespace("")
        assert type(a) is not type(b)
        assert a.structural_fingerprint() != b.structural_fingerprint()


class TestStructuralFingerprintDeepClone:
    def test_clone_simple(self):
        orig = cst.SimpleWhitespace("  ")
        cloned = orig.deep_clone()
        assert orig.structural_fingerprint() == cloned.structural_fingerprint()

    def test_clone_nested(self):
        orig = cst.TrailingWhitespace(
            whitespace=cst.SimpleWhitespace("  "),
            comment=cst.Comment("# hi"),
            newline=cst.Newline("\n"),
        )
        cloned = orig.deep_clone()
        assert orig.structural_fingerprint() == cloned.structural_fingerprint()

    def test_clone_with_sequence(self):
        orig = cst.SimpleStatementLine(body=[cst.Pass(), cst.Pass()])
        cloned = orig.deep_clone()
        assert orig.structural_fingerprint() == cloned.structural_fingerprint()

    def test_clone_identity_hash_unchanged(self):
        orig = cst.SimpleWhitespace("")
        cloned = orig.deep_clone()
        assert hash(orig) == id(orig)
        assert hash(cloned) == id(cloned)
        assert hash(orig) != hash(cloned)
        assert orig != cloned


class TestStructuralFingerprintCrossParse:
    def _can_parse(self):
        try:
            from libcst import native  # noqa: F401
            return True
        except ImportError:
            return False

    def test_expression_parse(self):
        if not self._can_parse():
            pytest.skip("native parser not available")
        a = cst.parse_expression("1+2")
        b = cst.parse_expression("1+2")
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_different_expressions(self):
        if not self._can_parse():
            pytest.skip("native parser not available")
        a = cst.parse_expression("1+2")
        b = cst.parse_expression("3+4")
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_name_node(self):
        a = cst.Name("foo")
        b = cst.Name("foo")
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_name_different_value(self):
        a = cst.Name("foo")
        b = cst.Name("bar")
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_manually_constructed_same_structure(self):
        a = cst.BinaryOperation(
            left=cst.Integer("1"),
            operator=cst.Add(),
            right=cst.Integer("2"),
        )
        b = cst.BinaryOperation(
            left=cst.Integer("1"),
            operator=cst.Add(),
            right=cst.Integer("2"),
        )
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_manually_constructed_different_structure(self):
        a = cst.BinaryOperation(
            left=cst.Integer("1"),
            operator=cst.Add(),
            right=cst.Integer("2"),
        )
        b = cst.BinaryOperation(
            left=cst.Integer("3"),
            operator=cst.Add(),
            right=cst.Integer("4"),
        )
        assert a.structural_fingerprint() != b.structural_fingerprint()


class TestStructuralFingerprintMaybeSentinel:
    def test_maybe_sentinel_default(self):
        a = cst.Pass()
        b = cst.Pass()
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_maybe_sentinel_vs_explicit_comma(self):
        a = cst.Pass(semicolon=MaybeSentinel.DEFAULT)
        b = cst.Pass(semicolon=cst.Semicolon())
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_arg_maybe_sentinel(self):
        a = cst.Arg(value=cst.Name("x"))
        b = cst.Arg(value=cst.Name("x"))
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_arg_explicit_comma(self):
        a = cst.Arg(value=cst.Name("x"), comma=cst.Comma())
        b = cst.Arg(value=cst.Name("x"), comma=cst.Comma())
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_arg_maybe_sentinel_vs_comma(self):
        a = cst.Arg(value=cst.Name("x"))
        b = cst.Arg(value=cst.Name("x"), comma=cst.Comma())
        assert a.structural_fingerprint() != b.structural_fingerprint()


class TestStructuralFingerprintSequenceFields:
    def test_empty_sequence(self):
        a = cst.SimpleStatementLine(body=[])
        b = cst.SimpleStatementLine(body=[])
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_sequence_same(self):
        a = cst.SimpleStatementLine(body=[cst.Pass()])
        b = cst.SimpleStatementLine(body=[cst.Pass()])
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_sequence_different_length(self):
        a = cst.SimpleStatementLine(body=[cst.Pass()])
        b = cst.SimpleStatementLine(body=[cst.Pass(), cst.Pass()])
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_sequence_different_element(self):
        a = cst.SimpleStatementLine(body=[cst.Pass()])
        b = cst.SimpleStatementLine(body=[cst.Break()])
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_tuple_vs_list_sequence(self):
        a = cst.SimpleStatementLine(body=[cst.Pass()])
        b = cst.SimpleStatementLine(body=(cst.Pass(),))
        assert a.structural_fingerprint() == b.structural_fingerprint()


class TestStructuralFingerprintEmptyFields:
    def test_none_field_same(self):
        a = cst.EmptyLine(comment=None)
        b = cst.EmptyLine(comment=None)
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_none_vs_value(self):
        a = cst.EmptyLine(comment=None)
        b = cst.EmptyLine(comment=cst.Comment("# x"))
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_empty_whitespace(self):
        a = cst.SimpleWhitespace("")
        b = cst.SimpleWhitespace("")
        assert a.structural_fingerprint() == b.structural_fingerprint()


class TestStructuralFingerprintNested:
    def test_deeply_nested(self):
        a = cst.TrailingWhitespace(
            whitespace=cst.SimpleWhitespace(" "),
            comment=cst.Comment("# test"),
            newline=cst.Newline("\n"),
        )
        b = cst.TrailingWhitespace(
            whitespace=cst.SimpleWhitespace(" "),
            comment=cst.Comment("# test"),
            newline=cst.Newline("\n"),
        )
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_deeply_nested_difference(self):
        a = cst.TrailingWhitespace(
            whitespace=cst.SimpleWhitespace(" "),
            comment=cst.Comment("# test1"),
            newline=cst.Newline("\n"),
        )
        b = cst.TrailingWhitespace(
            whitespace=cst.SimpleWhitespace(" "),
            comment=cst.Comment("# test2"),
            newline=cst.Newline("\n"),
        )
        assert a.structural_fingerprint() != b.structural_fingerprint()

    def test_nested_with_sequence_and_maybe_sentinel(self):
        a = cst.SimpleStatementLine(body=[cst.Pass()])
        b = cst.SimpleStatementLine(body=[cst.Pass()])
        assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_leaf_difference_in_nested(self):
        a = cst.SimpleStatementLine(body=[cst.Pass()])
        b = cst.SimpleStatementLine(body=[cst.Pass(semicolon=cst.Semicolon())])
        assert a.structural_fingerprint() != b.structural_fingerprint()


class TestStructuralFingerprintDeepEqualsConsistency:
    def test_equal_implies_same_fingerprint(self):
        pairs = [
            (cst.SimpleWhitespace(""), cst.SimpleWhitespace("")),
            (
                cst.EmptyLine(whitespace=cst.SimpleWhitespace("")),
                cst.EmptyLine(whitespace=cst.SimpleWhitespace("")),
            ),
            (
                cst.SimpleStatementLine(body=[cst.Pass()]),
                cst.SimpleStatementLine(body=(cst.Pass(),)),
            ),
            (cst.Name("foo"), cst.Name("foo")),
        ]
        for a, b in pairs:
            assert a.deep_equals(b), f"Expected {a} to deep_equal {b}"
            assert a.structural_fingerprint() == b.structural_fingerprint()

    def test_not_equal_implies_different_fingerprint(self):
        pairs = [
            (cst.SimpleWhitespace(" "), cst.SimpleWhitespace("     ")),
            (
                cst.EmptyLine(whitespace=cst.SimpleWhitespace(" ")),
                cst.EmptyLine(whitespace=cst.SimpleWhitespace("       ")),
            ),
            (
                cst.SimpleStatementLine(body=[cst.Pass(semicolon=cst.Semicolon())]),
                cst.SimpleStatementLine(body=[cst.Pass(semicolon=cst.Semicolon())] * 2),
            ),
            (cst.Name("foo"), cst.Name("bar")),
        ]
        for a, b in pairs:
            assert not a.deep_equals(b)
            assert a.structural_fingerprint() != b.structural_fingerprint()


class TestStructuralFingerprintAsKey:
    def test_dict_key(self):
        node = cst.Name("x")
        fp = node.structural_fingerprint()
        d = {fp: "hello"}
        same_node = cst.Name("x")
        assert d[same_node.structural_fingerprint()] == "hello"

    def test_set_membership(self):
        a = cst.Name("x")
        b = cst.Name("x")
        c = cst.Name("y")
        s = {a.structural_fingerprint(), c.structural_fingerprint()}
        assert b.structural_fingerprint() in s
        assert a.structural_fingerprint() in s

    def test_dedup_in_set(self):
        nodes = [cst.Name("x") for _ in range(5)]
        fps = {n.structural_fingerprint() for n in nodes}
        assert len(fps) == 1

    def test_distinct_in_set(self):
        nodes = [cst.Name(chr(ord("a") + i)) for i in range(5)]
        fps = {n.structural_fingerprint() for n in nodes}
        assert len(fps) == 5

    def test_fingerprint_hash_stable(self):
        node = cst.Name("test")
        fp = node.structural_fingerprint()
        h1 = hash(fp)
        h2 = hash(fp)
        assert h1 == h2

    def test_codemod_cache_pattern(self):
        cache = {}
        node = cst.SimpleStatementLine(body=[cst.Pass()])
        fp = node.structural_fingerprint()
        cache[fp] = "processed"
        cloned = node.deep_clone()
        assert cache[cloned.structural_fingerprint()] == "processed"


class TestStructuralFingerprintIdentityHashUnchanged:
    def test_identity_hash_still_id(self):
        sw = cst.SimpleWhitespace("")
        assert hash(sw) == id(sw)

    def test_identity_eq_still_is(self):
        a = cst.SimpleWhitespace("")
        b = cst.SimpleWhitespace("")
        assert a == a
        assert a != b
        assert a is not b

    def test_node_in_set_by_identity(self):
        a = cst.Name("x")
        b = cst.Name("x")
        s = {a}
        assert a in s
        assert b not in s

    def test_node_as_dict_key_by_identity(self):
        a = cst.Name("x")
        b = cst.Name("x")
        d = {a: "val_a"}
        assert d[a] == "val_a"
        with pytest.raises(KeyError):
            d[b]


class TestStructuralFingerprintRepr:
    def test_repr_includes_tuple(self):
        node = cst.Name("x")
        fp = node.structural_fingerprint()
        r = repr(fp)
        assert r.startswith("StructuralFingerprint(")
