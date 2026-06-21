# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Sequence

import libcst as cst
from libcst._flatten_sentinel import FlattenSentinel
from libcst._maybe_sentinel import MaybeSentinel
from libcst._nodes.base import _NOOPVisitor, CSTNode
from libcst._removal_sentinel import RemovalSentinel
from libcst.testing.utils import data_provider, UnitTest


_NON_ASCII_SRC = "变量 = '你好 \U0001f600'  # 中文注释"


class FingerprintTest(UnitTest):
    def test_public_api_exported(self) -> None:
        self.assertIn("fingerprint", cst.__all__)
        self.assertTrue(callable(cst.fingerprint))

    def test_method_exists_on_cstnode(self) -> None:
        node = cst.SimpleWhitespace("")
        self.assertTrue(callable(node.fingerprint))

    def test_returns_int(self) -> None:
        node = cst.SimpleWhitespace("")
        self.assertIsInstance(node.fingerprint(), int)
        self.assertIsInstance(cst.fingerprint(None), int)

    def test_identity_hash_unchanged(self) -> None:
        sw1 = cst.SimpleWhitespace("")
        sw2 = cst.SimpleWhitespace("")
        self.assertNotEqual(hash(sw1), hash(sw2))
        self.assertNotEqual(sw1, sw2)
        self.assertEqual(hash(sw1), id(sw1))

    def test_bool_not_confused_with_int(self) -> None:
        self.assertNotEqual(cst.fingerprint(True), cst.fingerprint(1))
        self.assertNotEqual(cst.fingerprint(False), cst.fingerprint(0))
        self.assertEqual(cst.fingerprint(True), cst.fingerprint(True))
        self.assertEqual(cst.fingerprint(False), cst.fingerprint(False))

    def test_str_not_confused_with_bytes(self) -> None:
        self.assertNotEqual(cst.fingerprint("hello"), cst.fingerprint(b"hello"))
        self.assertEqual(cst.fingerprint("hello"), cst.fingerprint("hello"))

    def test_none_fingerprint(self) -> None:
        self.assertEqual(cst.fingerprint(None), cst.fingerprint(None))

    def test_maybe_sentinel(self) -> None:
        self.assertEqual(
            cst.fingerprint(MaybeSentinel.DEFAULT),
            cst.fingerprint(MaybeSentinel.DEFAULT),
        )

    def test_removal_sentinel(self) -> None:
        self.assertEqual(
            cst.fingerprint(RemovalSentinel.REMOVE),
            cst.fingerprint(RemovalSentinel.REMOVE),
        )

    def test_flatten_sentinel(self) -> None:
        pass_node = cst.Pass()
        fs1 = FlattenSentinel([pass_node])
        fs2 = FlattenSentinel([pass_node.deep_clone()])
        self.assertEqual(cst.fingerprint(fs1), cst.fingerprint(fs2))

    def test_sequence_treated_uniformly(self) -> None:
        pass_node = cst.Pass()
        tup = (pass_node,)
        lst = [pass_node.deep_clone()]
        self.assertEqual(cst.fingerprint(tup), cst.fingerprint(lst))

    def test_empty_sequence(self) -> None:
        self.assertEqual(cst.fingerprint(()), cst.fingerprint([]))

    def test_nested_sequence(self) -> None:
        sw = cst.SimpleWhitespace(" ")
        nested1 = [(sw,), (sw.deep_clone(),)]
        nested2 = [[sw.deep_clone()], [sw]]
        self.assertEqual(cst.fingerprint(nested1), cst.fingerprint(nested2))

    def test_deep_clone_same_fingerprint(self) -> None:
        m = cst.parse_module("x = 1")
        self.assertEqual(m.fingerprint(), m.deep_clone().fingerprint())

    def test_multiple_deep_clone_same_fingerprint(self) -> None:
        m = cst.parse_module("x = 1\ny = 2\nz = 3")
        cloned = m.deep_clone().deep_clone().deep_clone()
        self.assertEqual(m.fingerprint(), cloned.fingerprint())

    def test_parse_twice_same_fingerprint(self) -> None:
        src = "def foo():\n    return 42\n"
        m1 = cst.parse_module(src)
        m2 = cst.parse_module(src)
        self.assertEqual(m1.fingerprint(), m2.fingerprint())

    def test_with_changes_revert_same_fingerprint(self) -> None:
        m = cst.parse_module("answer = 42")
        name_node = m.body[0].body[0].targets[0].target
        self.assertIsInstance(name_node, cst.Name)
        changed = name_node.with_changes(value="question")
        reverted = changed.with_changes(value="answer")
        self.assertEqual(name_node.fingerprint(), reverted.fingerprint())

    def test_noop_visitor_same_fingerprint(self) -> None:
        m = cst.parse_module("x = 1\ny = 2\n")
        visited = m.visit(_NOOPVisitor())
        self.assertEqual(m.fingerprint(), visited.fingerprint())

    def test_code_roundtrip_same_fingerprint(self) -> None:
        src = "def foo(x: int) -> str:\n    return str(x)\n"
        m1 = cst.parse_module(src)
        code = m1.code
        m2 = cst.parse_module(code)
        self.assertEqual(m1.fingerprint(), m2.fingerprint())

    def test_deep_equals_implies_same_fingerprint(self) -> None:
        sw1 = cst.SimpleWhitespace("")
        sw2 = cst.SimpleWhitespace("")
        self.assertTrue(sw1.deep_equals(sw2))
        self.assertEqual(sw1.fingerprint(), sw2.fingerprint())

        stmt1 = cst.SimpleStatementLine(body=[cst.Pass()])
        stmt2 = cst.SimpleStatementLine(body=(cst.Pass(),))
        self.assertTrue(stmt1.deep_equals(stmt2))
        self.assertEqual(stmt1.fingerprint(), stmt2.fingerprint())

    def test_different_content_different_fingerprint(self) -> None:
        sw1 = cst.SimpleWhitespace(" ")
        sw2 = cst.SimpleWhitespace("  ")
        self.assertNotEqual(sw1.fingerprint(), sw2.fingerprint())

    def test_non_ascii_source_stable(self) -> None:
        m1 = cst.parse_module(_NON_ASCII_SRC)
        m2 = cst.parse_module(_NON_ASCII_SRC)
        self.assertEqual(m1.fingerprint(), m2.fingerprint())

    def test_pickle_roundtrip(self) -> None:
        m = cst.parse_module("x = 1\ny = 2\n")
        pickled = pickle.dumps(m)
        m2 = pickle.loads(pickled)
        self.assertEqual(m.fingerprint(), m2.fingerprint())

    def test_compare_false_field_skipped(self) -> None:
        @dataclass(frozen=True)
        class _FakeNode(CSTNode):
            real: str
            hidden: str = field(default="secret", compare=False)

            def _visit_and_replace_children(self, visitor):
                return self

            def _codegen_impl(self, state):
                state.add_token(self.real)

        a = _FakeNode(real="hello")
        b = _FakeNode(real="hello", hidden="different")
        self.assertTrue(a.deep_equals(b))
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_cross_module_same_name_different_fingerprint(self) -> None:
        @dataclass(frozen=True)
        class _LocalSimpleWhitespace(CSTNode):
            value: str

            def _visit_and_replace_children(self, visitor):
                return self

            def _codegen_impl(self, state):
                state.add_token(self.value)

        fake = _LocalSimpleWhitespace(value="")
        real = cst.SimpleWhitespace("")
        self.assertNotEqual(type(fake).__module__, type(real).__module__)
        self.assertNotEqual(fake.fingerprint(), real.fingerprint())

    def test_cross_process_different_pythonhashseed(self) -> None:
        test_script = f"""
import libcst as cst
src = {repr(_NON_ASCII_SRC + '\\ndef foo(x):\\n    return x + 1\\n')}
m = cst.parse_module(src)
print(m.fingerprint())
"""
        env1 = os.environ.copy()
        env1["PYTHONHASHSEED"] = "1"
        result1 = subprocess.run(
            [sys.executable, "-c", test_script],
            env=env1,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result1.returncode, 0, msg=result1.stderr)
        fp1 = int(result1.stdout.strip())

        env2 = os.environ.copy()
        env2["PYTHONHASHSEED"] = "4294967295"
        result2 = subprocess.run(
            [sys.executable, "-c", test_script],
            env=env2,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result2.returncode, 0, msg=result2.stderr)
        fp2 = int(result2.stdout.strip())

        self.assertEqual(fp1, fp2)

    def test_function_and_method_same_result(self) -> None:
        m = cst.parse_module("x = 1")
        self.assertEqual(cst.fingerprint(m), m.fingerprint())

    def test_optional_field_none(self) -> None:
        el1 = cst.EmptyLine(comment=None)
        el2 = cst.EmptyLine(comment=None)
        self.assertTrue(el1.deep_equals(el2))
        self.assertEqual(el1.fingerprint(), el2.fingerprint())
