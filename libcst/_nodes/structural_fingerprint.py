# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from typing import Sequence

from libcst._maybe_sentinel import MaybeSentinel
from libcst._nodes.base import CSTNode


def structural_fingerprint(node: CSTNode) -> "StructuralFingerprint":
    return StructuralFingerprint(_fingerprint_cst_node(node))


def _fingerprint_cst_node(node: CSTNode) -> tuple:
    parts = [type(node).__qualname__]
    for f in (f for f in fields(node) if f.compare is True):
        val = getattr(node, f.name)
        parts.append(_fingerprint_value(val))
    return tuple(parts)


def _fingerprint_value(val: object) -> object:
    if isinstance(val, CSTNode):
        return _fingerprint_cst_node(val)
    if isinstance(val, MaybeSentinel):
        return ("MaybeSentinel", val.name)
    if isinstance(val, (str, bytes, bool, int, float)) or val is None:
        return val
    if isinstance(val, Sequence):
        return tuple(_fingerprint_value(v) for v in val)
    return val


class StructuralFingerprint:
    __slots__ = ("_tuple", "_hash")

    def __init__(self, t: tuple) -> None:
        object.__setattr__(self, "_tuple", t)
        object.__setattr__(self, "_hash", hash(t))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StructuralFingerprint):
            return self._tuple == other._tuple
        return NotImplemented

    def __repr__(self) -> str:
        return f"StructuralFingerprint({self._tuple!r})"
