# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Provides the implementation of structural fingerprint for CSTNode.

The fingerprint is:
- Cross-process deterministic: same code yields same fingerprint regardless
  of PYTHONHASHSEED
- Independent of Python's builtin hash()
- Consistent with deep_equals: two nodes that are deep_equals have the
  same fingerprint
- Type-aware: distinguishes nodes from different modules even if they
  share a class name
"""

from dataclasses import fields
from typing import Sequence

from libcst._flatten_sentinel import FlattenSentinel
from libcst._maybe_sentinel import MaybeSentinel
from libcst._nodes.base import CSTNode
from libcst._removal_sentinel import RemovalSentinel


_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = 0xFFFFFFFFFFFFFFFF


def _fnv1a_64(data: bytes) -> int:
    """
    FNV-1a 64-bit hash. Deterministic across all Python processes.
    Does not use Python's builtin hash().
    """
    h = _FNV_OFFSET_BASIS
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _MASK
    return h


def _combine(h1: int, h2: int) -> int:
    """
    Combine two hashes in a way that is order-dependent.
    """
    h = (h1 ^ h2) & _MASK
    h = (h * _FNV_PRIME) & _MASK
    return h


_PREFIX_NONE = b"\x00"
_PREFIX_BOOL = b"\x01"
_PREFIX_INT = b"\x02"
_PREFIX_BYTES = b"\x03"
_PREFIX_STR = b"\x04"
_PREFIX_MAYBE_SENTINEL = b"\x05"
_PREFIX_REMOVAL_SENTINEL = b"\x06"
_PREFIX_FLATTEN_SENTINEL = b"\x07"
_PREFIX_SEQUENCE = b"\x08"
_PREFIX_CST_NODE = b"\x09"


def _hash_int(value: int) -> int:
    """
    Hash an integer deterministically.
    """
    if value == 0:
        h = _fnv1a_64(_PREFIX_INT)
        return _combine(h, _fnv1a_64(b"\x00"))
    sign = 1 if value >= 0 else 0
    absval = abs(value)
    byte_length = (absval.bit_length() + 7) // 8
    data = bytes([sign]) + absval.to_bytes(byte_length, "big")
    h = _fnv1a_64(_PREFIX_INT)
    return _combine(h, _fnv1a_64(data))


def _hash_bool(value: bool) -> int:
    """
    Hash a boolean. Must be distinct from int hashing.
    """
    h = _fnv1a_64(_PREFIX_BOOL)
    return _combine(h, _fnv1a_64(b"\x01" if value else b"\x00"))


def _hash_none() -> int:
    """
    Hash None.
    """
    return _fnv1a_64(_PREFIX_NONE)


def _hash_bytes(value: bytes) -> int:
    """
    Hash bytes.
    """
    h = _fnv1a_64(_PREFIX_BYTES)
    h = _combine(h, _hash_int(len(value)))
    h = _combine(h, _fnv1a_64(value))
    return h


def _hash_str(value: str) -> int:
    """
    Hash a string deterministically using UTF-8 encoding.
    """
    encoded = value.encode("utf-8")
    h = _fnv1a_64(_PREFIX_STR)
    h = _combine(h, _hash_int(len(encoded)))
    h = _combine(h, _fnv1a_64(encoded))
    return h


def _hash_type_identity(cls: type) -> int:
    """
    Hash a type's identity including its module and qualified name.
    This ensures types from different modules with the same name get
    different fingerprints.
    """
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", cls.__name__)
    h = _hash_str(module)
    h = _combine(h, _hash_str(qualname))
    return h


def _hash_maybe_sentinel(value: MaybeSentinel) -> int:
    """
    Hash MaybeSentinel.
    """
    h = _fnv1a_64(_PREFIX_MAYBE_SENTINEL)
    h = _combine(h, _hash_str(value.name))
    return h


def _hash_removal_sentinel(value: RemovalSentinel) -> int:
    """
    Hash RemovalSentinel.
    """
    h = _fnv1a_64(_PREFIX_REMOVAL_SENTINEL)
    h = _combine(h, _hash_str(value.name))
    return h


def _hash_flatten_sentinel(value: "FlattenSentinel[object]") -> int:
    """
    Hash FlattenSentinel by hashing its contents as a sequence.
    """
    h = _fnv1a_64(_PREFIX_FLATTEN_SENTINEL)
    h = _combine(h, fingerprint_sequence(value))
    return h


def fingerprint_sequence(seq: Sequence[object]) -> int:
    """
    Hash a sequence deterministically, considering order and length.
    Sequences (except str and bytes) are treated uniformly like deep_equals does.
    """
    h = _fnv1a_64(_PREFIX_SEQUENCE)
    h = _combine(h, _hash_int(len(seq)))
    for item in seq:
        h = _combine(h, fingerprint(item))
    return h


def _fingerprint_cst_node(node: "CSTNode") -> int:
    """
    Hash a CSTNode by its type identity and compare-True fields.
    Only fields marked compare=True (the default) participate, matching
    deep_equals semantics.
    """
    h = _fnv1a_64(_PREFIX_CST_NODE)
    h = _combine(h, _hash_type_identity(type(node)))
    for field in (f for f in fields(node) if f.compare is True):
        field_value = getattr(node, field.name)
        h = _combine(h, _hash_str(field.name))
        h = _combine(h, fingerprint(field_value))
    return h


def fingerprint(value: object) -> int:
    """
    Compute a cross-process deterministic structural fingerprint.

    The fingerprint is consistent with :func:`deep_equals`: two values
    for which ``deep_equals(a, b)`` returns ``True`` will have the same
    fingerprint. Collisions are possible (like any hash function) but
    unlikely for normal inputs.

    The fingerprint does not depend on Python's :func:`hash`, object
    identity, or ``PYTHONHASHSEED``. It is safe to use across processes.

    Supported types:
    - ``None``
    - :class:`bool` (distinguished from :class:`int`)
    - :class:`int`
    - :class:`str` (UTF-8 encoded)
    - :class:`bytes`
    - :class:`Sequence` (list, tuple, etc.; excludes str/bytes)
    - :class:`MaybeSentinel`
    - :class:`RemovalSentinel`
    - :class:`FlattenSentinel`
    - :class:`CSTNode` (including all subclasses)
    """
    if value is None:
        return _hash_none()
    if isinstance(value, bool):
        return _hash_bool(value)
    if isinstance(value, int):
        return _hash_int(value)
    if isinstance(value, bytes):
        return _hash_bytes(value)
    if isinstance(value, str):
        return _hash_str(value)
    if isinstance(value, MaybeSentinel):
        return _hash_maybe_sentinel(value)
    if isinstance(value, RemovalSentinel):
        return _hash_removal_sentinel(value)
    if isinstance(value, FlattenSentinel):
        return _hash_flatten_sentinel(value)
    if isinstance(value, CSTNode):
        return _fingerprint_cst_node(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return fingerprint_sequence(value)
    raise TypeError(
        f"Cannot fingerprint object of type {type(value).__name__!r}"
    )


CSTNode.fingerprint = fingerprint
